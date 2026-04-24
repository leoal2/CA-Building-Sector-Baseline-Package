from __future__ import annotations

import argparse
import os
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import USEEIO as EIO
import rpy2.robjects as ro
from rpy2.robjects.packages import importr


US_MODEL_NAME = "bea_model_us_detailed_2017"
US_BEA_YEAR = 2017
US_GHG_YEAR = 2017

CA_MODEL_NAME = "bea_model_ca_summary_2022_after_TRACI"
CA_BEA_YEAR = 2022
CA_GHG_YEAR = 2022


def log(msg: str) -> None:
    print(msg, flush=True)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def standardize_bea_detail(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    s = s.str.replace(r"/(US|US-CA|RoUS|CA)$", "", regex=True)
    s = s.str.split("/", n=1).str[0].str.strip()
    return s


def tv_distance(p: np.ndarray, q: np.ndarray) -> float:
    return 0.5 * float(np.sum(np.abs(p - q)))


def setup_r(r_home: str, stateior_datadir: str):
    os.environ["R_HOME"] = r_home
    os.environ["STATEIOR_DATADIR"] = stateior_datadir
    os.environ["PATH"] = os.path.join(r_home, "bin", "x64") + ";" + os.environ.get("PATH", "")
    ro.r(f'Sys.setenv(STATEIOR_DATADIR = "{stateior_datadir.replace(os.sep, "/")}")')
    useeior = importr("useeior")
    return useeior


def load_model(name: str, bea_year: int, ghg_year: int, region: str = "US", detailed: bool = True):
    return EIO.Results(
        name=name,
        bea_year=bea_year,
        ghg_year=ghg_year,
        region=None if region == "US" else region,
        detailed=detailed,
        preserve=True,
    )


def load_dnb_cube(path: str) -> pd.DataFrame:
    raw = pd.read_excel(path, header=None, dtype=object)
    state_pat = re.compile(r"^US\d{2}:\s")

    best_row = None
    best_count = -1
    for r in range(min(50, raw.shape[0])):
        vals = raw.iloc[r, :].astype(str)
        cnt = int(vals.str.match(state_pat).sum())
        if cnt > best_count:
            best_row = r
            best_count = cnt

    if best_row is None or best_count < 10:
        raise ValueError("Could not identify the DnB state-header row.")

    state_row = best_row
    subheader_row = state_row + 1

    state_blocks: list[tuple[int, str]] = []
    for c in range(raw.shape[1]):
        v = raw.iat[state_row, c]
        if isinstance(v, str) and state_pat.match(v.strip()):
            state_blocks.append((c, v.strip()))

    ca_block = [b for b in state_blocks if "US06: CALIFORNIA" in b[1]]
    if not ca_block:
        raise ValueError("California block not found in DnB file.")
    ca_col = ca_block[0][0]

    sales_cols: list[tuple[int, str, int]] = []
    for start_col, label in state_blocks:
        window = raw.iloc[subheader_row, start_col:start_col + 6].astype(str).str.lower()
        sales_col = None
        for j, w in enumerate(window):
            if "sum(sales" in w:
                sales_col = start_col + j
                break
        if sales_col is not None:
            sales_cols.append((start_col, label, sales_col))

    if len(sales_cols) < 10:
        raise ValueError("Could not identify enough sales columns in DnB file.")

    ca_sales_col = [x for x in sales_cols if x[0] == ca_col][0][2]

    naics_col = 0
    probe_row = state_row - 2 if state_row >= 2 else state_row
    for c in range(min(10, raw.shape[1])):
        v = raw.iat[probe_row, c]
        if isinstance(v, str) and "axis" in v.lower():
            naics_col = c
            break

    data = raw.iloc[subheader_row + 1 :, :].copy()
    naics6 = data.iloc[:, naics_col].astype(str).str.extract(r"^\s*(\d{6})\s*:", expand=False)
    sales = data.iloc[:, [t[2] for t in sales_cols]].apply(pd.to_numeric, errors="coerce")
    ca_sales_series = pd.to_numeric(data.iloc[:, ca_sales_col], errors="coerce")

    out = pd.DataFrame(
        {
            "NAICS6": naics6,
            "DnB_US": sales.sum(axis=1, skipna=True),
            "DnB_CA": ca_sales_series,
        }
    )
    out = out.dropna(subset=["NAICS6"]).copy()
    out["NAICS6"] = out["NAICS6"].astype(str).str.zfill(6)
    out["DnB_RoUS"] = out["DnB_US"] - out["DnB_CA"]
    out = out[(out["DnB_US"].notna()) & (out["DnB_US"] != 0)].copy()
    out = out.groupby("NAICS6", as_index=False)[["DnB_US", "DnB_CA", "DnB_RoUS"]].sum()
    return out


def load_detailed_concordance(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str) if path.lower().endswith(".csv") else pd.read_excel(path, dtype=str)
    df.columns = [c.strip() for c in df.columns]

    naics_col = None
    for cand in ["2022 NAICS", "2017 NAICS", "NAICS6", "NAICS"]:
        if cand in df.columns:
            naics_col = cand
            break
    if naics_col is None:
        raise ValueError(f"No NAICS column found in concordance: {df.columns.tolist()}")
    for c in ["BEA_Detail", "BEA_Summary"]:
        if c not in df.columns:
            raise ValueError(f"Concordance missing required column: {c}")

    df = df[[naics_col, "BEA_Detail", "BEA_Summary"]].copy()
    df = df.rename(columns={naics_col: "NAICS6"})
    df["NAICS6"] = df["NAICS6"].astype(str).str.extract(r"(\d{6})", expand=False)
    df = df.dropna(subset=["NAICS6", "BEA_Detail", "BEA_Summary"]).copy()
    df["NAICS6"] = df["NAICS6"].astype(str).str.zfill(6)
    df["BEA_Detail"] = standardize_bea_detail(df["BEA_Detail"])
    df["BEA_Summary"] = df["BEA_Summary"].astype(str).str.strip()
    df = df.drop_duplicates(subset=["NAICS6", "BEA_Detail", "BEA_Summary"]).copy()
    n = df.groupby("NAICS6")["BEA_Detail"].transform("size")
    df["MapWeight"] = np.where(n > 0, 1.0 / n, 0.0)
    return df


def load_x_us_detail(us_model) -> pd.DataFrame:
    x_df = pd.DataFrame(us_model.x)
    x_df.index = x_df.index.astype(str)
    x_df.columns = ["x_2017"]
    x_df = x_df.reset_index().rename(columns={"index": "NAICS_Code"})
    x_df["BEA_Detail"] = standardize_bea_detail(x_df["NAICS_Code"])
    x_df["x_2017"] = pd.to_numeric(x_df["x_2017"], errors="coerce").fillna(0.0)

    cpi_matrix = us_model._model.rx2("MultiYearCommodityCPI")
    cpi_array = np.array(cpi_matrix).T
    years = list(cpi_matrix.colnames)
    sectors = list(cpi_matrix.rownames)
    idx_2017 = years.index("2017")
    idx_2022 = years.index("2022")

    ratios = {
        f"{code}/US": float(cpi_array[i, idx_2022] / cpi_array[i, idx_2017])
        for i, code in enumerate(sectors)
        if np.isfinite(cpi_array[i, idx_2017]) and cpi_array[i, idx_2017] != 0
    }

    x_df["X_US"] = x_df["x_2017"] * x_df["NAICS_Code"].map(ratios).fillna(1.0)
    return x_df.groupby("BEA_Detail", as_index=False)["X_US"].sum()


def load_x_ca_summary(ca_model) -> pd.DataFrame:
    x_df = pd.DataFrame(ca_model.x)
    x_df.index = x_df.index.astype(str)
    x_df.columns = ["x"]
    x_df = x_df.reset_index().rename(columns={"index": "Key"})
    x_df["BEA_Summary"] = x_df["Key"].str.split("/", n=1, expand=True)[0].str.strip()
    x_df["Region"] = x_df["Key"].str.split("/", n=1, expand=True)[1].str.strip()
    x_df = x_df[x_df["Region"].isin(["US-CA", "RoUS"])].copy()
    x_df["x"] = pd.to_numeric(x_df["x"], errors="coerce").fillna(0.0)
    return x_df.groupby(["BEA_Summary", "Region"], as_index=False)["x"].sum()


def load_d_us_detail(useeior, us_model) -> pd.DataFrame:
    adjusted = useeior.adjustResultMatrixPrice("D", 2022, False, us_model._model)
    d_array = np.array(adjusted)
    sector_keys = list(us_model._model.rx2["L"].rownames)
    if d_array.ndim != 2 or d_array.shape[0] != 1:
        raise ValueError(f"Unexpected shape for adjusted D: {d_array.shape}")

    out = pd.DataFrame(
        {
            "BEA_Detail": standardize_bea_detail(pd.Series(sector_keys)),
            "D_US": pd.to_numeric(d_array[0, :], errors="coerce"),
        }
    )
    out["D_US"] = out["D_US"].fillna(0.0)
    return out.groupby("BEA_Detail", as_index=False)["D_US"].mean()


def pick_ghg_row(df: pd.DataFrame):
    idx = df.index.astype(str)
    idx_lower = pd.Index([str(x).lower() for x in idx])

    preferred = [
        "greenhouse gases",
        "greenhouse gas",
        "ghg",
        "global warming",
        "climate change",
    ]
    for p in preferred:
        mask = idx_lower.str.contains(p, regex=False)
        if mask.any():
            return idx[mask.argmax()]

    return df.index[0]


def load_summary_d_from_ca_model(ca_model) -> pd.DataFrame:
    if ca_model.D is None:
        raise ValueError("CA model did not provide D.")

    d_raw = pd.DataFrame(ca_model.D)
    if d_raw.empty:
        raise ValueError("CA model D is empty.")

    if d_raw.shape[0] > 1:
        ghg_row = pick_ghg_row(d_raw)
        d = d_raw.loc[ghg_row].astype(float)
        log(f"Selected GHG row in CA summary D: {ghg_row}")
    else:
        d = d_raw.iloc[0].astype(float)

    d.index = d.index.astype(str).str.strip()

    out = pd.DataFrame(
        {
            "Key": d.index,
            "D": pd.to_numeric(d.values, errors="coerce"),
        }
    )

    out = out.dropna(subset=["Key"]).copy()
    out["BEA_Summary"] = out["Key"].str.split("/", n=1, expand=True)[0].str.strip()
    out["Region"] = out["Key"].str.split("/", n=1, expand=True)[1].str.strip()
    out = out[out["Region"].isin(["US-CA", "RoUS"])].copy()
    out["D"] = pd.to_numeric(out["D"], errors="coerce").fillna(0.0)
    out = out.groupby(["BEA_Summary", "Region"], as_index=False)["D"].mean()

    if out.empty:
        raise ValueError("Parsed CA summary D is empty after filtering to US-CA/RoUS.")

    return out


def load_mrr_emissions(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    df.columns = [c.strip() for c in df.columns]
    if "BEA_Detail" not in df.columns or "MRR_Emissions" not in df.columns:
        raise ValueError("MRR file must contain BEA_Detail and MRR_Emissions.")
    df["BEA_Detail"] = standardize_bea_detail(df["BEA_Detail"])
    df["MRR_kg"] = pd.to_numeric(df["MRR_Emissions"], errors="coerce").fillna(0.0) * 1000.0
    return df.groupby("BEA_Detail", as_index=False)["MRR_kg"].sum()


def build_backbone(dnb: pd.DataFrame, concordance: pd.DataFrame, x_us: pd.DataFrame) -> pd.DataFrame:
    tmp = concordance.merge(dnb, on="NAICS6", how="left")
    tmp[["DnB_US", "DnB_CA", "DnB_RoUS"]] = tmp[["DnB_US", "DnB_CA", "DnB_RoUS"]].fillna(0.0)

    for col in ["DnB_US", "DnB_CA", "DnB_RoUS"]:
        tmp[col] = tmp[col] * tmp["MapWeight"]

    out = tmp.groupby(["BEA_Summary", "BEA_Detail"], as_index=False)[["DnB_US", "DnB_CA", "DnB_RoUS"]].sum()
    out = out.merge(x_us, on="BEA_Detail", how="left")
    out["X_US"] = out["X_US"].fillna(0.0)
    return out


def compute_us_proxy_intensity(backbone: pd.DataFrame, d_us: pd.DataFrame) -> pd.DataFrame:
    tmp = backbone[["BEA_Summary", "BEA_Detail", "X_US"]].merge(d_us, on="BEA_Detail", how="left")
    tmp["D_US"] = tmp["D_US"].fillna(0.0)
    tmp["num"] = tmp["D_US"] * tmp["X_US"]
    out = tmp.groupby("BEA_Summary", as_index=False).agg(X_US_sum=("X_US", "sum"), num=("num", "sum"))
    out["D_US_proxy"] = np.where(out["X_US_sum"] > 0, out["num"] / out["X_US_sum"], 0.0)
    return out[["BEA_Summary", "D_US_proxy"]]


def compute_shares(backbone: pd.DataFrame, alpha_ca: float, alpha_rous_max: float, tv_cutoff: float) -> pd.DataFrame:
    out = backbone.copy()
    grp = out.groupby("BEA_Summary", as_index=False).agg(
        DnB_CA_sum=("DnB_CA", "sum"),
        DnB_RoUS_sum=("DnB_RoUS", "sum"),
        X_US_sum=("X_US", "sum"),
    )
    out = out.merge(grp, on="BEA_Summary", how="left")

    def share(num, den):
        return np.where(den > 0, num / den, 0.0)

    out["X_Share_US"] = share(out["X_US"], out["X_US_sum"])
    out["DnB_Share_CA"] = share(out["DnB_CA"], out["DnB_CA_sum"])
    out["DnB_Share_RoUS"] = share(out["DnB_RoUS"], out["DnB_RoUS_sum"])

    tv_rows = []
    for summary, df in out.groupby("BEA_Summary"):
        tv_rows.append(
            {
                "BEA_Summary": summary,
                "TV_CA_vs_US": tv_distance(
                    df["DnB_Share_CA"].to_numpy(dtype=float),
                    df["X_Share_US"].to_numpy(dtype=float),
                ),
                "TV_RoUS_vs_US": tv_distance(
                    df["DnB_Share_RoUS"].to_numpy(dtype=float),
                    df["X_Share_US"].to_numpy(dtype=float),
                ),
            }
        )

    out = out.merge(pd.DataFrame(tv_rows), on="BEA_Summary", how="left")

    def alpha_row(row: pd.Series, region: str) -> float:
        dnb_sum = row["DnB_CA_sum"] if region == "CA" else row["DnB_RoUS_sum"]
        x_sum = row["X_US_sum"]
        tv = row["TV_CA_vs_US"] if region == "CA" else row["TV_RoUS_vs_US"]

        if dnb_sum <= 0 and x_sum > 0:
            return 0.0
        if x_sum <= 0 and dnb_sum > 0:
            return 1.0
        if x_sum <= 0 and dnb_sum <= 0:
            return 0.0

        max_alpha = alpha_ca if region == "CA" else alpha_rous_max
        return max_alpha if tv <= tv_cutoff else 0.0

    out["alpha_CA"] = out.apply(lambda r: alpha_row(r, "CA"), axis=1)
    out["alpha_RoUS"] = out.apply(lambda r: alpha_row(r, "RoUS"), axis=1)
    out["CA_Share_Used"] = out["alpha_CA"] * out["DnB_Share_CA"] + (1.0 - out["alpha_CA"]) * out["X_Share_US"]
    out["RoUS_Share_Used"] = out["alpha_RoUS"] * out["DnB_Share_RoUS"] + (1.0 - out["alpha_RoUS"]) * out["X_Share_US"]

    for col in ["CA_Share_Used", "RoUS_Share_Used"]:
        s = out.groupby("BEA_Summary")[col].transform("sum")
        out[col] = np.where(s > 0, out[col] / s, 0.0)

    return out


def compute_detailed_outputs(backbone: pd.DataFrame, x_ca: pd.DataFrame) -> pd.DataFrame:
    out = backbone.copy()

    usca = x_ca[x_ca["Region"] == "US-CA"][["BEA_Summary", "x"]].rename(columns={"x": "X_CA_Summary_US_CA"})
    rous = x_ca[x_ca["Region"] == "RoUS"][["BEA_Summary", "x"]].rename(columns={"x": "X_CA_Summary_RoUS"})

    out = out.merge(usca, on="BEA_Summary", how="left")
    out = out.merge(rous, on="BEA_Summary", how="left")
    out[["X_CA_Summary_US_CA", "X_CA_Summary_RoUS"]] = out[["X_CA_Summary_US_CA", "X_CA_Summary_RoUS"]].fillna(0.0)

    out["DetailedOutput_CA_CA"] = out["X_CA_Summary_US_CA"] * out["CA_Share_Used"]
    out["DetailedOutput_CA_RoUS"] = out["X_CA_Summary_RoUS"] * out["RoUS_Share_Used"]
    return out


def build_emissions_targets(
    summary_d: pd.DataFrame,
    x_ca: pd.DataFrame,
    mrr_detail: pd.DataFrame,
    concordance: pd.DataFrame,
    us_proxy: pd.DataFrame,
) -> pd.DataFrame:
    base = x_ca.rename(columns={"x": "X_summary"}).merge(
        summary_d.rename(columns={"D": "D_model"}),
        on=["BEA_Summary", "Region"],
        how="left",
    )
    base["D_model"] = base["D_model"].fillna(0.0)
    base["E_model_kg"] = base["D_model"] * base["X_summary"]

    det_to_sum = concordance[["BEA_Detail", "BEA_Summary"]].drop_duplicates()
    mrr_sum = (
        mrr_detail.merge(det_to_sum, on="BEA_Detail", how="left")
        .groupby("BEA_Summary", as_index=False)["MRR_kg"]
        .sum()
    )

    ca = base[base["Region"] == "US-CA"][["BEA_Summary", "X_summary", "E_model_kg"]].merge(
        mrr_sum, on="BEA_Summary", how="left"
    )
    ca["MRR_kg"] = ca["MRR_kg"].fillna(0.0)
    ca = ca.merge(us_proxy, on="BEA_Summary", how="left")
    ca["D_US_proxy"] = ca["D_US_proxy"].fillna(0.0)
    ca["E_target_CA_kg"] = np.where(ca["MRR_kg"] > 0, ca["MRR_kg"], ca["E_model_kg"])

    needs_proxy = (
        (ca["MRR_kg"] <= 0)
        & (ca["E_model_kg"] <= 0)
        & (ca["X_summary"] > 0)
        & (ca["D_US_proxy"] > 0)
    )
    ca.loc[needs_proxy, "E_target_CA_kg"] = ca.loc[needs_proxy, "D_US_proxy"] * ca.loc[needs_proxy, "X_summary"]

    ro = base[base["Region"] == "RoUS"][["BEA_Summary", "X_summary", "E_model_kg"]].merge(
        us_proxy, on="BEA_Summary", how="left"
    )
    ro["D_US_proxy"] = ro["D_US_proxy"].fillna(0.0)
    ro["E_target_RoUS_kg"] = ro["E_model_kg"].fillna(0.0)

    needs_proxy_ro = (
        (ro["E_target_RoUS_kg"] <= 0)
        & (ro["X_summary"] > 0)
        & (ro["D_US_proxy"] > 0)
    )
    ro.loc[needs_proxy_ro, "E_target_RoUS_kg"] = ro.loc[needs_proxy_ro, "D_US_proxy"] * ro.loc[needs_proxy_ro, "X_summary"]

    out = ca[["BEA_Summary", "E_target_CA_kg"]].merge(
        ro[["BEA_Summary", "E_target_RoUS_kg"]],
        on="BEA_Summary",
        how="outer",
    )
    out[["E_target_CA_kg", "E_target_RoUS_kg"]] = out[["E_target_CA_kg", "E_target_RoUS_kg"]].fillna(0.0)
    return out


def calibrate_detailed_d(df: pd.DataFrame, d_us: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    out = df.merge(d_us, on="BEA_Detail", how="left").merge(targets, on="BEA_Summary", how="left")
    out["D_US"] = out["D_US"].fillna(0.0)
    out[["E_target_CA_kg", "E_target_RoUS_kg"]] = out[["E_target_CA_kg", "E_target_RoUS_kg"]].fillna(0.0)

    def scale_group(grp: pd.DataFrame, out_col: str, tgt_col: str, dcal_col: str) -> pd.DataFrame:
        outputs = pd.to_numeric(grp[out_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        dbase = pd.to_numeric(grp["D_US"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        target = float(pd.to_numeric(grp[tgt_col], errors="coerce").fillna(0.0).iloc[0])
        total_output = float(outputs.sum())
        baseline = float((dbase * outputs).sum())

        if target <= 0.0 or total_output <= 0.0:
            dcal = np.zeros_like(dbase, dtype=float)
        elif baseline > 0.0:
            dcal = dbase * (target / baseline)
        else:
            uniform = target / total_output
            dcal = np.where(outputs > 0.0, uniform, 0.0)

        grp[dcal_col] = np.where(dcal < 0.0, 0.0, dcal)
        return grp

    chunks = []
    for _, grp in out.groupby("BEA_Summary", sort=False):
        grp = scale_group(grp.copy(), "DetailedOutput_CA_CA", "E_target_CA_kg", "D_cal_CA")
        grp = scale_group(grp, "DetailedOutput_CA_RoUS", "E_target_RoUS_kg", "D_cal_RoUS")
        chunks.append(grp)

    out = pd.concat(chunks, ignore_index=True)
    return out


def build_outputs(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = df.copy()

    out["CA_Output_Weight"] = np.where(
        out["X_CA_Summary_US_CA"] > 0,
        out["DetailedOutput_CA_CA"] / out["X_CA_Summary_US_CA"],
        0.0,
    )
    out["RoUS_Output_Weight"] = np.where(
        out["X_CA_Summary_RoUS"] > 0,
        out["DetailedOutput_CA_RoUS"] / out["X_CA_Summary_RoUS"],
        0.0,
    )

    out["CA_Direct_Emissions"] = out["D_cal_CA"] * out["DetailedOutput_CA_CA"]
    out["RoUS_Direct_Emissions"] = out["D_cal_RoUS"] * out["DetailedOutput_CA_RoUS"]

    sum_em_ca = out.groupby("BEA_Summary")["CA_Direct_Emissions"].transform("sum")
    sum_em_ro = out.groupby("BEA_Summary")["RoUS_Direct_Emissions"].transform("sum")

    out["CA_Emissions_Weight"] = np.where(
        sum_em_ca > 0,
        out["CA_Direct_Emissions"] / sum_em_ca,
        out["CA_Output_Weight"],
    )
    out["RoUS_Emissions_Weight"] = np.where(
        sum_em_ro > 0,
        out["RoUS_Direct_Emissions"] / sum_em_ro,
        out["RoUS_Output_Weight"],
    )

    ca_cols = [
        "BEA_Summary",
        "BEA_Detail",
        "DnB_Share_CA",
        "X_Share_US",
        "alpha_CA",
        "CA_Share_Used",
        "DetailedOutput_CA_CA",
        "D_cal_CA",
        "CA_Direct_Emissions",
        "CA_Output_Weight",
        "CA_Emissions_Weight",
    ]
    ro_cols = [
        "BEA_Summary",
        "BEA_Detail",
        "DnB_Share_RoUS",
        "X_Share_US",
        "alpha_RoUS",
        "RoUS_Share_Used",
        "DetailedOutput_CA_RoUS",
        "D_cal_RoUS",
        "RoUS_Direct_Emissions",
        "RoUS_Output_Weight",
        "RoUS_Emissions_Weight",
    ]

    return out[ca_cols].copy(), out[ro_cols].copy()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build public-safe DnB-derived CA and RoUS weights.")
    p.add_argument("--dnb-raw", required=True)
    p.add_argument("--detailed-concordance", required=True)
    p.add_argument("--mrr-emissions", required=True)
    p.add_argument("--r-home", required=True)
    p.add_argument("--stateior-datadir", required=True)
    p.add_argument("--out-ca", required=True)
    p.add_argument("--out-rous", required=True)
    p.add_argument("--alpha-ca", type=float, default=1.0)
    p.add_argument("--alpha-rous-max", type=float, default=0.2)
    p.add_argument("--tv-cutoff", type=float, default=0.5)
    return p.parse_args()


def main() -> None:
    t0 = datetime.now()
    args = parse_args()

    useeior = setup_r(args.r_home, args.stateior_datadir)

    log("Loading USEEIO models")
    us_model = load_model(US_MODEL_NAME, US_BEA_YEAR, US_GHG_YEAR, region="US", detailed=True)
    ca_model = load_model(CA_MODEL_NAME, CA_BEA_YEAR, CA_GHG_YEAR, region="CA", detailed=False)

    log("Loading inputs")
    dnb = load_dnb_cube(args.dnb_raw)
    concordance = load_detailed_concordance(args.detailed_concordance)
    mrr = load_mrr_emissions(args.mrr_emissions)
    x_us = load_x_us_detail(us_model)
    x_ca = load_x_ca_summary(ca_model)
    d_us = load_d_us_detail(useeior, us_model)
    summary_d = load_summary_d_from_ca_model(ca_model)

    log("Building weights")
    backbone = build_backbone(dnb, concordance, x_us)
    us_proxy = compute_us_proxy_intensity(backbone, d_us)
    backbone = compute_shares(backbone, args.alpha_ca, args.alpha_rous_max, args.tv_cutoff)
    backbone = compute_detailed_outputs(backbone, x_ca)
    targets = build_emissions_targets(summary_d, x_ca, mrr, concordance, us_proxy)
    calibrated = calibrate_detailed_d(backbone, d_us, targets)
    ca_out, rous_out = build_outputs(calibrated)

    out_ca = Path(args.out_ca)
    out_rous = Path(args.out_rous)
    ensure_parent(out_ca)
    ensure_parent(out_rous)

    ca_out.to_csv(out_ca, index=False)
    rous_out.to_csv(out_rous, index=False)

    log(f"Wrote {out_ca}")
    log(f"Wrote {out_rous}")
    log(f"Done in {datetime.now() - t0}")


if __name__ == "__main__":
    main()