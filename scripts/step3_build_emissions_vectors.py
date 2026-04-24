from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import USEEIO as EIO


TWO_REGION_MODEL_NAME = "bea_model_ca_summary_2022_after_TRACI"
BEA_YEAR = 2022
GHG_YEAR = 2022
TARGET_YEAR = "2022"

CLEAN_NEGATIVE_ROUS_WEIGHTS = True


def log(msg: str) -> None:
    print(msg, flush=True)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def clean_naics(val) -> str:
    if pd.isna(val):
        return ""
    s = str(val).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s


def split_key(key: str) -> tuple[str, str]:
    s = str(key).strip()
    if "/" in s:
        a, b = s.split("/", 1)
        return a.strip(), b.strip()
    return s, ""


def standardize_bea_detail(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    s = s.str.replace(r"/(US|US-CA|RoUS|CA)$", "", regex=True)
    s = s.str.split("/", n=1).str[0].str.strip()
    return s


def as_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def setup_env(r_home: str, stateior_datadir: str) -> None:
    os.environ["R_HOME"] = r_home
    os.environ["STATEIOR_DATADIR"] = stateior_datadir
    os.environ["PATH"] = os.path.join(r_home, "bin", "x64") + ";" + os.environ.get("PATH", "")


def pick_ghg_row(df: pd.DataFrame) -> str:
    idx = [str(i) for i in df.index]
    patterns = [r"greenhouse", r"\bghg\b", r"co2", r"carbon"]
    for pat in patterns:
        for i in idx:
            if pd.notna(i) and __import__("re").search(pat, i, __import__("re").IGNORECASE):
                return i
    return str(df.index[0])


def load_step2_outputs(step2_file: str):
    xls = pd.ExcelFile(step2_file)
    required = ["CA_Detailed_L", "CA_Detailed_x", "CA_Detailed_f"]
    for s in required:
        if s not in xls.sheet_names:
            raise ValueError(f"Step 2 workbook missing required sheet: {s}")

    L = pd.read_excel(xls, sheet_name="CA_Detailed_L", index_col=0)
    x_raw = pd.read_excel(xls, sheet_name="CA_Detailed_x")
    f_raw = pd.read_excel(xls, sheet_name="CA_Detailed_f")

    if not {"NAICS_Code", "x"}.issubset(x_raw.columns):
        raise ValueError("CA_Detailed_x must contain columns NAICS_Code and x")
    if not {"NAICS_Code", "f"}.issubset(f_raw.columns):
        raise ValueError("CA_Detailed_f must contain columns NAICS_Code and f")

    L.index = L.index.astype(str).str.strip()
    L.columns = L.columns.astype(str).str.strip()
    x_raw["NAICS_Code"] = x_raw["NAICS_Code"].astype(str).str.strip()
    f_raw["NAICS_Code"] = f_raw["NAICS_Code"].astype(str).str.strip()

    x_aligned = x_raw.set_index("NAICS_Code")["x"].reindex(L.columns).fillna(0.0)
    f_aligned = f_raw.set_index("NAICS_Code")["f"].reindex(L.columns).fillna(0.0)

    x_det = x_aligned.reset_index().rename(columns={"index": "NAICS_Code", "x": "x"})
    f_det = f_aligned.reset_index().rename(columns={"index": "NAICS_Code", "f": "f"})
    for df in (x_det, f_det):
        df["Detail"] = df["NAICS_Code"].apply(lambda k: split_key(k)[0])
        df["Region"] = df["NAICS_Code"].apply(lambda k: split_key(k)[1])

    return L, x_det, f_det


def load_inventory(inventory_file: str) -> pd.DataFrame:
    inv = pd.read_excel(inventory_file)
    if "NAICS_Code" not in inv.columns:
        raise ValueError("Inventory file must contain NAICS_Code")
    if TARGET_YEAR not in inv.columns:
        raise ValueError(f"Inventory file missing column {TARGET_YEAR}")

    if "Is_Household_F010" in inv.columns:
        inv = inv.loc[inv["Is_Household_F010"] != 1].copy()

    inv["NAICS_str"] = inv["NAICS_Code"].apply(clean_naics)
    inv[TARGET_YEAR] = inv[TARGET_YEAR].apply(as_float)

    inv_agg = (
        inv.groupby("NAICS_str", as_index=False)[TARGET_YEAR]
        .sum()
        .rename(columns={TARGET_YEAR: "E_Inv_MMT"})
    )
    return inv_agg


def load_concordance(concordance_file: str) -> pd.DataFrame:
    cw = pd.read_csv(concordance_file)
    needed = ["2022 NAICS", "BEA_Detail", "BEA_Summary"]
    for c in needed:
        if c not in cw.columns:
            raise ValueError(f"Concordance missing required column: {c}")

    cw = cw[needed].copy()
    cw["NAICS6"] = cw["2022 NAICS"].apply(clean_naics)
    cw["BEA_Detail"] = standardize_bea_detail(cw["BEA_Detail"])
    cw["BEA_Summary"] = cw["BEA_Summary"].astype(str).str.strip()
    return cw[["NAICS6", "BEA_Detail", "BEA_Summary"]].drop_duplicates().copy()


def load_step1_weights(dnb_ca_csv: str, dnb_rous_csv: str):
    dnb_ca = pd.read_csv(dnb_ca_csv)
    dnb_rous = pd.read_csv(dnb_rous_csv)

    for col in ["BEA_Detail", "BEA_Summary", "CA_Emissions_Weight"]:
        if col not in dnb_ca.columns:
            raise ValueError(f"CA Step 1 CSV missing column: {col}")
    for col in ["BEA_Detail", "BEA_Summary", "RoUS_Emissions_Weight"]:
        if col not in dnb_rous.columns:
            raise ValueError(f"RoUS Step 1 CSV missing column: {col}")

    dnb_ca["BEA_Detail"] = standardize_bea_detail(dnb_ca["BEA_Detail"])
    dnb_ca["BEA_Summary"] = dnb_ca["BEA_Summary"].astype(str).str.strip()
    dnb_ca["CA_Emissions_Weight"] = pd.to_numeric(dnb_ca["CA_Emissions_Weight"], errors="coerce").fillna(0.0)

    dnb_rous["BEA_Detail"] = standardize_bea_detail(dnb_rous["BEA_Detail"])
    dnb_rous["BEA_Summary"] = dnb_rous["BEA_Summary"].astype(str).str.strip()
    dnb_rous["RoUS_Emissions_Weight"] = pd.to_numeric(dnb_rous["RoUS_Emissions_Weight"], errors="coerce").fillna(0.0)

    return dnb_ca, dnb_rous


def map_inventory_to_summary(inv_agg: pd.DataFrame, cw: pd.DataFrame, dnb_ca: pd.DataFrame) -> pd.DataFrame:
    w_detail = dnb_ca.set_index("BEA_Detail")["CA_Emissions_Weight"].to_dict()
    mapped = []

    for r in inv_agg.itertuples(index=False):
        na = r.NAICS_str
        E = float(r.E_Inv_MMT)

        children = cw[cw["NAICS6"].str.startswith(na)]
        if children.empty:
            continue

        sum_w = {}
        for _, ch in children.iterrows():
            bd = ch["BEA_Detail"]
            bs = ch["BEA_Summary"]
            w = w_detail.get(bd, np.nan)
            if pd.isna(w) or w <= 0:
                w = 1.0
            sum_w[bs] = sum_w.get(bs, 0.0) + float(w)

        total_w = float(sum(sum_w.values()))
        if total_w <= 0:
            uniq = sorted(children["BEA_Summary"].unique().tolist())
            for bs in uniq:
                mapped.append({"BEA_Summary": bs, "E_CA_Inv_MMT": E / len(uniq)})
        else:
            for bs, w in sum_w.items():
                mapped.append({"BEA_Summary": bs, "E_CA_Inv_MMT": E * (w / total_w)})

    if not mapped:
        return pd.DataFrame(columns=["BEA_Summary", "E_CA_Inv_MMT"])

    return pd.DataFrame(mapped).groupby("BEA_Summary", as_index=False)["E_CA_Inv_MMT"].sum()


def allocate_summary_to_detail(
    dnb_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    weight_col: str,
    summary_value_col: str,
    out_detail_value_col: str,
    clean_negatives: bool = False,
) -> pd.DataFrame:
    dnb = dnb_df.copy()
    dnb[weight_col] = pd.to_numeric(dnb[weight_col], errors="coerce").fillna(0.0)

    if clean_negatives:
        dnb.loc[dnb[weight_col] < 0, weight_col] = 0.0
        s = dnb.groupby("BEA_Summary")[weight_col].transform("sum")
        dnb[weight_col] = np.where(s > 0, dnb[weight_col] / s, 0.0)

    merged = dnb.merge(summary_df, on="BEA_Summary", how="left")
    merged[summary_value_col] = pd.to_numeric(merged[summary_value_col], errors="coerce").fillna(0.0)
    merged[out_detail_value_col] = merged[summary_value_col] * merged[weight_col]

    grp = merged.groupby("BEA_Summary", as_index=False).agg(
        total=(summary_value_col, "first"),
        allocated=(out_detail_value_col, "sum"),
        n_details=("BEA_Detail", "count"),
    )
    grp["gap"] = grp["total"] - grp["allocated"]
    needs = grp[(grp["gap"].abs() > 1e-9) & (grp["total"].abs() > 1e-12)].copy()

    extra_rows = []
    for r in needs.itertuples(index=False):
        bs = r.BEA_Summary
        total = float(r.total)
        block = merged[merged["BEA_Summary"] == bs]

        if block.empty:
            extra_rows.append(
                {
                    "BEA_Detail": f"UNALLOCATED_{bs}",
                    "BEA_Summary": bs,
                    out_detail_value_col: total,
                }
            )
            continue

        usable = block["BEA_Detail"].dropna().astype(str).tolist()
        usable = [u for u in usable if u]
        if not usable:
            extra_rows.append(
                {
                    "BEA_Detail": f"UNALLOCATED_{bs}",
                    "BEA_Summary": bs,
                    out_detail_value_col: total,
                }
            )
            continue

        equal_share = total / len(usable)
        merged.loc[merged["BEA_Summary"] == bs, out_detail_value_col] = 0.0
        for bd in usable:
            extra_rows.append(
                {
                    "BEA_Detail": bd,
                    "BEA_Summary": bs,
                    out_detail_value_col: equal_share,
                }
            )

    base = merged[["BEA_Detail", "BEA_Summary", out_detail_value_col]].copy()
    if extra_rows:
        extra = pd.DataFrame(extra_rows)
        out = pd.concat([base, extra], ignore_index=True)
        out = out.groupby(["BEA_Detail", "BEA_Summary"], as_index=False)[out_detail_value_col].sum()
    else:
        out = base.groupby(["BEA_Detail", "BEA_Summary"], as_index=False)[out_detail_value_col].sum()

    return out


def build_ca_detail_emissions(inv_summary: pd.DataFrame, dnb_ca: pd.DataFrame) -> pd.DataFrame:
    return allocate_summary_to_detail(
        dnb_df=dnb_ca,
        summary_df=inv_summary,
        weight_col="CA_Emissions_Weight",
        summary_value_col="E_CA_Inv_MMT",
        out_detail_value_col="E_CA_detail_MMT",
        clean_negatives=False,
    )


def extract_rous_summary_totals_mmt() -> pd.DataFrame:
    model = EIO.Results(
        name=TWO_REGION_MODEL_NAME,
        bea_year=BEA_YEAR,
        ghg_year=GHG_YEAR,
        region="CA",
        detailed=False,
        preserve=True,
    )

    if model.D is None or model.x is None:
        raise ValueError("Two-region summary model missing D or x")

    D = pd.DataFrame(model.D)
    x = model.x["x"].copy()

    x_aligned = x.reindex(D.columns).fillna(0.0)
    ghg_row = pick_ghg_row(D)
    d = D.loc[ghg_row].astype(float)

    e_mmt = (d * x_aligned) / 1e9
    keys = e_mmt.index.astype(str)
    mask_rous = pd.Series(keys).str.endswith("/RoUS").values
    e_rous = e_mmt[mask_rous]

    rous_df = pd.DataFrame(
        {
            "Key": e_rous.index.astype(str),
            "BEA_Summary": e_rous.index.astype(str).str.split("/").str[0],
            "E_RoUS_MMT": e_rous.values,
        }
    )
    return rous_df.groupby("BEA_Summary", as_index=False)["E_RoUS_MMT"].sum()


def build_rous_detail_emissions(rous_summary: pd.DataFrame, dnb_rous: pd.DataFrame) -> pd.DataFrame:
    return allocate_summary_to_detail(
        dnb_df=dnb_rous,
        summary_df=rous_summary,
        weight_col="RoUS_Emissions_Weight",
        summary_value_col="E_RoUS_MMT",
        out_detail_value_col="E_RoUS_detail_MMT",
        clean_negatives=CLEAN_NEGATIVE_ROUS_WEIGHTS,
    )


def build_D_from_detail_emissions(x_det: pd.DataFrame, ca_detail: pd.DataFrame, rous_detail: pd.DataFrame) -> pd.DataFrame:
    E_CA = ca_detail.groupby("BEA_Detail", as_index=False)["E_CA_detail_MMT"].sum().set_index("BEA_Detail")["E_CA_detail_MMT"].to_dict()
    E_RO = rous_detail.groupby("BEA_Detail", as_index=False)["E_RoUS_detail_MMT"].sum().set_index("BEA_Detail")["E_RoUS_detail_MMT"].to_dict()

    rec = []
    for r in x_det.itertuples(index=False):
        key = r.NAICS_Code
        detail = r.Detail
        region = r.Region
        xval = float(r.x)

        if region == "US-CA":
            E = float(E_CA.get(detail, 0.0))
        elif region == "RoUS":
            E = float(E_RO.get(detail, 0.0))
        else:
            E = 0.0

        D_mmt = (E / xval) if xval > 0 else 0.0
        rec.append(
            {
                "NAICS_Code": key,
                "D_MMT_per_$": D_mmt,
                "D_kg_per_$": D_mmt * 1e9,
            }
        )

    D_long = pd.DataFrame(rec)
    D_row = pd.DataFrame(
        [
            D_long["D_MMT_per_$"].to_numpy(dtype=float),
            D_long["D_kg_per_$"].to_numpy(dtype=float),
        ],
        index=["CO2e_MMT_per_$", "CO2e_kg_per_$"],
        columns=D_long["NAICS_Code"].astype(str).tolist(),
    )
    return D_row


def compute_N(L: pd.DataFrame, D_row: pd.DataFrame) -> pd.DataFrame:
    L_mat = L.values.astype(float)
    D_mat = D_row.values.astype(float)
    N_mat = D_mat @ L_mat
    return pd.DataFrame(N_mat, index=D_row.index, columns=L.columns)


def export_outputs(out_excel: str, D_row: pd.DataFrame, N_row: pd.DataFrame) -> None:
    out_path = Path(out_excel)
    ensure_parent(out_path)

    D_out = D_row.reset_index().rename(columns={"index": "Flow"})
    N_out = N_row.reset_index().rename(columns={"index": "Flow"})

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        D_out.to_excel(writer, sheet_name="CA_Detailed_D", index=False)
        N_out.to_excel(writer, sheet_name="CA_Detailed_N", index=False)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build detailed D and N from Step 2, inventory, concordance, and Step 1 weights.")
    p.add_argument("--step2-file", required=True)
    p.add_argument("--inventory-file", required=True)
    p.add_argument("--concordance-file", required=True)
    p.add_argument("--dnb-ca", required=True)
    p.add_argument("--dnb-rous", required=True)
    p.add_argument("--r-home", required=True)
    p.add_argument("--stateior-datadir", required=True)
    p.add_argument("--out-step3", required=True)
    return p.parse_args()


def main() -> None:
    t0 = datetime.now()
    args = parse_args()

    setup_env(args.r_home, args.stateior_datadir)

    log("Loading Step 2 outputs")
    L, x_det, _ = load_step2_outputs(args.step2_file)

    log("Loading inventory, concordance, and Step 1 weights")
    inv_agg = load_inventory(args.inventory_file)
    cw = load_concordance(args.concordance_file)
    dnb_ca, dnb_rous = load_step1_weights(args.dnb_ca, args.dnb_rous)

    log("Building CA detail emissions")
    inv_summary = map_inventory_to_summary(inv_agg, cw, dnb_ca)
    ca_detail = build_ca_detail_emissions(inv_summary, dnb_ca)

    log("Building RoUS detail emissions")
    rous_summary = extract_rous_summary_totals_mmt()
    rous_detail = build_rous_detail_emissions(rous_summary, dnb_rous)

    log("Building D and N")
    D_row = build_D_from_detail_emissions(x_det, ca_detail, rous_detail)
    N_row = compute_N(L, D_row)

    export_outputs(args.out_step3, D_row, N_row)

    log(f"Wrote {args.out_step3}")
    log(f"Done in {datetime.now() - t0}")


if __name__ == "__main__":
    main()