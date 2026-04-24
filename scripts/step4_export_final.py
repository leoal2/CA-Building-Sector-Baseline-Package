from __future__ import annotations

import argparse
import os
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import USEEIO as EIO

SUMMARY_MODEL_NAME = "bea_model_ca_summary_2022_after_TRACI"
BEA_YEAR = 2022
GHG_YEAR = 2022


def log(msg: str) -> None:
    print(msg, flush=True)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def split_key(key: str) -> tuple[str, str]:
    s = str(key).strip()
    if "/" in s:
        a, b = s.split("/", 1)
        return a.strip(), b.strip()
    return s, ""


def split_key_index(index_like):
    idx = pd.Index(index_like).astype(str)
    s = idx.to_series(index=idx)
    sp = s.str.split("/", n=1, expand=True)
    detail = sp[0]
    region = sp[1] if sp.shape[1] > 1 else pd.Series([""] * len(s), index=s.index)
    return detail, region


def build_region_block_order(keys, region_first=("US-CA", "RoUS")):
    detail, region = split_key_index(keys)
    df = pd.DataFrame({
        "Key": list(pd.Index(keys).astype(str)),
        "Detail": detail.values,
        "Region": region.values,
    })
    rank = {r: i for i, r in enumerate(region_first)}
    df["RegionRank"] = df["Region"].map(rank).fillna(999).astype(int)
    df = df.sort_values(["RegionRank", "Detail", "Key"], ascending=True)
    return df["Key"].tolist()


def pick_ghg_row(df: pd.DataFrame) -> str:
    idx = [str(i) for i in df.index]
    patterns = [r"greenhouse", r"\bghg\b", r"co2", r"carbon"]
    for pat in patterns:
        for i in idx:
            if re.search(pat, i, re.IGNORECASE):
                return i
    return str(df.index[0])


def setup_env(r_home: str, stateior_datadir: str) -> None:
    os.environ["R_HOME"] = r_home
    os.environ["STATEIOR_DATADIR"] = stateior_datadir
    os.environ["PATH"] = os.path.join(r_home, "bin", "x64") + ";" + os.environ.get("PATH", "")


def load_step2(step2_file: str):
    xls = pd.ExcelFile(step2_file)
    required = ["CA_Detailed_x", "CA_Detailed_f", "CA_Detailed_Z", "CA_Detailed_A", "CA_Detailed_L"]
    for s in required:
        if s not in xls.sheet_names:
            raise ValueError(f"Step 2 workbook missing sheet: {s}")

    x_df = pd.read_excel(xls, sheet_name="CA_Detailed_x")
    f_df = pd.read_excel(xls, sheet_name="CA_Detailed_f")
    Z = pd.read_excel(xls, sheet_name="CA_Detailed_Z", index_col=0)
    A = pd.read_excel(xls, sheet_name="CA_Detailed_A", index_col=0)
    L = pd.read_excel(xls, sheet_name="CA_Detailed_L", index_col=0)

    if not {"NAICS_Code", "x"}.issubset(x_df.columns):
        raise ValueError("CA_Detailed_x must contain NAICS_Code and x")
    if not {"NAICS_Code", "f"}.issubset(f_df.columns):
        raise ValueError("CA_Detailed_f must contain NAICS_Code and f")

    x = x_df.set_index("NAICS_Code")["x"].astype(float)
    f = f_df.set_index("NAICS_Code")["f"].astype(float)

    for obj in [x, f]:
        obj.index = obj.index.astype(str).str.strip()
    for M in [Z, A, L]:
        M.index = M.index.astype(str).str.strip()
        M.columns = M.columns.astype(str).str.strip()

    if not (Z.index.equals(Z.columns) and A.index.equals(A.columns) and L.index.equals(L.columns)):
        raise ValueError("One of Z/A/L is not square with matching row/column keys")
    if not (Z.index.equals(A.index) and A.index.equals(L.index)):
        raise ValueError("Z/A/L do not share identical key ordering")

    universe = list(L.index)
    x = x.reindex(universe).fillna(0.0)
    f = f.reindex(universe).fillna(0.0)
    return universe, x, f, Z, A, L


def load_step3(step3_file: str, step2_universe):
    xls = pd.ExcelFile(step3_file)
    required = ["CA_Detailed_D", "CA_Detailed_N"]
    for s in required:
        if s not in xls.sheet_names:
            raise ValueError(f"Step 3 workbook missing sheet: {s}")

    D_df = pd.read_excel(xls, sheet_name="CA_Detailed_D", index_col=0)
    N_df = pd.read_excel(xls, sheet_name="CA_Detailed_N", index_col=0)

    def extract_rows(df: pd.DataFrame, name: str):
        idx = df.index.astype(str)
        if "CO2e_MMT_per_$" in idx:
            mmt = df.loc["CO2e_MMT_per_$"].astype(float)
        else:
            mmt = df.iloc[0].astype(float)
            log(f"Warning: {name} missing CO2e_MMT_per_$ row; using first row.")

        if "CO2e_kg_per_$" in idx:
            kg = df.loc["CO2e_kg_per_$"].astype(float)
        else:
            kg = (mmt * 1e9).astype(float)
            log(f"Warning: {name} missing CO2e_kg_per_$ row; creating from MMT row.")
        return mmt, kg

    D_mmt, D_kg = extract_rows(D_df, "D")
    N_mmt, N_kg = extract_rows(N_df, "N")

    miss_d = [k for k in step2_universe if k not in D_mmt.index]
    miss_n = [k for k in step2_universe if k not in N_mmt.index]
    if miss_d or miss_n:
        raise ValueError(f"Step 3 missing Step 2 keys. Missing in D={len(miss_d)}, missing in N={len(miss_n)}")

    D_mmt = D_mmt.reindex(step2_universe).fillna(0.0)
    D_kg = D_kg.reindex(step2_universe).fillna(0.0)
    N_mmt = N_mmt.reindex(step2_universe).fillna(0.0)
    N_kg = N_kg.reindex(step2_universe).fillna(0.0)
    return D_mmt, D_kg, N_mmt, N_kg


def reorder_detail(step2_universe, x, f, Z, A, L, D_mmt, D_kg, N_mmt, N_kg):
    desired = build_region_block_order(step2_universe, region_first=("US-CA", "RoUS"))
    x = x.reindex(desired).fillna(0.0)
    f = f.reindex(desired).fillna(0.0)
    Z = Z.reindex(index=desired, columns=desired)
    A = A.reindex(index=desired, columns=desired)
    L = L.reindex(index=desired, columns=desired)
    D_mmt = D_mmt.reindex(desired).fillna(0.0)
    D_kg = D_kg.reindex(desired).fillna(0.0)
    N_mmt = N_mmt.reindex(desired).fillna(0.0)
    N_kg = N_kg.reindex(desired).fillna(0.0)
    return desired, x, f, Z, A, L, D_mmt, D_kg, N_mmt, N_kg


def compute_detail_emissions(x, f, D_mmt, N_mmt):
    E_D = D_mmt * x
    E_N = N_mmt * f
    return E_D, E_N


def build_summary_two_region():
    model = EIO.Results(
        name=SUMMARY_MODEL_NAME,
        bea_year=BEA_YEAR,
        ghg_year=GHG_YEAR,
        region="CA",
        detailed=False,
        preserve=True,
    )

    A = pd.DataFrame(model.A)
    L = pd.DataFrame(model.L)
    x_df = pd.DataFrame(model.x)

    A.index = A.index.astype(str).str.strip()
    A.columns = A.columns.astype(str).str.strip()
    L.index = L.index.astype(str).str.strip()
    L.columns = L.columns.astype(str).str.strip()
    x_df.index = x_df.index.astype(str).str.strip()

    desired = build_region_block_order(list(A.index), region_first=("US-CA", "RoUS"))
    A = A.reindex(index=desired, columns=desired)
    L = L.reindex(index=desired, columns=desired)

    x_series = x_df["x"].astype(float) if "x" in x_df.columns else x_df.iloc[:, 0].astype(float)
    x_series = x_series.reindex(desired).fillna(0.0)
    x_out = pd.DataFrame({"x": x_series.values}, index=desired)

    # Correct summary f source: official CA consumption vector already contains both /US-CA and /RoUS keys
    ca_consumption = model.consumption.get("CA")
    if ca_consumption is None or ca_consumption.empty:
        raise ValueError("Summary model consumption['CA'] is missing or empty.")
    ca_consumption = ca_consumption.copy()
    ca_consumption.index = ca_consumption.index.astype(str).str.strip()
    if "TotalDemand" not in ca_consumption.columns:
        raise ValueError("Summary model consumption['CA'] is missing TotalDemand column.")

    f_series = ca_consumption["TotalDemand"].astype(float).reindex(desired).fillna(0.0)
    f_out = pd.DataFrame({"f": f_series.values}, index=desired)

    D_vec = None
    D_series_kg = None
    Dx_df = None

    if model.D is not None:
        D_raw = pd.DataFrame(model.D)
        d = D_raw.loc[pick_ghg_row(D_raw)].astype(float) if D_raw.shape[0] > 1 else D_raw.iloc[0].astype(float)
        d.index = d.index.astype(str).str.strip()
        D_series_kg = d.reindex(desired).fillna(0.0)
        D_vec = pd.DataFrame({"Key": desired, "D_kg_per_$": D_series_kg.values})
        Dx_df = pd.DataFrame({
            "Key": desired,
            "x": x_series.values,
            "D_kg_per_$": D_series_kg.values,
            "E_D_MMT": (D_series_kg * x_series / 1e9).values,
        })

    N_vec = None
    Nf_df = None
    N_series_kg = None

    if getattr(model, "N", None) is not None:
        try:
            N_raw = pd.DataFrame(model.N)
            n = N_raw.loc[pick_ghg_row(N_raw)].astype(float) if N_raw.shape[0] > 1 else N_raw.iloc[0].astype(float)
            n.index = n.index.astype(str).str.strip()
            N_series_kg = n.reindex(desired).fillna(0.0)
        except Exception:
            N_series_kg = None

    if N_series_kg is None and D_series_kg is not None:
        D_row = D_series_kg.reindex(desired).to_numpy(dtype=float).reshape(1, -1)
        L_mat = L.values.astype(float)
        N_series_kg = pd.Series((D_row @ L_mat).ravel(), index=desired, dtype=float)

    if N_series_kg is not None:
        N_vec = pd.DataFrame({"Key": desired, "N_kg_per_$": N_series_kg.values})
        Nf_df = pd.DataFrame({
            "Key": desired,
            "f": f_series.values,
            "N_kg_per_$": N_series_kg.values,
            "E_N_MMT": (N_series_kg * f_series / 1e9).values,
        })

    return {
        "A": A,
        "L": L,
        "x": x_out,
        "f": f_out,
        "D_vec": D_vec,
        "N_vec": N_vec,
        "Dx": Dx_df,
        "Nf": Nf_df,
    }


def export_final(out_file: str, codes, x, f, Z, A, L, D_mmt, D_kg, N_mmt, N_kg, E_D, E_N, summary_data):
    detail, region = split_key_index(codes)
    base = pd.DataFrame({"Key": codes, "BEA_Detail": detail.values, "Region": region.values}).set_index("Key")

    df_x = base.copy()
    df_x["x"] = x.values

    df_f = base.copy()
    df_f["f"] = f.values

    df_D = base.copy()
    df_D["D_MMT_per_$"] = D_mmt.values
    df_D["D_kg_per_$"] = D_kg.values

    df_N = base.copy()
    df_N["N_MMT_per_$"] = N_mmt.values
    df_N["N_kg_per_$"] = N_kg.values

    df_Dx = base.copy()
    df_Dx["x"] = x.values
    df_Dx["D_MMT_per_$"] = D_mmt.values
    df_Dx["D_kg_per_$"] = D_kg.values
    df_Dx["E_D_MMT"] = E_D.values

    df_Nf = base.copy()
    df_Nf["f"] = f.values
    df_Nf["N_MMT_per_$"] = N_mmt.values
    df_Nf["N_kg_per_$"] = N_kg.values
    df_Nf["E_N_MMT"] = E_N.values

    out_path = Path(out_file)
    ensure_parent(out_path)

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        df_x.to_excel(writer, sheet_name="CA_Detail_x")
        df_f.to_excel(writer, sheet_name="CA_Detail_f")
        Z.to_excel(writer, sheet_name="CA_Detail_Z")
        A.to_excel(writer, sheet_name="CA_Detail_A")
        L.to_excel(writer, sheet_name="CA_Detail_L")
        df_D.to_excel(writer, sheet_name="CA_Detail_D")
        df_N.to_excel(writer, sheet_name="CA_Detail_N")
        df_Dx.to_excel(writer, sheet_name="CA_Detail_Dx_MMT")
        df_Nf.to_excel(writer, sheet_name="CA_Detail_Nf_MMT")

        if summary_data is not None:
            summary_data["x"].to_excel(writer, sheet_name="CA_Summary_x")
            if summary_data.get("f") is not None:
                summary_data["f"].to_excel(writer, sheet_name="CA_Summary_f")
            summary_data["A"].to_excel(writer, sheet_name="CA_Summary_A")
            summary_data["L"].to_excel(writer, sheet_name="CA_Summary_L")
            if summary_data.get("D_vec") is not None:
                summary_data["D_vec"].to_excel(writer, sheet_name="CA_Summary_D_kg_per_$", index=False)
            if summary_data.get("N_vec") is not None:
                summary_data["N_vec"].to_excel(writer, sheet_name="CA_Summary_N_kg_per_$", index=False)
            if summary_data.get("Dx") is not None:
                summary_data["Dx"].to_excel(writer, sheet_name="CA_Summary_Dx_MMT", index=False)
            if summary_data.get("Nf") is not None:
                summary_data["Nf"].to_excel(writer, sheet_name="CA_Summary_Nf_MMT", index=False)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Consolidate cleaned Step 2 and Step 3 outputs into final workbook.")
    p.add_argument("--step2-file", required=True)
    p.add_argument("--step3-file", required=True)
    p.add_argument("--r-home", required=True)
    p.add_argument("--stateior-datadir", required=True)
    p.add_argument("--out-final", required=True)
    return p.parse_args()


def main() -> None:
    t0 = datetime.now()
    args = parse_args()

    setup_env(args.r_home, args.stateior_datadir)

    log("Loading Step 2")
    universe, x, f, Z, A, L = load_step2(args.step2_file)

    log("Loading Step 3")
    D_mmt, D_kg, N_mmt, N_kg = load_step3(args.step3_file, universe)

    log("Reordering detailed outputs to region-block order")
    codes, x, f, Z, A, L, D_mmt, D_kg, N_mmt, N_kg = reorder_detail(
        universe, x, f, Z, A, L, D_mmt, D_kg, N_mmt, N_kg
    )

    E_D, E_N = compute_detail_emissions(x, f, D_mmt, N_mmt)

    log("Building summary two-region outputs")
    summary_data = build_summary_two_region()

    export_final(
        args.out_final,
        codes,
        x,
        f,
        Z,
        A,
        L,
        D_mmt,
        D_kg,
        N_mmt,
        N_kg,
        E_D,
        E_N,
        summary_data,
    )

    log(f"Wrote {args.out_final}")
    log(f"Done in {datetime.now() - t0}")


if __name__ == "__main__":
    main()