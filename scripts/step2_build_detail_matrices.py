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

REMI_TO_USD_MULTIPLIER = 1_000_000.0
UNDESIRED_BEA_DETAILS = {"331314", "S00101", "S00201", "S00202"}
COVERAGE_TARGET = 0.95
IO_BALANCE_WARN = 1.05
A_COLSUM_THRESHOLD = 1.0
A_COLSUM_CAP = 0.999
COND_MAX_FATAL = 1e14
RHO_FATAL = 1.02
RESID_MAX_ABS_FATAL = 1e-6
RESID_FRO_REL_FATAL = 1e-8
L_MAX_ABS_FATAL = 1e6


def log(msg: str) -> None:
    print(msg, flush=True)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def standardize_bea_detail(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    s = s.str.replace(r"/(US|US-CA|RoUS|CA)$", "", regex=True)
    s = s.str.split("/", n=1).str[0].str.strip()
    return s


def split_key(key: str) -> tuple[str, str]:
    a, b = str(key).split("/", 1)
    return a.strip(), b.strip()


def setup_r(r_home: str, stateior_datadir: str):
    os.environ["R_HOME"] = r_home
    os.environ["STATEIOR_DATADIR"] = stateior_datadir
    os.environ["PATH"] = os.path.join(r_home, "bin", "x64") + ";" + os.environ.get("PATH", "")
    ro.r(f'Sys.setenv(STATEIOR_DATADIR = "{stateior_datadir.replace(os.sep, "/")}")')
    return importr("useeior")


def load_model(name: str, bea_year: int, ghg_year: int, region: str = "US", detailed: bool = True):
    return EIO.Results(
        name=name,
        bea_year=bea_year,
        ghg_year=ghg_year,
        region=None if region == "US" else region,
        detailed=detailed,
        preserve=True,
    )


# ---------------- REMI / FD helpers ----------------

def norm_text(x):
    if pd.isna(x):
        return None
    return re.sub(r"\s+", " ", str(x).strip())


def split_numeric_tokens(s):
    if pd.isna(s):
        return []
    return re.findall(r"\d+", str(s))


def canon_code(s):
    toks = split_numeric_tokens(s)
    return "|".join(toks) if toks else None


def code_parts(code_canon):
    if code_canon is None or pd.isna(code_canon):
        return []
    return str(code_canon).split("|")


def is_digit_str(x):
    return x is not None and (not pd.isna(x)) and str(x).strip().isdigit()


def extract_paren_code(label):
    if pd.isna(label):
        return None
    m = re.search(r"\(([^()]*)\)\s*$", str(label))
    return canon_code(m.group(1)) if m else None


def normalize_lookup_name(label):
    if pd.isna(label):
        return None
    s = str(label).strip()
    s = re.sub(r"\s*\([^)]*\)\s*$", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


def allowed_prefix_lengths_for_single(code_str):
    if not is_digit_str(code_str):
        return []
    code_str = str(code_str).strip()
    L = len(code_str)
    if L <= 2:
        return [L]
    if L == 3:
        return [3]
    return list(range(L, 2, -1))


def get_prefix_candidates_single(code_str):
    code_str = str(code_str).strip()
    return list(dict.fromkeys([code_str[:n] for n in allowed_prefix_lengths_for_single(code_str)]))


def get_combo_prefix_candidates_constrained(code_canon):
    parts = code_parts(code_canon)
    if not parts:
        return []
    if not all(is_digit_str(p) for p in parts):
        return [code_canon]
    max_len = max(len(p) for p in parts)
    out = [code_canon]
    for n in range(max_len, 2, -1):
        if all(n in allowed_prefix_lengths_for_single(p) for p in parts):
            out.append("|".join([p[:n] for p in parts]))
    return list(dict.fromkeys(out))


def load_remi_input(path: str):
    remi_df = pd.read_excel(path, sheet_name="CA_REMI_FD_MIN")
    cross_df = pd.read_excel(path, sheet_name="Crosswalk_MIN")
    trade_df = pd.read_excel(path, sheet_name="TradeShares_2022_MIN")

    remi_df["Industry"] = remi_df["Industry"].map(norm_text)
    remi_df["REMI_CODE_canon"] = remi_df["REMI CODE"].map(canon_code)
    remi_df["FinalDemand__2022"] = pd.to_numeric(remi_df["FinalDemand__2022"], errors="coerce").fillna(0.0)

    cross_df["REMI Lookup"] = cross_df["REMI Lookup"].map(norm_text)
    cross_df["REMI_Lookup_NameOnly"] = cross_df["REMI Lookup"].map(normalize_lookup_name)
    cross_df["Lookup_Code_canon"] = cross_df["REMI Lookup"].map(extract_paren_code)
    cross_df["BEA_Detail"] = cross_df["BEA_Detail"].astype(str).str.strip()

    trade_label_col = "Industries" if "Industries" in trade_df.columns else "Industry"
    trade_df[trade_label_col] = trade_df[trade_label_col].map(norm_text)
    trade_df["Trade_Code_canon"] = trade_df[trade_label_col].map(extract_paren_code)
    trade_df["TradeShare"] = pd.to_numeric(trade_df["TradeShare_2022"], errors="coerce")
    return remi_df, cross_df, trade_df


def load_trade_map(trade_df: pd.DataFrame) -> dict[str, float]:
    return (
        trade_df.dropna(subset=["Trade_Code_canon", "TradeShare"])
        .groupby("Trade_Code_canon")["TradeShare"]
        .mean()
        .to_dict()
    )


def get_trade_share_for_remi_code(remi_code_canon, trade_map):
    if remi_code_canon is None or pd.isna(remi_code_canon):
        return None
    remi_code_canon = str(remi_code_canon)
    if remi_code_canon in trade_map:
        return float(trade_map[remi_code_canon])
    parts = remi_code_canon.split("|")
    if len(parts) > 1:
        shares = []
        for p in parts:
            if p in trade_map:
                shares.append(float(trade_map[p]))
            elif is_digit_str(p) and len(p) >= 4 and p[:3] in trade_map:
                shares.append(float(trade_map[p[:3]]))
        return float(np.mean(shares)) if shares else None
    code = parts[0]
    if code in trade_map:
        return float(trade_map[code])
    if is_digit_str(code) and len(code) >= 4 and code[:3] in trade_map:
        return float(trade_map[code[:3]])
    return None


def extract_us_consumption_complete(us_model):
    demandvectors = us_model._model.rx2["DemandVectors"].rx2["vectors"]
    demand = demandvectors.rx2["2017_US_Consumption_Complete"]
    out = pd.DataFrame({
        "Detail": [str(n).replace("/US", "").strip() for n in demand.names],
        "US_Detailed_Consumption_Complete": pd.to_numeric(list(demand), errors="coerce"),
    })
    out["US_Detailed_Consumption_Complete"] = out["US_Detailed_Consumption_Complete"].fillna(0.0)
    return out


def extract_ca_summary_consumption(ca_model):
    ca_consumption = ca_model.consumption.get("CA")
    if ca_consumption is None:
        raise RuntimeError("Could not find CA summary consumption.")
    ca_summary = ca_consumption.copy().reset_index().rename(columns={"index": "SummaryRegion", "TotalDemand": "CA_Summary_Consumption_Complete"})
    ca_summary = ca_summary[ca_summary["SummaryRegion"].astype(str).str.contains("/")].copy()
    split_df = ca_summary["SummaryRegion"].astype(str).str.rsplit("/", n=1, expand=True)
    ca_summary["Summary"] = split_df[0].str.strip()
    ca_summary["Region"] = split_df[1].str.strip().replace({"CA": "US-CA"})
    ca_summary["CA_Summary_Consumption_Complete"] = pd.to_numeric(ca_summary["CA_Summary_Consumption_Complete"], errors="coerce").fillna(0.0)
    return ca_summary[["Summary", "Region", "CA_Summary_Consumption_Complete"]].copy()


def recreate_ca_scaled_fd(us_model, ca_model):
    us_consumption = extract_us_consumption_complete(us_model)
    ca_summary = extract_ca_summary_consumption(ca_model)
    crosswalk_df = pd.DataFrame(us_model.crosswalk).copy()
    crosswalk_clean = crosswalk_df[["Summary", "Detail"]].drop_duplicates().copy()
    crosswalk_clean["Summary"] = crosswalk_clean["Summary"].astype(str).str.strip()
    crosswalk_clean["Detail"] = crosswalk_clean["Detail"].astype(str).str.strip()

    us_default = crosswalk_clean.merge(us_consumption, on="Detail", how="left")
    us_default["US_Detailed_Consumption_Complete"] = us_default["US_Detailed_Consumption_Complete"].fillna(0.0)
    us_default["Summary_US_Total"] = us_default.groupby("Summary")["US_Detailed_Consumption_Complete"].transform("sum")
    us_default["Consumption Weighting"] = np.where(
        us_default["Summary_US_Total"] > 0,
        us_default["US_Detailed_Consumption_Complete"] / us_default["Summary_US_Total"],
        0.0,
    )

    scaled_df = ca_summary.merge(us_default[["Summary", "Detail", "Consumption Weighting"]], on="Summary", how="inner")
    scaled_df["Estimated Consumption (USD)"] = scaled_df["CA_Summary_Consumption_Complete"] * scaled_df["Consumption Weighting"]
    scaled_df["Detailed"] = scaled_df["Detail"] + "/US"
    scaled_df["Detailed Lookup"] = scaled_df["Detail"] + "/" + scaled_df["Region"]
    scaled_df["Lookup"] = scaled_df["Summary"] + "/" + scaled_df["Region"]
    scaled_df["BEA_Detail"] = scaled_df["Detail"]
    scaled_df = scaled_df[["Detailed", "Region", "Detailed Lookup", "Summary", "Lookup", "Consumption Weighting", "Estimated Consumption (USD)", "BEA_Detail"]].copy()
    scaled_df = scaled_df[~scaled_df["BEA_Detail"].isin(UNDESIRED_BEA_DETAILS)].copy()
    return scaled_df.sort_values(["Region", "Detailed Lookup", "Summary", "Lookup"]).reset_index(drop=True)


def build_crosswalk_tables(cross_df, scaled_df):
    code_to_bea_exact = (
        cross_df.dropna(subset=["Lookup_Code_canon", "BEA_Detail"])
        .groupby("Lookup_Code_canon")["BEA_Detail"]
        .agg(lambda s: sorted(pd.unique(s)))
        .to_dict()
    )
    prefix_to_bea = {}
    for _, r in cross_df.dropna(subset=["Lookup_Code_canon", "BEA_Detail"]).iterrows():
        key = r["Lookup_Code_canon"]
        bea = r["BEA_Detail"]
        if key is None:
            continue
        cands = get_combo_prefix_candidates_constrained(str(key)) if "|" in str(key) else get_prefix_candidates_single(str(key))
        for cand in cands:
            prefix_to_bea.setdefault(cand, set()).add(bea)
    prefix_to_bea = {k: sorted(v) for k, v in prefix_to_bea.items()}
    nameonly_to_bea = (
        cross_df.dropna(subset=["REMI_Lookup_NameOnly", "BEA_Detail"])
        .groupby("REMI_Lookup_NameOnly")["BEA_Detail"]
        .agg(lambda s: sorted(pd.unique(s)))
        .to_dict()
    )
    bea_in_scaled = set(scaled_df["BEA_Detail"].dropna().unique())
    return code_to_bea_exact, prefix_to_bea, nameonly_to_bea, bea_in_scaled


manual_summary_targets = {
    ("Motor vehicle and parts dealers", "336"): ["441"],
    ("Truck transportation", "484"): ["484"],
    ("Web search portals, libraries, archives, and other information services", "519290"): ["514"],
    ("Consumer goods rental and general rental centers", "5322|5323"): ["532RL"],
    ("Scenic and sightseeing transportation and support activities for transportation", "487|488"): ["487OS"],
    ("Civic, social, professional, and similar organizations", "9134|8139"): ["81"],
    ("Federal Military", "92"): ["GFGD"],
    ("Federal Civilian", "92"): ["GFGN", "GFE"],
    ("State and Local Government", "92"): ["GSLG", "GSLE"],
}


def forbidden_summaries_for_row(industry_name):
    ind = (industry_name or "").lower()
    forbidden = set()
    if "excluding motor vehicle" in ind or "excluding motor vehicles" in ind:
        forbidden.add("441")
    return forbidden


def try_code_match(remi_code_canon, code_to_bea_exact, prefix_to_bea):
    if remi_code_canon is None or pd.isna(remi_code_canon):
        return []
    remi_code_canon = str(remi_code_canon)
    if remi_code_canon in code_to_bea_exact:
        return code_to_bea_exact[remi_code_canon]
    cands = get_combo_prefix_candidates_constrained(remi_code_canon) if "|" in remi_code_canon else get_prefix_candidates_single(remi_code_canon)
    for cand in cands:
        if cand != remi_code_canon and cand in prefix_to_bea:
            return prefix_to_bea[cand]
    return []


def allocate_over_candidate_scaled(candidate_scaled):
    df = candidate_scaled.copy()
    total_est = df["Estimated Consumption (USD)"].sum()
    if total_est > 0:
        df["Allocation_Share"] = df["Estimated Consumption (USD)"] / total_est
        return df
    total_weight = df["Consumption Weighting"].sum()
    if total_weight > 0:
        df["Allocation_Share"] = df["Consumption Weighting"] / total_weight
        return df
    df["Allocation_Share"] = 1.0 / len(df)
    return df


def within_region_weights(df_region):
    if df_region.empty:
        return df_region.assign(_w=0.0)
    est = df_region["Estimated Consumption (USD)"].sum()
    if est > 0:
        return df_region.assign(_w=df_region["Estimated Consumption (USD)"] / est)
    w = df_region["Consumption Weighting"].sum()
    if w > 0:
        return df_region.assign(_w=df_region["Consumption Weighting"] / w)
    return df_region.assign(_w=1.0 / len(df_region))


def allocate_over_candidate_trade_shares(candidate_scaled, remi_code_canon, trade_map):
    s = get_trade_share_for_remi_code(remi_code_canon, trade_map)
    if s is None or pd.isna(s):
        return allocate_over_candidate_scaled(candidate_scaled)
    df = candidate_scaled.copy()
    alloc = []
    df_us = within_region_weights(df[df["Region"] == "US-CA"].copy())
    df_ro = within_region_weights(df[df["Region"] == "RoUS"].copy())
    if not df_us.empty:
        tmp = df_us.copy()
        tmp["Allocation_Share"] = tmp["_w"] * s
        alloc.append(tmp.drop(columns=["_w"]))
    if not df_ro.empty:
        tmp = df_ro.copy()
        tmp["Allocation_Share"] = tmp["_w"] * (1.0 - s)
        alloc.append(tmp.drop(columns=["_w"]))
    if not alloc:
        return allocate_over_candidate_scaled(df)
    out = pd.concat(alloc, ignore_index=True)
    total = out["Allocation_Share"].sum()
    if total > 0:
        out["Allocation_Share"] = out["Allocation_Share"] / total
    return out


def build_detailed_fd(remi_df, cross_df, trade_df, us_model, ca_model):
    scaled_df = recreate_ca_scaled_fd(us_model, ca_model)
    code_to_bea_exact, prefix_to_bea, nameonly_to_bea, bea_in_scaled = build_crosswalk_tables(cross_df, scaled_df)
    trade_map = load_trade_map(trade_df)

    rows = []
    for _, rr in remi_df.iterrows():
        industry = rr["Industry"]
        remi_code = rr["REMI_CODE_canon"]
        fd_million = float(rr["FinalDemand__2022"])
        if fd_million == 0 or pd.isna(fd_million):
            continue

        forced = manual_summary_targets.get((industry, remi_code))
        forbidden = forbidden_summaries_for_row(industry)
        bea_candidates = try_code_match(remi_code, code_to_bea_exact, prefix_to_bea)
        if not bea_candidates and industry in nameonly_to_bea:
            bea_candidates = nameonly_to_bea[industry]
        bea_candidates = [b for b in bea_candidates if b in bea_in_scaled]

        if forced:
            candidate_scaled = scaled_df[scaled_df["Summary"].isin(forced)].copy()
        else:
            candidate_scaled = scaled_df[scaled_df["BEA_Detail"].isin(bea_candidates)].copy()
            if forbidden:
                candidate_scaled = candidate_scaled[~candidate_scaled["Summary"].isin(forbidden)].copy()
        if candidate_scaled.empty:
            continue

        alloc_df = allocate_over_candidate_trade_shares(candidate_scaled, remi_code, trade_map)
        alloc_df["REMI_FD_USD_TS"] = alloc_df["Allocation_Share"] * fd_million * REMI_TO_USD_MULTIPLIER
        rows.append(alloc_df)

    if not rows:
        raise RuntimeError("Detailed REMI final demand build produced no rows.")

    final_df = pd.concat(rows, ignore_index=True)
    final_df = (
        final_df.groupby(["Detailed Lookup", "Region", "Summary", "Lookup", "Consumption Weighting", "BEA_Detail"], as_index=False)
        .agg(REMI_FD_USD_TS=("REMI_FD_USD_TS", "sum"))
    )
    final_df["Detailed"] = final_df["BEA_Detail"] + "/US"
    return final_df[["Detailed", "Region", "Detailed Lookup", "Summary", "Lookup", "Consumption Weighting", "REMI_FD_USD_TS", "BEA_Detail"]].sort_values(["Region", "Detailed Lookup"]).reset_index(drop=True)


# ---------------- Detailed system builders ----------------

def find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for cand in candidates:
        for c in df.columns:
            if c.lower() == cand.lower():
                return c
    for cand in candidates:
        for c in df.columns:
            if cand.lower() in c.lower():
                return c
    return None


def load_step1_csvs(ca_csv: str, rous_csv: str):
    dnb_ca = pd.read_csv(ca_csv)
    dnb_rous = pd.read_csv(rous_csv)
    dnb_ca["Detail"] = standardize_bea_detail(dnb_ca[find_column(dnb_ca, ["BEA_Detail", "Detail"])])
    dnb_rous["Detail"] = standardize_bea_detail(dnb_rous[find_column(dnb_rous, ["BEA_Detail", "Detail"])])
    return dnb_ca, dnb_rous


def build_crosswalk_and_us_details(us_model):
    cw = pd.DataFrame(us_model.crosswalk).copy()
    us_index = list(us_model.L.index)
    detail_codes_us_unique = sorted(set([str(code).replace("/US", "") for code in us_index]))
    cw_valid = cw[cw["Detail"].isin(detail_codes_us_unique)].copy()
    return cw_valid, detail_codes_us_unique


def build_canonical_detail_universe(cw_valid, detail_codes_us_unique, dnb_ca, dnb_rous):
    dnb_details = set(dnb_ca["Detail"].dropna()) | set(dnb_rous["Detail"].dropna())
    canonical = sorted(set(detail_codes_us_unique) & set(cw_valid["Detail"].unique()) & dnb_details)
    if not canonical:
        raise RuntimeError("Canonical detail universe is empty.")
    detail_to_summary = dict(zip(cw_valid[cw_valid["Detail"].isin(canonical)]["Detail"], cw_valid[cw_valid["Detail"].isin(canonical)]["Summary"]))
    return canonical, detail_to_summary


def build_dnb_x_raw(dnb_ca, dnb_rous, canonical_details):
    ca = dnb_ca[dnb_ca["Detail"].isin(canonical_details)].copy()
    ro = dnb_rous[dnb_rous["Detail"].isin(canonical_details)].copy()
    ca_raw = ca[["Detail", "DetailedOutput_CA_CA"]].rename(columns={"DetailedOutput_CA_CA": "x_raw"})
    ca_raw["Region"] = "US-CA"
    ro_raw = ro[["Detail", "DetailedOutput_CA_RoUS"]].rename(columns={"DetailedOutput_CA_RoUS": "x_raw"})
    ro_raw["Region"] = "RoUS"
    return pd.concat([ca_raw, ro_raw], ignore_index=True).groupby(["Detail", "Region"], as_index=False)["x_raw"].sum()


def build_x_det(x_raw, ca_model, detail_to_summary):
    x_ca = pd.DataFrame(ca_model.x, columns=["x"])
    x_ca["Summary"] = x_ca.index.to_series().str.rsplit("/", n=1).str[0]
    x_ca["Region"] = x_ca.index.to_series().str.rsplit("/", n=1).str[1]
    x_ca_group = x_ca.groupby(["Summary", "Region"], as_index=False)["x"].sum().rename(columns={"x": "X_target"})

    tidy_x = x_raw.copy()
    tidy_x["Summary"] = tidy_x["Detail"].map(detail_to_summary)
    x_raw_group = tidy_x.groupby(["Summary", "Region"], as_index=False)["x_raw"].sum().rename(columns={"x_raw": "X_raw"})
    scales = x_ca_group.merge(x_raw_group, on=["Summary", "Region"], how="left")
    scales["X_raw"] = scales["X_raw"].fillna(0.0)
    scales["k"] = np.where(scales["X_raw"] > 0, scales["X_target"] / scales["X_raw"], 0.0)
    tidy_x = tidy_x.merge(scales[["Summary", "Region", "k"]], on=["Summary", "Region"], how="left")
    tidy_x["k"] = tidy_x["k"].fillna(0.0)
    tidy_x["x_det"] = tidy_x["x_raw"] * tidy_x["k"]
    tidy_x["NAICS_Code"] = tidy_x["Detail"] + "/" + tidy_x["Region"]

    x_det_vec = tidy_x[["NAICS_Code", "x_det"]].groupby("NAICS_Code", as_index=False)["x_det"].sum().rename(columns={"x_det": "x"})
    return tidy_x, x_det_vec


def build_f_det_vec(fd_df, canonical_details, x_det_vec):
    allowed = {f"{d}/US-CA" for d in canonical_details} | {f"{d}/RoUS" for d in canonical_details}
    kept = fd_df[["Detailed Lookup", "REMI_FD_USD_TS"]].rename(columns={"Detailed Lookup": "NAICS_Code", "REMI_FD_USD_TS": "f"}).copy()
    kept["NAICS_Code"] = kept["NAICS_Code"].astype(str).str.strip()
    kept["f"] = pd.to_numeric(kept["f"], errors="coerce").fillna(0.0)
    kept = kept[kept["NAICS_Code"].isin(allowed)].groupby("NAICS_Code", as_index=False)["f"].sum()
    f_det_vec = pd.DataFrame({"NAICS_Code": x_det_vec["NAICS_Code"].astype(str).tolist()}).merge(kept, on="NAICS_Code", how="left")
    f_det_vec["f"] = f_det_vec["f"].fillna(0.0)
    return f_det_vec


def build_us_z_detail(us_model, canonical_details):
    A_us = pd.DataFrame(us_model.A)
    x_us = pd.DataFrame(us_model.x, columns=["x"])
    A_us.index = [str(k) for k in A_us.index]
    A_us.columns = [str(k) for k in A_us.columns]
    x_us.index = [str(k) for k in x_us.index]
    keep = [f"{d}/US" for d in canonical_details]
    A_sub = A_us.loc[keep, keep].copy()
    x_sub = x_us.loc[keep, "x"].astype(float)
    Z = A_sub.multiply(x_sub, axis=1)
    Z.index = [k.replace("/US", "") for k in Z.index]
    Z.columns = [k.replace("/US", "") for k in Z.columns]
    return Z


def build_ca_z_summary(ca_model):
    A = pd.DataFrame(ca_model.A)
    x = pd.DataFrame(ca_model.x, columns=["x"])
    A.index = [str(k) for k in A.index]
    A.columns = [str(k) for k in A.columns]
    x.index = [str(k) for k in x.index]
    common = [k for k in A.columns if k in x.index]
    A = A.loc[common, common].copy()
    x = x.loc[common, "x"].astype(float)
    return A.multiply(x, axis=1)


def disaggregate_z_to_detail(Z_us_det, Z_ca_sum, canonical_details, detail_to_summary):
    keys = [f"{d}/US-CA" for d in canonical_details] + [f"{d}/RoUS" for d in canonical_details]
    Z_det = pd.DataFrame(0.0, index=keys, columns=keys)
    by_summary = {}
    for d in canonical_details:
        s = detail_to_summary.get(d)
        if s is not None:
            by_summary.setdefault(s, []).append(d)

    summaries = sorted(by_summary)
    for s_sup in summaries:
        sup_details = [d for d in by_summary[s_sup] if d in Z_us_det.index]
        if not sup_details:
            continue
        for s_buy in summaries:
            buy_details = [d for d in by_summary[s_buy] if d in Z_us_det.columns]
            if not buy_details:
                continue
            us_block = Z_us_det.loc[sup_details, buy_details].copy()
            block_total = float(us_block.to_numpy().sum())
            if block_total == 0:
                continue
            comp = us_block / block_total
            for r_sup in ["US-CA", "RoUS"]:
                for r_buy in ["US-CA", "RoUS"]:
                    sum_row = f"{s_sup}/{r_sup}"
                    sum_col = f"{s_buy}/{r_buy}"
                    if sum_row not in Z_ca_sum.index or sum_col not in Z_ca_sum.columns:
                        continue
                    total = float(Z_ca_sum.loc[sum_row, sum_col])
                    if total == 0:
                        continue
                    row_keys = [f"{d}/{r_sup}" for d in sup_details]
                    col_keys = [f"{d}/{r_buy}" for d in buy_details]
                    Z_det.loc[row_keys, col_keys] = comp.to_numpy(dtype=float) * total
    return Z_det


def build_a_from_zx(Z_det: pd.DataFrame, x_det_vec: pd.DataFrame):
    x_map = dict(zip(x_det_vec["NAICS_Code"], x_det_vec["x"]))
    X = np.array([float(x_map.get(c, 0.0)) for c in Z_det.columns], dtype=float)
    A = Z_det.to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        A = np.divide(A, X[np.newaxis, :], out=np.zeros_like(A), where=X[np.newaxis, :] != 0)

    A_df = pd.DataFrame(A, index=Z_det.index, columns=Z_det.columns)
    col_sums = A_df.sum(axis=0).to_numpy(dtype=float)
    factors = np.ones_like(col_sums)
    mask = col_sums > A_COLSUM_THRESHOLD
    factors[mask] = A_COLSUM_CAP / col_sums[mask]
    if int(mask.sum()) > 0:
        log(f"Regularizing A columns with colsum > {A_COLSUM_THRESHOLD}: {int(mask.sum())} columns")
    factor_series = pd.Series(factors, index=A_df.columns)
    A_df = A_df.mul(factor_series, axis=1)
    return A_df


def spectral_radius_estimate(A: np.ndarray) -> float:
    vals = np.linalg.eigvals(A)
    return float(np.max(np.abs(vals)))


def recompute_L(A_df: pd.DataFrame):
    A = A_df.to_numpy(dtype=float)
    M = np.eye(A.shape[0]) - A
    cond = float(np.linalg.cond(M))
    if not np.isfinite(cond) or cond > COND_MAX_FATAL:
        raise RuntimeError(f"(I-A) is singular or ill-conditioned: cond={cond}")
    rho = spectral_radius_estimate(A)
    if rho > RHO_FATAL:
        raise RuntimeError(f"spectral radius(A) too large: {rho}")
    L = np.linalg.inv(M)
    R = M @ L - np.eye(A.shape[0])
    resid_max_abs = float(np.max(np.abs(R)))
    resid_fro_rel = float(np.linalg.norm(R, ord="fro") / np.linalg.norm(np.eye(A.shape[0]), ord="fro"))
    L_max_abs = float(np.max(np.abs(L)))
    if resid_max_abs > RESID_MAX_ABS_FATAL and resid_fro_rel > RESID_FRO_REL_FATAL:
        raise RuntimeError(f"L residual failed: max_abs={resid_max_abs}, fro_rel={resid_fro_rel}")
    if not np.isfinite(L_max_abs) or L_max_abs > L_MAX_ABS_FATAL:
        raise RuntimeError(f"L exploded: max_abs={L_max_abs}")
    return pd.DataFrame(L, index=A_df.index, columns=A_df.columns)


def load_remi_detail_shares(remi_input_file: str):
    scaling = pd.read_excel(remi_input_file, sheet_name="Crosswalk_MIN")
    trade = pd.read_excel(remi_input_file, sheet_name="TradeShares_2022_MIN")
    scaling_map = scaling[["REMI Lookup", "BEA_Detail"]].copy()
    scaling_map["REMI Lookup"] = scaling_map["REMI Lookup"].map(norm_text)
    scaling_map["BEA_Detail"] = scaling_map["BEA_Detail"].map(norm_text)
    scaling_map = scaling_map[(scaling_map["REMI Lookup"] != "") & (scaling_map["BEA_Detail"] != "")].drop_duplicates()
    trade_label_col = "Industries" if "Industries" in trade.columns else "Industry"
    trade[trade_label_col] = trade[trade_label_col].map(norm_text)
    trade["Trade_Code_canon"] = trade[trade_label_col].map(extract_paren_code)
    trade["TradeShare"] = pd.to_numeric(trade["TradeShare_2022"], errors="coerce")
    remi_share_map = trade.dropna(subset=["Trade_Code_canon", "TradeShare"]).groupby("Trade_Code_canon")["TradeShare"].mean().to_dict()
    detail_share_map = {}
    for bea, grp in scaling_map.groupby("BEA_Detail"):
        shares = []
        for lookup in grp["REMI Lookup"]:
            code = extract_paren_code(lookup)
            s = get_trade_share_for_remi_code(code, remi_share_map)
            if s is not None and np.isfinite(s):
                shares.append(float(s))
        if shares:
            detail_share_map[bea] = float(np.mean(shares))
    return detail_share_map


def qa_io_balance(Z: pd.DataFrame, x_det_vec: pd.DataFrame):
    x_map = dict(zip(x_det_vec["NAICS_Code"], x_det_vec["x"]))
    x_vec = np.array([float(x_map.get(c, np.nan)) for c in Z.columns], dtype=float)
    colsumZ = np.nansum(Z.to_numpy(dtype=float), axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where((x_vec > 0) & np.isfinite(x_vec), colsumZ / x_vec, np.nan)
    finite = ratio[np.isfinite(ratio)]
    if finite.size == 0:
        log("IO balance diagnostic: no finite ratios")
        return
    mx = float(np.nanmax(finite))
    med = float(np.nanmedian(finite))
    p95 = float(np.nanpercentile(finite, 95))
    log(f"IO balance diagnostic: median={med:.6g}, p95={p95:.6g}, max={mx:.6g}")
    if mx > IO_BALANCE_WARN:
        idx = np.argsort(-finite)[:10]
        ratio_series = pd.Series(ratio, index=Z.columns).sort_values(ascending=False).head(10)
        log("Top 10 colsum(Z)/x ratios:")
        for k, v in ratio_series.items():
            log(f"  {k}: {v:.6g}")


def compute_supplier_to_ca(Z: pd.DataFrame):
    row_base = [split_key(k)[0] for k in Z.index]
    row_reg = [split_key(k)[1] for k in Z.index]
    col_reg = [split_key(k)[1] for k in Z.columns]
    ca_cols = [j for j, r in enumerate(col_reg) if r == "US-CA"]
    rows = []
    M = Z.to_numpy(dtype=float)
    for i, bea in enumerate(row_base):
        if row_reg[i] not in {"US-CA", "RoUS"}:
            continue
        rows.append({"BEA_Detail": bea, "Region": row_reg[i], "Z_to_CA": float(np.sum(M[i, ca_cols]))})
    df = pd.DataFrame(rows)
    piv = df.pivot_table(index="BEA_Detail", columns="Region", values="Z_to_CA", aggfunc="sum", fill_value=0.0).reset_index()
    for c in ["US-CA", "RoUS"]:
        if c not in piv.columns:
            piv[c] = 0.0
    piv = piv.rename(columns={"US-CA": "Z_USCA_to_CA", "RoUS": "Z_RoUS_to_CA"})
    piv["Z_tot_to_CA"] = piv["Z_USCA_to_CA"] + piv["Z_RoUS_to_CA"]
    piv = piv.sort_values("Z_tot_to_CA", ascending=False).reset_index(drop=True)
    total = float(piv["Z_tot_to_CA"].sum())
    piv["cum_share"] = piv["Z_tot_to_CA"].cumsum() / (total if total > 0 else 1.0)
    return piv


def select_suppliers_by_coverage(piv: pd.DataFrame):
    selected = piv[piv["cum_share"] <= COVERAGE_TARGET].copy()
    if selected.empty and not piv.empty:
        selected = piv.iloc[[0]].copy()
    elif len(selected) < len(piv):
        selected = piv.iloc[: len(selected) + 1].copy()
    return selected


def apply_resplit_per_column(Z: pd.DataFrame, selected: pd.DataFrame, detail_share_map: dict[str, float]):
    Z_adj = Z.copy()
    row_idx = {k: i for i, k in enumerate(Z.index)}
    col_reg = [split_key(k)[1] for k in Z.columns]
    ca_cols = [j for j, r in enumerate(col_reg) if r == "US-CA"]
    M = Z_adj.to_numpy(dtype=float)
    stats = []
    for _, rr in selected.iterrows():
        bea = rr["BEA_Detail"]
        share = detail_share_map.get(bea)
        us_key = f"{bea}/US-CA"
        ro_key = f"{bea}/RoUS"
        if share is None or us_key not in row_idx or ro_key not in row_idx:
            continue
        i_us = row_idx[us_key]
        i_ro = row_idx[ro_key]
        changed = 0
        mass = 0.0
        for j in ca_cols:
            tot = float(M[i_us, j] + M[i_ro, j])
            if tot == 0:
                continue
            M[i_us, j] = share * tot
            M[i_ro, j] = tot - M[i_us, j]
            changed += 1
            mass += tot
        stats.append({"BEA_Detail": bea, "Applied_Share_USCA": share, "Changed_CA_Buyer_Cols": changed, "Resplit_Total_Mass": mass})
    return pd.DataFrame(M, index=Z.index, columns=Z.columns), pd.DataFrame(stats)


def export_step2(path: str, x_det_vec: pd.DataFrame, f_det_vec: pd.DataFrame, Z_det: pd.DataFrame, A_det: pd.DataFrame, L_det: pd.DataFrame, fd_df: pd.DataFrame, resplit_stats: pd.DataFrame):
    ensure_parent(Path(path))
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        x_det_vec.to_excel(writer, sheet_name="CA_Detailed_x", index=False)
        f_det_vec.to_excel(writer, sheet_name="CA_Detailed_f", index=False)
        Z_det.reset_index().rename(columns={"index": "NAICS_Code"}).to_excel(writer, sheet_name="CA_Detailed_Z", index=False)
        A_det.reset_index().rename(columns={"index": "NAICS_Code"}).to_excel(writer, sheet_name="CA_Detailed_A", index=False)
        L_det.reset_index().rename(columns={"index": "NAICS_Code"}).to_excel(writer, sheet_name="CA_Detailed_L", index=False)



def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merged Step 2: build detailed FD, detailed x/Z/A/L, and mandatory REMI trade-share reconciliation.")
    p.add_argument("--dnb-ca", required=True)
    p.add_argument("--dnb-rous", required=True)
    p.add_argument("--remi-input", required=True)
    p.add_argument("--r-home", required=True)
    p.add_argument("--stateior-datadir", required=True)
    p.add_argument("--out-step2", required=True)
    return p.parse_args()


def main() -> None:
    t0 = datetime.now()
    args = parse_args()
    setup_r(args.r_home, args.stateior_datadir)

    log("Loading USEEIO models")
    us_model = load_model(US_MODEL_NAME, US_BEA_YEAR, US_GHG_YEAR, region="US", detailed=True)
    ca_model = load_model(CA_MODEL_NAME, CA_BEA_YEAR, CA_GHG_YEAR, region="CA", detailed=False)

    log("Loading Step 1 outputs and REMI input")
    dnb_ca, dnb_rous = load_step1_csvs(args.dnb_ca, args.dnb_rous)
    remi_df, cross_df, trade_df = load_remi_input(args.remi_input)

    log("Building detailed REMI final demand in memory")
    fd_df = build_detailed_fd(remi_df, cross_df, trade_df, us_model, ca_model)

    log("Building canonical detail universe")
    cw_valid, detail_codes_us_unique = build_crosswalk_and_us_details(us_model)
    canonical_details, detail_to_summary = build_canonical_detail_universe(cw_valid, detail_codes_us_unique, dnb_ca, dnb_rous)

    log("Building detailed x and f")
    x_raw = build_dnb_x_raw(dnb_ca, dnb_rous, canonical_details)
    tidy_x, x_det_vec = build_x_det(x_raw, ca_model, detail_to_summary)
    f_det_vec = build_f_det_vec(fd_df, canonical_details, x_det_vec)

    log("Building detailed Z and A before REMI reconciliation")
    Z_us_det = build_us_z_detail(us_model, canonical_details)
    Z_ca_sum = build_ca_z_summary(ca_model)
    Z_det = disaggregate_z_to_detail(Z_us_det, Z_ca_sum, canonical_details, detail_to_summary)
    qa_io_balance(Z_det, x_det_vec)
    A_det = build_a_from_zx(Z_det, x_det_vec)

    log("Applying mandatory REMI trade-share reconciliation")
    detail_share_map = load_remi_detail_shares(args.remi_input)
    supplier_piv = compute_supplier_to_ca(Z_det)
    selected = select_suppliers_by_coverage(supplier_piv)
    Z_det, resplit_stats = apply_resplit_per_column(Z_det, selected, detail_share_map)
    qa_io_balance(Z_det, x_det_vec)
    A_det = build_a_from_zx(Z_det, x_det_vec)

    log("Recomputing Leontief inverse")
    L_det = recompute_L(A_det)

    export_step2(args.out_step2, x_det_vec, f_det_vec, Z_det, A_det, L_det, fd_df, resplit_stats)
    log(f"Wrote {args.out_step2}")
    log(f"Done in {datetime.now() - t0}")


if __name__ == "__main__":
    main()
