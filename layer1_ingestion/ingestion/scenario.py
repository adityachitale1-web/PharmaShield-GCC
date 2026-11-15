"""
Scenario layer + Layer-1 hand-off table.

1. Validates the explicitly-labelled SCENARIO assumptions (Plan section 3, item 3):
     reference/scenario_corridor_assumptions.csv  (Hormuz sea / Gulf air shares, freight per vial)
     reference/price_benchmarks.csv               (public CMS ASP reimbursement benchmarks)
2. Builds data/interim/quotes_2026_carboplatin.csv  (C1 inputs: unit price + freight + insurance)
   - insurance rate is DATA-DRIVEN: median insurance/value ratio from cleaned SCMS lines
3. Builds data/interim/layer1_scoring_panel_seed.csv - one row per qualified labeler with
   plant country (+ evidence), corridor share, LPI/WGI, quote, capacity and preliminary
   (exact-key) links to shortages / recalls / inspections / SCMS history.
   Layer 2 (Member 2B) replaces the preliminary exact-key links with rapidfuzz resolution.

Data issues handled: shares outside [0,1] or not summing to 1, duplicate countries,
missing default row, labelers without plant country (falls back to override file, then
'Unknown' + flag), quotes for labelers that no longer qualify (orphans), etc.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .common import QLOG, Col, company_core, enforce_schema, write_table
from .config import SETTINGS

log = logging.getLogger("pharmashield.ingestion.scenario")
SRC = "scenario"
DEFAULT_CAPACITY = 4000  # SCENARIO: vials/month per qualified maker unless overridden


def corridor_table(ref_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(ref_dir / "scenario_corridor_assumptions.csv")
    problems = []
    if df["iso3"].duplicated().any():
        problems.append(f"duplicate iso3 {df.loc[df.iso3.duplicated(), 'iso3'].tolist()}")
    if "*" not in set(df["iso3"]):
        problems.append("missing default row iso3='*'")
    shares = ["air_share", "sea_share", "land_share", "hormuz_sea_fraction", "gulf_air_fraction"]
    for c in shares:
        if ((df[c] < 0) | (df[c] > 1)).any():
            problems.append(f"{c} outside [0,1]")
    tot = df[["air_share", "sea_share", "land_share"]].sum(axis=1)
    if (~np.isclose(tot, 1.0, atol=1e-6)).any():
        problems.append(f"mode shares do not sum to 1 for {df.loc[~np.isclose(tot, 1), 'iso3'].tolist()}")
    if (df[["freight_usd_per_vial_air", "freight_usd_per_vial_sea"]] < 0).any().any():
        problems.append("negative freight")
    QLOG.check(SRC, "corridor_assumption_validation", len(problems),
               severity="error" if problems else "info", detail="; ".join(problems))
    if problems:
        raise ValueError("scenario_corridor_assumptions.csv invalid: " + "; ".join(problems))
    df["hormuz_air_share"] = (df["sea_share"] * df["hormuz_sea_fraction"]
                              + df["air_share"] * df["gulf_air_fraction"]).clip(0, 1)
    df["freight_usd_per_vial"] = (df["air_share"] * df["freight_usd_per_vial_air"]
                                  + (df["sea_share"] + df["land_share"]) * df["freight_usd_per_vial_sea"])
    df["is_scenario_assumption"] = True
    return df


def _benchmark_for(core: str, bench: pd.DataFrame) -> pd.Series:
    specific = bench[bench["applies_to_labeler_core"].ne("*")]
    for _, b in specific.iterrows():
        if isinstance(core, str) and b["applies_to_labeler_core"].upper() in core:
            return b
    return bench[bench["applies_to_labeler_core"].eq("*")].iloc[0]


def _insurance_rate(interim: Path) -> tuple[float, str]:
    p = interim / "scms_shipments_clean.csv"
    if p.exists():
        d = pd.read_csv(p, usecols=["insurance_usd", "line_item_value", "insurance_imputed"], low_memory=False)
        d = d[(d["line_item_value"] > 0) & (~d["insurance_imputed"].astype(bool))]
        r = (d["insurance_usd"] / d["line_item_value"]).median()
        if np.isfinite(r):
            return float(r), "median insurance/value ratio, cleaned SCMS lines (non-imputed)"
    return 0.005, "SCENARIO default 0.5% (SCMS not available)"


def _read(interim: Path, name: str) -> pd.DataFrame:
    p = interim / f"{name}.csv"
    return pd.read_csv(p, low_memory=False) if p.exists() and p.stat().st_size > 1 else pd.DataFrame()


def build(ref_dir: Path, interim: Path) -> dict[str, pd.DataFrame]:
    sku = SETTINGS.sku
    corr = corridor_table(ref_dir)
    bench = pd.read_csv(ref_dir / "price_benchmarks.csv")
    over = pd.read_csv(ref_dir / "manufacturer_overrides.csv", dtype=str)
    labs = _read(interim, "fda_ndc_qualified_labelers")
    if labs.empty:
        QLOG.check(SRC, "qualified_labelers_available", 1, severity="error",
                   detail="run openFDA ingestion first (or offline cache)")
        return {}
    labels = _read(interim, "fda_label_manufacturing_sites")
    wb = _read(interim, "wb_country_indicators")

    # ---- plant country: override file > SPL label text > Unknown -------------------
    country = {}
    if not labels.empty:
        for _, lab in labs.iterrows():
            sets = set(str(lab.get("spl_set_ids", "")).split("|"))
            hit = labels[labels["spl_set_id"].isin(sets) & labels["label_country"].notna()]
            if hit.empty:  # fall back to same company key on the label
                hit = labels[(labels["labeler_key"] == lab["labeler_key"]) & labels["label_country"].notna()]
            if not hit.empty:
                top = hit["label_iso3"].value_counts()
                iso = top.index[0]
                row = hit[hit["label_iso3"] == iso].iloc[0]
                country[lab["labeler_key"]] = (row["label_country"], iso, "spl_label_text",
                                               str(row["manufactured_by_snippet"])[:200],
                                               len(top) > 1)
    panel = labs[["sku_id", "labeler_key", "labeler_name_fda", "labeler_core", "parent_key",
                  "n_products", "product_ndcs", "marketing_categories"]].copy()
    panel["criticality_tier"] = sku.criticality_tier
    panel["manufacturing_country"] = panel["labeler_key"].map(lambda k: country.get(k, (None,))[0])
    panel["manufacturing_iso3"] = panel["labeler_key"].map(lambda k: country.get(k, (None, None))[1])
    panel["country_source"] = panel["labeler_key"].map(lambda k: country.get(k, (None, None, None))[2])
    panel["country_evidence"] = panel["labeler_key"].map(lambda k: country.get(k, (None,) * 4)[3])
    panel["country_multiple_sites"] = panel["labeler_key"].map(lambda k: country.get(k, (None,) * 5)[4])
    panel["monthly_capacity_units"] = DEFAULT_CAPACITY
    panel["capacity_source"] = "SCENARIO default"
    if not over.empty:
        o = over.set_index("labeler_key")
        for i, r in panel.iterrows():
            if r["labeler_key"] in o.index:
                ov = o.loc[r["labeler_key"]]
                if isinstance(ov.get("manufacturing_country"), str) and ov["manufacturing_country"]:
                    panel.loc[i, ["manufacturing_country", "manufacturing_iso3", "country_source",
                                  "country_evidence"]] = [ov["manufacturing_country"], ov["iso3"],
                                                          "manual_override", ov.get("evidence")]
                if pd.notna(ov.get("monthly_capacity_units")):
                    panel.loc[i, ["monthly_capacity_units", "capacity_source"]] = \
                        [int(float(ov["monthly_capacity_units"])), "manual_override"]
    unk = panel["manufacturing_iso3"].isna()
    QLOG.check(SRC, "labelers_without_plant_country", int(unk.sum()), len(panel),
               action="DEFAULT corridor row used; add to reference/manufacturer_overrides.csv",
               severity="warning" if unk.any() else "info",
               detail=", ".join(panel.loc[unk, "labeler_name_fda"]))
    panel["manufacturing_country"] = panel["manufacturing_country"].fillna("Unknown")

    # ---- C5 inputs -------------------------------------------------------------------
    c = corr.set_index("iso3")
    look = panel["manufacturing_iso3"].where(panel["manufacturing_iso3"].isin(c.index), "*").fillna("*")
    panel["corridor_row_used"] = look
    panel["hormuz_air_share"] = look.map(c["hormuz_air_share"]).round(4)
    panel["freight_usd_per_vial"] = look.map(c["freight_usd_per_vial"]).round(4)
    if not wb.empty:
        w = wb.set_index("iso3")
        panel["lpi_score"] = panel["manufacturing_iso3"].map(w["lpi_score"])
        panel["lpi_year"] = panel["manufacturing_iso3"].map(w["lpi_score_year"])
        panel["wgi_political_stability"] = panel["manufacturing_iso3"].map(w["wgi_political_stability"])
    else:
        panel[["lpi_score", "lpi_year", "wgi_political_stability"]] = np.nan
    QLOG.check(SRC, "labelers_missing_lpi", int(panel["lpi_score"].isna().sum()), len(panel),
               action="Layer 2: impute regional median + flag",
               severity="warning" if panel["lpi_score"].isna().any() else "info")

    # ---- C1 quotes ---------------------------------------------------------------------
    ins_rate, ins_basis = _insurance_rate(interim)
    q = []
    for _, r in panel.iterrows():
        b = _benchmark_for(r["labeler_core"], bench)
        unit = float(b["payment_limit_usd"]) * sku.reference_vial_mg / float(b["unit_mg"])
        q.append({"sku_id": r["sku_id"], "labeler_key": r["labeler_key"], "labeler_name_fda": r["labeler_name_fda"],
                  "reference_vial": f"{sku.reference_vial_mg} mg / {sku.reference_vial_mg / sku.concentration_mg_per_ml:.0f} mL",
                  "base_unit_price_usd": round(unit, 2), "price_benchmark_id": b["benchmark_id"],
                  "freight_usd_per_vial": r["freight_usd_per_vial"],
                  "insurance_usd_per_vial": round(unit * ins_rate, 4), "insurance_rate": round(ins_rate, 5),
                  "insurance_basis": ins_basis, "corridor_row_used": r["corridor_row_used"],
                  "is_scenario_assumption": True})
    quotes = pd.DataFrame(q)
    quotes["landed_cost_usd_per_vial"] = (quotes["base_unit_price_usd"] + quotes["freight_usd_per_vial"]
                                          + quotes["insurance_usd_per_vial"]).round(4)
    user_q = ref_dir / "quotes_2026_carboplatin_override.csv"
    if user_q.exists():  # team-supplied tender quotes win, validated
        uq = pd.read_csv(user_q)
        orphan = ~uq["labeler_key"].isin(quotes["labeler_key"])
        QLOG.check(SRC, "override_quotes_for_non_qualified_labelers", int(orphan.sum()), len(uq),
                   action="ignored", severity="warning" if orphan.any() else "info")
        uq = uq[~orphan].drop_duplicates("labeler_key", keep="last").set_index("labeler_key")
        for col in ["base_unit_price_usd", "freight_usd_per_vial", "insurance_usd_per_vial"]:
            if col in uq:
                m = quotes["labeler_key"].map(uq[col])
                bad = m.notna() & ((m < 0) | (m > 1e5))
                quotes[col] = np.where(m.notna() & ~bad, m, quotes[col])
        quotes["landed_cost_usd_per_vial"] = quotes[["base_unit_price_usd", "freight_usd_per_vial",
                                                     "insurance_usd_per_vial"]].sum(axis=1).round(4)
    quotes = enforce_schema(quotes, [Col("labeler_key", "str", nullable=False, unique=True),
                                     Col("landed_cost_usd_per_vial", "float", nullable=False, min=0)], SRC)
    panel = panel.merge(quotes[["labeler_key", "base_unit_price_usd", "landed_cost_usd_per_vial"]],
                        on="labeler_key", how="left")

    # ---- preliminary exact-key evidence links (Layer 2 refines with rapidfuzz) ----------
    def _hits(df: pd.DataFrame, key_cols: list[str], mask=None) -> pd.Series:
        if df.empty:
            return pd.Series(0, index=panel.index)
        d = df if mask is None else df[mask(df)]
        keys = pd.concat([d[k] for k in key_cols if k in d], ignore_index=True).dropna()
        cores = keys.map(company_core).value_counts()
        return panel["labeler_core"].map(cores).fillna(0).astype(int)

    sh = _read(interim, "fda_shortages_sku")
    rc = _read(interim, "fda_recalls_5yr")
    ins = _read(interim, "fda_inspections_drugs_5yr")
    scms = _read(interim, "scms_shipments_clean")
    panel["prelim_active_shortage_records"] = _hits(sh, ["company_name", "parent_key"],
                                                    lambda d: d["is_active"].astype(bool))
    panel["prelim_class_i_ii_recalls_5yr"] = _hits(rc, ["recalling_firm"],
                                                   lambda d: d["is_class_i_or_ii"].astype(bool))
    panel["prelim_oai_5yr"] = _hits(ins, ["legal_name"], lambda d: d["classification_code"].eq("OAI"))
    panel["prelim_vai_5yr"] = _hits(ins, ["legal_name"], lambda d: d["classification_code"].eq("VAI"))
    if not scms.empty:
        hist = pd.concat([scms["vendor"].map(company_core), scms["manufacturing_site"].map(company_core)])
        first_tok = hist.dropna().str.split().str[0].value_counts()
        panel["prelim_scms_shipments"] = panel["labeler_core"].str.split().str[0].map(first_tok).fillna(0).astype(int)
    else:
        panel["prelim_scms_shipments"] = 0
    panel["is_probation"] = panel["prelim_scms_shipments"] < SETTINGS.scms.probation_min_shipments
    panel["prelim_match_method"] = "exact core-key (replace with rapidfuzz in Layer 2)"
    QLOG.check(SRC, "probation_labelers(<5 historical shipments)", int(panel["is_probation"].sum()), len(panel))
    panel = enforce_schema(panel, [
        Col("sku_id", "str", nullable=False), Col("criticality_tier", "int", allowed=(1, 2, 3)),
        Col("labeler_key", "str", nullable=False, unique=True), Col("labeler_name_fda", "str", nullable=False),
        Col("manufacturing_country", "str", nullable=False), Col("is_probation", "bool", nullable=False),
        Col("monthly_capacity_units", "int", nullable=False, min=0),
        Col("hormuz_air_share", "float", nullable=False, min=0, max=1),
        Col("lpi_score", "float", min=1, max=5)], SRC)
    return {"quotes_2026_carboplatin": quotes, "scenario_corridors_validated": corr,
            "layer1_scoring_panel_seed": panel}


def run(ref_dir: Path, interim: Path, **_) -> dict:
    try:
        tabs = build(ref_dir, interim)
    except Exception as e:
        log.exception("scenario/panel build failed")
        QLOG.status(SRC, "FAILED", note=str(e)[:200])
        return {}
    out = {k: write_table(v, k) for k, v in tabs.items()}
    if tabs:
        QLOG.status(SRC, "OK", rows=len(tabs["layer1_scoring_panel_seed"]),
                    note="panel seed + quotes built (SCENARIO-labelled)")
    return out
