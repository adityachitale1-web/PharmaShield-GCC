"""
Scenario layer + locked-schema hand-off.

1. Validates the explicitly-labelled SCENARIO assumptions (plan section 3, item 3):
     reference/scenario_corridor_assumptions.csv  (Hormuz sea / Gulf air shares, freight per vial)
     reference/price_benchmarks.csv               (public CMS ASP reimbursement benchmarks)
2. Builds quotes_2026_carboplatin.csv (plan 4.3, C1): base unit price anchored to the public
   reimbursement benchmark + deterministic origin-country air/sea freight + insurance.
3. Links every panel labeler to shortages, recalls, inspections and SCMS history using
   normalised corporate suffixes + rapidfuzz token_sort_ratio >= 88 (plan 5.2 step 3) and the
   corporate alias table; every candidate link is written to entity_match_candidates.csv for
   the plan's manual check.
4. Writes data/processed/suppliers_matrix.csv in the LOCKED schema of plan 5.1 (same column
   names, order and types). Columns owned by Layer 1 are filled; C1 and C5 are computed from
   the plan's closed-form formulas (4.3, Normal mode); C2 / C4 / shap_top3_reasons belong to the
   ML and NLP engines and are left empty for them; C3 defaults to 0.0 for probation vendors.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .common import QLOG, Col, company_core, company_key, enforce_schema, write_table
from .config import SETTINGS

try:
    from rapidfuzz import fuzz
except ImportError:  # pipeline still runs; matching falls back to exact / phrase rules
    fuzz = None

log = logging.getLogger("pharmashield.ingestion.scenario")
SRC = "scenario"
DEFAULT_CAPACITY = 4000  # SCENARIO: vials/month per maker unless overridden

LOCKED_SCHEMA = [  # plan 5.1 - order and names are fixed
    Col("sku_id", "str", nullable=False),
    Col("criticality_tier", "int", nullable=False, allowed=(1, 2, 3)),
    Col("manufacturer_id", "str", nullable=False, unique=True),
    Col("labeler_name_fda", "str", nullable=False),
    Col("manufacturing_country", "str", nullable=False),
    Col("is_probation", "bool", nullable=False),
    Col("monthly_capacity_units", "int", nullable=False, min=0),
    Col("C1_landed_cost", "float", nullable=False, min=0, max=1),
    Col("C2_delivery_risk", "float", min=0, max=1),      # filled by XGBoost (Member 4A)
    Col("C3_price_integrity", "float", min=0, max=1),    # filled by Isolation Forest (Member 4A)
    Col("C4_compliance_risk", "float", min=0, max=1),    # filled by NLP + FDA composite (Member 4B)
    Col("C5_route_exposure", "float", nullable=False, min=0, max=1),
    Col("hormuz_air_share", "float", nullable=False, min=0, max=1),
    Col("active_fda_shortage", "bool", nullable=False),
    Col("oai_count_5yr", "int", nullable=False, min=0),
    Col("shap_top3_reasons", "str", nullable=False),     # JSON string; "[]" until SHAP runs
]
LOCKED_COLUMNS = [c.name for c in LOCKED_SCHEMA]


# --------------------------------------------------------------------------- #
def corridor_table(ref_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(ref_dir / "scenario_corridor_assumptions.csv")
    problems = []
    if df["iso3"].duplicated().any():
        problems.append(f"duplicate iso3 {df.loc[df.iso3.duplicated(), 'iso3'].tolist()}")
    if "*" not in set(df["iso3"]):
        problems.append("missing default row iso3='*'")
    for c in ["air_share", "sea_share", "land_share", "hormuz_sea_fraction", "gulf_air_fraction"]:
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
    for _, b in bench[bench["applies_to_labeler_core"].ne("*")].iterrows():
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
    if not p.exists() or p.stat().st_size <= 1:
        return pd.DataFrame()
    try:
        return pd.read_csv(p, low_memory=False)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


# --------------------------------------------------------------------------- #
# Entity linking (plan 5.2 step 3)
# --------------------------------------------------------------------------- #
def _alias_map(ref_dir: Path, panel: pd.DataFrame) -> dict[str, list[tuple[str, str]]]:
    p = ref_dir / "corporate_aliases.csv"
    al = pd.read_csv(p, dtype=str) if p.exists() else pd.DataFrame(
        columns=["labeler_core_contains", "alias_name", "relationship"])
    out = {}
    for _, lab in panel.iterrows():
        names = [(lab["labeler_key"], "self")]
        if isinstance(lab.get("parent_key"), str):
            names.append((lab["parent_key"], "parent_in_fda_name"))
        for _, a in al.iterrows():
            if a["labeler_core_contains"].upper() in str(lab["labeler_key"]):
                names.append((company_key(a["alias_name"]), a["relationship"]))
        out[lab["labeler_key"]] = [(n, r) for n, r in dict(names).items() if isinstance(n, str)]
    return out


def match_names(panel: pd.DataFrame, raw_names: pd.Series, source: str,
                aliases: dict[str, list[tuple[str, str]]]) -> pd.DataFrame:
    """One row per accepted (labeler, external name) link, with method and score."""
    thr = SETTINGS.sku.fuzzy_match_threshold
    uniq = [n for n in pd.Series(raw_names).dropna().astype(str).unique() if n.strip()]
    ext = [(n, company_key(n), company_core(n)) for n in uniq]
    rows = []
    for lab_key, targets in aliases.items():
        prepared = []
        for t, rel in targets:
            tcore = company_core(t)
            phrase = re.compile(rf"(?<![A-Z0-9]){re.escape(tcore)}(?![A-Z0-9])") \
                if isinstance(tcore, str) and len(tcore) >= 4 else None
            prepared.append((t, rel, tcore, phrase))
        for raw, k, c in ext:
            if not isinstance(k, str):
                continue
            best = None
            for t, rel, tcore, phrase in prepared:
                if c == tcore:
                    cand = ("exact_normalised_name", 100.0)
                else:
                    score = float(fuzz.token_sort_ratio(t, k)) if fuzz else 0.0
                    if score >= thr:
                        cand = ("rapidfuzz_token_sort_ratio", score)
                    elif phrase is not None and phrase.search(k):
                        cand = ("core_name_contained", 90.0)
                    else:
                        continue
                if best is None or cand[1] > best[1]:
                    best = (*cand, t, rel)
            if best:
                rows.append({"labeler_key": lab_key, "source": source, "external_name": raw,
                             "match_method": best[0], "score": best[1], "matched_via": best[2],
                             "relationship": best[3],
                             "needs_manual_check": best[0] != "exact_normalised_name" or best[3] != "self"})
    return pd.DataFrame(rows, columns=["labeler_key", "source", "external_name", "match_method", "score",
                                       "matched_via", "relationship", "needs_manual_check"])


def _count(panel: pd.DataFrame, links: pd.DataFrame, df: pd.DataFrame, name_cols: list[str]) -> pd.Series:
    """Rows of df whose name (any of name_cols) is linked to each labeler."""
    if df.empty or links.empty:
        return pd.Series(0, index=panel.index)
    by_lab = links.groupby("labeler_key")["external_name"].agg(set)
    cols = [c for c in name_cols if c in df.columns]
    out = []
    for k in panel["labeler_key"]:
        names = by_lab.get(k, set())
        mask = pd.Series(False, index=df.index)
        for c in cols:
            mask |= df[c].astype(str).isin(names)
        out.append(int(mask.sum()))
    return pd.Series(out, index=panel.index)


# --------------------------------------------------------------------------- #
def build(ref_dir: Path, interim: Path) -> dict[str, pd.DataFrame]:
    sku = SETTINGS.sku
    corr = corridor_table(ref_dir)
    bench = pd.read_csv(ref_dir / "price_benchmarks.csv")
    over = pd.read_csv(ref_dir / "manufacturer_overrides.csv", dtype=str)
    hints_p = ref_dir / "labeler_country_hints.csv"
    hints = pd.read_csv(hints_p, dtype=str) if hints_p.exists() else pd.DataFrame()
    labs = _read(interim, "fda_ndc_qualified_labelers")
    if labs.empty:
        QLOG.check(SRC, "panel_labelers_available", 1, severity="error",
                   detail="run openFDA ingestion first (or offline cache)")
        return {}
    labels = _read(interim, "fda_label_manufacturing_sites")
    wb = _read(interim, "wb_country_indicators")

    panel = labs[["sku_id", "labeler_key", "labeler_name_fda", "labeler_core", "parent_key",
                  "authorization_type", "n_products", "product_ndcs", "marketing_categories"]].copy()
    panel["criticality_tier"] = sku.criticality_tier

    # ---- plant country: manual override > SPL label text > HQ hint > Unknown ----------
    res = {}
    for _, lab in labs.iterrows():
        k = lab["labeler_key"]
        if not labels.empty:
            sets = set(str(lab.get("spl_set_ids", "")).split("|"))
            hit = labels[labels["spl_set_id"].isin(sets) & labels["label_country"].notna()]
            if hit.empty:
                hit = labels[(labels["labeler_key"] == k) & labels["label_country"].notna()]
            if not hit.empty:
                iso = hit["label_iso3"].value_counts().index[0]
                row = hit[hit["label_iso3"] == iso].iloc[0]
                res[k] = (row["label_country"], iso, "spl_label_text", str(row["manufactured_by_snippet"])[:200])
                continue
        for _, h in hints.iterrows():
            if h["labeler_core_contains"].upper() in str(k):
                res[k] = (h["manufacturing_country"], h["iso3"], "hq_hint_verify", h["basis"])
                break
    if not over.empty:
        for _, ov in over.iterrows():
            if isinstance(ov.get("manufacturing_country"), str) and ov["manufacturing_country"].strip():
                res[ov["labeler_key"]] = (ov["manufacturing_country"], ov["iso3"], "manual_override",
                                          ov.get("evidence"))
    for j, col in enumerate(["manufacturing_country", "manufacturing_iso3", "country_source", "country_evidence"]):
        panel[col] = panel["labeler_key"].map(lambda k, j=j: res.get(k, (None,) * 4)[j])
    unk = panel["manufacturing_iso3"].isna()
    QLOG.check(SRC, "labelers_without_plant_country", int(unk.sum()), len(panel),
               action="DEFAULT corridor row used; add to reference/manufacturer_overrides.csv",
               severity="warning" if unk.any() else "info", detail=", ".join(panel.loc[unk, "labeler_name_fda"]))
    hint = panel["country_source"].eq("hq_hint_verify")
    QLOG.check(SRC, "plant_country_from_hq_hint", int(hint.sum()), len(panel), action="flagged - verify plant",
               detail=", ".join(panel.loc[hint, "labeler_name_fda"]))
    panel["manufacturing_country"] = panel["manufacturing_country"].fillna("Unknown")

    panel["monthly_capacity_units"] = DEFAULT_CAPACITY
    panel["capacity_source"] = "SCENARIO default"
    if not over.empty and "monthly_capacity_units" in over:
        cap = pd.to_numeric(over.set_index("labeler_key")["monthly_capacity_units"], errors="coerce").dropna()
        m = panel["labeler_key"].map(cap)
        panel.loc[m.notna(), "capacity_source"] = "manual_override"
        panel["monthly_capacity_units"] = m.fillna(panel["monthly_capacity_units"]).astype(int)

    # ---- corridor + World Bank ------------------------------------------------------------
    c = corr.set_index("iso3")
    look = panel["manufacturing_iso3"].where(panel["manufacturing_iso3"].isin(c.index), "*").fillna("*")
    panel["corridor_row_used"] = look
    panel["hormuz_air_share"] = look.map(c["hormuz_air_share"]).round(4)
    panel["air_share"] = look.map(c["air_share"]).round(4)
    panel["freight_usd_per_vial"] = look.map(c["freight_usd_per_vial"]).round(4)
    if not wb.empty:
        w = wb.set_index("iso3")
        panel["lpi_score"] = panel["manufacturing_iso3"].map(w["lpi_score"])
        panel["lpi_year"] = panel["manufacturing_iso3"].map(w["lpi_score_year"])
        panel["wgi_political_stability"] = panel["manufacturing_iso3"].map(w["wgi_political_stability"])
        fallback_lpi = float(w["lpi_score"].median())
    else:
        panel[["lpi_score", "lpi_year", "wgi_political_stability"]] = np.nan
        fallback_lpi = 3.0
    panel["lpi_imputed"] = panel["lpi_score"].isna()
    QLOG.check(SRC, "labelers_missing_lpi", int(panel["lpi_imputed"].sum()), len(panel),
               action=f"imputed with all-country median LPI {fallback_lpi:.2f} + flagged",
               severity="warning" if panel["lpi_imputed"].any() else "info")
    panel["lpi_score"] = panel["lpi_score"].fillna(fallback_lpi)

    # ---- C1 quotes (plan 4.3) -----------------------------------------------------------------
    ins_rate, ins_basis = _insurance_rate(interim)
    q = []
    for _, r in panel.iterrows():
        b = _benchmark_for(r["labeler_core"], bench)
        unit = float(b["payment_limit_usd"]) * sku.reference_vial_mg / float(b["unit_mg"])
        q.append({"sku_id": r["sku_id"], "labeler_key": r["labeler_key"], "labeler_name_fda": r["labeler_name_fda"],
                  "reference_vial": f"{sku.reference_vial_mg} mg / "
                                    f"{sku.reference_vial_mg / sku.concentration_mg_per_ml:.0f} mL",
                  "base_unit_price_usd": round(unit, 2), "price_benchmark_id": b["benchmark_id"],
                  "manufacturing_country": r["manufacturing_country"],
                  "freight_usd_per_vial": r["freight_usd_per_vial"],
                  "insurance_usd_per_vial": round(unit * ins_rate, 4), "insurance_rate": round(ins_rate, 5),
                  "insurance_basis": ins_basis, "corridor_row_used": r["corridor_row_used"],
                  "is_scenario_assumption": True})
    quotes = pd.DataFrame(q)
    user_q = ref_dir / "quotes_2026_carboplatin_override.csv"
    if user_q.exists():  # team-supplied tender quotes win, validated
        uq = pd.read_csv(user_q)
        orphan = ~uq["labeler_key"].isin(quotes["labeler_key"])
        QLOG.check(SRC, "override_quotes_for_labelers_not_in_panel", int(orphan.sum()), len(uq),
                   action="ignored", severity="warning" if orphan.any() else "info")
        uq = uq[~orphan].drop_duplicates("labeler_key", keep="last").set_index("labeler_key")
        for col in ["base_unit_price_usd", "freight_usd_per_vial", "insurance_usd_per_vial"]:
            if col in uq:
                m = pd.to_numeric(quotes["labeler_key"].map(uq[col]), errors="coerce")
                ok = m.notna() & (m >= 0) & (m <= 1e5)
                quotes[col] = np.where(ok, m, quotes[col])
    quotes["landed_cost_usd_per_vial"] = quotes[["base_unit_price_usd", "freight_usd_per_vial",
                                                 "insurance_usd_per_vial"]].sum(axis=1).round(4)
    quotes = enforce_schema(quotes, [Col("labeler_key", "str", nullable=False, unique=True),
                                     Col("landed_cost_usd_per_vial", "float", nullable=False, min=0)], SRC)
    panel = panel.merge(quotes[["labeler_key", "base_unit_price_usd", "landed_cost_usd_per_vial"]],
                        on="labeler_key", how="left")

    # ---- entity links (rapidfuzz >= 88 + aliases) --------------------------------------------------
    sh = _read(interim, "fda_shortages_sku")
    rc = _read(interim, "fda_recalls_5yr")
    ins = _read(interim, "fda_inspections_drugs_5yr")
    if not ins.empty and "fei_number" in ins:   # FEI is an identifier, not a number
        ins["fei_number"] = ins["fei_number"].astype("string").str.replace(r"\.0$", "", regex=True)
    scms = _read(interim, "scms_shipments_clean")
    wl = _read(interim, "warning_letters_corpus")
    ct = _read(interim, "contracts_corpus")
    aliases = _alias_map(ref_dir, panel)
    QLOG.check(SRC, "rapidfuzz_available", 0 if fuzz else 1, severity="info" if fuzz else "warning",
               action="" if fuzz else "pip install rapidfuzz - falling back to exact/phrase matching")

    def col(df, c):
        return df[c] if (not df.empty and c in df.columns) else pd.Series(dtype=str)

    L = {
        "shortages": match_names(panel, col(sh, "company_name"), "openfda_shortages", aliases),
        "recalls": match_names(panel, col(rc, "recalling_firm"), "openfda_recalls", aliases),
        "inspections": match_names(panel, col(ins, "legal_name"), "fda_inspections", aliases),
        "scms": match_names(panel, pd.concat([col(scms, "vendor"), col(scms, "manufacturing_site")]),
                            "scms", aliases),
        "warning_letters": match_names(panel, col(wl, "company_name"), "warning_letters", aliases),
        "contracts": match_names(panel, col(ct, "supplier_name"), "contracts", aliases),
    }
    internal_po = _read(interim, "internal_erp_purchase_orders")
    L["internal"] = match_names(panel, col(internal_po, "supplier_name"), "internal_gpo", aliases)
    # manual verification (plan 5.2 step 3): reviewer decisions override the automatic links
    dec_p = ref_dir / "entity_match_decisions.csv"
    dec = pd.read_csv(dec_p, dtype=str) if dec_p.exists() else pd.DataFrame(
        columns=["labeler_key", "external_name", "decision"])
    dec["decision"] = dec["decision"].fillna("").str.strip().str.lower()
    dmap = {(r["labeler_key"], r["external_name"]): r["decision"] for _, r in dec.iterrows()}
    n_rej = 0
    for name, df_ in L.items():
        df_["manual_decision"] = [dmap.get((a, b), "") for a, b in zip(df_["labeler_key"], df_["external_name"])]
        rej = df_["manual_decision"].eq("reject")
        n_rej += int(rej.sum())
        df_.loc[df_["manual_decision"].eq("accept"), "needs_manual_check"] = False
        L[name] = df_[~rej].copy()
    QLOG.check(SRC, "entity_links_rejected_by_reviewer", n_rej, action="removed before counting",
               detail=f"{len(dec)} decision row(s) in reference/entity_match_decisions.csv")
    links = pd.concat(L.values(), ignore_index=True)
    QLOG.check(SRC, "entity_links_found", len(links), detail=str(links["source"].value_counts().to_dict()))
    QLOG.check(SRC, "entity_links_needing_manual_check", int(links["needs_manual_check"].sum()), len(links),
               action="review entity_match_candidates.csv (plan 5.2 step 3)")

    active = sh[sh["is_active"].astype(bool)] if not sh.empty else sh
    panel["active_shortage_records"] = _count(panel, L["shortages"], active, ["company_name"])
    panel["active_fda_shortage"] = panel["active_shortage_records"] > 0
    c12 = rc[rc["is_class_i_or_ii"].astype(bool)] if not rc.empty else rc
    panel["recalls_class_i_ii_5yr"] = _count(panel, L["recalls"], c12, ["recalling_firm"])
    panel["oai_count_5yr"] = _count(panel, L["inspections"],
                                    ins[ins["classification_code"].eq("OAI")] if not ins.empty else ins,
                                    ["legal_name"])
    panel["vai_count_5yr"] = _count(panel, L["inspections"],
                                    ins[ins["classification_code"].eq("VAI")] if not ins.empty else ins,
                                    ["legal_name"])
    panel["scms_historical_shipments"] = _count(panel, L["scms"], scms, ["vendor", "manufacturing_site"])
    panel["warning_letters_linked"] = _count(panel, L["warning_letters"], wl, ["company_name"])
    panel["contract_on_file"] = _count(panel, L["contracts"], ct, ["supplier_name"]) > 0
    # plan 5.1: probation = fewer than 5 historical shipments in internal/SCMS logs
    panel["internal_historical_deliveries"] = _count(panel, L["internal"], internal_po, ["supplier_name"])
    panel["historical_shipments_total"] = panel["scms_historical_shipments"] + panel["internal_historical_deliveries"]
    panel["is_probation"] = panel["historical_shipments_total"] < SETTINGS.scms.probation_min_shipments
    QLOG.check(SRC, "probation_labelers(<5 historical shipments)", int(panel["is_probation"].sum()), len(panel),
               detail=", ".join(panel.loc[~panel["is_probation"], "labeler_name_fda"]) + " have history")
    if ins.empty:
        QLOG.check(SRC, "oai_count_without_inspection_data", len(panel), severity="warning",
                   action="oai_count_5yr = 0 for all rows until inspections load")

    # ---- manual-check shortlist: top N non-exact links, compliance sources first (plan step 3) ------
    prio = {"fda_inspections": 0, "openfda_recalls": 1, "openfda_shortages": 2, "warning_letters": 3,
            "scms": 4, "internal_gpo": 5, "contracts": 6}
    todo = links[links["needs_manual_check"]].copy()
    todo["_p"] = todo["source"].map(prio).fillna(9)
    top_n = (todo.sort_values(["_p", "score", "labeler_key"]).drop(columns="_p")
             .head(sku.manual_check_top_n).assign(decision="", reviewer_note=""))

    # ---- facility-level inspections (plan 4.1: C4 = worst severity across all plants) ------------------
    fac_cols = ["labeler_key", "fei_number", "legal_name", "country", "n_classifications", "oai_5yr", "vai_5yr",
                "nai_5yr", "last_inspection_end_date", "worst_classification", "match_method"]
    fac = pd.DataFrame(columns=fac_cols)
    if not ins.empty and not L["inspections"].empty:
        j = L["inspections"].merge(ins, left_on="external_name", right_on="legal_name")
        if not j.empty:
            rank = {"NAI": 0, "VAI": 1, "OAI": 2}
            j["_r"] = j["classification_code"].map(rank)
            fac = j.groupby(["labeler_key", "fei_number", "legal_name"], dropna=False).agg(
                country=("country", "first"), n_classifications=("classification_code", "size"),
                oai_5yr=("classification_code", lambda x: int((x == "OAI").sum())),
                vai_5yr=("classification_code", lambda x: int((x == "VAI").sum())),
                nai_5yr=("classification_code", lambda x: int((x == "NAI").sum())),
                last_inspection_end_date=("inspection_end_date", "max"),
                worst_classification=("_r", lambda x: {v: k for k, v in rank.items()}.get(x.max())),
                match_method=("match_method", "first")).reset_index()[fac_cols]
    panel["n_fda_inspected_plants"] = panel["labeler_key"].map(fac.groupby("labeler_key").size()).fillna(0).astype(int)

    # ---- feature rows for C2 feature-based inference (plan 3.1) --------------------------------------
    ctry_pri = _read(interim, "scms_country_priors_train")
    corr_pri = _read(interim, "scms_corridor_features_train")
    inf = pd.DataFrame({
        "sku_id": panel["sku_id"], "manufacturer_id": panel["labeler_key"],
        "labeler_name_fda": panel["labeler_name_fda"],
        "manufacturing_country": panel["manufacturing_country"], "manufacturing_iso3": panel["manufacturing_iso3"],
        "shipment_mode": np.where(panel["air_share"] >= 0.5, "Air", "Ocean"),
        "is_air": panel["air_share"] >= 0.5, "air_share": panel["air_share"],
        "freight_to_value_ratio": (panel["freight_usd_per_vial"] / panel["base_unit_price_usd"]).round(6),
        "line_item_quantity": sku.monthly_demand_units,
        "line_item_value": (sku.monthly_demand_units * panel["base_unit_price_usd"]).round(2),
        "unit_price": panel["base_unit_price_usd"],
        "weight_kg": round(sku.monthly_demand_units * sku.vial_gross_weight_kg, 1),
        "vendor_inco_term": sku.panel_inco_term, "product_group": "ONCOLOGY_INJ",
    })
    if not scms.empty and {"late", "delay_days", "time_split"} <= set(scms.columns) and not L["scms"].empty:
        by_lab = L["scms"].groupby("labeler_key")["external_name"].agg(set)
        tr = scms[scms["time_split"] == "train"]
        pri = []
        for k in panel["labeler_key"]:
            names = by_lab.get(k, set())
            rows_ = tr[tr["vendor"].astype(str).isin(names) | tr["manufacturing_site"].astype(str).isin(names)]
            pri.append((len(rows_), 1 - rows_["late"].mean() if len(rows_) else np.nan,
                        rows_["delay_days"].mean() if len(rows_) else np.nan,
                        rows_["delay_days"].quantile(0.9) if len(rows_) else np.nan))
        inf[["site_prior_n_shipments", "site_prior_on_time_rate", "site_prior_mean_delay_days",
             "site_prior_p90_delay_days"]] = pd.DataFrame(pri, index=inf.index)
    else:
        inf["site_prior_n_shipments"] = 0
        for c_ in ["site_prior_on_time_rate", "site_prior_mean_delay_days", "site_prior_p90_delay_days"]:
            inf[c_] = np.nan
    inf["site_prior_available"] = inf["site_prior_n_shipments"] >= SETTINGS.scms.probation_min_shipments
    if not ctry_pri.empty:
        cp = ctry_pri.set_index("manufacturing_country")
        for c_ in ["country_n_shipments", "country_on_time_rate", "country_air_share",
                   "country_mean_freight_to_value"]:
            inf[c_] = inf["manufacturing_country"].map(cp[c_]) if c_ in cp else np.nan
    if not corr_pri.empty:
        kp = corr_pri.set_index(["manufacturing_country", "shipment_mode"])
        idx = list(zip(inf["manufacturing_country"], inf["shipment_mode"]))
        for c_ in ["corridor_n_shipments", "corridor_on_time_rate", "corridor_p90_delay_days"]:
            inf[c_] = [kp[c_].get(i, np.nan) for i in idx]
    inf["scenario_fields"] = "shipment_mode|air_share|freight_to_value_ratio|weight_kg|vendor_inco_term"
    QLOG.check(SRC, "c2_inference_rows_without_site_prior", int((~inf["site_prior_available"]).sum()), len(inf),
               action="use country / corridor prior (plan 3.1 'where available')")

    # ---- locked-schema suppliers_matrix (plan 5.1) ----------------------------------------------------
    lc = panel["landed_cost_usd_per_vial"]
    rng = lc.max() - lc.min()
    m = pd.DataFrame({
        "sku_id": panel["sku_id"], "criticality_tier": panel["criticality_tier"],
        "manufacturer_id": panel["labeler_key"], "labeler_name_fda": panel["labeler_name_fda"],
        "manufacturing_country": panel["manufacturing_country"], "is_probation": panel["is_probation"],
        "monthly_capacity_units": panel["monthly_capacity_units"],
        # C1 (4.3): unit landed cost, min-max normalised to [0,1]; zero-guarded
        "C1_landed_cost": ((lc - lc.min()) / rng).round(6) if rng > 0 else 0.0,
        "C2_delivery_risk": np.nan,
        # C3 (4.3): probation suppliers default to 0.0; others computed by Isolation Forest
        "C3_price_integrity": np.where(panel["is_probation"], 0.0, np.nan),
        "C4_compliance_risk": np.nan,
        # C5 (4.3), Normal mode: min(1, StressFactor x share) x (1 - LPI / 5), StressFactor = 1.0
        "C5_route_exposure": (np.minimum(1.0, 1.0 * panel["hormuz_air_share"])
                              * (1 - panel["lpi_score"] / 5.0)).round(6),
        "hormuz_air_share": panel["hormuz_air_share"], "active_fda_shortage": panel["active_fda_shortage"],
        "oai_count_5yr": panel["oai_count_5yr"], "shap_top3_reasons": json.dumps([]),
    })
    assert list(m.columns) == LOCKED_COLUMNS
    # supporting raw inputs, appended AFTER the 16 locked columns (never between them)
    # Crisis mode (plan 4.2): corridor exposure x 1.5 stress factor, capped at 1.0
    panel["C5_route_exposure_crisis"] = (np.minimum(1.0, 1.5 * panel["hormuz_air_share"])
                                         * (1 - panel["lpi_score"] / 5.0)).round(6)
    # FDA-only part of C4 (plan 4.3); the two NLP terms (0.20 + 0.20) are added by Member 4B
    panel["C4_fda_component_partial"] = (0.30 * (panel["oai_count_5yr"] > 0) + 0.10 * (panel["vai_count_5yr"] > 0)
                                         + 0.20 * np.minimum(1, panel["recalls_class_i_ii_5yr"] / 2)).round(4)
    for extra in ["landed_cost_usd_per_vial", "C5_route_exposure_crisis", "C4_fda_component_partial",
                  "lpi_score", "lpi_imputed", "wgi_political_stability",
                  "vai_count_5yr", "recalls_class_i_ii_5yr", "n_fda_inspected_plants", "active_shortage_records",
                  "scms_historical_shipments", "internal_historical_deliveries", "warning_letters_linked",
                  "contract_on_file", "authorization_type", "country_source", "capacity_source"]:
        m[extra] = panel[extra].values
    m = m.rename(columns={"landed_cost_usd_per_vial": "C1_landed_cost_usd_raw"})
    m = enforce_schema(m, LOCKED_SCHEMA, "suppliers_matrix")
    assert list(m.columns[:len(LOCKED_COLUMNS)]) == LOCKED_COLUMNS
    for c_ in ["C2_delivery_risk", "C4_compliance_risk"]:
        QLOG.check("suppliers_matrix", f"{c_}_pending_downstream_engine", int(m[c_].isna().sum()), len(m),
                   action="left empty for Layer 3 (not a Layer 1 output)")
    return {"quotes_2026_carboplatin": quotes, "scenario_corridors_validated": corr,
            "layer1_scoring_panel_seed": panel, "entity_match_candidates": links,
            "entity_match_top20_manual_check": top_n, "panel_facility_inspections": fac,
            "panel_inference_features": inf, "suppliers_matrix": m}


def plan_vs_live(interim: Path, matrix: pd.DataFrame) -> pd.DataFrame:
    """Figures the plan quotes (section 3) next to what this run actually returned."""
    def n(name):
        return len(_read(interim, name))
    onc = _read(interim, "fda_shortages_oncology_current")
    labs = _read(interim, "fda_ndc_qualified_labelers")
    fin = int((labs["authorization_type"] == "finished_injection").sum()) if len(labs) else 0
    rows = [
        ("SCMS shipments in training layer", "~10,300", n("scms_shipments_clean"),
         "raw file has 10,324 lines; rejected rows are in data/quarantine"),
        ("Carboplatin labelers in openFDA NDC directory", "18", len(labs),
         f"{fin} hold a finished-injection listing, {len(labs) - fin} are bulk-API only"),
        ("Carboplatin shortage records (openFDA)", "35", n("fda_shortages_sku"),
         "after keeping the latest status row per company + package"),
        ("Active oncology shortages", "84", len(onc),
         f"{onc['generic_name'].str.lower().nunique() if len(onc) else 0} distinct molecules; "
         "the plan does not say whether 84 counts records or molecules"),
        ("FDA CGMP warning letters collected", "50-100", n("warning_letters_corpus"), "51 URLs shipped"),
        ("Manufacturers in suppliers_matrix.csv", "18", len(matrix), "one row per manufacturer x SKU"),
    ]
    out = pd.DataFrame(rows, columns=["item", "plan_figure", "this_run", "note"])
    def status(r):
        if r["this_run"] == 0:
            return "no data this run"
        nums = [int(x.replace(",", "")) for x in re.findall(r"[\d,]+", r["plan_figure"])]
        if "~" in r["plan_figure"]:
            return "matches" if abs(r["this_run"] - nums[0]) <= 0.02 * nums[0] else "differs"
        if len(nums) == 2:
            return "matches" if nums[0] <= r["this_run"] <= nums[1] else "differs"
        return "matches" if r["this_run"] == nums[0] else "differs - use the live figure on slides"
    out["status"] = out.apply(status, axis=1)
    return out


def run(ref_dir: Path, interim: Path, **_) -> dict:
    try:
        tabs = build(ref_dir, interim)
    except Exception as e:
        log.exception("scenario/panel build failed")
        QLOG.status(SRC, "FAILED", note=str(e)[:200])
        return {}
    if not tabs:
        QLOG.status(SRC, "FAILED", note="no panel labelers available")
        return {}
    processed = SETTINGS.paths.processed
    out = {k: write_table(v, k) for k, v in tabs.items() if k != "suppliers_matrix"}
    out["suppliers_matrix"] = write_table(tabs["suppliers_matrix"], "suppliers_matrix", processed)
    out["quotes_2026_carboplatin_processed"] = write_table(tabs["quotes_2026_carboplatin"],
                                                           "quotes_2026_carboplatin", processed)
    m = tabs["suppliers_matrix"]
    pvl = plan_vs_live(interim, m)
    out["plan_vs_live_reconciliation"] = write_table(pvl, "plan_vs_live_reconciliation")
    diff = pvl[pvl["status"].str.startswith("differs")]
    QLOG.check(SRC, "plan_figures_that_differ_from_live_data", len(diff), len(pvl),
               severity="warning" if len(diff) else "info", action="quote the live figure on slides",
               detail="; ".join(f"{r['item']}: plan {r['plan_figure']} vs {r['this_run']}" for _, r in diff.iterrows()))
    QLOG.status(SRC, "OK", rows=len(tabs["layer1_scoring_panel_seed"]), note="quotes + entity links built")
    QLOG.status("suppliers_matrix", "OK", rows=len(m),
                note=f"locked schema (plan 5.1); {int(m.is_probation.sum())} probation; C2/C4 pending Layer 3")
    return out
