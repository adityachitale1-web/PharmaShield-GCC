"""
Sources 2-5 - openFDA live feeds (scoring layer).

  * NDC Directory      -> hard qualification gate per SKU (UAE registration proxy)
  * Drug Shortages     -> live alert feed + active-shortage penalty (C4 / dashboard tab 2)
  * Drug Enforcement   -> Class I/II/III recalls, last 5 years (C4)
  * Drug Label (SPL)   -> "Manufactured by: ... India" text -> plant country per labeler
                          (feeds manufacturing_country, C2 inference and C5 route exposure)

Data issues handled:
  * pagination (limit 1000, skip cap 25,000 -> automatic date windowing for recalls)
  * 404 NOT_FOUND = zero matches (not an error); 429/5xx back-off; cached replay
  * the same product_ndc appearing twice in the NDC index (multiple SPL versions)
  * non-target rows: bulk API powder, repackagers, wrong concentration, expired listings,
    combination products  -> written to quarantine with the exact exclusion reason
  * strength strings in many shapes ('10 mg/mL', '450 mg/45mL', '80 mg/8mL') -> parsed to mg/mL
  * shortage duplicates per package and status-history rows -> latest update wins
  * date formats differ per endpoint (YYYYMMDD vs MM/DD/YYYY) -> explicit parsing
  * company names: 'Hospira, Inc., a Pfizer Company' -> key 'HOSPIRA' + parent 'PFIZER'
"""
from __future__ import annotations

import logging
import re
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from .common import QLOG, Col, HttpClient, clean_text, company_core, company_key, enforce_schema, \
    parent_company, quarantine, write_table
from .config import SETTINGS

log = logging.getLogger("pharmashield.ingestion.openfda")


# --------------------------------------------------------------------------- #
# Generic paginated fetch
# --------------------------------------------------------------------------- #
def _is_not_found(resp) -> bool:
    if resp.status_code != 404:
        return False
    try:
        return resp.json().get("error", {}).get("code") == "NOT_FOUND"
    except ValueError:
        return False


def fetch_all(http: HttpClient, endpoint: str, search: str, sort: str | None = None) -> tuple[list, dict]:
    """Use plain spaces in `search` ('a AND b'); requests encodes them as '+'.
    A literal '+' would be sent as %2B and silently break the openFDA query."""
    cfg = SETTINGS.openfda
    url = f"{cfg.base_url}/{endpoint}.json"
    results, skip, meta = [], 0, {}
    while True:
        params = {"search": search, "limit": cfg.page_limit, "skip": skip}
        if sort:
            params["sort"] = sort
        if cfg.api_key:
            params["api_key"] = cfg.api_key
        payload = http.request_json(f"openfda/{endpoint.replace('/', '_')}", url, params=params,
                                    empty_on=_is_not_found)
        if not payload:
            break
        meta = payload.get("meta", {})
        page = payload.get("results", [])
        results.extend(page)
        total = meta.get("results", {}).get("total", len(results))
        skip += cfg.page_limit
        if len(results) >= total or not page:
            break
        if skip > cfg.max_skip:
            log.error("%s: total=%s exceeds openFDA skip cap; narrow the search window", endpoint, total)
            QLOG.check(f"openfda_{endpoint}", "skip_cap_truncation", total - len(results), total,
                       severity="error", action="narrow query window")
            break
    log.info("openFDA %s [%s]: %d records (last_updated=%s)", endpoint, search, len(results),
             meta.get("last_updated"))
    return results, meta


def _first(v):
    if isinstance(v, list):
        return v[0] if v else None
    return v


def _join(v):
    if isinstance(v, list):
        return "|".join(sorted({str(x) for x in v if x is not None}))
    return v


def _yyyymmdd(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s.astype("string").str.slice(0, 8), format="%Y%m%d", errors="coerce")


# --------------------------------------------------------------------------- #
# NDC directory -> qualification gate
# --------------------------------------------------------------------------- #
_STRENGTH_RE = re.compile(r"([\d.,]+)\s*(mg|g|mcg|ug)\s*/\s*([\d.,]*)\s*(ml|l)\b", re.I)
_UNIT_MG = {"mg": 1.0, "g": 1000.0, "mcg": 0.001, "ug": 0.001}
_UNIT_ML = {"ml": 1.0, "l": 1000.0}


def strength_mg_per_ml(strength: str | None) -> float | None:
    """'10 mg/mL' -> 10 ; '450 mg/45mL' -> 10 ; '1 g/100 mL' -> 10 ; unparseable -> None."""
    if not isinstance(strength, str):
        return None
    m = _STRENGTH_RE.search(strength.replace(" ", " "))
    if not m:
        return None
    num, mu, den, vu = m.groups()
    try:
        mg = float(num.replace(",", "")) * _UNIT_MG[mu.lower()]
        ml = (float(den.replace(",", "")) if den else 1.0) * _UNIT_ML[vu.lower()]
        return round(mg / ml, 4) if ml else None
    except ValueError:
        return None


def ndc_products(http: HttpClient) -> pd.DataFrame:
    sku = SETTINGS.sku
    queries = [f"generic_name:{sku.molecule.lower()}",
               f"active_ingredients.name:{sku.molecule.lower()}",
               "brand_name:paraplatin"]
    raw, seen_ids = [], set()
    for q in queries:
        res, _ = fetch_all(http, "drug/ndc", q)
        for r in res:
            pid = r.get("product_id") or (r.get("product_ndc"), r.get("spl_id"))
            if pid not in seen_ids:
                seen_ids.add(pid)
                raw.append(r)
    QLOG.check("openfda_ndc", "records_fetched_union_of_queries", len(raw), detail=" | ".join(queries))
    rows = []
    for r in raw:
        ings = r.get("active_ingredients") or []
        of = r.get("openfda") or {}
        pk = r.get("packaging") or []
        rows.append({
            "product_ndc": r.get("product_ndc"), "product_id": r.get("product_id"), "spl_id": r.get("spl_id"),
            "labeler_name_fda": clean_text(r.get("labeler_name")), "brand_name": r.get("brand_name"),
            "generic_name": r.get("generic_name"), "dosage_form": r.get("dosage_form"),
            "route": _join(r.get("route")), "product_type": r.get("product_type"),
            "marketing_category": r.get("marketing_category"), "application_number": r.get("application_number"),
            "finished": r.get("finished"), "n_active_ingredients": len(ings),
            "active_ingredient_names": "|".join(i.get("name", "") for i in ings),
            "strength": ings[0].get("strength") if ings else None,
            "marketing_start_date": r.get("marketing_start_date"),
            "marketing_end_date": r.get("marketing_end_date"),
            "listing_expiration_date": r.get("listing_expiration_date"),
            "package_ndcs": "|".join(p.get("package_ndc", "") for p in pk),
            "package_descriptions": " || ".join(p.get("description", "") for p in pk),
            "spl_set_id": _first(of.get("spl_set_id")),
            "is_original_packager": _first(of.get("is_original_packager")),
            "openfda_manufacturer_name": _first(of.get("manufacturer_name")),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    for c in ["marketing_start_date", "marketing_end_date", "listing_expiration_date"]:
        df[c] = _yyyymmdd(df[c])
    df["concentration_mg_per_ml"] = df["strength"].map(strength_mg_per_ml)
    df["labeler_key"] = df["labeler_name_fda"].map(company_key)
    df["labeler_core"] = df["labeler_name_fda"].map(company_core)
    df["parent_key"] = df["labeler_name_fda"].map(parent_company)

    # ---- duplicates: same product_ndc listed under >1 SPL version -------------
    df = df.sort_values(["product_ndc", "listing_expiration_date", "marketing_start_date"],
                        ascending=[True, False, False])
    dup = df.duplicated("product_ndc", keep="first")
    quarantine("openfda_ndc", df[dup], "duplicate_product_ndc_older_spl_version")
    QLOG.check("openfda_ndc", "duplicate_product_ndc", dup.sum(), len(df), action="kept latest listing",
               severity="warning" if dup.any() else "info")
    df = df[~dup].copy()

    # ---- qualification rules ---------------------------------------------------
    today = pd.Timestamp(SETTINGS.as_of)
    mol = sku.molecule.upper()
    rules = {
        "not_human_rx": ~df["product_type"].fillna("").str.upper().eq("HUMAN PRESCRIPTION DRUG"),
        "not_finished_dosage_form": df["finished"].eq(False),
        "not_injection": ~df["dosage_form"].fillna("").str.upper().str.contains("INJECTION"),
        "combination_or_wrong_molecule": ~(df["n_active_ingredients"].eq(1)
                                           & df["active_ingredient_names"].str.upper().str.contains(mol)),
        "concentration_mismatch": pd.Series(~np.isclose(df["concentration_mg_per_ml"].astype(float),
                                                        sku.concentration_mg_per_ml, atol=0.01), index=df.index),
        "listing_expired": df["listing_expiration_date"].notna() & (df["listing_expiration_date"] < today),
        "marketing_ended": df["marketing_end_date"].notna() & (df["marketing_end_date"] < today),
        "unapproved_marketing_category": ~df["marketing_category"].fillna("").str.upper()
            .isin(["ANDA", "NDA", "BLA", "NDA AUTHORIZED GENERIC"]),
    }
    reasons = pd.Series("", index=df.index)
    for name, mask in rules.items():
        mask = mask.fillna(True) if name == "concentration_mismatch" else mask.fillna(False)
        reasons = reasons.where(~mask, reasons + name + ";")
        QLOG.check("openfda_ndc", f"excluded:{name}", int(mask.sum()), len(df), action="excluded from SKU gate")
    df["exclusion_reasons"] = reasons.str.rstrip(";")
    df["qualifies_for_sku"] = df["exclusion_reasons"].eq("")
    df["repackager_flag"] = df["is_original_packager"].eq(False)
    # bulk-API listings of the same molecule (e.g. Cipla / Hetero carboplatin powder): not a
    # finished-injection authorization, but part of the plan's "labelers in the NDC directory"
    is_bulk = (df["finished"].eq(False) | df["product_type"].fillna("").str.upper().str.contains("BULK")
               | df["dosage_form"].fillna("").str.upper().str.contains("POWDER"))
    live = ~rules["listing_expired"].fillna(False) & ~rules["marketing_ended"].fillna(False)
    single = ~rules["combination_or_wrong_molecule"].fillna(False)
    df["authorization_type"] = np.where(df["qualifies_for_sku"], "finished_injection",
                                        np.where(is_bulk & live & single, "bulk_api", "excluded"))
    if sku.panel_scope == "all_ndc_labelers":
        df["in_scoring_panel"] = df["authorization_type"].ne("excluded")
    else:
        df["in_scoring_panel"] = df["qualifies_for_sku"]
    QLOG.check("openfda_ndc", "products_in_scoring_panel", int(df["in_scoring_panel"].sum()), len(df),
               detail=f"panel_scope={sku.panel_scope}; "
                      f"{df['authorization_type'].value_counts().to_dict()}")
    quarantine("openfda_ndc", df[~df["in_scoring_panel"]], "not_in_scoring_panel (see exclusion_reasons)")
    df.insert(0, "sku_id", sku.sku_id)
    return df


def ndc_labelers(products: pd.DataFrame) -> pd.DataFrame:
    q = products[products["in_scoring_panel"]]
    if q.empty:
        QLOG.check("openfda_ndc", "qualified_labelers", 0, severity="error",
                   detail="No labeler passed the SKU gate - check query/filters")
        return pd.DataFrame()
    g = q.groupby(["sku_id", "labeler_key"], dropna=False)
    lab = g.agg(
        labeler_name_fda=("labeler_name_fda", lambda s: s.mode().iat[0]),
        labeler_name_variants=("labeler_name_fda", lambda s: "|".join(sorted(set(s)))),
        labeler_core=("labeler_core", "first"),
        parent_key=("parent_key", "first"),
        n_products=("product_ndc", "nunique"),
        product_ndcs=("product_ndc", lambda s: "|".join(sorted(set(s)))),
        package_ndcs=("package_ndcs", lambda s: "|".join(sorted({p for x in s for p in str(x).split("|") if p}))),
        application_numbers=("application_number", lambda s: "|".join(sorted(set(s.dropna())))),
        marketing_categories=("marketing_category", lambda s: "|".join(sorted(set(s.dropna())))),
        spl_set_ids=("spl_set_id", lambda s: "|".join(sorted(set(s.dropna())))),
        strengths=("strength", lambda s: "|".join(sorted(set(s.dropna())))),
        any_repackager_listing=("repackager_flag", "any"),
        holds_finished_injection_authorization=("qualifies_for_sku", "any"),
        earliest_marketing_start=("marketing_start_date", "min"),
        latest_listing_expiry=("listing_expiration_date", "max"),
    ).reset_index()
    lab["authorization_type"] = np.where(lab["holds_finished_injection_authorization"],
                                         "finished_injection", "bulk_api_only")
    QLOG.check("openfda_ndc", "labelers_bulk_api_only", int((lab["authorization_type"] == "bulk_api_only").sum()),
               len(lab), action="kept in panel + flagged (no finished-injection authorization)",
               detail=", ".join(lab.loc[lab["authorization_type"] == "bulk_api_only", "labeler_name_fda"]))
    # same company under two spellings collapses here because the key ignores suffix/punctuation
    QLOG.check("openfda_ndc", "qualified_labelers", len(lab), detail=", ".join(lab["labeler_name_fda"]))
    QLOG.check("openfda_ndc", "labelers_with_multiple_name_spellings",
               int((lab["labeler_name_variants"].str.count(r"\|") > 0).sum()), len(lab),
               action="collapsed on normalised key")
    schema = [Col("sku_id", "str", nullable=False), Col("labeler_key", "str", nullable=False, unique=True),
              Col("labeler_name_fda", "str", nullable=False), Col("n_products", "int", min=1)]
    return enforce_schema(lab, schema, "openfda_ndc")


# --------------------------------------------------------------------------- #
# Drug label -> plant country
# --------------------------------------------------------------------------- #
_MFG_RE = re.compile(r"(manufactured\s+by|mfd\.?\s+by|made\s+in|manufactured\s+in|"
                     r"product\s+of|manufacturer\s*:)\s*:?\s*(.{0,260})", re.I | re.S)


def _country_lookup(ref_dir: Path) -> list[tuple[re.Pattern, str, str]]:
    ref = pd.read_csv(ref_dir / "countries.csv", dtype=str, keep_default_na=False)
    pats = []
    for _, r in ref.iterrows():
        names = [r["country"]] + [a for a in r["aliases"].split("|") if a]
        for n in sorted(set(names), key=len, reverse=True):
            # word-boundary, case-sensitive for short ALL-CAPS codes (avoid 'us' in text)
            flags = 0 if (len(n) <= 3 and n.isupper()) else re.I
            pats.append((re.compile(rf"(?<![A-Za-z]){re.escape(n)}(?![A-Za-z])", flags), r["country"], r["iso3"]))
    return pats


def label_manufacturing_sites(http: HttpClient, ref_dir: Path) -> pd.DataFrame:
    sku = SETTINGS.sku
    res, _ = fetch_all(http, "drug/label", f"openfda.generic_name:{sku.molecule.lower()}")
    pats = _country_lookup(ref_dir)
    text_fields = ["spl_unclassified_section", "package_label_principal_display_panel",
                   "how_supplied", "storage_and_handling", "information_for_patients"]
    rows = []
    for r in res:
        of = r.get("openfda") or {}
        blob = " ".join(" ".join(r.get(f, []) if isinstance(r.get(f), list) else [str(r.get(f, ""))])
                        for f in text_fields)
        blob = re.sub(r"\s+", " ", blob)
        snippets = [m.group(0)[:300] for m in _MFG_RE.finditer(blob)]
        found = []
        for snip in snippets:
            is_for = bool(re.match(r"\s*manufactured\s+for|distributed\s+by", snip, re.I))
            for pat, ctry, iso in pats:
                m = pat.search(snip)
                if m:
                    found.append((m.start(), ctry, iso, is_for))
            # US addresses usually omit the country: 'Lake Forest, IL 60045'
            us = re.search(r",\s*[A-Z]{2}\s+\d{5}(?:-\d{4})?\b", snip)
            if us and not any(f[1] == "United States" for f in found):
                found.append((us.start(), "United States", "USA", is_for))
            # earliest-mentioned country in each snippet = address country
        found.sort()
        mfg = [f for f in found if not f[3]]
        best = mfg[0] if mfg else (found[0] if found else None)
        rows.append({
            "spl_set_id": r.get("set_id"), "spl_id": r.get("id"), "effective_time": r.get("effective_time"),
            "openfda_manufacturer_name": _first(of.get("manufacturer_name")),
            "labeler_key": company_key(_first(of.get("manufacturer_name"))),
            "product_ndcs": _join(of.get("product_ndc")),
            "label_country": best[1] if best else None, "label_iso3": best[2] if best else None,
            "all_countries_mentioned": "|".join(sorted({f[1] for f in found})),
            "manufactured_by_snippet": snippets[0] if snippets else None,
            "n_snippets": len(snippets),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["effective_time"] = _yyyymmdd(df["effective_time"])
    df = df.sort_values("effective_time", ascending=False)
    dup = df.duplicated("spl_set_id", keep="first")
    quarantine("openfda_label", df[dup], "older_label_version")
    df = df[~dup]
    multi = df["all_countries_mentioned"].str.contains(r"\|", na=False)
    QLOG.check("openfda_label", "labels_without_country", int(df["label_country"].isna().sum()), len(df),
               action="manual override via reference/manufacturer_overrides.csv",
               severity="warning" if df["label_country"].isna().any() else "info")
    QLOG.check("openfda_label", "labels_mentioning_multiple_countries", int(multi.sum()), len(df),
               action="first 'manufactured by' country used; review snippet")
    return df


# --------------------------------------------------------------------------- #
# Shortages
# --------------------------------------------------------------------------- #
def _shortage_frame(res: list) -> pd.DataFrame:
    rows = []
    for r in res:
        of = r.get("openfda") or {}
        rows.append({
            "generic_name": clean_text(r.get("generic_name")), "company_name": clean_text(r.get("company_name")),
            "status": clean_text(r.get("status")), "availability": clean_text(r.get("availability")),
            "shortage_reason": clean_text(r.get("shortage_reason")), "dosage_form": r.get("dosage_form"),
            "presentation": clean_text(r.get("presentation")), "package_ndc": r.get("package_ndc"),
            "therapeutic_category": _join(r.get("therapeutic_category")),
            "initial_posting_date": r.get("initial_posting_date"), "update_date": r.get("update_date"),
            "update_type": r.get("update_type"), "related_info": clean_text(r.get("related_info")),
            "openfda_manufacturer_name": _first(of.get("manufacturer_name")),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    for c in ["initial_posting_date", "update_date"]:
        df[c] = pd.to_datetime(df[c], format="%m/%d/%Y", errors="coerce")
    df["status"] = df["status"].str.title()
    df["company_key"] = df["company_name"].map(company_key)
    df["company_core"] = df["company_name"].map(company_core)
    df["parent_key"] = df["company_name"].map(parent_company)
    df["is_active"] = df["status"].isin(["Current", "To Be Discontinued"]) | \
        df["availability"].fillna("").str.contains("Unavailable|Limited", case=False)
    df["is_active"] &= ~df["status"].eq("Resolved")
    df["concentration_mg_per_ml"] = df["presentation"].map(strength_mg_per_ml)
    return df


def shortages(http: HttpClient) -> tuple[pd.DataFrame, pd.DataFrame]:
    sku = SETTINGS.sku
    res, _ = fetch_all(http, "drug/shortages", f"generic_name:{sku.molecule.lower()}")
    sku_df = _shortage_frame(res)
    res_onc, _ = fetch_all(http, "drug/shortages",
                           f'therapeutic_category:"{sku.therapeutic_category}" AND status:"Current"')
    onc = _shortage_frame(res_onc)
    out = []
    for name, df in (("openfda_shortages_sku", sku_df), ("openfda_shortages_oncology", onc)):
        if df.empty:
            QLOG.check(name, "records", 0, severity="warning")
            out.append(df)
            continue
        n = len(df)
        exact = df.duplicated(keep="first")
        df = df[~exact]
        df = df.sort_values("update_date", ascending=False)
        key = ["company_key", "package_ndc", "presentation"]
        dup = df.duplicated(key, keep="first")
        quarantine(name, df[dup], "older_status_row_same_package")
        df = df[~dup]
        QLOG.check(name, "duplicates_removed", int(exact.sum() + dup.sum()), n,
                   action="kept latest update_date per company+package")
        bad_dates = df["update_date"].isna()
        QLOG.check(name, "unparseable_update_date", int(bad_dates.sum()), len(df),
                   severity="warning" if bad_dates.any() else "info")
        future = df["update_date"] > pd.Timestamp(SETTINGS.as_of) + pd.Timedelta(days=1)
        QLOG.check(name, "update_date_in_future", int(future.sum()), len(df),
                   severity="warning" if future.any() else "info")
        out.append(df)
    sku_df, onc = out
    if not onc.empty:
        QLOG.check("openfda_shortages_oncology", "distinct_active_oncology_molecules",
                   onc["generic_name"].str.lower().nunique(), len(onc))
    if not sku_df.empty:
        sku_df.insert(0, "sku_id", sku.sku_id)
        sku_df["matches_sku_concentration"] = np.isclose(
            sku_df["concentration_mg_per_ml"].astype(float), sku.concentration_mg_per_ml, atol=0.01)
    return sku_df, onc


# --------------------------------------------------------------------------- #
# Enforcement (recalls)
# --------------------------------------------------------------------------- #
def recalls(http: HttpClient) -> pd.DataFrame:
    cfg = SETTINGS.openfda
    end = SETTINGS.as_of
    start = date(end.year - cfg.recall_lookback_years, end.month, min(end.day, 28))
    res = []
    # one window per calendar year keeps every query far below the 25k skip cap
    y0 = start
    while y0 <= end:
        y1 = min(date(y0.year, 12, 31), end)
        q = f"report_date:[{y0:%Y%m%d} TO {y1:%Y%m%d}]"
        page, _ = fetch_all(http, "drug/enforcement", q)
        res.extend(page)
        y0 = y1 + timedelta(days=1)
    rows = [{k: (clean_text(v) if isinstance(v, str) else v) for k, v in r.items() if k != "openfda"}
            for r in res]
    df = pd.DataFrame(rows)
    if df.empty:
        QLOG.check("openfda_recalls", "records", 0, severity="warning")
        return df
    n = len(df)
    for c in ["recall_initiation_date", "center_classification_date", "report_date", "termination_date"]:
        if c in df:
            df[c] = _yyyymmdd(df[c])
    dup = df.duplicated("recall_number", keep="last")
    quarantine("openfda_recalls", df[dup], "duplicate_recall_number")
    df = df[~dup].copy()
    QLOG.check("openfda_recalls", "duplicate_recall_number", int(dup.sum()), n, action="kept last")
    df["recall_class"] = df["classification"].str.extract(r"Class\s+(I{1,3})", expand=False)
    bad = df["recall_class"].isna()
    QLOG.check("openfda_recalls", "unparseable_classification", int(bad.sum()), len(df),
               severity="warning" if bad.any() else "info")
    df["is_class_i_or_ii"] = df["recall_class"].isin(["I", "II"])
    desc = df.get("product_description", pd.Series("", index=df.index)).fillna("")
    df["is_sterile_injectable"] = desc.str.contains(r"inject|infusion|intravenous|\bIV\b|vial|sterile",
                                                    case=False, regex=True)
    df["mentions_target_molecule"] = desc.str.contains(SETTINGS.sku.molecule, case=False)
    reason = df.get("reason_for_recall", pd.Series("", index=df.index)).fillna("")
    df["reason_sterility"] = reason.str.contains(r"steril|particulate|microb|endotoxin|contamina", case=False)
    df["reason_cgmp"] = reason.str.contains(r"CGMP|good manufacturing", case=False)
    df["firm_key"] = df["recalling_firm"].map(company_key)
    df["firm_core"] = df["recalling_firm"].map(company_core)
    df["parent_key"] = df["recalling_firm"].map(parent_company)
    schema = [Col("recall_number", "str", nullable=False, unique=True),
              Col("recalling_firm", "str", nullable=False), Col("report_date", "date", nullable=False),
              Col("recall_class", "str", allowed=("I", "II", "III"))]
    return enforce_schema(df, schema, "openfda_recalls")


# --------------------------------------------------------------------------- #
def run(http: HttpClient, ref_dir: Path, **_) -> dict:
    out = {}
    try:
        prods = ndc_products(http)
    except Exception as e:
        log.exception("NDC directory failed")
        prods = pd.DataFrame()
        QLOG.check("openfda_ndc", "fetch_failed", 1, severity="error", detail=str(e)[:200])
    if prods.empty:
        QLOG.status("openfda_ndc", "FAILED", note="no NDC records returned")
    else:
        labs = ndc_labelers(prods)
        out["fda_ndc_products"] = write_table(prods, "fda_ndc_products")
        out["fda_ndc_qualified_labelers"] = write_table(labs, "fda_ndc_qualified_labelers")
        QLOG.status("openfda_ndc", "OK", rows=len(labs),
                    note=f"{len(labs)} labelers in panel "
                         f"({int((labs.authorization_type == 'finished_injection').sum())} finished injection, "
                         f"{int((labs.authorization_type == 'bulk_api_only').sum())} bulk API only)")

    for name, fn, status_key in (("fda_label_manufacturing_sites",
                                  lambda: label_manufacturing_sites(http, ref_dir), "openfda_label"),
                                 ("fda_recalls_5yr", lambda: recalls(http), "openfda_recalls")):
        try:
            df = fn()
            out[name] = write_table(df, name)
            QLOG.status(status_key, "OK", rows=len(df))
        except Exception as e:  # one failing feed must not kill the run
            log.exception("%s failed", name)
            QLOG.status(status_key, "FAILED", note=str(e)[:200])

    try:
        sku_sh, onc_sh = shortages(http)
        out["fda_shortages_sku"] = write_table(sku_sh, "fda_shortages_sku")
        out["fda_shortages_oncology_current"] = write_table(onc_sh, "fda_shortages_oncology_current")
        QLOG.status("openfda_shortages", "OK", rows=len(sku_sh),
                    note=f"{len(sku_sh)} SKU records ({int(sku_sh['is_active'].sum()) if len(sku_sh) else 0} active); "
                         f"oncology current: {len(onc_sh)} records / "
                         f"{onc_sh['generic_name'].str.lower().nunique() if len(onc_sh) else 0} molecules")
    except Exception as e:
        log.exception("shortages failed")
        QLOG.status("openfda_shortages", "FAILED", note=str(e)[:200])
    return out
