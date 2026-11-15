"""
Source 6 - FDA Data Dashboard inspection classifications (NAI / VAI / OAI) -> C4.

Two ingestion paths, tried in order:
  A. API  POST https://api-datadashboard.fda.gov/v1/inspections_classifications
     needs free credentials (env FDA_DD_USER / FDA_DD_KEY, from the OII Unified Logon).
  B. Manual export: download "Inspections" from datadashboard.fda.gov (filter Product Type =
     Drugs) and drop the .xlsx/.csv in data/manual/fda_inspections/. Several files are fine;
     overlapping exports are de-duplicated.

Data issues handled:
  * API short field names vs export long names ('FEINumber' vs 'FEI Number') -> one schema
  * classification spelled 3 ways ('OAI', 'Official Action Indicated (OAI)', 'Official Action')
  * one inspection -> several project-area rows (not duplicates!) vs true duplicate rows
  * FEI stored as float / with '.0' / leading zeros -> canonical digit string
  * look-back window (5 yrs) applied on inspection end date relative to run date
  * missing source -> empty table with full schema + loud warning (downstream never crashes)
"""
from __future__ import annotations

import logging
import re
from datetime import date
from pathlib import Path

import pandas as pd

from .common import QLOG, Col, HttpClient, clean_text, company_core, company_key, enforce_schema, \
    quarantine, snake, write_table
from .config import SETTINGS

log = logging.getLogger("pharmashield.ingestion.inspections")
SRC = "fda_inspections"

ALIASES = {
    "fei_number": ["feinumber", "fei_number", "fei"],
    "legal_name": ["legalname", "legal_name", "firm_name"],
    "city": ["city", "city_name"],
    "state": ["state", "inspection_state", "statecode", "inspection_state_code"],
    "country": ["countryname", "country_name", "country_area", "country"],
    "country_code": ["countrycode", "country_code"],
    "inspection_id": ["inspectionid", "inspection_id"],
    "inspection_end_date": ["inspectionenddate", "inspection_end_date"],
    "fiscal_year": ["fiscalyear", "fiscal_year"],
    "classification": ["classification", "classificationcode", "classification_code"],
    "project_area": ["projectarea", "project_area"],
    "product_type": ["producttype", "product_type"],
    "posted_citations": ["postedcitations", "posted_citations"],
    "additional_details": ["additionaldetails", "additional_details"],
}
OUT_COLS = list(ALIASES) + ["classification_code", "firm_key", "firm_core", "source_file"]


def _canon(df: pd.DataFrame) -> pd.DataFrame:
    cols = {snake(c): c for c in df.columns}
    out = pd.DataFrame(index=df.index)
    for canon, alts in ALIASES.items():
        hit = next((cols[a] for a in alts if a in cols), None)
        out[canon] = df[hit] if hit is not None else pd.NA
    return out


def _class_code(v) -> str | None:
    if not isinstance(v, str):
        return None
    u = v.upper()
    for code, word in (("OAI", "OFFICIAL"), ("VAI", "VOLUNTARY"), ("NAI", "NO ACTION")):
        if re.search(rf"\b{code}\b", u) or word in u:
            return code
    return None


def _fei(v) -> str | None:
    if pd.isna(v):
        return None
    s = re.sub(r"\.0+$", "", str(v).strip())
    s = re.sub(r"\D", "", s)
    return s.lstrip("0") or None


def _from_api(http: HttpClient) -> pd.DataFrame:
    cfg = SETTINGS.dashboard
    start_date = date(SETTINGS.as_of.year - cfg.lookback_years, SETTINGS.as_of.month, 1)
    headers = {"Content-Type": "application/json", "Authorization-User": cfg.user,
               "Authorization-Key": cfg.key}
    rows, start = [], 1
    while True:
        body = {"start": start, "rows": cfg.rows_per_page, "returntotalcount": True,
                "sort": "InspectionEndDate", "sortorder": "DESC",
                "filters": {"ProductType": ["Drugs"], "InspectionEndDateFrom": [start_date.isoformat()]},
                "columns": []}
        payload = http.request_json("fda_dashboard/inspections", cfg.endpoint, method="POST",
                                    body=body, headers=headers)
        # This API reports success as statuscode 400 + message "Success."
        if not payload or str(payload.get("message", "")).lower().rstrip(".") != "success":
            raise RuntimeError(f"Dashboard API error: {str(payload)[:300]}")
        page = payload.get("result", [])
        rows.extend(page)
        total = payload.get("totalrecordcount", len(rows))
        if not page or len(rows) >= total:
            break
        start += cfg.rows_per_page
    df = pd.DataFrame(rows)
    df["source_file"] = "api"
    return df


def _from_manual(manual_dir: Path) -> pd.DataFrame:
    files = sorted(p for p in (manual_dir / "fda_inspections").glob("*")
                   if p.suffix.lower() in {".csv", ".xlsx", ".xls"})
    frames = []
    for p in files:
        try:
            if p.suffix.lower() == ".csv":
                d = pd.read_csv(p, dtype=str, encoding="utf-8-sig")
            else:
                # dashboard exports sometimes carry title rows above the header -> locate it
                probe = pd.read_excel(p, header=None, dtype=str, nrows=15)
                hdr = next((i for i, r in probe.iterrows()
                            if r.astype(str).str.contains("FEI", case=False).any()), 0)
                d = pd.read_excel(p, header=hdr, dtype=str)
            d["source_file"] = p.name
            frames.append(d)
            log.info("inspections: read %s (%d rows)", p.name, len(d))
        except Exception as e:
            QLOG.check(SRC, f"unreadable_file:{p.name}", 1, severity="error", detail=str(e)[:200])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build(http: HttpClient, manual_dir: Path) -> tuple[pd.DataFrame, str]:
    cfg = SETTINGS.dashboard
    raw, mode = pd.DataFrame(), "none"
    if cfg.user and cfg.key and not http.offline:
        try:
            raw, mode = _from_api(http), "api"
        except Exception as e:
            log.warning("Dashboard API failed (%s) - falling back to manual export", e)
    if raw.empty:
        raw = _from_manual(manual_dir)
        mode = "manual_export" if not raw.empty else "none"
    if raw.empty:
        QLOG.check(SRC, "source_available", 1, severity="error",
                   action="C4 OAI/VAI component will be 0 - add export or API credentials",
                   detail="see README section 'FDA inspections'")
        return pd.DataFrame(columns=OUT_COLS), mode

    src_file = raw.get("source_file")
    df = _canon(raw)
    df["source_file"] = src_file
    n = len(df)
    for c in df.columns:
        df[c] = df[c].map(clean_text)
    df["fei_number"] = df["fei_number"].map(_fei)
    df["classification_code"] = df["classification"].map(_class_code)
    df["inspection_end_date"] = pd.to_datetime(df["inspection_end_date"], errors="coerce")

    bad = df["fei_number"].isna() | df["classification_code"].isna() | df["inspection_end_date"].isna()
    quarantine(SRC, df[bad], "missing_fei_or_classification_or_date")
    QLOG.check(SRC, "missing_fei/classification/date", int(bad.sum()), n, action="quarantined",
               severity="warning" if bad.any() else "info")
    df = df[~bad].copy()

    # true duplicates = same inspection + project area + classification (overlapping exports)
    key = ["fei_number", "inspection_id", "project_area", "classification_code"]
    dup = df.duplicated(key, keep="first")
    quarantine(SRC, df[dup], "duplicate_inspection_project_area")
    QLOG.check(SRC, "duplicate_rows", int(dup.sum()), n, action="removed (multi project-area rows kept)")
    df = df[~dup].copy()

    pt = df["product_type"].fillna("")
    non_drug = ~pt.str.contains("drug", case=False) & pt.ne("")
    quarantine(SRC, df[non_drug], "non_drug_product_type")
    df = df[~non_drug]
    QLOG.check(SRC, "non_drug_rows_removed", int(non_drug.sum()), n)

    cutoff = pd.Timestamp(SETTINGS.as_of) - pd.DateOffset(years=cfg.lookback_years)
    old = df["inspection_end_date"] < cutoff
    QLOG.check(SRC, f"older_than_{cfg.lookback_years}y_removed", int(old.sum()), n)
    df = df[~old].copy()
    future = df["inspection_end_date"] > pd.Timestamp(SETTINGS.as_of)
    QLOG.check(SRC, "future_dated_inspections", int(future.sum()), len(df),
               severity="warning" if future.any() else "info")

    df["firm_key"] = df["legal_name"].map(company_key)
    df["firm_core"] = df["legal_name"].map(company_core)
    QLOG.check(SRC, "classification_mix", int((df["classification_code"] == "OAI").sum()), len(df),
               detail=str(df["classification_code"].value_counts().to_dict()))
    schema = [Col("fei_number", "str", nullable=False), Col("legal_name", "str", nullable=False),
              Col("inspection_end_date", "date", nullable=False),
              Col("classification_code", "str", nullable=False, allowed=("NAI", "VAI", "OAI"))]
    return enforce_schema(df[OUT_COLS], schema, SRC), mode


def run(http: HttpClient, manual_dir: Path, **_) -> dict:
    df, mode = build(http, manual_dir)
    path = write_table(df, "fda_inspections_drugs_5yr")
    QLOG.status(SRC, "OK" if len(df) else "MISSING", rows=len(df), note=f"mode={mode}")
    return {"fda_inspections_drugs_5yr": path}
