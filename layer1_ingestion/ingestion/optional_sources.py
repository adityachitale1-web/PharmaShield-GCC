"""
Optional Layer-1 inputs named in the plan that are not available for the prototype:

  * GHSC-PSM (plan 3.2, "Optional Stretch"): modern USAID health-commodity deliveries used only
    for an out-of-time XGBoost check. If a CSV is dropped in data/manual/ghsc_psm/ it is run
    through the same cleaner as SCMS (same columns are required) and tagged time_split =
    'out_of_range'. First item in the plan's contingency cut order, so absence is not an error.
  * Internal GPO data (plan 2, Layer 1 "Internal Data"): ERP purchase orders, delivery logs and
    Tatmeen GS1 scan timestamps. Confidential, so plan 3 uses SCMS as the explicit proxy. If the
    files are supplied they are validated, de-duplicated and written to data/interim/.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from . import scms
from .common import QLOG, clean_text, company_key, quarantine, snake, write_table

log = logging.getLogger("pharmashield.ingestion.optional")

INTERNAL = {
    "erp_purchase_orders": dict(required=["po_id", "sku_id", "supplier_name", "order_date", "quantity",
                                          "unit_price_usd"], key=["po_id", "sku_id"], dates=["order_date"],
                                positive=["quantity"], non_negative=["unit_price_usd", "freight_usd", "insurance_usd"]),
    "delivery_logs": dict(required=["po_id", "scheduled_delivery_date", "delivered_date"], key=["po_id"],
                          dates=["scheduled_delivery_date", "delivered_date"], positive=[], non_negative=[]),
    "tatmeen_scans": dict(required=["gtin", "serial_number", "scan_timestamp", "event_type"],
                          key=["gtin", "serial_number", "scan_timestamp", "event_type"], dates=["scan_timestamp"],
                          positive=[], non_negative=[]),
}


def run_ghsc(manual_dir: Path, **_) -> dict:
    src = "ghsc_psm"
    files = sorted(p for p in (manual_dir / "ghsc_psm").glob("*") if p.suffix.lower() in {".csv", ".xlsx", ".xls"})
    if not files:
        QLOG.status(src, "SKIPPED", note="optional stretch dataset (plan 3.2) not supplied")
        return {}
    old = scms.SRC
    try:
        scms.SRC = src  # log + quarantine under its own name
        clean = scms.clean_scms(files[-1])
        QLOG.status(src, "OK", rows=len(clean), note=f"late rate {clean['late'].mean():.1%} (out-of-time check)")
        return {"ghsc_psm_shipments_clean": write_table(clean, "ghsc_psm_shipments_clean")}
    except Exception as e:
        QLOG.check(src, "optional_file_not_usable", 1, severity="warning", detail=str(e)[:250])
        QLOG.status(src, "SKIPPED", note=f"file present but not usable: {str(e)[:120]}")
        return {}
    finally:
        scms.SRC = old


def run_internal(manual_dir: Path, **_) -> dict:
    src, out, n_total = "internal_gpo", {}, 0
    d = manual_dir / "internal"
    for name, spec in INTERNAL.items():
        p = d / f"{name}.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p, dtype=str, encoding="utf-8-sig")
        df.columns = [snake(c) for c in df.columns]
        missing = [c for c in spec["required"] if c not in df.columns]
        if missing:
            QLOG.check(src, f"{name}:missing_required_columns", len(missing), severity="error",
                       detail=", ".join(missing), action="file skipped")
            continue
        n = len(df)
        df = df.apply(lambda s: s.map(clean_text))
        blank = df[spec["required"]].isna().any(axis=1)
        quarantine(src, df[blank].assign(_file=name), "missing_required_value")
        df = df[~blank]
        dup = df.duplicated(spec["key"], keep="last")
        quarantine(src, df[dup].assign(_file=name), "duplicate_key")
        df = df[~dup].copy()
        for c in spec["dates"]:
            df[c] = pd.to_datetime(df[c], errors="coerce", utc=(c == "scan_timestamp"))
        bad = df[spec["dates"]].isna().any(axis=1)
        for c in spec["positive"] + [c for c in spec["non_negative"] if c in df.columns]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            bad |= df[c].isna() | (df[c] <= 0 if c in spec["positive"] else df[c] < 0)
        quarantine(src, df[bad].assign(_file=name), "unparseable_date_or_invalid_number")
        df = df[~bad].copy()
        if "supplier_name" in df:
            df["supplier_key"] = df["supplier_name"].map(company_key)
        QLOG.check(src, f"{name}:rows_rejected", int(n - len(df)), n,
                   action="quarantined (missing / duplicate / invalid)")
        out[f"internal_{name}"] = write_table(df, f"internal_{name}")
        n_total += len(df)
    if out:
        QLOG.status(src, "OK", rows=n_total, note=f"{len(out)} internal file(s) ingested")
    else:
        QLOG.status(src, "PROXY", note="confidential GPO ERP / Tatmeen data not supplied - SCMS is the proxy (plan 3)")
    return out
