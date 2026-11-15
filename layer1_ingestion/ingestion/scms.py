"""
Source 1 - USAID SCMS Delivery History (training layer for C2 XGBoost and C3 Isolation Forest).

Implements Plan section 5.2 steps 1-2:
  1. Clean training data (dates with errors='coerce', freight/weight text coercion,
     binary target late = delivered > scheduled, shape + class balance at every step).
  2. Engineer site & corridor features (on-time rate, mean / p90 delay, freight-to-value,
     unit-price variance, air share per manufacturing site) - TRAIN PERIOD ONLY so the
     2014-2015 out-of-time test set never leaks into site priors.

Data issues handled (each is logged in the quality report):
  * BOM / encoding / mojibake in headers and text; column-name drift (fuzzy header map)
  * exact duplicate lines (ID differs, everything else identical) -> quarantined
  * duplicate primary keys (ID)
  * sentinel strings in date fields ('Pre-PQ Process', 'Date Not Captured', 'N/A - From RDC')
  * freight / weight text: 'Freight Included in Commodity Cost', 'Invoiced Separately',
    'Weight Captured Separately', and cross-references 'See DN-304 (ID#:10589)' / 'See ASN-...'
    -> referenced shipment cost is allocated across all lines of that shipment by value share
  * missing insurance -> imputed from same-year median insurance rate, flagged
  * impossible values (negative qty/value), zero-value lines, impossible date order
    (delivered before PO, PO before PQ), implausible delays (> 365 days)
  * statistical outliers (unit price, freight ratio, weight/unit) -> FLAGGED, NOT DROPPED,
    because they are exactly the signal the C3 Isolation Forest must see
  * site -> manufacturing country via reference/scms_site_country.csv (SCMS 'Country' is the
    DESTINATION country, not the plant country)
"""
from __future__ import annotations

import difflib
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .common import QLOG, Col, clean_text, company_core, company_key, enforce_schema, \
    quarantine, robust_z, sha256_file, snake, write_table
from .config import SETTINGS

log = logging.getLogger("pharmashield.ingestion.scms")
SRC = "scms"

# canonical snake_case name -> raw header as published by USAID
EXPECTED = {
    "id": "ID", "project_code": "Project Code", "pq": "PQ #", "po_so": "PO / SO #",
    "asn_dn": "ASN/DN #", "country": "Country", "managed_by": "Managed By",
    "fulfill_via": "Fulfill Via", "vendor_inco_term": "Vendor INCO Term",
    "shipment_mode": "Shipment Mode", "pq_first_sent_to_client_date": "PQ First Sent to Client Date",
    "po_sent_to_vendor_date": "PO Sent to Vendor Date", "scheduled_delivery_date": "Scheduled Delivery Date",
    "delivered_to_client_date": "Delivered to Client Date", "delivery_recorded_date": "Delivery Recorded Date",
    "product_group": "Product Group", "sub_classification": "Sub Classification", "vendor": "Vendor",
    "item_description": "Item Description", "molecule_test_type": "Molecule/Test Type", "brand": "Brand",
    "dosage": "Dosage", "dosage_form": "Dosage Form", "unit_of_measure_per_pack": "Unit of Measure (Per Pack)",
    "line_item_quantity": "Line Item Quantity", "line_item_value": "Line Item Value",
    "pack_price": "Pack Price", "unit_price": "Unit Price", "manufacturing_site": "Manufacturing Site",
    "first_line_designation": "First Line Designation", "weight_kilograms": "Weight (Kilograms)",
    "freight_cost_usd": "Freight Cost (USD)", "line_item_insurance_usd": "Line Item Insurance (USD)",
}
CRITICAL = ["id", "scheduled_delivery_date", "delivered_to_client_date", "shipment_mode",
            "line_item_quantity", "line_item_value", "unit_price", "manufacturing_site",
            "freight_cost_usd", "weight_kilograms", "vendor_inco_term", "product_group"]

DATE_FORMATS = {  # explicit formats -> no silent day/month swaps
    "scheduled_delivery_date": "%d-%b-%y", "delivered_to_client_date": "%d-%b-%y",
    "delivery_recorded_date": "%d-%b-%y", "pq_first_sent_to_client_date": "%m/%d/%y",
    "po_sent_to_vendor_date": "%m/%d/%y",
}
DATE_SENTINELS = {"pre-pq process": "pre_pq", "date not captured": "not_captured",
                  "n/a - from rdc": "from_rdc", "see asn": "see_asn", "": "blank"}
REF_RE = re.compile(r"see\s+(?:dn|asn)[-\s]*\S*\s*\(id#?\s*:?\s*(\d+)\)", re.I)
MODE_MAP = {"air": "Air", "air charter": "Air Charter", "ocean": "Ocean", "truck": "Truck"}


# --------------------------------------------------------------------------- #
def _read_raw(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path, dtype=str)
    else:
        last = None
        for enc in ("utf-8-sig", "cp1252", "latin-1"):
            try:
                df = pd.read_csv(path, dtype=str, encoding=enc, keep_default_na=False,
                                 na_values=["", "NA", "N/A", "nan", "NaN", "null"])
                break
            except UnicodeDecodeError as e:
                last = e
        else:
            raise last  # type: ignore[misc]
    return df


def _map_headers(df: pd.DataFrame) -> pd.DataFrame:
    """Map whatever headers arrived (Kaggle / USAID / re-saved Excel) to canonical names."""
    incoming = {c: snake(c) for c in df.columns}
    canon_by_snake = {snake(v): k for k, v in EXPECTED.items()}
    rename, unmatched = {}, []
    for raw, s in incoming.items():
        if s in canon_by_snake:
            rename[raw] = canon_by_snake[s]
        elif s in EXPECTED:
            rename[raw] = s
        else:
            close = difflib.get_close_matches(s, list(canon_by_snake), n=1, cutoff=0.85)
            if close:
                rename[raw] = canon_by_snake[close[0]]
                log.warning("header '%s' fuzzy-mapped to '%s'", raw, rename[raw])
            else:
                unmatched.append(raw)
    df = df.rename(columns=rename)
    missing = [c for c in CRITICAL if c not in df.columns]
    QLOG.check(SRC, "missing_critical_columns", len(missing), severity="error" if missing else "info",
               detail=", ".join(missing), action="abort" if missing else "")
    if missing:
        raise ValueError(f"SCMS file is missing critical columns: {missing}. "
                         f"Unrecognised headers: {unmatched}")
    if unmatched:
        QLOG.check(SRC, "unrecognised_columns_kept", len(unmatched), severity="info",
                   detail=", ".join(unmatched))
    return df


def _parse_date(df: pd.DataFrame, col: str) -> None:
    raw = df[col].astype("string").str.strip()
    status = raw.str.lower().map(lambda v: DATE_SENTINELS.get(v, None) if isinstance(v, str) else "blank")
    parsed = pd.to_datetime(raw, format=DATE_FORMATS[col], errors="coerce")
    # second chance for re-saved files (ISO / other formats) - only where the strict format failed
    retry = parsed.isna() & status.isna() & raw.notna()
    if retry.any():
        parsed.loc[retry] = pd.to_datetime(raw[retry], errors="coerce", dayfirst=False)
    unparsed = parsed.isna() & status.isna() & raw.notna()
    df[col] = parsed
    df[f"{col}_status"] = np.where(parsed.notna(), "ok", status.fillna("unparseable"))
    QLOG.check(SRC, f"date_unparseable:{col}", unparsed.sum(), len(df),
               action="set NaT", severity="warning" if unparsed.any() else "info",
               detail=str(raw[unparsed].unique()[:5].tolist()))
    sentinel = df[f"{col}_status"].isin(set(DATE_SENTINELS.values()))
    if sentinel.any():
        QLOG.check(SRC, f"date_sentinel:{col}", sentinel.sum(), len(df), action="set NaT + status column",
                   detail=str(df.loc[sentinel, f"{col}_status"].value_counts().to_dict()))


def _num(s: pd.Series) -> pd.Series:
    out = pd.to_numeric(s.astype("string").str.replace(r"[,$\s]", "", regex=True), errors="coerce")
    return out.astype("float64")


def _parse_cost_field(df: pd.DataFrame, col: str, kind: str) -> None:
    """Freight/weight: numeric | included | separate | reference to another line's shipment."""
    raw = df[col].astype("string").str.strip()
    value = _num(raw)
    ref = raw.str.extract(REF_RE, expand=False)
    low = raw.str.lower().fillna("")
    status = np.select(
        [value.notna(), ref.notna(), low.str.contains("included in commodity"),
         low.str.contains("invoiced separately|captured separately"), raw.isna()],
        ["numeric", "reference", "included_in_commodity", "captured_separately", "missing"],
        default="unparseable")
    df[f"{kind}_raw_numeric"] = value
    df[f"{kind}_ref_id"] = ref
    df[f"{kind}_status"] = status
    QLOG.check(SRC, f"{kind}_text_values", (status != "numeric").sum(), len(df), action="categorised",
               detail=str(pd.Series(status).value_counts().to_dict()))
    neg = value < 0
    if neg.any():
        QLOG.check(SRC, f"{kind}_negative", neg.sum(), len(df), action="set NaN", severity="warning")
        df.loc[neg, f"{kind}_raw_numeric"] = np.nan


def _allocate_references(df: pd.DataFrame, kind: str) -> None:
    """
    'See DN-304 (ID#:10589)' means this line travelled in the same shipment as line 10589,
    whose freight/weight figure covers the whole shipment. Allocate that figure across the
    referenced line + all lines pointing at it, proportional to line value (equal split if
    values are zero). Unresolvable references stay NaN with status 'reference_unresolved'.
    """
    ids = df["id"].astype(str)
    val_by_id = pd.Series(df[f"{kind}_raw_numeric"].values, index=ids)
    is_ref = df[f"{kind}_status"].eq("reference")
    anchor = df[f"{kind}_ref_id"].where(is_ref)
    # chase chains (A -> B -> C) up to 5 hops
    status_by_id = pd.Series(df[f"{kind}_status"].values, index=ids)
    ref_by_id = pd.Series(df[f"{kind}_ref_id"].values, index=ids)
    for _ in range(5):
        nxt = anchor.map(lambda a: ref_by_id.get(a) if status_by_id.get(a) == "reference" else None)
        if nxt.isna().all():
            break
        anchor = nxt.combine_first(anchor)
    anchor_val = anchor.map(val_by_id)
    group = anchor.where(is_ref, ids.where(df[f"{kind}_status"].eq("numeric")))
    shipment_total = group.map(lambda g: val_by_id.get(g, np.nan) if isinstance(g, str) else np.nan)
    in_group = group.notna() & shipment_total.notna()
    w = df["line_item_value"].astype("float64").clip(lower=0).fillna(0)
    g_sum = w.groupby(group).transform("sum").astype("float64")
    g_n = w.groupby(group).transform("count").astype("float64")
    share = np.where((g_sum > 0).fillna(False), w / g_sum.replace(0, np.nan), 1 / g_n)
    alloc = pd.Series(np.where(in_group, shipment_total * share, np.nan), index=df.index)
    df[f"{kind}_usd" if kind == "freight" else "weight_kg"] = alloc
    df[f"{kind}_allocated"] = in_group & (g_n > 1)
    anchor_known = anchor.isin(set(ids))
    unresolved = is_ref & ~anchor_known
    inherits = is_ref & anchor_known & anchor_val.isna()
    df.loc[inherits, f"{kind}_status"] = "ref_" + anchor[inherits].map(status_by_id).astype(str)
    df.loc[unresolved, f"{kind}_status"] = "reference_unresolved"
    QLOG.check(SRC, f"{kind}_references_unresolved", unresolved.sum(), int(is_ref.sum()),
               action="NaN", severity="warning" if unresolved.any() else "info")
    QLOG.check(SRC, f"{kind}_references_to_non_numeric_line", inherits.sum(), int(is_ref.sum()),
               action="NaN; status inherited from referenced line (e.g. invoiced separately)")
    QLOG.check(SRC, f"{kind}_shared_shipment_allocated", int(df[f"{kind}_allocated"].sum()), len(df),
               action="allocated by line-value share")


def _site_country(df: pd.DataFrame, ref_path: Path) -> None:
    ref = pd.read_csv(ref_path, dtype=str, encoding="utf-8")
    norm = lambda s: re.sub(r"[^a-z0-9]", "", str(clean_text(s)).lower())
    ref["k"] = ref["manufacturing_site"].map(norm)
    lut = ref.set_index("k")[["manufacturing_country", "iso3", "confidence"]]
    k = df["manufacturing_site"].map(norm)
    df["manufacturing_country"] = k.map(lut["manufacturing_country"])
    df["manufacturing_iso3"] = k.map(lut["iso3"])
    df["site_country_confidence"] = k.map(lut["confidence"]).fillna("unmapped")
    unmapped = df["site_country_confidence"].eq("unmapped")
    QLOG.check(SRC, "site_country_unmapped", unmapped.sum(), len(df),
               action="country='Unknown' - add row to reference/scms_site_country.csv",
               severity="warning" if unmapped.any() else "info",
               detail=str(df.loc[unmapped, "manufacturing_site"].unique()[:10].tolist()))
    weak = df["site_country_confidence"].isin(["none", "low"])
    QLOG.check(SRC, "site_country_low_or_no_confidence", weak.sum(), len(df),
               action="kept; verify rows marked low/none in reference file",
               detail=str(df.loc[weak, "manufacturing_site"].value_counts().head(8).to_dict()))
    df["manufacturing_country"] = df["manufacturing_country"].fillna("Unknown")


# --------------------------------------------------------------------------- #
def clean_scms(path: Path, ref_dir: Path | None = None) -> pd.DataFrame:
    cfg = SETTINGS.scms
    ref_dir = ref_dir or SETTINGS.paths.reference
    df = _read_raw(path)
    n0 = len(df)
    QLOG.check(SRC, "rows_read", n0, detail=f"{path.name} sha256={sha256_file(path)[:12]}")
    log.info("SCMS read: shape=%s", df.shape)

    df = _map_headers(df)
    for c in df.columns:
        if pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_string_dtype(df[c]):
            df[c] = df[c].map(clean_text)

    # ---- empty / key checks -------------------------------------------------
    blank = df[CRITICAL].isna().all(axis=1)
    quarantine(SRC, df[blank], "blank_row")
    df = df[~blank]
    QLOG.check(SRC, "blank_rows", blank.sum(), n0, action="quarantined")

    missing_id = df["id"].isna()
    quarantine(SRC, df[missing_id], "missing_id")
    df = df[~missing_id]
    dup_id = df["id"].duplicated(keep="first")
    quarantine(SRC, df[dup_id], "duplicate_id")
    df = df[~dup_id]
    QLOG.check(SRC, "duplicate_primary_key_id", dup_id.sum(), n0, action="quarantined (kept first)")

    # exact duplicate content with a different ID (double-keyed line)
    content_cols = [c for c in df.columns if c != "id"]
    dup_content = df.duplicated(subset=content_cols, keep="first")
    first_id = df.groupby(content_cols, dropna=False)["id"].transform("first")
    q = df[dup_content].copy()
    q["duplicate_of_id"] = first_id[dup_content]
    quarantine(SRC, q, "duplicate_content_different_id")
    df = df[~dup_content].copy()
    QLOG.check(SRC, "exact_duplicate_lines", dup_content.sum(), n0, action="quarantined (kept lowest ID)",
               severity="warning" if dup_content.any() else "info")
    log.info("after dedupe: shape=%s", df.shape)

    # ---- dates ---------------------------------------------------------------
    for col in DATE_FORMATS:
        if col in df.columns:
            _parse_date(df, col)
    no_target = df["scheduled_delivery_date"].isna() | df["delivered_to_client_date"].isna()
    quarantine(SRC, df[no_target], "missing_target_dates")
    df = df[~no_target].copy()
    QLOG.check(SRC, "missing_scheduled_or_delivered_date", no_target.sum(), n0,
               action="quarantined (no label)", severity="warning" if no_target.any() else "info")

    # ---- numerics -----------------------------------------------------------
    for c in ["line_item_quantity", "line_item_value", "pack_price", "unit_price",
              "unit_of_measure_per_pack", "line_item_insurance_usd"]:
        if c in df.columns:
            raw = df[c]
            df[c] = _num(raw)
            bad = df[c].isna() & raw.notna()
            QLOG.check(SRC, f"non_numeric:{c}", bad.sum(), len(df), action="NaN",
                       severity="warning" if bad.any() else "info")
    impossible = (df["line_item_quantity"] <= 0) | (df["line_item_value"] < 0) | (df["unit_price"] < 0)
    quarantine(SRC, df[impossible], "non_positive_quantity_or_negative_value")
    df = df[~impossible].copy()
    QLOG.check(SRC, "impossible_qty_or_value", impossible.sum(), n0, action="quarantined")

    df["flag_zero_value_line"] = df["line_item_value"].eq(0) | df["unit_price"].eq(0)
    QLOG.check(SRC, "zero_value_or_zero_price_lines", df["flag_zero_value_line"].sum(), len(df),
               action="kept + flagged (donations/samples; excluded from price features)")

    # reconcile line value with qty x pack price, unit price with pack price / pack size
    exp_val = df["line_item_quantity"] * df["pack_price"]
    rel = (exp_val - df["line_item_value"]).abs() / df["line_item_value"].replace(0, np.nan)
    df["flag_value_mismatch"] = (rel > cfg.value_reconcile_tolerance).fillna(False)
    exp_unit = df["pack_price"] / df["unit_of_measure_per_pack"].replace(0, np.nan)
    unit_gap = (exp_unit - df["unit_price"]).abs()
    df["flag_unit_price_mismatch"] = ((unit_gap > 0.01) &
                                      (unit_gap / df["unit_price"].replace(0, np.nan) > 0.5)).fillna(False)
    QLOG.check(SRC, "line_value_vs_qty_x_pack_price_mismatch", df["flag_value_mismatch"].sum(), len(df),
               action="flagged")
    QLOG.check(SRC, "unit_price_vs_pack_price_mismatch(>50%)", df["flag_unit_price_mismatch"].sum(),
               len(df), action="flagged (small gaps are 2-dp rounding)")

    # ---- freight / weight / insurance ----------------------------------------
    _parse_cost_field(df, "freight_cost_usd", "freight")
    _parse_cost_field(df, "weight_kilograms", "weight")
    _allocate_references(df, "freight")
    _allocate_references(df, "weight")
    df.loc[df["freight_status"].eq("included_in_commodity"), "freight_usd"] = np.nan

    df["insurance_usd"] = df["line_item_insurance_usd"]
    year = df["scheduled_delivery_date"].dt.year
    rate = (df["insurance_usd"] / df["line_item_value"].replace(0, np.nan))
    yr_rate = rate.groupby(year).transform("median").fillna(rate.median())
    df["insurance_imputed"] = df["insurance_usd"].isna()
    df.loc[df["insurance_imputed"], "insurance_usd"] = (df["line_item_value"] * yr_rate)[df["insurance_imputed"]]
    QLOG.check(SRC, "insurance_missing_imputed", df["insurance_imputed"].sum(), len(df),
               action="imputed = line value x same-year median insurance rate (flagged)")

    # ---- categoricals ----------------------------------------------------------
    df["shipment_mode"] = df["shipment_mode"].str.lower().map(MODE_MAP).fillna("Unknown")
    QLOG.check(SRC, "shipment_mode_missing", df["shipment_mode"].eq("Unknown").sum(), len(df),
               action="category 'Unknown' (not imputed - avoids inventing air/sea)")
    df["is_air"] = df["shipment_mode"].isin(["Air", "Air Charter"])
    df["vendor_inco_term"] = df["vendor_inco_term"].str.upper().replace({"N/A - FROM RDC": "RDC"})
    df["fulfill_via"] = df["fulfill_via"].replace({"From RDC": "RDC"})
    _site_country(df, ref_dir / "scms_site_country.csv")
    df["vendor_key"] = df["vendor"].map(company_key)
    df["site_core_key"] = df["manufacturing_site"].map(company_core)

    # ---- logical date order ----------------------------------------------------
    d = df
    order_viol = pd.Series(False, index=d.index)
    if "po_sent_to_vendor_date" in d:
        order_viol |= (d["delivered_to_client_date"] < d["po_sent_to_vendor_date"]).fillna(False)
    if "pq_first_sent_to_client_date" in d and "po_sent_to_vendor_date" in d:
        order_viol |= (d["po_sent_to_vendor_date"] < d["pq_first_sent_to_client_date"]).fillna(False)
    if "delivery_recorded_date" in d:
        d["flag_recorded_before_delivered"] = (d["delivery_recorded_date"] < d["delivered_to_client_date"]).fillna(False)
    d["flag_date_order_violation"] = order_viol
    QLOG.check(SRC, "date_order_violation(delivered<PO or PO<PQ)", order_viol.sum(), len(d),
               action="kept + flagged (exclude in ML if desired)", severity="warning" if order_viol.any() else "info")

    # ---- target ----------------------------------------------------------------
    d["delay_days"] = (d["delivered_to_client_date"] - d["scheduled_delivery_date"]).dt.days
    implausible = d["delay_days"].abs() > cfg.max_abs_delay_days
    quarantine(SRC, d[implausible], f"implausible_delay_gt_{cfg.max_abs_delay_days}d")
    d = d[~implausible].copy()
    QLOG.check(SRC, f"implausible_delay_abs_gt_{cfg.max_abs_delay_days}d", implausible.sum(), n0,
               action="quarantined (likely keying error)", severity="warning" if implausible.any() else "info")
    d["late"] = (d["delay_days"] > 0).astype(int)
    d["scheduled_year"] = d["scheduled_delivery_date"].dt.year
    d["time_split"] = np.where(d["scheduled_year"] <= cfg.train_end_year, "train",
                               np.where(d["scheduled_year"].isin(cfg.test_years), "test", "out_of_range"))
    bal = d.groupby("time_split")["late"].agg(["size", "mean"]).round(4).to_dict("index")
    QLOG.check(SRC, "class_balance_late", int(d["late"].sum()), len(d), detail=f"by split: {bal}")
    log.info("target built: shape=%s late_rate=%.3f split=%s", d.shape, d["late"].mean(), bal)

    # ---- engineered line features + outlier FLAGS --------------------------------
    priced = ~d["flag_zero_value_line"]
    d["freight_to_value_ratio"] = (d["freight_usd"] / d["line_item_value"]).where(priced)
    d["weight_kg_per_unit"] = d["weight_kg"] / d["line_item_quantity"]
    d["landed_cost_per_unit"] = ((d["line_item_value"] + d["freight_usd"].fillna(0) + d["insurance_usd"].fillna(0))
                                 / d["line_item_quantity"]).where(priced)
    extreme_ratio = d["freight_to_value_ratio"] > 10
    QLOG.check(SRC, "freight_to_value_ratio_gt_10x", extreme_ratio.sum(), len(d),
               action="kept + flagged (tiny-value lines)", severity="info")
    for feat in ["unit_price", "freight_to_value_ratio", "weight_kg_per_unit", "line_item_value"]:
        logv = np.log1p(d[feat].where(d[feat] > 0))
        z = logv.groupby(d["product_group"]).transform(robust_z)
        d[f"outlier_{feat}"] = (z.abs() > cfg.robust_z_threshold).fillna(False)
        QLOG.check(SRC, f"robust_outlier:{feat}", d[f"outlier_{feat}"].sum(), len(d),
                   action="FLAGGED only - retained as Isolation Forest signal")
    d["outlier_any"] = d.filter(like="outlier_").any(axis=1)

    schema = [
        Col("id", "str", nullable=False, unique=True),
        Col("scheduled_delivery_date", "date", nullable=False),
        Col("delivered_to_client_date", "date", nullable=False),
        Col("delay_days", "int", nullable=False, min=-cfg.max_abs_delay_days, max=cfg.max_abs_delay_days),
        Col("late", "int", nullable=False, allowed=(0, 1)),
        Col("time_split", "str", nullable=False, allowed=("train", "test", "out_of_range")),
        Col("shipment_mode", "str", nullable=False, allowed=("Air", "Air Charter", "Ocean", "Truck", "Unknown")),
        Col("is_air", "bool", nullable=False),
        Col("vendor_inco_term", "str"),
        Col("manufacturing_site", "str"),
        Col("manufacturing_country", "str", nullable=False),
        Col("manufacturing_iso3", "str"),
        Col("product_group", "str", nullable=False),
        Col("line_item_quantity", "float", nullable=False, min=0),
        Col("line_item_value", "float", nullable=False, min=0),
        Col("unit_price", "float", min=0),
        Col("freight_usd", "float", min=0),
        Col("freight_status", "str", nullable=False),
        Col("weight_kg", "float", min=0),
        Col("insurance_usd", "float", min=0),
        Col("freight_to_value_ratio", "float", min=0),
        Col("landed_cost_per_unit", "float", min=0),
    ]
    d = enforce_schema(d, schema, SRC)
    QLOG.check(SRC, "rows_out_clean", len(d), n0, detail=f"retained {len(d)}/{n0}")
    return d


def site_features(clean: pd.DataFrame) -> pd.DataFrame:
    """Plan 5.2 step 2. Computed on TRAIN rows only (no look-ahead into the 2014-15 test set)."""
    tr = clean[clean["time_split"] == "train"]
    g = tr.groupby(["manufacturing_site", "manufacturing_country", "manufacturing_iso3"], dropna=False)
    feats = g.agg(
        site_n_shipments=("id", "size"),
        site_on_time_rate=("late", lambda s: 1 - s.mean()),
        site_mean_delay_days=("delay_days", "mean"),
        site_p90_delay_days=("delay_days", lambda s: s.quantile(0.9)),
        site_mean_freight_to_value=("freight_to_value_ratio", "mean"),
        site_unit_price_cv=("unit_price", lambda s: s.std() / s.mean() if s.mean() else np.nan),
        site_air_share=("is_air", "mean"),
        first_shipment=("scheduled_delivery_date", "min"),
        last_shipment=("scheduled_delivery_date", "max"),
    ).reset_index()
    feats["site_low_history"] = feats["site_n_shipments"] < SETTINGS.scms.probation_min_shipments
    ctry = tr.groupby("manufacturing_country").agg(
        country_n_shipments=("id", "size"), country_on_time_rate=("late", lambda s: 1 - s.mean()),
        country_air_share=("is_air", "mean"), country_mean_freight_to_value=("freight_to_value_ratio", "mean"),
    ).reset_index()
    feats = feats.merge(ctry, on="manufacturing_country", how="left")
    QLOG.check(SRC, "site_features_low_history_sites", feats["site_low_history"].sum(), len(feats),
               action="flagged (use country prior; probation logic in Layer 4)")
    return feats, ctry


def run(manual_dir: Path, offline: bool = False, **_) -> dict:
    candidates = sorted([p for p in (manual_dir / "scms").glob("*")
                         if p.suffix.lower() in {".csv", ".xlsx", ".xls"}])
    if not candidates:
        QLOG.status(SRC, "MISSING", note="put SCMS_Delivery_History_Dataset.csv in data/manual/scms/")
        QLOG.check(SRC, "input_file_present", 1, severity="error",
                   detail="Download from Kaggle (divyeshardeshana/supply-chain-shipment-pricing-data)")
        return {}
    path = candidates[-1]
    clean = clean_scms(path)
    feats, ctry = site_features(clean)
    out = {"scms_shipments_clean": write_table(clean, "scms_shipments_clean"),
           "scms_site_features_train": write_table(feats, "scms_site_features_train"),
           "scms_country_priors_train": write_table(ctry, "scms_country_priors_train")}
    QLOG.status(SRC, "OK", rows=len(clean), note=f"late rate {clean['late'].mean():.1%}; "
                f"{(clean.time_split == 'train').sum()} train / {(clean.time_split == 'test').sum()} test")
    return out
