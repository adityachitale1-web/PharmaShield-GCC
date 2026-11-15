# PharmaShield GCC — Layer 1: Multi-Source Data Ingestion & Collection

Group 3 · AI in Operations · Master Implementation Plan §2 (Layer 1), §3 (data strategy), §5.2 steps 1–3.

One command pulls, cleans, validates and documents every Layer-1 source and writes
analysis-ready tables for Layer 2 (entity fusion), Layer 3 (C1–C5 engines) and the Streamlit cockpit.

```bash
pip install -r requirements.txt
python run_ingestion.py                 # live run (≈2–4 min; openFDA + World Bank)
python run_ingestion.py --offline       # replay cached API snapshots, no network (demo / audit)
python -m pytest -q                     # 20 offline tests
```

Optional:
`--sources scms openfda worldbank` (run a subset) · `--crawl-warning-letters 80` ·
`--as-of 2026-11-14` (fix look-back windows) · `--strict` (exit 1 on any error, for CI) ·
`export OPENFDA_API_KEY=...` (higher quota) · `export FDA_DD_USER=... FDA_DD_KEY=...` (inspections API).

---

## What it ingests

| # | Source | How | Feeds |
|---|---|---|---|
| 1 | USAID SCMS Delivery History (10,324 lines) | file in `data/manual/scms/` (already included) | C2 XGBoost, C3 Isolation Forest |
| 2 | openFDA NDC Directory | live API | SKU qualification gate |
| 3 | openFDA Drug Shortages | live API (SKU + all current oncology) | dashboard alerts, C4 penalty |
| 4 | openFDA Drug Enforcement (recalls, 5 yrs) | live API, yearly windows | C4 recall component |
| 5 | openFDA Drug Label (SPL text) | live API | plant country per labeler ("Manufactured by: … India") |
| 6 | FDA Data Dashboard inspections (NAI/VAI/OAI) | automatic download of FDA's published dataset (or API / manual export) | C4 OAI/VAI component |
| 7 | World Bank LPI + WGI Political Stability | live API | C5 route exposure |
| 8 | FDA CGMP warning letters + supplier contracts | saved files / URL list / optional crawl | C4 NLP (Member 4B) |
| 9 | Scenario assumptions (corridor shares, CMS ASP price benchmark) | `reference/*.csv` | C1, C5 |

Full catalogue with URLs, fields and licences: **[DATA_SOURCES.md](DATA_SOURCES.md)**. Line-by-line check against the plan: **[PLAN_COMPLIANCE.md](PLAN_COMPLIANCE.md)**.

## The hand-off file (`data/processed/`)

`suppliers_matrix.csv` uses the **locked schema of plan §5.1** (16 columns, fixed names and order). Layer 1 fills
identity, country, probation, capacity, C1, C5, corridor share, shortage flag and OAI count. `C2_delivery_risk`,
`C4_compliance_risk` and `shap_top3_reasons` are left empty for Members 4A / 4B to fill in the same file.
`quotes_2026_carboplatin.csv` (plan §4.3) sits next to it.

## What comes out (`data/interim/`)

| File | Grain | Consumer |
|---|---|---|
| `scms_shipments_clean.csv` | SCMS line (target `late`, `delay_days`, `time_split` train ≤2013 / test 2014–15, freight/weight resolved, outlier flags) | Member 4A |
| `scms_site_features_train.csv`, `scms_country_priors_train.csv` | site / country — **train period only** (no leakage) | Member 4A |
| `fda_ndc_products.csv` | product NDC with `qualifies_for_sku` + `exclusion_reasons` | Member 2B |
| `fda_ndc_qualified_labelers.csv` | one row per qualified labeler | Member 2B |
| `fda_label_manufacturing_sites.csv` | SPL label → plant country + evidence snippet | Member 2B |
| `fda_shortages_sku.csv`, `fda_shortages_oncology_current.csv` | shortage record (latest status) | Members 2B, 5 |
| `fda_recalls_5yr.csv` | recall number (class, sterility/CGMP reason flags) | Member 2B |
| `fda_inspections_drugs_5yr.csv` | inspection × project area | Member 2B |
| `wb_country_indicators.csv` | country (latest non-null LPI + WGI, with year) | Member 2B |
| `warning_letters_corpus.csv`, `contracts_corpus.csv` | document (PII-redacted text + metadata) | Member 4B |
| `quotes_2026_carboplatin.csv` | labeler (C1 inputs, SCENARIO-labelled) | Member 3 |
| `layer1_scoring_panel_seed.csv` | labeler, all raw panel fields | Member 2B |
| `entity_match_candidates.csv` | labeler × external name, match method and score, `needs_manual_check` | Member 2B |
| `ghsc_psm_shipments_clean.csv`, `internal_*.csv` | only if those optional files are supplied | Members 4A, 2B |

Also: `data/raw/` (immutable API snapshots), `data/quarantine/*_rejected.csv` (every rejected row + reason),
`data/quality/quality_report.md` (all checks), `data/manifest.json` (row counts + SHA-256 per output).

## How data issues are handled

Principle: **nothing is silently dropped or invented.** Bad rows go to quarantine with a reason code;
suspicious-but-real values are kept and flagged; every decision is counted in the quality report.

| Issue | Treatment |
|---|---|
| Exact duplicates (SCMS: 4 lines double-keyed under different IDs) | quarantined, `duplicate_of_id` recorded |
| Duplicate keys across pages/windows/SPL versions (NDC, recalls, shortages, labels, inspections) | deterministic keep-latest rule per source |
| Text in numeric fields (`Freight Included in Commodity Cost`, `Invoiced Separately`, `Weight Captured Separately`) | NaN + explicit `*_status` category (never coerced to 0) |
| Cross-line references (`See DN-304 (ID#:10589)`, 2,441 lines) | chased (incl. chains) and the shipment total **allocated across lines by value share** |
| Sentinel dates (`Pre-PQ Process`, `Date Not Captured`, `N/A - From RDC`) | NaT + `*_status` column; explicit date formats (no day/month swaps) |
| Impossible values (qty ≤ 0, negative value) | quarantined |
| Implausible delays (> 365 days, 4 lines) / impossible date order (10 lines) | quarantined / flagged |
| Statistical outliers (unit price, freight ratio, weight per unit) | MAD robust-z within product group, **flagged not removed** — they are the C3 signal |
| Missing insurance (287) | imputed from same-year median rate, `insurance_imputed=True` |
| Missing shipment mode (360) | `Unknown` category — not guessed |
| Encoding: BOM header, mojibake (`CÃ´te d'Ivoire`) | UTF-8-sig read + repair |
| Header drift (re-saved / Kaggle variants) | fuzzy header mapping; abort with clear message if critical columns missing |
| Non-target NDC rows (bulk API powder, wrong concentration, expired listing, combos, repackagers) | excluded with reason; strengths like `450 mg/45mL` parsed to mg/mL |
| Company name variants (`Hospira, Inc., a Pfizer Company`) | `company_key` / `company_core` / `parent_key` for Layer-2 rapidfuzz |
| World Bank aggregates, sparse LPI survey years, out-of-range values | aggregates removed, latest non-null + year kept, out-of-range quarantined |
| API failures (429/5xx, timeouts, truncated JSON, openFDA 404 = "no matches") | exponential back-off + Retry-After; falls back to last cached snapshot; one failing feed never stops the run |
| PII in letters/contracts (UAE PDPL, Decree-Law 45/2021) | e-mails, phone numbers, salutation names redacted before writing |
| Look-ahead leakage | site/country priors computed on train period only |

## Inputs

Nothing has to be uploaded. The repo ships the SCMS file, 51 FDA warning-letter URLs (`data/manual/warning_letters/urls.csv`)
and 11 synthetic supplier contracts (`data/manual/contracts/`), and FDA inspections are downloaded automatically.
Optional extras: more letters or contracts in the same folders; `reference/manufacturer_overrides.csv` for plant country
and capacity; `reference/quotes_2026_carboplatin_override.csv` for real tender quotes; `data/manual/ghsc_psm/` and
`data/manual/internal/` for the optional datasets.

## Hand-off notes for teammates

* **Member 2B:** review `entity_match_candidates.csv` where `needs_manual_check` is true (plan step 3). Parent links come
  from `reference/corporate_aliases.csv` (e.g. Eugia → Aurobindo, Accord → Intas).
* **Member 4A:** use `time_split`; don't drop `outlier_*` rows before fitting Isolation Forest. Write `C2`, `C3` and
  `shap_top3_reasons` into `suppliers_matrix.csv`.
* **Member 4B:** corpora are `warning_letters_corpus.csv` and `contracts_corpus.csv`; C4 inputs are in the matrix's
  supporting columns.
* **Member 3:** `monthly_capacity_units` is a SCENARIO default of 4,000 until overrides are filled.
* **Slides:** take labeler and shortage counts from the final live run, not from the plan text (see PLAN_COMPLIANCE.md).

## Layout

```
run_ingestion.py            CLI orchestrator → manifest + quality report
ingestion/config.py         every tunable (SKU, windows, thresholds, endpoints)
ingestion/common.py         HTTP client (retry/cache/offline), QualityLog, quarantine, schema contracts, name keys
ingestion/scms.py           SCMS cleaning + train-only site features
ingestion/openfda.py        NDC gate, label plant country, shortages, recalls
ingestion/fda_inspections.py
ingestion/worldbank.py
ingestion/documents.py      warning letters + contracts (PII redaction)
ingestion/scenario.py       scenario validation, C1 quotes, entity links, suppliers_matrix (locked schema)
ingestion/optional_sources.py  GHSC-PSM and internal GPO files (optional)
reference/                  site→country map, countries, corridor + price assumptions, overrides
tests/test_ingestion.py     offline tests (fake APIs + messy synthetic SCMS + real SCMS)
```

GenAI disclosure (plan §6.1): this code was generated with Claude (Anthropic) and reviewed by the team — mark it per the
red-dashed-outline protocol and log the prompt in the Prompt Log PDF.
