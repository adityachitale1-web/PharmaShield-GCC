# Layer 1 vs. the Master Implementation Plan — line-by-line check

Status key: ✅ implemented and tested · 🟡 implemented, needs a live run to confirm · ⬜ not a Layer 1 output (owner named)

## §5.1 Locked schema — `data/processed/suppliers_matrix.csv`

Column names, order and types are enforced in code (`scenario.LOCKED_SCHEMA`) and by a test that compares against the plan's list.

| # | Column | Type | Filled by Layer 1? | How |
|---|---|---|---|---|
| 1 | `sku_id` | str | ✅ | `CARBOPLATIN_INJ_10MG_ML` |
| 2 | `criticality_tier` | int | ✅ | 1 (Life-Critical) |
| 3 | `manufacturer_id` | str | ✅ | canonical name after suffix normalisation; links listed in `entity_match_candidates.csv` for manual verification |
| 4 | `labeler_name_fda` | str | ✅ | exact `labeler_name` from openFDA NDC |
| 5 | `manufacturing_country` | str | ✅ | manual override > SPL label "Manufactured by" text > HQ hint (flagged) > `Unknown` |
| 6 | `is_probation` | bool | ✅ | fewer than 5 historical shipments in SCMS (vendor or plant site, incl. parent alias) |
| 7 | `monthly_capacity_units` | int | ✅ | SCENARIO default 4,000; override file |
| 8 | `C1_landed_cost` | float [0,1] | ✅ | min-max of (unit price + freight + insurance) per vial; raw USD in `C1_landed_cost_usd_raw` |
| 9 | `C2_delivery_risk` | float [0,1] | ⬜ Member 4A | empty — XGBoost `predict_proba` |
| 10 | `C3_price_integrity` | float [0,1] | partly | 0.0 for probation vendors (plan default); others ⬜ Member 4A |
| 11 | `C4_compliance_risk` | float [0,1] | ⬜ Member 4B | empty — inputs provided: `oai_count_5yr`, `vai_count_5yr`, `recalls_class_i_ii_5yr`, letters, contracts |
| 12 | `C5_route_exposure` | float [0,1] | ✅ | `min(1, 1.0 × hormuz_air_share) × (1 − LPI ÷ 5)` (Normal mode) |
| 13 | `hormuz_air_share` | float [0,1] | ✅ | SCENARIO corridor table |
| 14 | `active_fda_shortage` | bool | ✅ | any active openFDA shortage record linked to the labeler |
| 15 | `oai_count_5yr` | int | 🟡 | OAI rows in the FDA inspections dataset, last 5 years |
| 16 | `shap_top3_reasons` | str (JSON) | ⬜ Member 4A | `"[]"` placeholder (valid JSON) |

Supporting columns are appended after column 16, never between the locked ones.

## §3.2 Dataset inventory

| Plan dataset | Status | Where |
|---|---|---|
| USAID SCMS Delivery History | ✅ real file, 10,324 rows | `ingestion/scms.py` |
| openFDA NDC Directory API | 🟡 | `ingestion/openfda.py` |
| openFDA Drug Shortages API | 🟡 | `ingestion/openfda.py` |
| openFDA Drug Enforcement API | 🟡 | `ingestion/openfda.py` |
| FDA Data Dashboard Inspections (bulk) | 🟡 automatic download, or API, or manual export | `ingestion/fda_inspections.py` |
| FDA CGMP Warning Letters | 🟡 51 real letter URLs shipped | `data/manual/warning_letters/urls.csv` |
| World Bank LPI & WGI | 🟡 | `ingestion/worldbank.py` |
| USAID GHSC-PSM (optional stretch) | ✅ optional loader; skipped if no file (first item in the plan's cut order) | `ingestion/optional_sources.py` |
| Supplier MSAs (synthetic) | ✅ 11 shipped | `data/manual/contracts/` |
| Internal GPO ERP / delivery logs / Tatmeen scans (§2) | ✅ optional loader; SCMS is the proxy per §3 | `ingestion/optional_sources.py` |
| Scenario layer: 2026 quotes + corridor shares (§3 item 3) | ✅ labelled SCENARIO | `reference/`, `data/processed/quotes_2026_carboplatin.csv` |

## §5.2 Build steps owned by data engineering

| Step | Plan wording | Status |
|---|---|---|
| 1 | Parse both dates with `errors='coerce'`, dropping text such as "See ASN" | ✅ |
| 1 | Coerce non-numeric `Freight Cost (USD)` to numeric | ✅ `freight_cost_usd_coerced` (plan-literal) + `freight_usd` (shared shipments allocated) |
| 1 | `late = delivered > scheduled` | ✅ |
| 1 | Print shape and class balance at every step | ✅ 8 `STEP` lines, also in the quality report |
| 2 | On-time rate, mean and p90 delay, freight-to-value ratio, unit price variance, air share per site | ✅ train period only |
| 3 | Live openFDA NDC + Shortages for Carboplatin Injection labelers | 🟡 |
| 3 | Normalise corporate suffixes; `rapidfuzz token_sort_ratio ≥ 88` + manual check | ✅ candidates file flags every non-exact link |
| 3 | Link to inspections, recalls, World Bank LPI/WGI | ✅ |
| 4 | Chronological split: train ≤ 2013, test 2014–2015 | ✅ `time_split` (7,771 / 2,545) |

## One point where the plan and the live data disagree

The plan says the NDC directory returned **18 labelers including Cipla and Hetero**. A check on 15 Nov 2026 showed 40
human-prescription carboplatin product rows from **11** labelers, plus 8 bulk-powder rows; Cipla and Hetero were not
among the 11. The likely explanation is that the 18 counts bulk-API labelers too. The pipeline therefore follows the plan
(`panel_scope = "all_ndc_labelers"`) and tags every row `finished_injection` or `bulk_api_only`. The exact count will be
whatever the live run returns; switch to `finished_injection_only` in `config.py` for the strict §4.1 gate.
