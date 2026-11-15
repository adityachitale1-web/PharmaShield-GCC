# PharmaShield GCC — Layer 1 Data Source Catalogue

All sources are public. Live endpoints were checked on 15 Nov 2026. The "SCENARIO" rows are labelled
assumptions, as required by plan §3 (item 3); they must be presented as such on the slides.

| # | Source | Endpoint / location | Access | Freshness | Key fields used | Pipeline role | Plan Q |
|---|---|---|---|---|---|---|---|
| 1 | **USAID SCMS Delivery History** (Supply Chain Shipment Pricing Data) | Kaggle `divyeshardeshana/supply-chain-shipment-pricing-data` (original: catalog.data.gov). Included at `data/manual/scms/`, SHA-256 `918b992d…8c8673` | file | historical 2006–2015, 10,324 lines × 33 cols | Scheduled / Delivered dates, Shipment Mode, Vendor INCO Term, Manufacturing Site, Line Item Qty/Value, Unit/Pack Price, Weight, Freight, Insurance, Product Group | training layer: C2 XGBoost target `late`, C3 Isolation Forest features | Q1, Q3, Q5 |
| 2 | **openFDA NDC Directory** | `https://api.fda.gov/drug/ndc.json` | open API (key optional) | daily | labeler_name, product_ndc, generic_name, dosage_form, active_ingredients.strength, product_type, marketing_category, application_number, listing_expiration_date, finished, openfda.spl_set_id / is_original_packager | hard qualification gate per SKU (UAE registration proxy) | Q1, Q2 |
| 3 | **openFDA Drug Shortages** | `https://api.fda.gov/drug/shortages.json` | open API | daily | generic_name, company_name, status, availability, shortage_reason, presentation, package_ndc, update_date, therapeutic_category | live alert banner; `active_fda_shortage`; C4 penalty | Q2, Q3, Q7 |
| 4 | **openFDA Drug Enforcement (recalls)** | `https://api.fda.gov/drug/enforcement.json` | open API | weekly | recall_number, recalling_firm, classification (Class I/II/III), report_date, reason_for_recall, product_description, status | C4 recall term `min(1, Class I/II ÷ 2)` | Q2, Q6 |
| 5 | **openFDA Drug Label (SPL)** | `https://api.fda.gov/drug/label.json` | open API | weekly | set_id, effective_time, openfda.manufacturer_name, "Manufactured by / Mfd. by / Made in" text | plant country per labeler → C2 inference, C5 corridor | Q2 |
| 6 | **FDA Data Dashboard — Inspection Classifications** | API `https://api-datadashboard.fda.gov/v1/inspections_classifications` (POST; headers `Authorization-User`, `Authorization-Key`) **or** export from `https://datadashboard.fda.gov` → Inspections | free credentials / manual export | weekly | FEI Number, Legal Name, Country, Inspection ID, Inspection End Date, Classification (NAI/VAI/OAI), Project Area, Product Type | C4 `0.30·I(OAI>0) + 0.10·I(VAI>0)` | Q2, Q6 |
| 7a | **World Bank LPI — overall** | `https://api.worldbank.org/v2/country/all/indicator/LP.LPI.OVRL.XQ` | open API | survey years (latest = 2022 edition, published 2023) | countryiso3code, date, value (1–5) | C5 `× (1 − LPI ÷ 5)` | Q2, Q5 |
| 7b | **World Bank WGI — Political Stability & Absence of Violence (estimate)** | same API; code `PV.EST` / `GOV_WGI_PV.EST` (source 3), tried in order | open API | annual | value (−2.5…+2.5) | macro stability context | Q2, Q5 |
| 8a | **FDA CGMP Warning Letters** | `https://www.fda.gov/inspections-compliance-enforcement-and-criminal-investigations/compliance-actions-and-activities/warning-letters` | public web pages | ongoing | full text, letter date, MARCS-CMS no., FEI, recipient firm | C4 NLP severity (sterility, data integrity, process validation, quality unit) | Q4 |
| 8b | **Supplier Master Service Agreements** | synthetic, produced by Member 4B → `data/manual/contracts/` | team-generated | n/a | full text; force majeure / lead time / penalty / single-source clauses | C4 LLM contract-risk term | Q4 |
| 9a | **CMS Medicare Part B ASP payment limits** (public reimbursement benchmark) | J9045 carboplatin 50 mg = **$3.101** (Oct–Dec 2026); J9278 carboplatin (Avyxa) 1 mg = **$5.226** — `reference/price_benchmarks.csv` | public file | quarterly | payment limit per HCPCS unit | C1 base unit price (450 mg vial = $27.91 multi-source generic) | Q1 |
| 9b | **SCENARIO: Hormuz sea / Gulf air corridor shares, freight per vial** | `reference/scenario_corridor_assumptions.csv` | team assumption | per run | air/sea/land split, Hormuz fraction of sea, Gulf-airspace fraction of air, freight USD/vial | C5 `hormuz_air_share`; C1 freight | Q5, Q7 |
| 9c | **SCENARIO: monthly plant capacity** | default 4,000 vials/month; override in `reference/manufacturer_overrides.csv` | team assumption | per run | monthly_capacity_units | LP hard upper bound | Q5 |
| — | Insurance rate for C1 | derived from SCMS (median insurance ÷ line value, non-imputed lines) | data-driven | — | — | C1 insurance per vial | Q1 |
| opt | USAID GHSC-PSM health commodity deliveries (stretch, plan §3.2) | not ingested — first item in the plan's cut order | — | 2024 | — | out-of-time check | Q3 |

## Reference tables in `reference/`

| File | Content | Basis |
|---|---|---|
| `scms_site_country.csv` | all 88 SCMS manufacturing sites → country + ISO3 + confidence | country in site name (high), known HQ/plant (medium), unverified (low/none). SCMS `Country` is the **destination**, so this table is required for the C2 "manufacturing country" feature |
| `countries.csv` | country names, aliases, ISO3 | used to read plant country from label text and to join World Bank data |
| `price_benchmarks.csv` | CMS ASP payment limits | source URLs in the file |
| `scenario_corridor_assumptions.csv` | corridor + freight assumptions per origin country, with `*` default row | SCENARIO; validated (shares in [0,1], modes sum to 1) on every run |
| `manufacturer_overrides.csv` | manual plant country / capacity per labeler | team-verified evidence (e.g. the DailyMed label "Manufactured by" line) |

## Terms & caveats

* openFDA: public-domain US government data; openFDA notes it is unvalidated and not for clinical decisions.
  Without an API key the quota is 240 requests/min and 1,000/day per IP, which is plenty for one run (≈15 calls).
* FDA Data Dashboard: not every inspection is included; a classification is posted only once all its project areas are final.
* World Bank: CC BY 4.0. LPI is published only in survey years, so the year is kept next to every value.
* Warning letters are public, but they name individuals. Salutation names, e-mails and phone numbers are redacted
  (UAE PDPL data minimisation, plan §6.1).
* SCMS covers ARV/HIV test-kit logistics, not oncology. Plan §3.1 handles this with feature-based inference,
  not vendor lookup.
