# PharmaShield GCC

AI-driven pharmaceutical supplier selection, multi-source risk profiling and dual-source order allocation
for a UAE healthcare GPO. Group 3 · AI in Operations · SP Jain MAIB.

Demonstration SKU: **Carboplatin Injection 10 mg/mL** (Tier 1, life-critical oncology).

## Architecture (Master Implementation Plan §2)

| Layer | Folder | Owner | Status |
|---|---|---|---|
| 1. Multi-source data ingestion | [`layer1_ingestion/`](layer1_ingestion/) | Member 2A | ✅ done |
| 2. Unified manufacturer × SKU risk profile (rapidfuzz entity resolution → `suppliers_matrix.csv`) | `layer2_fusion/` | Member 2B | to do |
| 3. AI risk engines C1–C5 (XGBoost + SHAP, Isolation Forest, NLP) | `layer3_engines/` | Members 4A, 4B | to do |
| 4. Hybrid AHP + Entropy TOPSIS and PuLP order split | `layer4_decision/` | Member 3 | to do |
| 5. Streamlit cockpit, governance and audit log | `app/` | Member 5 | to do |

## Run it on Streamlit

**Streamlit Community Cloud:** share.streamlit.io → *Create app* → repo `adityachitale1-web/PharmaShield-GCC`,
branch `main`, main file `streamlit_app.py` → *Deploy*. Optional secrets (App settings → Secrets):

```toml
OPENFDA_API_KEY = "..."      # higher openFDA quota
FDA_DD_USER = "you@example.com"  # FDA Data Dashboard inspections API
FDA_DD_KEY = "..."
```

**Locally:** `pip install -r requirements.txt` then `streamlit run streamlit_app.py`.

In the app: press **Run Layer 1**. No uploads are needed; all inputs ship with the repo or download automatically. Tabs show source status, every data-quality check, the scoring panel seed, all output
tables, quarantined rows and a one-click zip download of the outputs.

Dashboard tabs: Overview (source status, shortage alert, plan-vs-live figures) · **Data schema (§5.1)** with a pass / fail
per column · suppliers_matrix.csv · Data feeds (§3.2) · 360° manufacturer profile · Build steps 1–3 · Data quality ·
Output tables · Quarantine · Download.

Notebook: `notebooks/01_layer1_data_ingestion.ipynb`. GenAI prompt log: `docs/GENAI_PROMPT_LOG_layer1.md`.
Line-by-line plan check: `layer1_ingestion/PLAN_COMPLIANCE.md`.

## Quick start (Layer 1, command line)

```bash
cd layer1_ingestion
pip install -r requirements.txt
python run_ingestion.py          # live pull: openFDA + World Bank + SCMS
python run_ingestion.py --offline   # replay cached snapshots, no network
python -m pytest -q
```

The locked-schema hand-off is `data/processed/suppliers_matrix.csv` at the repo root (plan §5.1); other tables are in `layer1_ingestion/data/interim/`. See the
[Layer 1 README](layer1_ingestion/README.md) and the [data source catalogue](layer1_ingestion/DATA_SOURCES.md).

## Conventions

* Generated data (`data/processed`, `layer1_ingestion/data/{raw,interim,quarantine,quality}`) is not committed. Re-run the pipeline instead.
* Every layer reads the previous layer's `data/interim/` tables; don't edit them by hand.
* GenAI disclosure (plan §6.1): AI-assisted code is marked and logged in the GenAI Prompt Log.
