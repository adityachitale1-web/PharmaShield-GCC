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

## Quick start (Layer 1)

```bash
cd layer1_ingestion
pip install -r requirements.txt
python run_ingestion.py          # live pull: openFDA + World Bank + SCMS
python run_ingestion.py --offline   # replay cached snapshots, no network
python -m pytest -q
```

Outputs land in `layer1_ingestion/data/interim/`. See the
[Layer 1 README](layer1_ingestion/README.md) and the [data source catalogue](layer1_ingestion/DATA_SOURCES.md).

## Conventions

* Generated data (`data/raw`, `data/interim`, `data/quarantine`, `data/quality`) is not committed. Re-run the pipeline instead.
* Every layer reads the previous layer's `data/interim/` tables; don't edit them by hand.
* GenAI disclosure (plan §6.1): AI-assisted code is marked and logged in the GenAI Prompt Log.
