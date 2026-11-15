"""
PharmaShield GCC - Layer 1 Data Ingestion console (Streamlit).

Deploy on Streamlit Community Cloud with main file path `streamlit_app.py`.
Optional secrets (App settings -> Secrets):
    OPENFDA_API_KEY = "..."
    FDA_DD_USER = "you@example.com"
    FDA_DD_KEY = "..."
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import zipfile
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
L1 = ROOT / "layer1_ingestion"
sys.path.insert(0, str(L1))

# secrets -> env BEFORE the ingestion config is imported (it reads env at import time)
try:
    for k in ("OPENFDA_API_KEY", "FDA_DD_USER", "FDA_DD_KEY"):
        if k in st.secrets and not os.getenv(k):
            os.environ[k] = str(st.secrets[k])
except Exception:  # no secrets file configured
    pass

import run_ingestion  # noqa: E402
from ingestion import common  # noqa: E402
from ingestion.config import SETTINGS  # noqa: E402

P = SETTINGS.paths
SOURCES = list(run_ingestion.SOURCES)

st.set_page_config(page_title="PharmaShield GCC · Layer 1 Ingestion", page_icon="💊", layout="wide")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def reset_state() -> None:
    common.QLOG.records.clear()
    common.QLOG.source_status.clear()
    common._QUARANTINE.clear()


def save_uploads(files, subdir: str) -> int:
    if not files:
        return 0
    d = P.manual / subdir
    d.mkdir(parents=True, exist_ok=True)
    for f in files:
        (d / Path(f.name).name).write_bytes(f.getbuffer())
    return len(files)


@st.cache_data(show_spinner=False)
def read_csv(path: str, mtime: float) -> pd.DataFrame:  # mtime busts the cache after a re-run
    return pd.read_csv(path, low_memory=False)


def load(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size <= 1:
        return pd.DataFrame()
    try:
        return read_csv(str(path), path.stat().st_mtime)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def report() -> dict | None:
    p = P.quality / "quality_report.json"
    return json.loads(p.read_text()) if p.exists() else None


def zip_outputs() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for folder in (P.interim, P.quarantine, P.quality):
            for f in sorted(folder.glob("*")) if folder.exists() else []:
                z.write(f, f.relative_to(P.data))
        if (P.data / "manifest.json").exists():
            z.write(P.data / "manifest.json", "manifest.json")
    return buf.getvalue()


# --------------------------------------------------------------------------- #
# sidebar: run controls
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("Run ingestion")
    sources = st.multiselect("Sources", SOURCES, default=SOURCES,
                             help="'scenario' builds the panel seed from the other outputs, so keep it last.")
    offline = st.toggle("Offline (replay cached API snapshots)", value=False)
    as_of = st.date_input("As-of date", value=date.today())
    crawl = st.number_input("Crawl FDA warning letters (0 = off)", 0, 200, 0, step=10)

    st.subheader("Manual inputs")
    insp = st.file_uploader("FDA inspections export (.xlsx/.csv)", type=["xlsx", "xls", "csv"],
                            accept_multiple_files=True)
    wl = st.file_uploader("Warning letters (urls.csv or saved .html/.pdf/.txt)",
                          type=["csv", "txt", "html", "htm", "pdf"], accept_multiple_files=True)
    ct = st.file_uploader("Supplier contracts (.txt/.md/.docx/.pdf)", type=["txt", "md", "docx", "pdf"],
                          accept_multiple_files=True)
    run = st.button("▶ Run Layer 1", type="primary", width="stretch")

    keys = [k for k in ("OPENFDA_API_KEY", "FDA_DD_USER") if os.getenv(k)]
    st.caption(f"Secrets loaded: {', '.join(keys) if keys else 'none (public API limits)'}")

if run:
    n = save_uploads(insp, "fda_inspections") + save_uploads(wl, "warning_letters") + save_uploads(ct, "contracts")
    ordered = [s for s in SOURCES if s in sources]
    args = ["--sources", *ordered, "--as-of", as_of.isoformat(), "--log-level", "WARNING"]
    if offline:
        args.append("--offline")
    if crawl:
        args += ["--crawl-warning-letters", str(int(crawl))]
    reset_state()
    out = io.StringIO()
    with st.status(f"Running {len(ordered)} source(s){f' · {n} uploaded file(s)' if n else ''}…",
                   expanded=True) as status:
        st.write("Pulling and validating: " + ", ".join(ordered))
        try:
            with contextlib.redirect_stdout(out):
                code = run_ingestion.main(args)
            status.update(label="Ingestion finished", state="complete" if code == 0 else "error")
        except Exception as e:  # never leave the user with a blank page
            status.update(label=f"Run crashed: {e}", state="error")
            st.exception(e)
    st.session_state["last_summary"] = out.getvalue()
    read_csv.clear()

# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
st.title("PharmaShield GCC · Layer 1 — Multi-Source Data Ingestion")
st.caption("SCMS · openFDA NDC / Shortages / Recalls / Labels · FDA inspections · World Bank LPI & WGI · "
           "warning letters & contracts · scenario inputs. Target SKU: "
           f"**{SETTINGS.sku.sku_id}** (Tier {SETTINGS.sku.criticality_tier}).")

rep = report()
if rep is None:
    st.info("No run yet. Choose sources in the sidebar and press **Run Layer 1** "
            "(a full live run takes about 2–4 minutes).")
    st.stop()

checks = pd.DataFrame(rep["checks"])
status_df = pd.DataFrame([{"source": k, **v} for k, v in rep["sources"].items()])
man = rep.get("manifest", {})

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Sources OK", f"{(status_df['status'] == 'OK').sum()}/{len(status_df)}")
c2.metric("Checks run", len(checks))
c3.metric("Warnings", int((checks["severity"] == "warning").sum()) if len(checks) else 0)
c4.metric("Errors", int((checks["severity"] == "error").sum()) if len(checks) else 0)
c5.metric("Rows quarantined", sum(man.get("quarantined_rows", {}).values()))
st.caption(f"Last run (UTC): {rep['generated_at_utc']} · as-of {man.get('as_of')} · "
           f"offline={man.get('offline')} · HTTP calls {man.get('http_calls')} · "
           f"cache replays {man.get('cache_replays')}")

tab_over, tab_q, tab_panel, tab_tables, tab_quar, tab_dl = st.tabs(
    ["Source status", "Data quality", "Scoring panel seed", "Output tables", "Quarantine", "Download"])

with tab_over:
    def _colour(v):
        return {"OK": "background-color:#d4edda", "FAILED": "background-color:#f8d7da",
                "MISSING": "background-color:#fff3cd", "EMPTY": "background-color:#fff3cd"}.get(v, "")
    show = status_df.reindex(columns=["source", "status", "rows", "note"])
    st.dataframe(show.style.map(_colour, subset=["status"]), width="stretch", hide_index=True)
    if st.session_state.get("last_summary"):
        with st.expander("Console summary of this run"):
            st.code(st.session_state["last_summary"])
    shortages = load(P.interim / "fda_shortages_sku.csv")
    if len(shortages) and "is_active" in shortages:
        act = shortages[shortages["is_active"].astype(bool)]
        if len(act):
            st.error(f"🚨 {len(act)} active FDA shortage record(s) for {SETTINGS.sku.molecule.title()}: "
                     + ", ".join(sorted(act["company_name"].dropna().unique())))

with tab_q:
    if checks.empty:
        st.write("No checks recorded.")
    else:
        sev = st.multiselect("Severity", ["error", "warning", "info", "pass"], default=["error", "warning", "info"])
        src = st.multiselect("Source", sorted(checks["source"].unique()))
        view = checks[checks["severity"].isin(sev)]
        if src:
            view = view[view["source"].isin(src)]
        order = {"error": 0, "warning": 1, "info": 2, "pass": 3}
        view = view.sort_values("severity", key=lambda s: s.map(order))
        st.dataframe(view, width="stretch", hide_index=True,
                     column_config={"detail": st.column_config.TextColumn(width="large")})

with tab_panel:
    panel = load(P.interim / "layer1_scoring_panel_seed.csv")
    if panel.empty:
        st.info("Panel seed not built yet - run 'openfda', 'worldbank' and 'scenario'.")
    else:
        st.markdown("One row per qualified labeler. `prelim_*` links are exact-key only; "
                    "Layer 2 replaces them with rapidfuzz entity resolution. SCENARIO columns are labelled assumptions.")
        cols = [c for c in ["labeler_name_fda", "manufacturing_country", "country_source", "hormuz_air_share",
                            "lpi_score", "wgi_political_stability", "base_unit_price_usd",
                            "landed_cost_usd_per_vial", "monthly_capacity_units", "prelim_active_shortage_records",
                            "prelim_class_i_ii_recalls_5yr", "prelim_oai_5yr", "prelim_vai_5yr",
                            "prelim_scms_shipments", "is_probation"] if c in panel]
        st.dataframe(panel[cols], width="stretch", hide_index=True)
        unk = panel[panel["manufacturing_country"].eq("Unknown")]
        if len(unk):
            st.warning("Plant country unresolved for: " + ", ".join(unk["labeler_name_fda"]) +
                       " → add rows to `layer1_ingestion/reference/manufacturer_overrides.csv`.")
        with st.expander("Evidence for plant country (label text)"):
            st.dataframe(panel[["labeler_name_fda", "country_evidence"]], width="stretch", hide_index=True)

with tab_tables:
    files = sorted(P.interim.glob("*.csv")) if P.interim.exists() else []
    if not files:
        st.write("No output tables yet.")
    else:
        pick = st.selectbox("Table", [f.stem for f in files])
        df = load(P.interim / f"{pick}.csv")
        st.caption(f"{len(df):,} rows × {df.shape[1]} columns")
        if "text" in df.columns:
            df = df.assign(text=df["text"].astype(str).str.slice(0, 300) + "…")
        st.dataframe(df.head(2000), width="stretch", hide_index=True)
        st.download_button(f"Download {pick}.csv", (P.interim / f"{pick}.csv").read_bytes(),
                           file_name=f"{pick}.csv", mime="text/csv")
        if pick == "scms_shipments_clean" and len(df):
            a, b = st.columns(2)
            a.bar_chart(df.groupby("time_split")["late"].mean().rename("late rate"))
            b.bar_chart(df.groupby("shipment_mode")["late"].mean().rename("late rate by mode"))

with tab_quar:
    files = sorted(P.quarantine.glob("*.csv")) if P.quarantine.exists() else []
    if not files:
        st.success("Nothing quarantined.")
    else:
        pick = st.selectbox("Quarantine file", [f.stem for f in files])
        q = load(P.quarantine / f"{pick}.csv")
        st.dataframe(q["_reject_reason"].value_counts().rename_axis("reason").reset_index(name="rows"),
                     hide_index=True)
        st.dataframe(q.head(1000), width="stretch", hide_index=True)

with tab_dl:
    st.write("All clean tables, quarantine files, the quality report and the manifest in one zip.")
    st.download_button("⬇ Download all Layer 1 outputs (.zip)", zip_outputs(),
                       file_name=f"pharmashield_layer1_outputs_{man.get('as_of')}.zip", mime="application/zip",
                       type="primary")
    qm = P.quality / "quality_report.md"
    if qm.exists():
        st.download_button("Download quality_report.md", qm.read_bytes(), file_name="quality_report.md")
