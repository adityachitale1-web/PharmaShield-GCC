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
MATRIX = P.processed / "suppliers_matrix.csv"

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


def interim(name: str) -> pd.DataFrame:
    return load(P.interim / f"{name}.csv")


def report() -> dict | None:
    p = P.quality / "quality_report.json"
    return json.loads(p.read_text()) if p.exists() else None


def zip_outputs() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for folder, prefix in ((P.processed, "processed"), (P.interim, "interim"),
                               (P.quarantine, "quarantine"), (P.quality, "quality")):
            for f in sorted(folder.glob("*")) if folder.exists() else []:
                if f.is_file():
                    z.write(f, f"data/{prefix}/{f.name}")
        if (P.data / "manifest.json").exists():
            z.write(P.data / "manifest.json", "data/manifest.json")
    return buf.getvalue()


def show(df, **kw) -> None:
    st.dataframe(df, width="stretch", hide_index=True, **kw)


def validate_matrix(sm: pd.DataFrame, schema: pd.DataFrame) -> pd.DataFrame:
    """Check the produced file against every rule of the locked schema (plan 5.1)."""
    out = []
    for _, r in schema.iterrows():
        col, pos, dtype = r["column"], int(r["position"]), r["dtype"]
        layer1 = r["filled_by"].startswith("Layer 1")
        if col not in sm.columns:
            out.append((pos, col, "MISSING", "column not in file", ""))
            continue
        s = sm[col]
        problems = []
        if list(sm.columns).index(col) != pos - 1:
            problems.append(f"at position {list(sm.columns).index(col) + 1}, expected {pos}")
        nn = int(s.notna().sum())
        if dtype in ("float", "int"):
            num = pd.to_numeric(s, errors="coerce")
            if num.notna().sum() != nn:
                problems.append("non-numeric values")
            if dtype == "int" and (num.dropna() % 1 != 0).any():
                problems.append("non-integer values")
            if col.startswith("C") or col == "hormuz_air_share":
                if ((num.dropna() < 0) | (num.dropna() > 1)).any():
                    problems.append("outside [0, 1]")
            if col == "criticality_tier" and not num.dropna().isin([1, 2, 3]).all():
                problems.append("tier not in {1, 2, 3}")
            if col in ("monthly_capacity_units", "oai_count_5yr") and (num.dropna() < 0).any():
                problems.append("negative")
        elif dtype == "bool":
            if not s.dropna().astype(str).str.lower().isin(["true", "false"]).all():
                problems.append("not boolean")
        elif col == "shap_top3_reasons":
            try:
                s.dropna().map(json.loads)
            except (ValueError, TypeError):
                problems.append("not valid JSON")
        if col == "manufacturer_id" and s.duplicated().any():
            problems.append("duplicate manufacturer_id")
        if layer1 and nn < len(sm):
            problems.append(f"{len(sm) - nn} empty")
        if problems:
            status = "FAIL"
        elif nn == len(sm):
            status = "PASS" if layer1 or col != "shap_top3_reasons" else "PENDING (Layer 3 - placeholder [])"
        else:
            status = "PENDING (Layer 3)" if nn == 0 else "PARTIAL (Layer 3 fills the rest)"
        out.append((pos, col, status, "; ".join(problems), f"{nn}/{len(sm)}"))
    return pd.DataFrame(out, columns=["position", "column", "validation", "problem", "filled"])


def colour_status(v):
    v = str(v)
    if v.startswith(("OK", "PASS", "matches")):
        return "background-color:#d4edda"
    if v.startswith(("FAIL", "MISSING", "FAILED")):
        return "background-color:#f8d7da"
    if v.startswith(("PENDING", "PARTIAL", "EMPTY", "differs", "SKIPPED", "PROXY", "no data")):
        return "background-color:#fff3cd"
    return ""


# --------------------------------------------------------------------------- #
# sidebar: run controls
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("Run ingestion")
    sources = st.multiselect("Sources", SOURCES, default=SOURCES,
                             help="'scenario' builds suppliers_matrix.csv from the other outputs, so keep it last.")
    offline = st.toggle("Offline (replay cached API snapshots)", value=False)
    as_of = st.date_input("As-of date", value=date.today())
    crawl = st.number_input("Crawl extra FDA warning letters (0 = off)", 0, 200, 0, step=10)

    st.subheader("Manual inputs (all optional)")
    st.caption("The repo already ships the SCMS file, 51 warning-letter URLs and 11 synthetic contracts; "
               "FDA inspections download automatically.")
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
st.caption(f"Target SKU **{SETTINGS.sku.sku_id}** · Tier {SETTINGS.sku.criticality_tier} (Life-Critical) · "
           "Master Implementation Plan §2 Layer 1, §3 data strategy, §5.1 locked schema, §5.2 steps 1–3.")

rep = report()
schema = pd.read_csv(P.reference / "locked_schema.csv")
inventory = pd.read_csv(P.reference / "dataset_inventory.csv", dtype=str)
sm = load(MATRIX)
checks = pd.DataFrame(rep["checks"]) if rep else pd.DataFrame()
status_df = pd.DataFrame([{"source": k, **v} for k, v in rep["sources"].items()]) if rep else pd.DataFrame()
man = rep.get("manifest", {}) if rep else {}

if rep is None:
    st.info("No run yet. Press **▶ Run Layer 1** in the sidebar (no uploads needed; a full live run takes a few "
            "minutes). The schema and data-feed tabs below already show what the run will produce.")
else:
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    ok = status_df["status"].isin(["OK", "SKIPPED", "PROXY"]).sum()
    c1.metric("Sources OK", f"{ok}/{len(status_df)}")
    c2.metric("Manufacturers in matrix", len(sm))
    val = validate_matrix(sm, schema) if len(sm) else pd.DataFrame()
    c3.metric("Schema columns valid", f"{int((val['validation'] != 'FAIL').sum()) if len(val) else 0}/16")
    c4.metric("Checks / warnings", f"{len(checks)} / {int((checks['severity'] == 'warning').sum())}")
    c5.metric("Errors", int((checks["severity"] == "error").sum()))
    c6.metric("Rows quarantined", sum(man.get("quarantined_rows", {}).values()))
    st.caption(f"Last run (UTC): {rep['generated_at_utc']} · as-of {man.get('as_of')} · offline={man.get('offline')} "
               f"· HTTP calls {man.get('http_calls')} · cache replays {man.get('cache_replays')}")

(tab_over, tab_schema, tab_matrix, tab_feeds, tab_360, tab_steps, tab_q, tab_tables, tab_quar, tab_dl) = st.tabs(
    ["Overview", "Data schema (§5.1)", "suppliers_matrix.csv", "Data feeds (§3.2)", "360° manufacturer profile",
     "Build steps 1–3", "Data quality", "Output tables", "Quarantine", "Download"])

# ------------------------------------------------------------------ Overview
with tab_over:
    if rep is None:
        st.write("Run the pipeline to see source status.")
    else:
        shortages = interim("fda_shortages_sku")
        if len(shortages) and "is_active" in shortages:
            act = shortages[shortages["is_active"].astype(bool)]
            if len(act):
                st.error(f"🚨 Live openFDA shortage alert: {len(act)} active record(s) for "
                         f"{SETTINGS.sku.molecule.title()} — "
                         + ", ".join(sorted(act["company_name"].dropna().unique())))
        st.subheader("Source status")
        show(status_df.reindex(columns=["source", "status", "rows", "note"])
             .style.map(colour_status, subset=["status"]))
        pvl = interim("plan_vs_live_reconciliation")
        if len(pvl):
            st.subheader("Figures quoted in the plan vs. this run")
            show(pvl.style.map(colour_status, subset=["status"]))
        if st.session_state.get("last_summary"):
            with st.expander("Console output of this run (includes shape + class balance at every step)"):
                st.code(st.session_state["last_summary"])

# ------------------------------------------------------------------ Schema
with tab_schema:
    st.subheader("Locked schema for `data/processed/suppliers_matrix.csv` (plan §5.1)")
    st.markdown("One master table, 16 columns in a fixed order, consumed directly by the TOPSIS/LP engine and the "
                "cockpit. Definitions below are the plan's wording.")
    DEF = "definition & validation rule (plan)"
    view = schema.rename(columns={"definition_and_validation_rule": DEF, "filled_by": "filled by",
                                  "how_layer1_fills_it": "how it is filled"})
    cfg = {DEF: st.column_config.TextColumn(width="large")}
    if len(sm):
        v = validate_matrix(sm, schema)
        view = view.merge(v[["column", "validation", "filled", "problem"]], on="column", how="left")
        n_fail = int((v["validation"] == "FAIL").sum())
        order_ok = list(sm.columns[:16]) == schema["column"].tolist()
        a, b, c = st.columns(3)
        a.metric("Columns present, in locked order", "16/16 ✓" if order_ok else "MISMATCH")
        b.metric("Columns failing validation", n_fail)
        c.metric("Awaiting Layer 3 (C2, C3, C4, SHAP)",
                 int(v["validation"].str.startswith(("PENDING", "PARTIAL")).sum()))
        if order_ok and n_fail == 0:
            st.success("The file matches the locked schema: names, order, types and value ranges all pass.")
        else:
            st.error("Schema problems found — see the `problem` column.")
        show(view.style.map(colour_status, subset=["validation"]), column_config=cfg)
    else:
        st.info("Not validated yet — run the pipeline. This is the contract the output will be checked against.")
        show(view, column_config=cfg)
    st.caption("Supporting columns (raw USD landed cost, LPI, VAI count, recalls, crisis-mode C5, FDA part of C4, …) "
               "are appended after column 16 and never inserted between the locked ones.")
    with st.expander("Clinical criticality tiers and SKU master (plan §4.2)"):
        show(pd.read_csv(P.reference / "sku_master.csv"))
        show(pd.read_csv(P.reference / "criticality_tiers.csv"))

# ------------------------------------------------------------------ Matrix
with tab_matrix:
    if sm.empty:
        st.info("suppliers_matrix.csv not built yet — run 'openfda', 'worldbank' and 'scenario'.")
    else:
        st.markdown("The 16 locked columns. **C2, C4 and SHAP reasons are empty by design**: they are produced by the "
                    "XGBoost and NLP engines (Members 4A / 4B), which write into this same file. C3 is 0.0 for "
                    "probation vendors (plan §4.3).")
        show(sm[list(sm.columns[:16])])
        st.download_button("Download suppliers_matrix.csv", MATRIX.read_bytes(), file_name="suppliers_matrix.csv",
                           mime="text/csv", type="primary")
        with st.expander("Supporting raw inputs (after the locked columns)"):
            show(sm[["manufacturer_id"] + list(sm.columns[16:])])
        unk = sm[sm["manufacturing_country"].eq("Unknown")]
        if len(unk):
            st.warning("Plant country unresolved for: " + ", ".join(unk["labeler_name_fda"]) +
                       " → add rows to `layer1_ingestion/reference/manufacturer_overrides.csv`.")
        if "country_source" in sm.columns:
            hq = sm[sm["country_source"].eq("hq_hint_verify")]
            if len(hq):
                st.warning("Plant country taken from company headquarters (verify): "
                           + ", ".join(hq["labeler_name_fda"]))
        quotes = load(P.processed / "quotes_2026_carboplatin.csv")
        if len(quotes):
            with st.expander("quotes_2026_carboplatin.csv — C1 inputs (SCENARIO-labelled, plan §4.3)"):
                show(quotes)

# ------------------------------------------------------------------ Feeds
with tab_feeds:
    st.subheader("Complete dataset inventory (plan §3.2) — 7 multi-source feeds + optional and scenario inputs")
    inv = inventory.copy()
    cols = ["feed_no", "dataset", "records_and_key_fields", "freshness", "pipeline_role", "questions"]
    cfg = {"records_and_key_fields": st.column_config.TextColumn(width="large")}
    if rep is not None:
        st_map = rep["sources"]

        def live(r):
            states = [st_map.get(k, {}).get("status", "not run") for k in r["status_keys"].split("|")]
            rows = sum(len(interim(t)) for t in r["output_tables"].split("|"))
            return pd.Series({"status this run": " / ".join(dict.fromkeys(states)), "rows ingested": rows})
        inv = pd.concat([inv, inv.apply(live, axis=1)], axis=1)
        show(inv[cols + ["status this run", "rows ingested", "output_tables"]]
             .style.map(colour_status, subset=["status this run"]), column_config=cfg)
    else:
        show(inv[cols + ["output_tables"]], column_config=cfg)

    st.subheader("Five risk dimensions and the feeds behind them")
    show(pd.DataFrame([
        ("C1 Landed cost", "Scenario 2026 tender quotes; SCMS (insurance rate)",
         "quotes_2026_carboplatin.csv → C1 filled"),
        ("C2 Delivery disruption", "SCMS (training); NDC + label text (plant country)",
         "scms_shipments_clean.csv, site / corridor features, panel_inference_features.csv"),
        ("C3 Price & freight integrity", "SCMS line items",
         "unit_price, freight_to_value_ratio, line_item_value by product_group"),
        ("C4 Compliance & contract", "Inspections, Enforcement recalls, Shortages, Warning letters, MSAs",
         "OAI / VAI counts, recalls, plant rollup, two text corpora, C4_fda_component_partial"),
        ("C5 Route exposure", "World Bank LPI & WGI; scenario corridor shares",
         "hormuz_air_share, lpi_score → C5 filled"),
    ], columns=["risk dimension", "feeds", "what Layer 1 delivers"]))

# ------------------------------------------------------------------ 360 profile
with tab_360:
    st.subheader("360° profile of one manufacturer — fused across SCMS, FDA, World Bank and documents (Q2)")
    links = interim("entity_match_candidates")
    if sm.empty:
        st.info("Run the pipeline to build profiles.")
    else:
        name = st.selectbox("Manufacturer", sm["labeler_name_fda"].tolist())
        row = sm[sm["labeler_name_fda"] == name].iloc[0]
        mid = row["manufacturer_id"]
        mine = links[links["labeler_key"] == mid] if len(links) else links

        def linked(source: str) -> set:
            return set(mine.loc[mine["source"] == source, "external_name"]) if len(mine) else set()

        def table_or_none(df: pd.DataFrame, columns: list[str], empty: str) -> None:
            if len(df):
                show(df[[c_ for c_ in columns if c_ in df.columns]])
            else:
                st.caption(empty)

        a, b, c, d, e = st.columns(5)
        a.metric("Plant country", row["manufacturing_country"])
        b.metric("Probation", "Yes" if bool(row["is_probation"]) else "No")
        c.metric("Active FDA shortage", "Yes" if bool(row["active_fda_shortage"]) else "No")
        d.metric("OAI (5 yr)", int(row["oai_count_5yr"]))
        e.metric("Landed cost / vial",
                 f"${row['C1_landed_cost_usd_raw']:,.2f}" if "C1_landed_cost_usd_raw" in sm.columns else "–")

        st.markdown("**Row in suppliers_matrix.csv**")
        show(sm[sm["labeler_name_fda"] == name][list(sm.columns[:16])])

        left, right = st.columns(2)
        with left:
            st.markdown("**FDA — marketing authorization (NDC directory)**")
            labs = interim("fda_ndc_qualified_labelers")
            if len(labs):
                keep = [c_ for c_ in ["labeler_name_fda", "authorization_type", "n_products", "product_ndcs",
                                      "strengths", "marketing_categories", "application_numbers"] if c_ in labs]
                one = labs[labs["labeler_key"] == mid][keep]
                if len(one):
                    show(pd.DataFrame({"field": keep, "value": one.iloc[0].astype(str).values}))
            seed = interim("layer1_scoring_panel_seed")
            srow = seed[seed["labeler_key"] == mid] if len(seed) else seed
            if len(srow):
                st.caption(f"Country source: {srow['country_source'].iat[0]} — {srow['country_evidence'].iat[0]}")
            st.markdown("**FDA — shortage records**")
            sh = interim("fda_shortages_sku")
            table_or_none(sh[sh["company_name"].isin(linked("openfda_shortages"))] if len(sh) else sh,
                          ["company_name", "status", "availability", "presentation", "shortage_reason",
                           "update_date"], "None linked.")
            st.markdown("**FDA — recalls (5 yr)**")
            rc = interim("fda_recalls_5yr")
            table_or_none(rc[rc["recalling_firm"].isin(linked("openfda_recalls"))] if len(rc) else rc,
                          ["recalling_firm", "classification", "report_date", "reason_for_recall",
                           "product_description"], "None linked.")
            st.markdown("**FDA — inspected plants (5 yr)**")
            fac = interim("panel_facility_inspections")
            table_or_none(fac[fac["labeler_key"] == mid] if len(fac) else fac,
                          ["fei_number", "legal_name", "country", "oai_5yr", "vai_5yr", "nai_5yr",
                           "last_inspection_end_date", "worst_classification"], "None linked.")
        with right:
            st.markdown("**SCMS — delivery history and C2 inference features**")
            inf = interim("panel_inference_features")
            irow = inf[inf["manufacturer_id"] == mid] if len(inf) else inf
            if len(irow):
                feat = irow.drop(columns=["sku_id", "manufacturer_id", "labeler_name_fda"]).iloc[0]
                show(pd.DataFrame({"feature": feat.index, "value": feat.astype(str).values}))
            names = sorted(linked("scms"))
            st.caption("Linked SCMS vendors / plants: " + ("; ".join(names) if names else "none (probation pathway)"))
            st.markdown("**World Bank — origin country and corridor**")
            sup = [c_ for c_ in ["lpi_score", "lpi_imputed", "wgi_political_stability", "hormuz_air_share",
                                 "C5_route_exposure", "C5_route_exposure_crisis"] if c_ in sm.columns]
            show(sm[sm["labeler_name_fda"] == name][sup])
            st.markdown("**Documents — warning letters and contract**")
            wlc = interim("warning_letters_corpus")
            table_or_none(wlc[wlc["company_name"].isin(linked("warning_letters"))] if len(wlc) else wlc,
                          ["company_name", "letter_date", "mentions_sterility", "mentions_data_integrity",
                           "source_url"], "No warning letter linked.")
            ctc = interim("contracts_corpus")
            table_or_none(ctc[ctc["supplier_name"].isin(linked("contracts"))] if len(ctc) else ctc,
                          ["supplier_name", "has_force_majeure", "has_lead_time_clause", "has_penalty_clause",
                           "mentions_single_source", "source_file"], "No contract linked.")
        with st.expander("How these records were linked (entity resolution audit)"):
            if len(mine):
                show(mine.drop(columns="labeler_key"))
            else:
                st.caption("No links.")

# ------------------------------------------------------------------ Build steps
with tab_steps:
    if rep is None:
        st.info("Run the pipeline to see the step evidence.")
    else:
        st.subheader("Step 1 — Clean SCMS training data")
        steps = checks[checks["check"].str.startswith("step:") & checks["source"].eq("scms")]
        if len(steps):
            show(pd.DataFrame({"step": steps["check"].str.replace("step:", "", regex=False),
                               "rows": steps["n_affected"], "shape and class balance": steps["detail"]}))
        clean = interim("scms_shipments_clean")
        if len(clean):
            a, b, c = st.columns(3)
            a.metric("Clean shipments", f"{len(clean):,}")
            b.metric("Late (delivered > scheduled)", f"{clean['late'].mean():.1%}")
            sp = clean["time_split"].value_counts()
            c.metric("Train ≤ 2013 / Test 2014–15", f"{sp.get('train', 0):,} / {sp.get('test', 0):,}")
            x, y = st.columns(2)
            x.bar_chart(clean.groupby("time_split")["late"].mean().rename("late rate by split"))
            y.bar_chart(clean.groupby("shipment_mode")["late"].mean().rename("late rate by shipment mode"))
            st.caption("Freight Cost (USD) after coercion: " + str(clean["freight_status"].value_counts().to_dict()))
        st.subheader("Step 2 — Site & corridor features (train period only)")
        sf, cf = interim("scms_site_features_train"), interim("scms_corridor_features_train")
        if len(sf):
            st.markdown("Per manufacturing site: on-time rate, mean and 90th-percentile delay, freight-to-value "
                        "ratio, unit price variance, air-freight share.")
            show(sf.sort_values("site_n_shipments", ascending=False))
        if len(cf):
            st.markdown("Per corridor (origin country × shipment mode):")
            show(cf.sort_values("corridor_n_shipments", ascending=False))
        st.subheader("Step 3 — 2026 oncology scoring panel and entity resolution")
        labs = interim("fda_ndc_qualified_labelers")
        if len(labs):
            st.markdown(f"**{len(labs)} labelers** returned by the live openFDA NDC query for "
                        f"{SETTINGS.sku.molecule.title()}.")
            show(labs[[c_ for c_ in ["labeler_name_fda", "authorization_type", "n_products", "strengths",
                                     "marketing_categories", "labeler_name_variants"] if c_ in labs]])
        top = interim("entity_match_top20_manual_check")
        if len(top):
            st.markdown(f"**Manual check on the top {len(top)} non-exact links** (rapidfuzz `token_sort_ratio ≥ "
                        f"{SETTINGS.sku.fuzzy_match_threshold}`, alias, or name-contained). Record accept / reject "
                        "in `layer1_ingestion/reference/entity_match_decisions.csv`; rejected links are dropped on "
                        "the next run.")
            show(top)
        inf = interim("panel_inference_features")
        if len(inf):
            st.markdown("**Feature rows for C2 feature-based inference (plan §3.1)** — one per manufacturer, same "
                        "feature names as the SCMS training table.")
            show(inf)

# ------------------------------------------------------------------ Quality
with tab_q:
    if checks.empty:
        st.write("No checks recorded yet.")
    else:
        sev = st.multiselect("Severity", ["error", "warning", "info", "pass"], default=["error", "warning", "info"])
        src = st.multiselect("Source", sorted(checks["source"].unique()))
        view = checks[checks["severity"].isin(sev) & ~checks["check"].str.startswith("step:")]
        if src:
            view = view[view["source"].isin(src)]
        order = {"error": 0, "warning": 1, "info": 2, "pass": 3}
        show(view.sort_values("severity", key=lambda s: s.map(order)),
             column_config={"detail": st.column_config.TextColumn(width="large")})

# ------------------------------------------------------------------ Tables
with tab_tables:
    files = sorted(P.interim.glob("*.csv")) if P.interim.exists() else []
    if not files:
        st.write("No output tables yet.")
    else:
        pick = st.selectbox("Table", [f.stem for f in files])
        df = interim(pick)
        st.caption(f"{len(df):,} rows × {df.shape[1]} columns")
        if "text" in df.columns:
            df = df.assign(text=df["text"].astype(str).str.slice(0, 300) + "…")
        show(df.head(2000))
        st.download_button(f"Download {pick}.csv", (P.interim / f"{pick}.csv").read_bytes(),
                           file_name=f"{pick}.csv", mime="text/csv")

# ------------------------------------------------------------------ Quarantine
with tab_quar:
    files = sorted(P.quarantine.glob("*.csv")) if P.quarantine.exists() else []
    if not files:
        st.write("Nothing quarantined yet." if rep is None else "Nothing quarantined.")
    else:
        pick = st.selectbox("Quarantine file", [f.stem for f in files])
        q = load(P.quarantine / f"{pick}.csv")
        show(q["_reject_reason"].value_counts().rename_axis("reason").reset_index(name="rows"))
        show(q.head(1000))

# ------------------------------------------------------------------ Download
with tab_dl:
    if rep is None:
        st.write("Run the pipeline first.")
    else:
        st.write("Everything in one zip: `data/processed/suppliers_matrix.csv`, `quotes_2026_carboplatin.csv`, all "
                 "clean tables, quarantine files, the quality report and the manifest.")
        st.download_button("⬇ Download all Layer 1 outputs (.zip)", zip_outputs(),
                           file_name=f"pharmashield_layer1_outputs_{man.get('as_of')}.zip", mime="application/zip",
                           type="primary")
        qm = P.quality / "quality_report.md"
        if qm.exists():
            st.download_button("Download quality_report.md", qm.read_bytes(), file_name="quality_report.md")
