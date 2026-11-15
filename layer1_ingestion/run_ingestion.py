#!/usr/bin/env python3
"""
PharmaShield GCC - Layer 1: Multi-Source Data Ingestion & Collection
=====================================================================

    python run_ingestion.py                       # all sources, live APIs
    python run_ingestion.py --offline             # replay cached API snapshots (no network)
    python run_ingestion.py --sources scms openfda worldbank
    python run_ingestion.py --crawl-warning-letters 80
    python run_ingestion.py --strict              # non-zero exit if any source FAILED / error check

Outputs
    data/raw/         immutable JSON snapshots of every API response (audit trail)
    data/processed/   suppliers_matrix.csv in the locked schema (plan 5.1) + quotes_2026_carboplatin.csv
    data/interim/     clean, schema-validated tables for Layers 2-4
    data/quarantine/  every rejected row with its reason code (nothing silently dropped)
    data/quality/     quality_report.md / .json / quality_checks.csv
    data/manifest.json  run metadata, row counts, file hashes
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from datetime import date
from pathlib import Path

from ingestion import documents, fda_inspections, openfda, optional_sources, scenario, scms, worldbank
from ingestion.common import QLOG, HttpClient, flush_quarantine, log, setup_logging, sha256_file, utc_now
from ingestion.config import SETTINGS

# order matters: scenario/panel reads the outputs of the others
SOURCES = {
    "scms": scms.run,
    "ghsc_psm": optional_sources.run_ghsc,
    "internal": optional_sources.run_internal,
    "openfda": openfda.run,
    "inspections": fda_inspections.run,
    "worldbank": worldbank.run,
    "documents": documents.run,
    "scenario": scenario.run,
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", nargs="+", choices=list(SOURCES), default=list(SOURCES))
    ap.add_argument("--offline", action="store_true", help="replay cached raw snapshots only")
    ap.add_argument("--crawl-warning-letters", type=int, default=0, metavar="N",
                    help="best-effort crawl of N CGMP warning letters from fda.gov")
    ap.add_argument("--strict", action="store_true", help="exit 1 on any FAILED source or error check")
    ap.add_argument("--as-of", metavar="YYYY-MM-DD",
                    help="run date for look-back windows (offline mode defaults to last manifest)")
    ap.add_argument("--log-level", default="INFO")
    a = ap.parse_args(argv)

    P = SETTINGS.paths
    as_of = a.as_of
    if as_of is None and a.offline and (P.data / "manifest.json").exists():
        as_of = json.loads((P.data / "manifest.json").read_text()).get("as_of")
    if as_of:  # identical windows => identical cached requests => reproducible replay
        object.__setattr__(SETTINGS, "as_of", date.fromisoformat(as_of))
    for d in (P.raw, P.manual, P.interim, P.quarantine, P.quality, P.processed):
        d.mkdir(parents=True, exist_ok=True)
    for sub in ("scms", "fda_inspections", "warning_letters", "contracts", "ghsc_psm", "internal"):
        (P.manual / sub).mkdir(parents=True, exist_ok=True)
    setup_logging(a.log_level, P.quality / "ingestion.log")
    log.info("PharmaShield Layer 1 ingestion | as_of=%s | offline=%s | sources=%s",
             SETTINGS.as_of, a.offline, a.sources)

    http = HttpClient(P.raw, offline=a.offline)
    ctx = dict(http=http, manual_dir=P.manual, ref_dir=P.reference, interim=P.interim,
               offline=a.offline, crawl_warning_letters=a.crawl_warning_letters)
    outputs, timings = {}, {}
    for name in a.sources:
        t0 = time.time()
        try:
            outputs.update(SOURCES[name](**ctx) or {})
        except Exception as e:  # isolate failures per source
            log.exception("source '%s' crashed", name)
            QLOG.status(name, "FAILED", note=f"{type(e).__name__}: {str(e)[:200]}")
        timings[name] = round(time.time() - t0, 1)

    q_counts = flush_quarantine(P.quarantine)
    manifest = {
        "run_at_utc": utc_now(), "as_of": str(SETTINGS.as_of), "offline": a.offline,
        "python": platform.python_version(), "sku": SETTINGS.sku.__dict__,
        "http_calls": http.calls, "cache_replays": http.cache_hits, "timings_s": timings,
        "quarantined_rows": q_counts,
        "outputs": {k: {"path": os.path.relpath(v, P.root), "sha256": sha256_file(Path(v))}
                    for k, v in outputs.items() if v and Path(v).exists()},
    }
    (P.data / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    QLOG.write(P.quality, manifest)

    print("\n=== Layer 1 ingestion summary ===")
    for src, st in QLOG.source_status.items():
        print(f"  {src:28s} {st['status']:8s} rows={st.get('rows', '-'):<7} {st.get('note', '')}")
    errors = [r for r in QLOG.records if r["severity"] == "error"]
    warns = [r for r in QLOG.records if r["severity"] == "warning"]
    print(f"  checks: {len(QLOG.records)} | warnings: {len(warns)} | errors: {len(errors)}")
    print(f"  quarantined rows: {sum(q_counts.values())} -> data/quarantine/")
    print("  report: data/quality/quality_report.md\n")
    if (P.processed / "suppliers_matrix.csv").exists():
        print(f"  locked-schema hand-off: {P.processed / 'suppliers_matrix.csv'}\n")
    failed = any(st["status"] == "FAILED" for st in QLOG.source_status.values())
    return 1 if a.strict and (failed or errors) else 0


if __name__ == "__main__":
    sys.exit(main())
