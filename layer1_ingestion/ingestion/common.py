"""
Shared plumbing for every Layer-1 source:

* HttpClient   - retries, exponential back-off, Retry-After, polite pacing,
                 immutable raw-snapshot caching and an --offline replay mode.
* QualityLog   - one place where every check records what it found, so the
                 run ends with a single auditable data-quality report.
* quarantine() - rejected rows are never silently dropped; they are written
                 to data/quarantine/ with a reason code.
* Schema       - lightweight column/type/range contracts enforced before any
                 table is written to data/interim/.
* Name keys    - deterministic company-name normalisation that Layer 2
                 (rapidfuzz entity resolution) joins on.
"""
from __future__ import annotations

import hashlib
import json
import logging
import random
import re
import time
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd
import requests

from .config import SETTINGS

log = logging.getLogger("pharmashield.ingestion")


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
def setup_logging(level: str = "INFO", logfile: Path | None = None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if logfile:
        logfile.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(logfile, encoding="utf-8"))
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        handlers=handlers,
        force=True,
    )


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# Data-quality log
# --------------------------------------------------------------------------- #
@dataclass
class QualityLog:
    """Collects checks across all sources -> data/quality/quality_report.{json,md}."""

    records: list[dict] = field(default_factory=list)
    source_status: dict[str, dict] = field(default_factory=dict)

    def check(self, source: str, check: str, n_affected: int, n_total: int | None = None,
              action: str = "", severity: str = "info", detail: str = "") -> None:
        n_affected = int(n_affected)
        if n_affected == 0 and severity == "info" and not detail:
            severity = "pass"
        rec = dict(source=source, check=check, n_affected=n_affected,
                   n_total=None if n_total is None else int(n_total),
                   pct=None if not n_total else round(100 * n_affected / n_total, 2),
                   action=action, severity=severity, detail=detail)
        self.records.append(rec)
        lvl = {"error": logging.ERROR, "warning": logging.WARNING}.get(severity, logging.INFO)
        log.log(lvl, "[%s] %s: %s%s %s", source, check, n_affected,
                f"/{n_total}" if n_total else "", f"-> {action}" if action else "")

    def status(self, source: str, status: str, **info: Any) -> None:
        self.source_status[source] = {"status": status, **info}

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.records)

    def write(self, outdir: Path, manifest: dict) -> None:
        outdir.mkdir(parents=True, exist_ok=True)
        payload = {"generated_at_utc": utc_now(), "sources": self.source_status,
                   "checks": self.records, "manifest": manifest}
        (outdir / "quality_report.json").write_text(json.dumps(payload, indent=2, default=str))
        self.to_frame().to_csv(outdir / "quality_checks.csv", index=False)

        lines = ["# PharmaShield GCC - Layer 1 Data Quality Report", "",
                 f"Generated (UTC): {payload['generated_at_utc']}", "", "## Source status", "",
                 "| Source | Status | Rows out | Notes |", "|---|---|---|---|"]
        for src, st in self.source_status.items():
            lines.append(f"| {src} | {st.get('status')} | {st.get('rows', '')} | "
                         f"{str(st.get('note', '')).replace('|', '/')} |")
        lines += ["", "## Checks (non-passing first)", "",
                  "| Source | Check | Affected | % | Severity | Action | Detail |",
                  "|---|---|---|---|---|---|---|"]
        order = {"error": 0, "warning": 1, "info": 2, "pass": 3}
        for r in sorted(self.records, key=lambda r: order.get(r["severity"], 9)):
            lines.append(f"| {r['source']} | {r['check']} | {r['n_affected']}"
                         f"{'/' + str(r['n_total']) if r['n_total'] else ''} | "
                         f"{'' if r['pct'] is None else r['pct']} | {r['severity']} | "
                         f"{r['action']} | {str(r['detail']).replace('|', '/')[:200]} |")
        (outdir / "quality_report.md").write_text("\n".join(lines) + "\n")


QLOG = QualityLog()


# --------------------------------------------------------------------------- #
# Quarantine
# --------------------------------------------------------------------------- #
_QUARANTINE: dict[str, list[pd.DataFrame]] = {}


def quarantine(source: str, rows: pd.DataFrame, reason: str) -> None:
    """Stage rejected rows; flushed to data/quarantine/<source>_rejected.csv."""
    if rows is None or rows.empty:
        return
    tagged = rows.copy()
    tagged.insert(0, "_reject_reason", reason)
    _QUARANTINE.setdefault(source, []).append(tagged)


def flush_quarantine(outdir: Path) -> dict[str, int]:
    outdir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for source, frames in _QUARANTINE.items():
        df = pd.concat(frames, ignore_index=True, sort=False)
        df.to_csv(outdir / f"{source}_rejected.csv", index=False)
        counts[source] = len(df)
    _QUARANTINE.clear()
    return counts


# --------------------------------------------------------------------------- #
# HTTP client with caching + offline replay
# --------------------------------------------------------------------------- #
class OfflineCacheMiss(RuntimeError):
    pass


class HttpClient:
    """
    GET/POST JSON with retries. Every successful response is written to
    data/raw/<namespace>/<request-hash>__<timestamp>.json (immutable). With
    offline=True no network is used: the newest snapshot for the identical
    request is replayed, which makes runs reproducible for the Sunday-night
    audit (Plan 5.2 step 10) and survives API outages during the demo.
    """

    def __init__(self, raw_dir: Path, offline: bool = False, session: requests.Session | None = None):
        self.raw_dir = raw_dir
        self.offline = offline
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": SETTINGS.http.user_agent,
                                     "Accept": "application/json"})
        self._last_call = 0.0
        self.calls = 0
        self.cache_hits = 0

    @staticmethod
    def _key(method: str, url: str, params: dict | None, body: dict | None) -> str:
        safe_params = {k: v for k, v in (params or {}).items() if k.lower() not in {"api_key"}}
        blob = json.dumps([method, url, safe_params, body], sort_keys=True, default=str)
        return hashlib.sha1(blob.encode()).hexdigest()[:16]

    def _cache_path(self, namespace: str, key: str) -> Path:
        d = self.raw_dir / namespace
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{key}__{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"

    def _latest_cached(self, namespace: str, key: str) -> Path | None:
        d = self.raw_dir / namespace
        hits = sorted(d.glob(f"{key}__*.json")) if d.exists() else []
        return hits[-1] if hits else None

    def request_json(self, namespace: str, url: str, params: dict | None = None,
                     method: str = "GET", body: dict | None = None, headers: dict | None = None,
                     ok_statuses: Iterable[int] = (200,),
                     empty_on: Callable[[requests.Response], bool] | None = None) -> Any:
        key = self._key(method, url, params, body)
        if self.offline:
            cached = self._latest_cached(namespace, key)
            if cached is None:
                raise OfflineCacheMiss(f"No cached snapshot for {namespace} {url} {params}")
            self.cache_hits += 1
            return json.loads(cached.read_text())["payload"]

        cfg = SETTINGS.http
        last_err: Exception | None = None
        for attempt in range(cfg.max_retries + 1):
            wait = cfg.min_interval_s - (time.monotonic() - self._last_call)
            if wait > 0:
                time.sleep(wait)
            try:
                self._last_call = time.monotonic()
                self.calls += 1
                resp = self.session.request(method, url, params=params, json=body,
                                            headers=headers, timeout=cfg.timeout_s)
                if empty_on is not None and empty_on(resp):
                    payload: Any = None
                elif resp.status_code in ok_statuses:
                    payload = resp.json()
                elif resp.status_code in (429, 500, 502, 503, 504):
                    retry_after = resp.headers.get("Retry-After")
                    delay = float(retry_after) if retry_after and retry_after.isdigit() \
                        else cfg.backoff_base_s * (2 ** attempt) + random.uniform(0, 1)
                    log.warning("HTTP %s from %s - retry %d/%d in %.1fs", resp.status_code,
                                url, attempt + 1, cfg.max_retries, delay)
                    time.sleep(min(delay, 60))
                    continue
                else:
                    raise requests.HTTPError(f"HTTP {resp.status_code} for {resp.url}: "
                                             f"{resp.text[:300]}", response=resp)
                snap = {"fetched_at_utc": utc_now(), "method": method, "url": url,
                        "params": {k: v for k, v in (params or {}).items() if k != "api_key"},
                        "body": body, "status": resp.status_code, "payload": payload}
                self._cache_path(namespace, key).write_text(json.dumps(snap))
                return payload
            except (requests.ConnectionError, requests.Timeout, ValueError) as e:
                # ValueError covers truncated / non-JSON bodies.
                last_err = e
                delay = cfg.backoff_base_s * (2 ** attempt) + random.uniform(0, 1)
                log.warning("%s on %s - retry %d/%d in %.1fs", type(e).__name__, url,
                            attempt + 1, cfg.max_retries, delay)
                time.sleep(min(delay, 60))
        # Network exhausted -> fall back to last good snapshot rather than failing the run.
        cached = self._latest_cached(namespace, key)
        if cached is not None:
            log.error("Network failed for %s; replaying cached snapshot %s", url, cached.name)
            self.cache_hits += 1
            return json.loads(cached.read_text())["payload"]
        raise RuntimeError(f"Request failed after retries and no cache exists: {url}") from last_err

    def get_text(self, namespace: str, url: str) -> str | None:
        """Fetch an HTML page (warning letters). Cached like JSON."""
        key = self._key("GET", url, None, None)
        if self.offline:
            cached = self._latest_cached(namespace, key)
            return json.loads(cached.read_text())["payload"] if cached else None
        for attempt in range(SETTINGS.http.max_retries + 1):
            try:
                wait = SETTINGS.http.min_interval_s * 3 - (time.monotonic() - self._last_call)
                if wait > 0:
                    time.sleep(wait)
                self._last_call = time.monotonic()
                self.calls += 1
                r = self.session.get(url, timeout=SETTINGS.http.timeout_s,
                                     headers={"Accept": "text/html,application/xhtml+xml"})
                if r.status_code == 200:
                    snap = {"fetched_at_utc": utc_now(), "url": url, "payload": r.text}
                    self._cache_path(namespace, key).write_text(json.dumps(snap))
                    return r.text
                if r.status_code in (429, 500, 502, 503, 504):
                    time.sleep(SETTINGS.http.backoff_base_s * (2 ** attempt))
                    continue
                log.warning("HTTP %s for %s (not retried)", r.status_code, url)
                return None
            except (requests.ConnectionError, requests.Timeout) as e:
                log.warning("%s for %s", type(e).__name__, url)
                time.sleep(SETTINGS.http.backoff_base_s * (2 ** attempt))
        return None


# --------------------------------------------------------------------------- #
# Schema contracts
# --------------------------------------------------------------------------- #
@dataclass
class Col:
    name: str
    dtype: str                       # "str" | "int" | "float" | "bool" | "date"
    required: bool = True            # column must exist
    nullable: bool = True
    min: float | None = None
    max: float | None = None
    allowed: tuple | None = None
    unique: bool = False


def enforce_schema(df: pd.DataFrame, schema: list[Col], source: str) -> pd.DataFrame:
    """Coerce types, verify ranges/enums/uniqueness. Raises on contract breach."""
    df = df.copy()
    problems = []
    for c in schema:
        if c.name not in df.columns:
            if c.required:
                problems.append(f"missing column '{c.name}'")
                continue
            df[c.name] = pd.NA
        s = df[c.name]
        if c.dtype == "int":
            df[c.name] = pd.to_numeric(s, errors="coerce").round().astype("Int64")
        elif c.dtype == "float":
            df[c.name] = pd.to_numeric(s, errors="coerce").astype("float64")
        elif c.dtype == "bool":
            df[c.name] = s.map(lambda v: v if isinstance(v, (bool, np.bool_)) or pd.isna(v)
                               else str(v).strip().lower() in {"true", "1", "yes", "y"}).astype("boolean")
        elif c.dtype == "date":
            df[c.name] = pd.to_datetime(s, errors="coerce")
        else:
            df[c.name] = s.astype("string")
        s = df[c.name]
        if not c.nullable and s.isna().any():
            problems.append(f"'{c.name}' has {int(s.isna().sum())} nulls")
        if c.min is not None and (s.dropna() < c.min).any():
            problems.append(f"'{c.name}' below min {c.min}")
        if c.max is not None and (s.dropna() > c.max).any():
            problems.append(f"'{c.name}' above max {c.max}")
        if c.allowed is not None:
            bad = set(s.dropna().unique()) - set(c.allowed)
            if bad:
                problems.append(f"'{c.name}' unexpected values {sorted(map(str, bad))[:5]}")
        if c.unique and s.dropna().duplicated().any():
            problems.append(f"'{c.name}' not unique ({int(s.dropna().duplicated().sum())} dups)")
    QLOG.check(source, "schema_contract", len(problems), action="raise" if problems else "",
               severity="error" if problems else "info", detail="; ".join(problems))
    if problems:
        raise ValueError(f"[{source}] schema contract violated: " + "; ".join(problems))
    ordered = [c.name for c in schema]
    return df[ordered + [x for x in df.columns if x not in ordered]]


def write_table(df: pd.DataFrame, name: str, outdir: Path | None = None) -> Path:
    outdir = outdir or SETTINGS.paths.interim
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"{name}.csv"
    df.to_csv(path, index=False, date_format="%Y-%m-%d")
    log.info("wrote %s (%d rows x %d cols)", path.relative_to(SETTINGS.paths.root)
             if path.is_relative_to(SETTINGS.paths.root) else path, len(df), df.shape[1])
    return path


# --------------------------------------------------------------------------- #
# Text / name normalisation
# --------------------------------------------------------------------------- #
_LEGAL_SUFFIXES = r"""
    INCORPORATED|INC|LLC|L\.L\.C|LTD|LIMITED|PVT|PRIVATE|CORPORATION|CORP|CO|COMPANY|
    PLC|GMBH|AG|SA|S\.A|SPA|S\.P\.A|BV|B\.V|NV|N\.V|KG|LP|LLP|PTY|SRL|AB|AS|OY|KK
"""
_SUFFIX_RE = re.compile(rf"\b(?:{_LEGAL_SUFFIXES.replace(chr(10), '').replace(' ', '')})\b\.?")
_INDUSTRY_WORDS = re.compile(
    r"\b(PHARMACEUTICALS?|PHARMA|LABORATORIES|LABORATORY|LABS?|HEALTHCARE|MEDICINES|"
    r"PARENTERAL|INDUSTRIES|INTERNATIONAL|INSTITUTIONAL|GLOBAL|HOLDINGS?|GROUP|USA|US|AMERICA|NORTH)\b")


def fix_mojibake(s: str) -> str:
    """Repair UTF-8 read as Latin-1 (e.g. 'CÃ´te' -> 'Côte')."""
    if not isinstance(s, str) or not re.search(r"[ÃÂâ][\x80-\xbf€™œ\"]?", s):
        return s
    for enc in ("latin-1", "cp1252"):
        try:
            fixed = s.encode(enc).decode("utf-8")
            return fixed
        except (UnicodeEncodeError, UnicodeDecodeError):
            continue
    return s


def clean_text(s: Any) -> Any:
    if not isinstance(s, str):
        return s
    s = fix_mojibake(s)
    s = unicodedata.normalize("NFKC", s).replace(" ", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s if s else np.nan


def parent_company(name: str) -> str | None:
    """'Hospira, Inc., a Pfizer Company' -> 'PFIZER'."""
    if not isinstance(name, str):
        return None
    m = re.search(r",?\s*an?\s+([A-Za-z0-9&.\- ]+?)\s+company\b", name, flags=re.I)
    return company_key(m.group(1)) if m else None


def company_key(name: Any) -> Any:
    """Legal-suffix-stripped canonical key: 'Gland Pharma Limited' -> 'GLAND PHARMA'."""
    if not isinstance(name, str) or not name.strip():
        return np.nan
    s = unicodedata.normalize("NFKD", fix_mojibake(name)).encode("ascii", "ignore").decode()
    s = re.sub(r",?\s*an?\s+[A-Za-z0-9&.\- ]+?\s+company\b", "", s, flags=re.I)  # drop 'a Pfizer Company'
    s = re.sub(r"\(.*?\)", " ", s.upper())
    s = s.replace("&", " AND ")
    s = _SUFFIX_RE.sub(" ", s)
    s = re.sub(r"[^A-Z0-9 ]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s or np.nan


def company_core(name: Any) -> Any:
    """Aggressive key for fuzzy blocking: 'Teva Parenteral Medicines, Inc.' -> 'TEVA'."""
    k = company_key(name)
    if not isinstance(k, str):
        return k
    core = re.sub(r"\s+", " ", _INDUSTRY_WORDS.sub(" ", k)).strip()
    return core or k


def snake(col: str) -> str:
    col = clean_text(str(col)).replace("﻿", "")
    col = re.sub(r"\(usd\)", "usd", col, flags=re.I)
    col = re.sub(r"[^0-9a-zA-Z]+", "_", col).strip("_").lower()
    return col


def robust_z(x: pd.Series) -> pd.Series:
    """MAD-based z-score; safe when MAD == 0."""
    x = pd.to_numeric(x, errors="coerce")
    med = x.median()
    mad = (x - med).abs().median()
    if not np.isfinite(mad) or mad == 0:
        mad = (x - med).abs().mean() or np.nan
    return 0.6745 * (x - med) / mad
