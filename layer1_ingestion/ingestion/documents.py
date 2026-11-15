"""
Source 8 - Unstructured documents for the C4 NLP engine (Member 4B consumes these corpora).

  a) FDA CGMP Warning Letters (full text)
       inputs (any combination):
         - data/manual/warning_letters/urls.csv   (column 'url'; optional 'company')
         - data/manual/warning_letters/*.html|*.htm|*.txt|*.pdf  (saved letters)
         - --crawl-warning-letters N : best-effort crawl of the FDA index, CGMP drug letters only
  b) Supplier Master Service Agreements (synthetic for the prototype)
         - data/manual/contracts/*.txt|*.md|*.docx|*.pdf

Data issues handled:
  * boilerplate / navigation stripped from HTML; whitespace + unicode normalised
  * duplicates: same URL, same MARCS-CMS number, or identical normalised text hash
  * empty / truncated documents (< 800 chars) -> quarantined
  * PII minimisation (UAE PDPL, Federal Decree-Law 45/2021): e-mails, phone numbers and
    salutation names ('Dear Mr. X') are redacted before the corpus is written
  * metadata extracted with regex: letter date, MARCS-CMS ref, FEI, CGMP/sterility markers
"""
from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd

from .common import QLOG, HttpClient, company_key, quarantine, write_table

log = logging.getLogger("pharmashield.ingestion.documents")
FDA = "https://www.fda.gov"
WL_INDEX = (FDA + "/inspections-compliance-enforcement-and-criminal-investigations/"
            "compliance-actions-and-activities/warning-letters")
MIN_CHARS = 800

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE = re.compile(r"(?<!\d)(?:\+?\d{1,3}[\s.-]?)?(?:\(\d{2,4}\)|\d{2,4})[\s.-]\d{3,4}[\s.-]\d{3,4}(?!\d)")
_SALUTE = re.compile(r"\b(Dear|Attn:?|Attention:?)\s+(Mr|Mrs|Ms|Dr|Prof)\.?\s+[A-Z][\w'\-]+(?:\s+[A-Z][\w'\-]+)?",)
_DATE = re.compile(r"\b(January|February|March|April|May|June|July|August|September|October|November|"
                   r"December)\s+\d{1,2},\s+(19|20)\d{2}\b")
_MARCS = re.compile(r"MARCS-CMS\s*(?:No\.?|#|:)?\s*(\d{5,7})", re.I)
_FEI = re.compile(r"\bFEI\s*(?:No\.?|Number|#)?\s*:?\s*(\d{7,11})", re.I)


def redact(text: str) -> tuple[str, int]:
    n = 0
    for pat, rep in ((_EMAIL, "[EMAIL]"), (_PHONE, "[PHONE]")):
        text, k = pat.subn(rep, text)
        n += k
    text, k = _SALUTE.subn(lambda m: f"{m.group(1)} {m.group(2)}. [NAME]", text)
    return text, n + k


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _html_to_text(html: str) -> tuple[str, str | None]:
    try:
        from bs4 import BeautifulSoup
    except ImportError:  # minimal fallback
        txt = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", html, flags=re.S | re.I)
        return _norm(re.sub(r"<[^>]+>", " ", txt)), None
    soup = BeautifulSoup(html, "html.parser")
    for t in soup(["script", "style", "nav", "header", "footer", "noscript", "form", "aside"]):
        t.decompose()
    title = soup.find("h1")
    title = _norm(title.get_text(" ")) if title else None
    # FDA pages: main column holds the letter; otherwise pick the block with most paragraph text
    main = soup.find(attrs={"role": "main"}) or soup.find("article")
    if main is None:
        blocks = soup.find_all(["div", "section"])
        main = max(blocks, key=lambda b: sum(len(p.get_text()) for p in b.find_all("p", recursive=False)),
                   default=soup)
    return _norm(main.get_text(" ")), title


def _file_to_text(p: Path) -> tuple[str | None, str | None]:
    suf = p.suffix.lower()
    try:
        if suf in {".html", ".htm"}:
            return _html_to_text(p.read_text(errors="ignore"))
        if suf in {".txt", ".md"}:
            return _norm(p.read_text(errors="ignore")), None
        if suf == ".pdf":
            from pypdf import PdfReader
            return _norm(" ".join(pg.extract_text() or "" for pg in PdfReader(str(p)).pages)), None
        if suf == ".docx":
            import docx
            return _norm(" ".join(par.text for par in docx.Document(str(p)).paragraphs)), None
    except ImportError as e:
        QLOG.check("documents", f"missing_parser_for_{suf}", 1, severity="warning",
                   detail=f"pip install {'pypdf' if suf == '.pdf' else 'python-docx'} ({e})")
    except Exception as e:
        QLOG.check("documents", f"unreadable:{p.name}", 1, severity="warning", detail=str(e)[:200])
    return None, None


def crawl_index(http: HttpClient, max_letters: int) -> list[dict]:
    """Best-effort crawl of the public index table; filters to CGMP drug letters."""
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        log.warning("beautifulsoup4 not installed - skipping crawl")
        return []
    found, page = [], 0
    while len(found) < max_letters and page < 60:
        html = http.get_text("fda_warning_letters/index", f"{WL_INDEX}?page={page}")
        if not html:
            QLOG.check("warning_letters", "index_page_unreachable", 1, severity="warning",
                       detail="FDA may block automated clients; use urls.csv or saved HTML instead")
            break
        rows = BeautifulSoup(html, "html.parser").select("table tbody tr")
        if not rows:
            break
        for tr in rows:
            a = tr.select_one("td a[href]")
            subject = " ".join(td.get_text(" ", strip=True) for td in tr.find_all("td"))
            if a and re.search(r"CGMP|Current Good Manufacturing", subject, re.I) and \
                    not re.search(r"food|tobacco|device|cosmetic|dietary", subject, re.I):
                found.append({"url": urljoin(FDA, a["href"]), "company_hint": a.get_text(strip=True)})
        page += 1
    return found[:max_letters]


def _letter_record(text: str, title: str | None, url: str | None, file: str | None,
                   company_hint: str | None) -> dict:
    red, n_red = redact(text)
    # FDA page titles read 'Company Name - MARCS# - MM/DD/YYYY'; that beats a filename/CSV hint
    if title and " - " in title:
        company_hint = title.split(" - ")[0].strip()
    elif not company_hint:
        company_hint = title
    date = _DATE.search(text)
    marcs = _MARCS.search(text)
    fei = _FEI.search(text)
    return {
        "doc_id": hashlib.sha1((url or file or text[:200]).encode()).hexdigest()[:12],
        "source_url": url, "source_file": file, "title": title,
        "company_name": company_hint,
        "company_key": company_key(company_hint),
        "letter_date": pd.to_datetime(date.group(0), errors="coerce") if date else pd.NaT,
        "marcs_cms": marcs.group(1) if marcs else None,
        "fei_number": fei.group(1) if fei else None,
        "mentions_cgmp": bool(re.search(r"CGMP|current good manufacturing", text, re.I)),
        "mentions_sterility": bool(re.search(r"steril|aseptic|endotoxin|media fill|particulate", text, re.I)),
        "mentions_data_integrity": bool(re.search(r"data integrity|audit trail|21 CFR 211\.68|"
                                                  r"21 CFR 211\.194|backdat", text, re.I)),
        "n_chars": len(red), "pii_redactions": n_red,
        "text_sha256": hashlib.sha256(_norm(text).lower().encode()).hexdigest(),
        "text": red,
    }


def _dedupe(df: pd.DataFrame, source: str) -> pd.DataFrame:
    if df.empty:
        return df
    n = len(df)
    short = df["n_chars"] < MIN_CHARS
    quarantine(source, df[short].drop(columns="text"), f"too_short_lt_{MIN_CHARS}_chars")
    df = df[~short]
    masks = [df.duplicated("text_sha256")]
    if "source_url" in df:
        masks.append(df["source_url"].notna() & df.duplicated("source_url"))
    if "marcs_cms" in df:
        masks.append(df["marcs_cms"].notna() & df.duplicated("marcs_cms"))
    dup = pd.concat(masks, axis=1).any(axis=1)
    quarantine(source, df[dup].drop(columns="text"), "duplicate_document")
    QLOG.check(source, "too_short_documents", int(short.sum()), n, action="quarantined")
    QLOG.check(source, "duplicate_documents(url/marcs/text-hash)", int(dup.sum()), n, action="quarantined")
    QLOG.check(source, "pii_redactions", int(df["pii_redactions"].sum()), action="redacted (UAE PDPL)")
    return df[~dup]


def warning_letters(http: HttpClient, manual_dir: Path, crawl: int = 0) -> pd.DataFrame:
    d = manual_dir / "warning_letters"
    d.mkdir(parents=True, exist_ok=True)
    recs = []
    for p in sorted(d.glob("*")):
        if p.suffix.lower() in {".html", ".htm", ".txt", ".pdf"}:
            text, title = _file_to_text(p)
            if text:
                recs.append(_letter_record(text, title, None, p.name, p.stem.replace("_", " ")))
    targets = []
    for f in [d / "urls.csv", d / "urls.txt"]:
        if f.exists():
            if f.suffix == ".csv":
                u = pd.read_csv(f, dtype=str)
                targets += [{"url": r["url"], "company_hint": r.get("company")} for _, r in u.iterrows()
                            if isinstance(r.get("url"), str)]
            else:
                targets += [{"url": l.strip(), "company_hint": None} for l in f.read_text().splitlines()
                            if l.strip().startswith("http")]
    if crawl:
        targets += crawl_index(http, crawl)
    seen = set()
    for t in targets:
        if t["url"] in seen:
            continue
        seen.add(t["url"])
        html = http.get_text("fda_warning_letters/letters", t["url"])
        if html is None:
            QLOG.check("warning_letters", "letter_unreachable", 1, severity="warning", detail=t["url"])
            continue
        text, title = _html_to_text(html)
        recs.append(_letter_record(text, title, t["url"], None, t.get("company_hint")))
    df = pd.DataFrame(recs)
    if df.empty:
        QLOG.check("warning_letters", "documents_collected", 0, severity="warning",
                   action="NLP C4 component needs letters - add urls.csv or saved HTML")
        return df
    df = _dedupe(df, "warning_letters")
    QLOG.check("warning_letters", "documents_collected", len(df),
               detail=f"cgmp={int(df.mentions_cgmp.sum())} sterility={int(df.mentions_sterility.sum())} "
                      f"data_integrity={int(df.mentions_data_integrity.sum())}")
    return df


def contracts(manual_dir: Path) -> pd.DataFrame:
    d = manual_dir / "contracts"
    d.mkdir(parents=True, exist_ok=True)
    recs = []
    for p in sorted(d.glob("*")):
        if p.suffix.lower() not in {".txt", ".md", ".pdf", ".docx"}:
            continue
        text, _ = _file_to_text(p)
        if not text:
            continue
        red, n_red = redact(text)
        m = re.search(r"between\s+(.{3,80}?)\s*(?:\(|,)?\s*(?:the\s+)?[\"“]?(?:Supplier|Vendor|Manufacturer)",
                      text, re.I)
        supplier = m.group(1).strip(" ,\"“”") if m else p.stem.replace("_", " ")
        recs.append({"doc_id": hashlib.sha1(p.name.encode()).hexdigest()[:12], "source_file": p.name,
                     "supplier_name": supplier, "supplier_key": company_key(supplier),
                     "has_force_majeure": bool(re.search(r"force\s+majeure", text, re.I)),
                     "has_lead_time_clause": bool(re.search(r"lead[\s-]time", text, re.I)),
                     "has_penalty_clause": bool(re.search(r"penalt|liquidated damages", text, re.I)),
                     "mentions_single_source": bool(re.search(r"sole|single[\s-]source|exclusiv", text, re.I)),
                     "n_chars": len(red), "pii_redactions": n_red,
                     "text_sha256": hashlib.sha256(_norm(text).lower().encode()).hexdigest(), "text": red})
    df = pd.DataFrame(recs)
    if df.empty:
        QLOG.check("contracts", "documents_collected", 0, severity="info",
                   detail="drop synthetic MSAs into data/manual/contracts/")
        return df
    df = _dedupe(df.assign(source_url=None, marcs_cms=None), "contracts").drop(columns=["source_url", "marcs_cms"])
    QLOG.check("contracts", "documents_collected", len(df))
    return df


def run(http: HttpClient, manual_dir: Path, crawl_warning_letters: int = 0, **_) -> dict:
    out = {}
    wl = warning_letters(http, manual_dir, crawl_warning_letters)
    out["warning_letters_corpus"] = write_table(wl, "warning_letters_corpus")
    QLOG.status("warning_letters", "OK" if len(wl) else "EMPTY", rows=len(wl))
    ct = contracts(manual_dir)
    out["contracts_corpus"] = write_table(ct, "contracts_corpus")
    QLOG.status("contracts", "OK" if len(ct) else "EMPTY", rows=len(ct))
    return out
