"""
Source 7 - World Bank LPI (Logistics Performance Index, 1-5) and WGI Political Stability
(-2.5..+2.5) -> C5 route exposure and macro risk.

Data issues handled:
  * LPI is only published in survey years (2007, 2010, 2012, 2014, 2016, 2018, 2022-edition
    released 2023) -> latest NON-NULL value per country, with its year kept for transparency
  * regional / income aggregates mixed with countries ('World', 'South Asia') -> removed using
    the country metadata endpoint (region.id == 'NA' marks aggregates)
  * WGI code differs between World Bank source catalogues -> candidates tried in order;
    API error envelopes ({"message":[...]}) detected rather than parsed as data
  * out-of-range values -> quarantined; stale values (> 6 yrs) flagged
  * multi-page responses -> paginated
"""
from __future__ import annotations

import logging

import pandas as pd

from .common import QLOG, Col, HttpClient, enforce_schema, quarantine, write_table
from .config import SETTINGS

log = logging.getLogger("pharmashield.ingestion.worldbank")
SRC = "worldbank"


def _paged(http: HttpClient, path: str, params: dict) -> list:
    cfg = SETTINGS.worldbank
    out, page = [], 1
    while True:
        p = {"format": "json", "per_page": 20000, "page": page, **params}
        payload = http.request_json("worldbank", f"{cfg.base_url}/{path}", params=p)
        if not isinstance(payload, list) or len(payload) < 2 or payload[1] is None:
            msg = payload[0].get("message") if isinstance(payload, list) and payload and \
                isinstance(payload[0], dict) else payload
            raise RuntimeError(f"World Bank API returned no data for {path}: {str(msg)[:200]}")
        meta, rows = payload[0], payload[1]
        out.extend(rows)
        if page >= int(meta.get("pages", 1)):
            break
        page += 1
    return out


def _country_meta(http: HttpClient) -> pd.DataFrame:
    rows = _paged(http, "country", {})
    return pd.DataFrame([{"iso3": r["id"], "country_wb": r["name"],
                          "is_aggregate": (r.get("region") or {}).get("id") == "NA",
                          "region": (r.get("region") or {}).get("value"),
                          "income_level": (r.get("incomeLevel") or {}).get("value")} for r in rows])


def _indicator(http: HttpClient, code: str, source: str | None) -> pd.DataFrame:
    params = {"date": f"{SETTINGS.worldbank.first_year}:{SETTINGS.as_of.year}"}
    if source:
        params["source"] = source
    rows = _paged(http, f"country/all/indicator/{code}", params)
    df = pd.DataFrame([{"iso3": r.get("countryiso3code"), "year": r.get("date"), "value": r.get("value")}
                       for r in rows])
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.dropna(subset=["iso3", "value"]).query("iso3 != ''")


def _latest(df: pd.DataFrame, name: str, lo: float, hi: float) -> pd.DataFrame:
    n = len(df)
    dup = df.duplicated(["iso3", "year"], keep="last")
    df = df[~dup]
    oor = (df["value"] < lo) | (df["value"] > hi)
    quarantine(SRC, df[oor].assign(indicator=name), f"{name}_out_of_range")
    QLOG.check(SRC, f"{name}_out_of_range", int(oor.sum()), n, action="quarantined",
               severity="warning" if oor.any() else "info")
    df = df[~oor].sort_values(["iso3", "year"])
    last = df.groupby("iso3").tail(1).rename(columns={"value": name, "year": f"{name}_year"})
    return last


def build(http: HttpClient) -> pd.DataFrame:
    cfg = SETTINGS.worldbank
    meta = _country_meta(http)
    lpi = _latest(_indicator(http, cfg.lpi_indicator, None), "lpi_score", 1.0, 5.0)
    wgi, used = None, None
    for code, source in cfg.wgi_candidates:
        try:
            raw = _indicator(http, code, source)
            if len(raw):
                wgi, used = _latest(raw, "wgi_political_stability", -2.6, 2.6), f"{code} (source={source})"
                break
        except Exception as e:
            log.warning("WGI candidate %s/source=%s failed: %s", code, source, e)
    if wgi is None:
        QLOG.check(SRC, "wgi_unavailable", 1, severity="error", action="column left empty")
        wgi = pd.DataFrame(columns=["iso3", "wgi_political_stability", "wgi_political_stability_year"])
    df = meta.merge(lpi, on="iso3", how="left").merge(wgi, on="iso3", how="left")
    agg = df["is_aggregate"]
    QLOG.check(SRC, "aggregates_removed", int(agg.sum()), len(df), action="removed (regions/income groups)")
    df = df[~agg].drop(columns="is_aggregate")
    for col in ["lpi_score", "wgi_political_stability"]:
        QLOG.check(SRC, f"countries_missing_{col}", int(df[col].isna().sum()), len(df),
                   action="left NaN - Layer 2 should fall back to regional median and flag")
    stale = (SETTINGS.as_of.year - df["lpi_score_year"].astype("float")) > 6
    QLOG.check(SRC, "lpi_older_than_6y", int(stale.sum()), int(df["lpi_score"].notna().sum()),
               action="flagged", detail=f"WGI series used: {used}")
    df["lpi_stale"] = stale.fillna(False)
    schema = [Col("iso3", "str", nullable=False, unique=True),
              Col("lpi_score", "float", min=1, max=5),
              Col("wgi_political_stability", "float", min=-2.6, max=2.6)]
    return enforce_schema(df, schema, SRC)


def run(http: HttpClient, **_) -> dict:
    try:
        df = build(http)
        QLOG.status(SRC, "OK", rows=len(df), note=f"{int(df.lpi_score.notna().sum())} countries with LPI")
        return {"wb_country_indicators": write_table(df, "wb_country_indicators")}
    except Exception as e:
        log.exception("World Bank ingestion failed")
        QLOG.status(SRC, "FAILED", note=str(e)[:200])
        return {}
