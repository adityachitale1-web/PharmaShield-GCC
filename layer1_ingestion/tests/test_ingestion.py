"""
Offline test-suite for Layer 1. No network: a fake HTTP router serves payloads with the
same structure as the real openFDA / World Bank / Data Dashboard APIs, including the edge
cases the pipeline must survive.  Run:  python -m pytest -q
"""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import pandas as pd
import pytest
import requests

import ingestion.config as cfgmod
from ingestion import common, openfda, scms, worldbank
from ingestion.common import HttpClient, company_core, company_key, parent_company

ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------- #
# Fake HTTP
# --------------------------------------------------------------------------- #
class Resp:
    def __init__(self, status, payload=None, text=None):
        self.status_code, self._p, self.headers = status, payload, {}
        self.text = text if text is not None else json.dumps(payload)
        self.url = "fake"

    def json(self):
        if self._p is None:
            raise ValueError("no json")
        return self._p


def ndc_rec(labeler, ndc, strength="10 mg/mL", form="INJECTION, SOLUTION", ptype="HUMAN PRESCRIPTION DRUG",
            cat="ANDA", exp="20271231", set_id=None, ingredients=None, finished=True, of=True):
    return {"product_ndc": ndc, "product_id": f"{ndc}_{set_id or ndc}", "spl_id": f"spl-{ndc}",
            "labeler_name": labeler, "generic_name": "CARBOPLATIN", "brand_name": "Carboplatin",
            "dosage_form": form, "product_type": ptype, "marketing_category": cat, "finished": finished,
            "application_number": f"{cat}0001", "listing_expiration_date": exp, "marketing_start_date": "20200101",
            "active_ingredients": ingredients or [{"name": "CARBOPLATIN", "strength": strength}],
            "packaging": [{"package_ndc": f"{ndc}-01", "description": "1 VIAL in 1 CARTON"}],
            "openfda": ({"spl_set_id": [set_id or f"set-{labeler[:4]}"], "is_original_packager": [True],
                         "manufacturer_name": [labeler]} if of else {})}


NDC = [
    ndc_rec("Apotex Corp.", "60505-6282", set_id="set-apotex"),
    ndc_rec("Hospira, Inc.", "61703-360", set_id="set-hosp"),
    ndc_rec("Hospira, Inc.", "61703-360", set_id="set-hosp-old", exp="20251231"),   # duplicate NDC
    ndc_rec("Eugia US LLC", "55150-335", strength="450 mg/45mL", set_id="set-eugia"),
    ndc_rec("AVYXA Pharma, LLC", "83831-141", strength="80 mg/8mL", form="INJECTION", cat="NDA", of=False),
    ndc_rec("Gland Pharma Limited", "68083-190", set_id="set-gland"),
    ndc_rec("Bulk API Co", "11111-111", form="POWDER", ptype="BULK INGREDIENT", finished=False),
    ndc_rec("Expired Labs Inc", "22222-222", exp="20200101"),
    ndc_rec("Dilute Co", "33333-333", strength="5 mg/mL"),
    ndc_rec("Combo Pharma", "44444-444", ingredients=[{"name": "CARBOPLATIN", "strength": "10 mg/mL"},
                                                      {"name": "PACLITAXEL", "strength": "6 mg/mL"}]),
]
LABELS = [
    {"set_id": "set-apotex", "id": "l1", "effective_time": "20240110",
     "openfda": {"manufacturer_name": ["Apotex Corp."], "product_ndc": ["60505-6282"]},
     "spl_unclassified_section": ["Manufactured by: Teyro Labs Private Limited Plot No. A-9/1, SIPCOT "
                                  "Industrial Growth Centre Oragadam, Kanchipuram District, Tamil Nadu 601301, "
                                  "India (IND) Manufactured for: Apotex Corp. Weston Florida 33326"]},
    {"set_id": "set-gland", "id": "l2", "effective_time": "20250301",
     "openfda": {"manufacturer_name": ["Gland Pharma Limited"]},
     "package_label_principal_display_panel": ["Mfd. by: Gland Pharma Limited Hyderabad - 500043, India"]},
    {"set_id": "set-gland", "id": "l2-old", "effective_time": "20200301",   # older version -> dropped
     "openfda": {"manufacturer_name": ["Gland Pharma Limited"]},
     "spl_unclassified_section": ["Made in Germany"]},
]
SHORT = [
    {"generic_name": "Carboplatin Injection", "company_name": "Hospira, Inc., a Pfizer Company",
     "status": "Current", "availability": "Unavailable", "presentation": "Carboplatin, Injection, 150 mg/15 mL (10 mg/mL) (NDC 61703-150-05)",
     "package_ndc": "61703-150-05", "update_date": "08/07/2026", "initial_posting_date": "04/28/2023",
     "therapeutic_category": ["Oncology"], "shortage_reason": "Other"},
    {"generic_name": "Carboplatin Injection", "company_name": "Hospira, Inc., a Pfizer Company",
     "status": "Current", "availability": "Limited Availability", "presentation": "Carboplatin, Injection, 150 mg/15 mL (10 mg/mL) (NDC 61703-150-05)",
     "package_ndc": "61703-150-05", "update_date": "01/02/2026", "initial_posting_date": "04/28/2023",
     "therapeutic_category": ["Oncology"]},                                      # older status row
    {"generic_name": "Carboplatin Injection", "company_name": "Gland Pharma Limited", "status": "Resolved",
     "availability": "Available", "presentation": "Carboplatin, Injection, 450mg/45mL (NDC 54288-166-01)",
     "package_ndc": "54288-166-01", "update_date": "07/22/2026", "therapeutic_category": ["Oncology"]},
]
RECALLS = [
    {"recall_number": "D-0001-2024", "recalling_firm": "Hospira, Inc.", "classification": "Class II",
     "report_date": "20240105", "recall_initiation_date": "20231201", "status": "Ongoing",
     "product_description": "Carboplatin Injection 10 mg/mL vial", "reason_for_recall": "Presence of Particulate Matter"},
    {"recall_number": "D-0002-2025", "recalling_firm": "Gland Pharma Limited", "classification": "Class I",
     "report_date": "20250610", "status": "Terminated", "product_description": "Some tablet",
     "reason_for_recall": "CGMP deviations"},
]


def wb_page(rows, total=None):
    return [{"page": 1, "pages": 1, "per_page": 20000, "total": total or len(rows)}, rows]


WB_COUNTRIES = wb_page([
    {"id": "IND", "name": "India", "region": {"id": "SAS", "value": "South Asia"}, "incomeLevel": {"value": "LMC"}},
    {"id": "USA", "name": "United States", "region": {"id": "NAC", "value": "North America"}, "incomeLevel": {"value": "HIC"}},
    {"id": "WLD", "name": "World", "region": {"id": "NA", "value": "Aggregates"}, "incomeLevel": {"value": "Aggregates"}},
])
LPI = wb_page([{"countryiso3code": "IND", "date": "2022", "value": 3.4},
               {"countryiso3code": "IND", "date": "2018", "value": 3.18},
               {"countryiso3code": "IND", "date": "2021", "value": None},
               {"countryiso3code": "USA", "date": "2022", "value": 3.8},
               {"countryiso3code": "USA", "date": "2016", "value": 9.9},      # out of range -> quarantined
               {"countryiso3code": "WLD", "date": "2022", "value": 3.0}])
PV = wb_page([{"countryiso3code": "IND", "date": "2023", "value": -0.6},
              {"countryiso3code": "USA", "date": "2023", "value": 0.0}])
WB_ERR = [{"message": [{"id": "120", "key": "Invalid value", "value": "The provided parameter value is not valid"}]}]


def router(method, url, params=None, json=None, headers=None, timeout=None):
    p = params or {}
    s = p.get("search", "")
    if "api.fda.gov" in url:
        if p.get("skip", 0) > 0:
            return Resp(404, {"error": {"code": "NOT_FOUND"}})
        data = {"ndc": NDC if ("carboplatin" in s) else [], "label": LABELS, "shortages":
                SHORT if "carboplatin" in s else SHORT[:1], "enforcement": RECALLS}
        ep = re.search(r"drug/(\w+)\.json", url).group(1)
        res = data[ep]
        if not res:
            return Resp(404, {"error": {"code": "NOT_FOUND", "message": "No matches found!"}})
        return Resp(200, {"meta": {"last_updated": "2026-11-14", "results": {"total": len(res)}}, "results": res})
    if "api.worldbank.org" in url:
        if url.endswith("/country"):
            return Resp(200, WB_COUNTRIES)
        if "LP.LPI" in url:
            return Resp(200, LPI)
        if "PV.EST" in url and p.get("source") == "3" and "GOV_WGI" not in url:
            return Resp(200, WB_ERR)                                         # first candidate fails
        return Resp(200, PV)
    return Resp(404, None, text="not found")


@pytest.fixture()
def http(tmp_path, monkeypatch):
    monkeypatch.setattr(requests.Session, "request", lambda self, *a, **k: router(*a, **k))
    monkeypatch.setattr(common.time, "sleep", lambda s: None)
    return HttpClient(tmp_path / "raw")


@pytest.fixture(autouse=True)
def fresh_log():
    common.QLOG.records.clear()
    common.QLOG.source_status.clear()
    common._QUARANTINE.clear()
    yield


# --------------------------------------------------------------------------- #
# Unit tests
# --------------------------------------------------------------------------- #
def test_company_keys():
    assert company_key("Gland Pharma Limited") == "GLAND PHARMA"
    assert company_key("Hospira, Inc., a Pfizer Company") == "HOSPIRA"
    assert parent_company("Hospira, Inc., a Pfizer Company") == "PFIZER"
    assert company_core("Teva Parenteral Medicines, Inc.") == "TEVA MEDICINES" or \
        company_core("Teva Parenteral Medicines, Inc.").startswith("TEVA")
    assert company_key("Fresenius Kabi USA, LLC") == company_key("FRESENIUS KABI USA LLC")


@pytest.mark.parametrize("s,exp", [("10 mg/mL", 10), ("450 mg/45mL", 10), ("80 mg/8mL", 10),
                                   ("1 g/100 mL", 10), ("5 mg/mL", 5), ("tablet", None)])
def test_strength_parser(s, exp):
    assert openfda.strength_mg_per_ml(s) == exp


def test_ndc_gate(http):
    prods = openfda.ndc_products(http)
    labs = openfda.ndc_labelers(prods)
    q = set(labs["labeler_name_fda"])
    assert q == {"Apotex Corp.", "Hospira, Inc.", "Eugia US LLC", "AVYXA Pharma, LLC", "Gland Pharma Limited"}
    assert prods["product_ndc"].is_unique                           # duplicate NDC collapsed
    reasons = prods.set_index("labeler_name_fda")["exclusion_reasons"]
    assert "not_human_rx" in reasons["Bulk API Co"]
    assert "listing_expired" in reasons["Expired Labs Inc"]
    assert "concentration_mismatch" in reasons["Dilute Co"]
    assert "combination_or_wrong_molecule" in reasons["Combo Pharma"]


def test_label_country(http):
    lab = openfda.label_manufacturing_sites(http, ROOT / "reference")
    assert lab["spl_set_id"].is_unique                              # older version dropped
    by = lab.set_index("spl_set_id")["label_country"]
    assert by["set-apotex"] == "India"                               # 'made by' beats 'made for'
    assert by["set-gland"] == "India"


def test_shortages_dedupe(http):
    sku, onc = openfda.shortages(http)
    hosp = sku[sku["company_key"] == "HOSPIRA"]
    assert len(hosp) == 1 and hosp["availability"].iat[0] == "Unavailable"  # latest row kept
    assert hosp["parent_key"].iat[0] == "PFIZER"
    assert not sku.loc[sku.company_key == "GLAND PHARMA", "is_active"].iat[0]


def test_recalls_cross_window_duplicates(http):
    rc = openfda.recalls(http)          # same payload served for each yearly window
    assert rc["recall_number"].is_unique and len(rc) == 2
    assert set(rc["recall_class"]) == {"I", "II"}


def test_worldbank(http):
    wb = worldbank.build(http)
    assert "WLD" not in set(wb["iso3"])                              # aggregate removed
    ind = wb.set_index("iso3").loc["IND"]
    assert ind["lpi_score"] == 3.4 and int(ind["lpi_score_year"]) == 2022
    assert wb.set_index("iso3").loc["USA", "lpi_score"] == 3.8       # 9.9 quarantined
    assert ind["wgi_political_stability"] == -0.6                    # fallback candidate worked


def test_offline_replay(http, tmp_path, monkeypatch):
    openfda.ndc_products(http)
    off = HttpClient(tmp_path / "raw", offline=True)
    monkeypatch.setattr(requests.Session, "request", lambda *a, **k: (_ for _ in ()).throw(AssertionError("network")))
    assert len(openfda.ndc_products(off)) > 0


def test_retry_then_cache_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(common.time, "sleep", lambda s: None)
    calls = {"n": 0}

    def flaky(self, *a, **k):
        calls["n"] += 1
        return Resp(503, None, text="busy") if calls["n"] < 3 else router(*a, **k)
    monkeypatch.setattr(requests.Session, "request", flaky)
    h = HttpClient(tmp_path / "raw")
    res, _ = openfda.fetch_all(h, "drug/enforcement", "x")
    assert len(res) == 2 and calls["n"] >= 3


# --------------------------------------------------------------------------- #
# SCMS with deliberately messy synthetic data
# --------------------------------------------------------------------------- #
def _scms_row(i, **kw):
    base = {"ID": str(i), "Project Code": "P1", "PQ #": "PQ1", "PO / SO #": f"PO{i}", "ASN/DN #": f"DN-{i}",
            "Country": "Nigeria", "Managed By": "PMO", "Fulfill Via": "Direct Drop", "Vendor INCO Term": "EXW",
            "Shipment Mode": "Air", "PQ First Sent to Client Date": "1/10/12", "PO Sent to Vendor Date": "2/10/12",
            "Scheduled Delivery Date": "2-Jun-12", "Delivered to Client Date": "5-Jun-12",
            "Delivery Recorded Date": "6-Jun-12", "Product Group": "ARV", "Sub Classification": "Adult",
            "Vendor": "Aurobindo Pharma Limited", "Item Description": "Drug X", "Molecule/Test Type": "X",
            "Brand": "Generic", "Dosage": "300mg", "Dosage Form": "Tablet", "Unit of Measure (Per Pack)": "30",
            "Line Item Quantity": "100", "Line Item Value": "300", "Pack Price": "3", "Unit Price": "0.1",
            "Manufacturing Site": "Aurobindo Unit III, India", "First Line Designation": "Yes",
            "Weight (Kilograms)": "50", "Freight Cost (USD)": "100", "Line Item Insurance (USD)": "0.5"}
    base.update(kw)
    return base


def test_scms_messy(tmp_path):
    rows = [_scms_row(i) for i in range(1, 30)]
    rows[0].update({"Freight Cost (USD)": "1000", "Weight (Kilograms)": "200", "Line Item Value": "300"})
    rows[1].update({"Freight Cost (USD)": "See DN-1 (ID#:1)", "Weight (Kilograms)": "See DN-1 (ID#:1)",
                    "Line Item Value": "100"})
    rows[2].update({"Freight Cost (USD)": "Freight Included in Commodity Cost",
                    "Weight (Kilograms)": "Weight Captured Separately"})
    rows[3].update({"Freight Cost (USD)": "See ASN-9 (ID#:99999)"})                 # unresolvable ref
    rows[4].update({"PQ First Sent to Client Date": "Pre-PQ Process", "PO Sent to Vendor Date": "N/A - From RDC"})
    rows[5].update({"Line Item Quantity": "-5"})                                     # impossible
    rows[6].update({"Delivered to Client Date": "1-Jun-14"})                         # +730 days -> implausible
    rows[7].update({"Scheduled Delivery Date": "Date Not Captured"})                 # no label
    rows[8].update({"Country": "CÃ´te d'Ivoire", "Shipment Mode": ""})               # mojibake + missing mode
    rows[9].update({"Line Item Insurance (USD)": ""})                                # imputed
    rows[10].update({"Unit Price": "50", "Pack Price": "1500", "Line Item Value": "150000"})  # outlier
    rows[11].update({"Scheduled Delivery Date": "2-Jun-14", "Delivered to Client Date": "1-Jun-14"})  # test split
    dup = dict(rows[12]); dup["ID"] = "500"; rows.append(dup)                         # exact duplicate
    df = pd.DataFrame(rows).rename(columns={"ID": "﻿ID", "Weight (Kilograms)": "Weight (Kilogram)"})
    p = tmp_path / "scms" / "scms.csv"
    p.parent.mkdir()
    df.to_csv(p, index=False, encoding="utf-8")

    out = scms.clean_scms(p, ROOT / "reference")
    ids = set(out["id"])
    assert "500" not in ids                       # duplicate content quarantined
    assert not {"6", "7", "8"} & ids              # impossible qty / implausible delay / no label
    r = out.set_index("id")
    assert r.loc["1", "freight_usd"] == pytest.approx(750) and r.loc["2", "freight_usd"] == pytest.approx(250)
    assert r.loc["2", "weight_kg"] == pytest.approx(50)
    assert pd.isna(r.loc["3", "freight_usd"]) and r.loc["3", "freight_status"] == "included_in_commodity"
    assert r.loc["4", "freight_status"] == "reference_unresolved"
    assert r.loc["5", "pq_first_sent_to_client_date_status"] == "pre_pq"
    assert r.loc["9", "country"] == "Côte d'Ivoire" and r.loc["9", "shipment_mode"] == "Unknown"
    assert bool(r.loc["10", "insurance_imputed"]) and r.loc["10", "insurance_usd"] > 0
    assert bool(r.loc["11", "outlier_unit_price"])
    assert r.loc["12", "time_split"] == "test" and r.loc["1", "time_split"] == "train"
    assert set(out["late"]) <= {0, 1} and out["manufacturing_country"].eq("India").all()
    q = pd.concat(common._QUARANTINE["scms"])
    assert set(q["_reject_reason"]) >= {"duplicate_content_different_id", "missing_target_dates",
                                        "non_positive_quantity_or_negative_value"}


def test_scms_missing_columns(tmp_path):
    p = tmp_path / "bad.csv"
    pd.DataFrame([{"ID": 1, "Vendor": "x"}]).to_csv(p, index=False)
    with pytest.raises(ValueError, match="missing critical columns"):
        scms.clean_scms(p, ROOT / "reference")


REAL = ROOT / "data" / "manual" / "scms" / "SCMS_Delivery_History_Dataset.csv"


@pytest.mark.skipif(not REAL.exists(), reason="real SCMS file not present")
def test_scms_real_file():
    out = scms.clean_scms(REAL, ROOT / "reference")
    assert len(out) > 10_000 and out["id"].is_unique
    assert 0.10 < out["late"].mean() < 0.13
    feats, _ = scms.site_features(out)
    assert feats["first_shipment"].max() <= pd.Timestamp("2013-12-31")   # no test-period leakage
