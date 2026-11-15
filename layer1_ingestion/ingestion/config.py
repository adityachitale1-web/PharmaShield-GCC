"""
PharmaShield GCC - Layer 1 configuration.

Every tunable used by the ingestion layer lives here so the team changes ONE
file, not seven scripts. Values marked SCENARIO are explicitly-labelled
assumptions (Section 3, item 3 of the Master Implementation Plan).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Paths:
    root: Path = ROOT
    data: Path = ROOT / "data"
    raw: Path = ROOT / "data" / "raw"            # immutable API snapshots (JSON)
    manual: Path = ROOT / "data" / "manual"      # files the team drops in by hand
    interim: Path = ROOT / "data" / "interim"    # clean, validated Layer-1 outputs
    quarantine: Path = ROOT / "data" / "quarantine"  # rejected rows + reason codes
    quality: Path = ROOT / "data" / "quality"    # data-quality reports
    processed: Path = ROOT / "data" / "processed"  # locked-schema hand-off (plan 5.1)
    reference: Path = ROOT / "reference"         # curated lookup / scenario tables


@dataclass(frozen=True)
class TargetSKU:
    sku_id: str = "CARBOPLATIN_INJ_10MG_ML"
    molecule: str = "CARBOPLATIN"
    concentration_mg_per_ml: float = 10.0
    criticality_tier: int = 1                    # Tier 1: Life-Critical (oncology)
    reference_vial_mg: int = 450                 # 450 mg / 45 mL = costing unit "vial"
    therapeutic_category: str = "Oncology"
    # Plan 3 (item 2): the scoring panel is every labeler in the openFDA NDC directory for the
    # molecule ("18 authorized labelers ... including Cipla, Hetero"). That count includes
    # bulk-API (powder) listings. "all_ndc_labelers" follows the plan; each row is still tagged
    # finished_injection vs bulk_api_only. Use "finished_injection_only" for the strict gate.
    panel_scope: str = "all_ndc_labelers"
    fuzzy_match_threshold: int = 88               # plan 5.2 step 3: token_sort_ratio >= 88


@dataclass(frozen=True)
class OpenFDA:
    base_url: str = "https://api.fda.gov"
    api_key: str | None = os.getenv("OPENFDA_API_KEY")   # optional; raises daily quota
    page_limit: int = 1000                       # openFDA hard max per request
    max_skip: int = 25000                        # openFDA hard max for skip
    recall_lookback_years: int = 5


@dataclass(frozen=True)
class FDADataDashboard:
    endpoint: str = "https://api-datadashboard.fda.gov/v1/inspections_classifications"
    user: str | None = os.getenv("FDA_DD_USER")  # Authorization-User (email)
    key: str | None = os.getenv("FDA_DD_KEY")    # Authorization-Key
    rows_per_page: int = 5000
    lookback_years: int = 5


@dataclass(frozen=True)
class WorldBank:
    base_url: str = "https://api.worldbank.org/v2"
    lpi_indicator: str = "LP.LPI.OVRL.XQ"        # Logistics Performance Index, 1-5
    # WGI Political Stability; code differs between WB source catalogues, try in order.
    wgi_candidates: tuple = (("PV.EST", "3"), ("GOV_WGI_PV.EST", "3"), ("PV.EST", None))
    first_year: int = 2007


@dataclass(frozen=True)
class SCMSRules:
    train_end_year: int = 2013                   # Plan 5.2 step 4: train <= 2013
    test_years: tuple = (2014, 2015)
    max_abs_delay_days: int = 365                # beyond this the record is implausible
    robust_z_threshold: float = 3.5              # MAD-based outlier FLAG (never drop)
    value_reconcile_tolerance: float = 0.05      # qty x pack price vs line value
    probation_min_shipments: int = 5             # Plan 3.1 probation badge rule


@dataclass(frozen=True)
class HTTP:
    timeout_s: int = 45
    max_retries: int = 5
    backoff_base_s: float = 1.5
    min_interval_s: float = 0.30                 # polite pacing (openFDA: 240 req/min)
    user_agent: str = "PharmaShield-GCC-Academic-Prototype/1.0 (SP Jain MAIB, Group 3)"


@dataclass(frozen=True)
class Settings:
    paths: Paths = field(default_factory=Paths)
    sku: TargetSKU = field(default_factory=TargetSKU)
    openfda: OpenFDA = field(default_factory=OpenFDA)
    dashboard: FDADataDashboard = field(default_factory=FDADataDashboard)
    worldbank: WorldBank = field(default_factory=WorldBank)
    scms: SCMSRules = field(default_factory=SCMSRules)
    http: HTTP = field(default_factory=HTTP)
    as_of: date = field(default_factory=date.today)


SETTINGS = Settings()
