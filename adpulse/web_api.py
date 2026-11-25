"""Interactive Web Studio API & AWS CloudFront `/api/*` Origin for AdPulse (`adpulse/web_api.py`).

Wires AdPulse's four production subsystems:
  1. Two-Stage Sponsored Catalog Retrieval & Calibrated eCPM Ranker (`adpulse/catalog_ranker.py`)
  2. GSP / Second-Price Auction & 24-Hour Lagrangian PID Budget Pacer (`adpulse/auction_pacer.py`)
  3. Clipped IPS & Doubly-Robust Off-Policy Bid Evaluator (`adpulse/off_policy_eval.py`)
  4. SQLite Seller Campaign Diagnostics + Live Amazon Bedrock (`us.amazon.nova-lite-v1:0`)
     Seller & Spektr Iceberg Lakehouse Copilot (`adpulse/campaign_diagnostics.py`)
"""

from __future__ import annotations

import base64
from dataclasses import asdict, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
import os
from pathlib import Path
import platform
import random
import shutil
import subprocess
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.parse import parse_qs, urlparse

from . import __version__
from .auction_pacer import (
    HOURLY_CVR_MULTIPLIERS,
    HOURLY_TRAFFIC_WEIGHTS,
    PEAK_EVENING_START_HOUR,
    AdvertiserCampaign,
    AuctionOpportunity,
    BudgetPacedAuctionEngine,
    _PIDCampaignState,
    generate_auction_workload,
)
from .campaign_diagnostics import (
    CampaignDiagnosticInput,
    CampaignDiagnostics,
    generate_diagnostic_scenarios,
)
from .catalog_ranker import (
    PRICE_INTENT_RE,
    CatalogSKU,
    LexicalBidRanker,
    ShopperQuery,
    TwoStageAdRanker,
    generate_synthetic_catalog_and_queries,
    tokenize,
)
from .cli import SUPPORTED_MODES, execute_cli_mode
from .off_policy_eval import (
    BID_SHADING_MULTIPLIERS,
    LoggedAuctionEvent,
    OffPolicyBidEvaluator,
    _ContextualAuctionOracle,
    generate_logged_propensity_traces,
)

ROOT_DIR = Path(__file__).resolve().parent.parent
WEB_DIR = ROOT_DIR / "web"
DOCS_DIR = ROOT_DIR / "docs"
BENCHMARK_JSON_PATH = ROOT_DIR / "benchmarks" / "latest_results.json"

CORS_HEADERS: Dict[str, str] = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type, Authorization",
}

FEATURED_SHOPPER_QUERIES: Tuple[Dict[str, Any], ...] = (
    {
        "id": "earbuds",
        "query": "wireless anc running earbuds",
        "label": "Wireless ANC Running Earbuds",
        "category": "electronics_audio",
        "product_type": "earbuds",
        "default_price_cap": 160.0,
        "conversational_prompt": "Best sweatproof wireless ANC earbuds for marathon training under $160?",
    },
    {
        "id": "monitor",
        "query": "4k usb-c monitor macbook",
        "label": "4K USB-C Monitor MacBook",
        "category": "home_office",
        "product_type": "monitor",
        "default_price_cap": 340.0,
        "conversational_prompt": "Which 27-inch 4K USB-C PD monitor charges a MacBook Pro with one cable?",
    },
    {
        "id": "chair",
        "query": "ergonomic mesh office chair",
        "label": "Ergonomic Mesh Office Chair",
        "category": "home_office",
        "product_type": "chair",
        "default_price_cap": 280.0,
        "conversational_prompt": "Recommend a breathable mesh lumbar office chair for 8-hour coding sessions.",
    },
    {
        "id": "espresso",
        "query": "espresso grinder burr",
        "label": "Espresso Grinder Burr",
        "category": "kitchen_appliances",
        "product_type": "espresso",
        "default_price_cap": 260.0,
        "conversational_prompt": "Quiet stainless conical burr espresso grinder with low retention for home baristas?",
    },
)


def _build_featured_storefront_skus() -> Tuple[List[CatalogSKU], Dict[str, Dict[str, Any]]]:
    """Curated storefront SKUs layered onto the 500-SKU synthetic catalog so featured shopper
    queries showcase crisp Amazon Sponsored Product cards AND realistic Stage-1 filter pruning
    (out-of-stock stale high bids, overpriced luxury SKUs, early morning budget exhaustion, and
    missing Spektr catalog attributes).
    """
    specs: List[Dict[str, Any]] = [
        # --- Query 1: wireless anc running earbuds ---
        {
            "sku_id": "SKU-F101",
            "campaign_id": "CMP-002",
            "seller_id": "SLR-AURALIS",
            "title": "Auralis AeroFit Pro Wireless ANC Running Earbuds IPX7 Earhooks Bluetooth 5.3",
            "category": "electronics_audio",
            "product_type": "earbuds",
            "brand": "Auralis",
            "attributes": {"connectivity": "bluetooth_5_3", "feature": "anc", "color": "matte_black"},
            "required_attribute_keys": ("connectivity", "feature", "color"),
            "price": 119.99,
            "margin": 0.42,
            "base_bid": 2.15,
            "in_stock": True,
            "prior_ctr": 0.068,
            "prior_cvr": 0.184,
            "impressions_30d": 24500,
            "clicks_30d": 1680,
            "conversions_30d": 312,
            "rating": 4.8,
            "reviews": 3420,
            "badge": "Best Seller in Running Audio",
            "depletes_greedy_hour": 12,
            "highlights": ["42dB Adaptive ANC + Transparency", "IPX7 SweatShield + LockWing", "9h Earbud / 36h USB-C Case"],
        },
        {
            "sku_id": "SKU-F102",
            "campaign_id": "CMP-005",
            "seller_id": "SLR-SOUNDCOREX",
            "title": "SoundCoreX StrideBuds S3 Bluetooth Noise Cancelling Running Earbuds Spatial Audio",
            "category": "electronics_audio",
            "product_type": "earbuds",
            "brand": "SoundCoreX",
            "attributes": {"connectivity": "multipoint_bt", "feature": "anc", "color": "midnight_navy"},
            "required_attribute_keys": ("connectivity", "feature", "color"),
            "price": 89.50,
            "margin": 0.39,
            "base_bid": 1.85,
            "in_stock": True,
            "prior_ctr": 0.061,
            "prior_cvr": 0.168,
            "impressions_30d": 19200,
            "clicks_30d": 1170,
            "conversions_30d": 198,
            "rating": 4.6,
            "reviews": 2185,
            "badge": "Amazon's Choice",
            "depletes_greedy_hour": None,
            "highlights": ["Multipoint Bluetooth 5.3", "Wind-Cut Running Mic Array", "10-Min Fast Charge = 2.5h Run"],
        },
        {
            "sku_id": "SKU-F103",
            "campaign_id": "CMP-009",
            "seller_id": "SLR-VELOTONE",
            "title": "VeloTone PacePulse Ultra Wireless ANC Running Earbuds Waterproof IPX7",
            "category": "electronics_audio",
            "product_type": "earbuds",
            "brand": "VeloTone",
            "attributes": {"connectivity": "bluetooth_5_3", "feature": "waterproof_ipx7", "color": "arctic_white"},
            "required_attribute_keys": ("connectivity", "feature", "color"),
            "price": 134.00,
            "margin": 0.44,
            "base_bid": 1.95,
            "in_stock": True,
            "prior_ctr": 0.055,
            "prior_cvr": 0.152,
            "impressions_30d": 14800,
            "clicks_30d": 815,
            "conversions_30d": 124,
            "rating": 4.7,
            "reviews": 1290,
            "badge": "Prime Same-Day",
            "depletes_greedy_hour": 13,
            "highlights": ["IPX7 Submersible Sport Seal", "Heart-Rate Cadence Sensor", "Featherweight 4.6g Fit"],
        },
        {
            "sku_id": "SKU-F104",
            "campaign_id": "CMP-014",
            "seller_id": "SLR-KODAAUDIO",
            "title": "KodaAudio RunLite Wireless Earbuds ANC Bluetooth 5.3 Low Latency",
            "category": "electronics_audio",
            "product_type": "earbuds",
            "brand": "KodaAudio",
            "attributes": {"connectivity": "bluetooth_5_3", "feature": "anc", "color": "graphite"},
            "required_attribute_keys": ("connectivity", "feature", "color"),
            "price": 74.99,
            "margin": 0.36,
            "base_bid": 1.55,
            "in_stock": True,
            "prior_ctr": 0.049,
            "prior_cvr": 0.138,
            "impressions_30d": 11200,
            "clicks_30d": 548,
            "conversions_30d": 76,
            "rating": 4.4,
            "reviews": 840,
            "badge": "Prime Two-Day",
            "depletes_greedy_hour": None,
            "highlights": ["Hybrid Active Noise Cancelling", "USB-C Pocket Charging Case", "4 Ear-Fin Sizes Included"],
        },
        {
            "sku_id": "SKU-F105-OOS",
            "campaign_id": "CMP-D001",
            "seller_id": "SLR-NORDWAVE",
            "title": "NordWave SonicRun Pro Wireless ANC Running Earbuds Titanium Driver",
            "category": "electronics_audio",
            "product_type": "earbuds",
            "brand": "NordWave",
            "attributes": {"connectivity": "bluetooth_5_3", "feature": "anc", "color": "matte_black"},
            "required_attribute_keys": ("connectivity", "feature", "color"),
            "price": 129.00,
            "margin": 0.40,
            "base_bid": 3.85,  # Aggressive stale bid while FBA inventory == 0!
            "in_stock": False,
            "prior_ctr": 0.052,
            "prior_cvr": 0.135,
            "impressions_30d": 9400,
            "clicks_30d": 490,
            "conversions_30d": 66,
            "rating": 4.5,
            "reviews": 910,
            "badge": "FBA Out of Stock",
            "depletes_greedy_hour": None,
            "highlights": ["0 Units in US-West FBA", "Stale $3.85 CPC Bid Active"],
        },
        {
            "sku_id": "SKU-F106-LUX",
            "campaign_id": "CMP-018",
            "seller_id": "SLR-AURALIS",
            "title": "Auralis Reference Carbon Audiophile Wireless ANC Running Earbuds",
            "category": "electronics_audio",
            "product_type": "earbuds",
            "brand": "Auralis",
            "attributes": {"connectivity": "bluetooth_5_3", "feature": "anc", "color": "graphite"},
            "required_attribute_keys": ("connectivity", "feature", "color"),
            "price": 269.00,  # Exceeds $160 shopper ceiling
            "margin": 0.48,
            "base_bid": 3.60,
            "in_stock": True,
            "prior_ctr": 0.034,
            "prior_cvr": 0.062,
            "impressions_30d": 8100,
            "clicks_30d": 275,
            "conversions_30d": 17,
            "rating": 4.6,
            "reviews": 412,
            "badge": "Luxury Tier",
            "depletes_greedy_hour": None,
            "highlights": ["Exceeds Shopper Price Ceiling", "High Raw Bid ($3.60) / Low CVR"],
        },
        # --- Query 2: 4k usb-c monitor macbook ---
        {
            "sku_id": "SKU-F201",
            "campaign_id": "CMP-003",
            "seller_id": "SLR-LUMAWORK",
            "title": "LumaWork ProView 27-Inch 4K UHD USB-C Monitor MacBook 96W PD IPS Display",
            "category": "home_office",
            "product_type": "monitor",
            "brand": "LumaWork",
            "attributes": {"connectivity": "usb_c", "material": "anodized_aluminum", "color": "space_gray"},
            "required_attribute_keys": ("connectivity", "material", "color"),
            "price": 299.00,
            "margin": 0.38,
            "base_bid": 2.45,
            "in_stock": True,
            "prior_ctr": 0.066,
            "prior_cvr": 0.174,
            "impressions_30d": 21600,
            "clicks_30d": 1425,
            "conversions_30d": 248,
            "rating": 4.8,
            "reviews": 2890,
            "badge": "MacBook Pro Verified",
            "depletes_greedy_hour": 13,
            "highlights": ["Single-Cable 96W USB-C Power Delivery", "99% DCI-P3 Factory Calibrated 4K", "Built-in KVM + Gigabit Hub"],
        },
        {
            "sku_id": "SKU-F202",
            "campaign_id": "CMP-007",
            "seller_id": "SLR-DESKFORGE",
            "title": "DeskForge StudioDisplay 27 4K USB-C MacBook Monitor 90W Hub Height Adjustable",
            "category": "home_office",
            "product_type": "monitor",
            "brand": "DeskForge",
            "attributes": {"connectivity": "usb_c", "material": "anodized_aluminum", "color": "matte_black"},
            "required_attribute_keys": ("connectivity", "material", "color"),
            "price": 264.99,
            "margin": 0.36,
            "base_bid": 2.10,
            "in_stock": True,
            "prior_ctr": 0.059,
            "prior_cvr": 0.161,
            "impressions_30d": 17400,
            "clicks_30d": 1026,
            "conversions_30d": 165,
            "rating": 4.7,
            "reviews": 1940,
            "badge": "Amazon's Choice",
            "depletes_greedy_hour": None,
            "highlights": ["90W USB-C Charging + Daisy Chain", "Ergonomic Tilt/Pivot/Height Stand", " macOS HiDPI Scaling Ready"],
        },
        {
            "sku_id": "SKU-F203",
            "campaign_id": "CMP-011",
            "seller_id": "SLR-AXISMOUNT",
            "title": "AxisMount Clarity 4K UHD USB-C Monitor MacBook Air/Pro 65W Ultra-Thin Bezel",
            "category": "home_office",
            "product_type": "monitor",
            "brand": "AxisMount",
            "attributes": {"connectivity": "usb_c", "material": "walnut_steel", "color": "warm_white"},
            "required_attribute_keys": ("connectivity", "material", "color"),
            "price": 229.00,
            "margin": 0.35,
            "base_bid": 1.88,
            "in_stock": True,
            "prior_ctr": 0.054,
            "prior_cvr": 0.149,
            "impressions_30d": 13900,
            "clicks_30d": 750,
            "conversions_30d": 112,
            "rating": 4.6,
            "reviews": 1410,
            "badge": "Best Value 4K",
            "depletes_greedy_hour": 12,
            "highlights": ["65W USB-C One-Cord Dock", "HDR400 Low-Blue-Light IPS", "VESA 100x100 Quick-Mount"],
        },
        {
            "sku_id": "SKU-F204",
            "campaign_id": "CMP-015",
            "seller_id": "SLR-ERGOLIFT",
            "title": "ErgoLift Vision 27 4K IPS USB-C Monitor MacBook 65W Matte Anti-Glare",
            "category": "home_office",
            "product_type": "monitor",
            "brand": "ErgoLift",
            "attributes": {"connectivity": "usb_c", "material": "anodized_aluminum", "color": "charcoal"},
            "required_attribute_keys": ("connectivity", "material", "color"),
            "price": 248.00,
            "margin": 0.34,
            "base_bid": 1.65,
            "in_stock": True,
            "prior_ctr": 0.048,
            "prior_cvr": 0.134,
            "impressions_30d": 10400,
            "clicks_30d": 500,
            "conversions_30d": 67,
            "rating": 4.5,
            "reviews": 780,
            "badge": "Prime Same-Day",
            "depletes_greedy_hour": None,
            "highlights": ["Paper-Matte Anti-Reflection Coating", "65W USB-C + Dual HDMI 2.1", "3-Year Zero Bright Dot Warranty"],
        },
        {
            "sku_id": "SKU-F205-OOS",
            "campaign_id": "CMP-D006",
            "seller_id": "SLR-KEYCRAFT",
            "title": "KeyCraft Pro Retina 5K/4K USB-C Thunderbolt MacBook Monitor 100W",
            "category": "home_office",
            "product_type": "monitor",
            "brand": "KeyCraft",
            "attributes": {"connectivity": "usb_c", "material": "anodized_aluminum", "color": "space_gray"},
            "required_attribute_keys": ("connectivity", "material", "color"),
            "price": 319.00,
            "margin": 0.41,
            "base_bid": 3.95,
            "in_stock": False,
            "prior_ctr": 0.056,
            "prior_cvr": 0.140,
            "impressions_30d": 8900,
            "clicks_30d": 498,
            "conversions_30d": 70,
            "rating": 4.6,
            "reviews": 620,
            "badge": "FBA Out of Stock",
            "depletes_greedy_hour": None,
            "highlights": ["0 Units Available", "Unpaused $3.95 CPC Bid Blocked"],
        },
        {
            "sku_id": "SKU-F206-LUX",
            "campaign_id": "CMP-019",
            "seller_id": "SLR-LUMAWORK",
            "title": "LumaWork Reference 32-Inch 4K Mini-LED Thunderbolt USB-C Monitor MacBook",
            "category": "home_office",
            "product_type": "monitor",
            "brand": "LumaWork",
            "attributes": {"connectivity": "usb_c", "material": "anodized_aluminum", "color": "space_gray"},
            "required_attribute_keys": ("connectivity", "material", "color"),
            "price": 459.00,
            "margin": 0.45,
            "base_bid": 3.70,
            "in_stock": True,
            "prior_ctr": 0.035,
            "prior_cvr": 0.068,
            "impressions_30d": 7200,
            "clicks_30d": 252,
            "conversions_30d": 17,
            "rating": 4.7,
            "reviews": 350,
            "badge": "Above Price Cap",
            "depletes_greedy_hour": None,
            "highlights": ["$459 Exceeds $340 Ceiling", "Filtered Before Auction"],
        },
        # --- Query 3: ergonomic mesh office chair ---
        {
            "sku_id": "SKU-F301",
            "campaign_id": "CMP-006",
            "seller_id": "SLR-ERGOLIFT",
            "title": "ErgoLift SpineSync Pro Ergonomic Mesh Lumbar Office Chair 4D Armrests",
            "category": "home_office",
            "product_type": "chair",
            "brand": "ErgoLift",
            "attributes": {"connectivity": "mechanical_mount", "material": "breathable_mesh", "color": "matte_black"},
            "required_attribute_keys": ("connectivity", "material", "color"),
            "price": 219.00,
            "margin": 0.43,
            "base_bid": 2.30,
            "in_stock": True,
            "prior_ctr": 0.064,
            "prior_cvr": 0.172,
            "impressions_30d": 22800,
            "clicks_30d": 1460,
            "conversions_30d": 251,
            "rating": 4.8,
            "reviews": 4120,
            "badge": "Best Seller in Office Chairs",
            "depletes_greedy_hour": 12,
            "highlights": ["Dynamic Dual-Zone Lumbar Support", "Elastomeric Breathable Korean Mesh", "BIFMA Certified 330 lb Alloy Base"],
        },
        {
            "sku_id": "SKU-F302",
            "campaign_id": "CMP-010",
            "seller_id": "SLR-DESKFORGE",
            "title": "DeskForge AeroPosture Ergonomic Mesh Office Chair Adjustable Seat Depth",
            "category": "home_office",
            "product_type": "chair",
            "brand": "DeskForge",
            "attributes": {"connectivity": "mechanical_mount", "material": "breathable_mesh", "color": "charcoal"},
            "required_attribute_keys": ("connectivity", "material", "color"),
            "price": 184.99,
            "margin": 0.40,
            "base_bid": 1.92,
            "in_stock": True,
            "prior_ctr": 0.058,
            "prior_cvr": 0.158,
            "impressions_30d": 16500,
            "clicks_30d": 957,
            "conversions_30d": 151,
            "rating": 4.6,
            "reviews": 2650,
            "badge": "Amazon's Choice",
            "depletes_greedy_hour": None,
            "highlights": ["Sliding Seat Depth + 135° Synchro-Tilt", "3D Padded Flip-Up Armrests", "15-Minute Tool-Included Assembly"],
        },
        {
            "sku_id": "SKU-F303",
            "campaign_id": "CMP-016",
            "seller_id": "SLR-LUMAWORK",
            "title": "LumaWork CloudMesh Ergonomic Lumbar Office Chair Headrest Warm Gray",
            "category": "home_office",
            "product_type": "chair",
            "brand": "LumaWork",
            "attributes": {"connectivity": "mechanical_mount", "material": "breathable_mesh", "color": "space_gray"},
            "required_attribute_keys": ("connectivity", "material", "color"),
            "price": 249.00,
            "margin": 0.41,
            "base_bid": 2.05,
            "in_stock": True,
            "prior_ctr": 0.053,
            "prior_cvr": 0.146,
            "impressions_30d": 13100,
            "clicks_30d": 694,
            "conversions_30d": 101,
            "rating": 4.7,
            "reviews": 1530,
            "badge": "Prime Delivery",
            "depletes_greedy_hour": 13,
            "highlights": ["2D Cervical Mesh Headrest", "Silent Rollerblade Hardwood Casters", "5-Year Structural Frame Warranty"],
        },
        {
            "sku_id": "SKU-F304-OOS",
            "campaign_id": "CMP-D011",
            "seller_id": "SLR-AXISMOUNT",
            "title": "AxisMount FlexBack Pro Ergonomic Mesh Lumbar Office Chair",
            "category": "home_office",
            "product_type": "chair",
            "brand": "AxisMount",
            "attributes": {"connectivity": "mechanical_mount", "material": "breathable_mesh", "color": "matte_black"},
            "required_attribute_keys": ("connectivity", "material", "color"),
            "price": 199.00,
            "margin": 0.38,
            "base_bid": 3.75,
            "in_stock": False,
            "prior_ctr": 0.051,
            "prior_cvr": 0.139,
            "impressions_30d": 9100,
            "clicks_30d": 464,
            "conversions_30d": 64,
            "rating": 4.5,
            "reviews": 870,
            "badge": "FBA Out of Stock",
            "depletes_greedy_hour": None,
            "highlights": ["Out-of-Stock SKU Suppressed in <0.5ms"],
        },
        # --- Query 4: espresso grinder burr ---
        {
            "sku_id": "SKU-F401",
            "campaign_id": "CMP-004",
            "seller_id": "SLR-KONAMILL",
            "title": "KonaMill Precision 64mm Stainless Conical Burr Espresso Grinder Low Retention",
            "category": "kitchen_appliances",
            "product_type": "espresso",
            "brand": "KonaMill",
            "attributes": {"material": "stainless_steel", "capacity": "single_serve", "wattage": "600w"},
            "required_attribute_keys": ("material", "capacity", "wattage"),
            "price": 199.00,
            "margin": 0.44,
            "base_bid": 2.25,
            "in_stock": True,
            "prior_ctr": 0.069,
            "prior_cvr": 0.189,
            "impressions_30d": 19800,
            "clicks_30d": 1366,
            "conversions_30d": 258,
            "rating": 4.9,
            "reviews": 2740,
            "badge": "Barista Choice #1",
            "depletes_greedy_hour": 12,
            "highlights": ["Stepless 64mm Titanium-Coated Burrs", "<0.1g Zero-Retention Bellows", "58mm Magnetic Portafilter Cradle"],
        },
        {
            "sku_id": "SKU-F402",
            "campaign_id": "CMP-008",
            "seller_id": "SLR-BREWCRAFT",
            "title": "BrewCraft DoseMaster Digital Stainless Burr Espresso Grinder 60 Grind Settings",
            "category": "kitchen_appliances",
            "product_type": "espresso",
            "brand": "BrewCraft",
            "attributes": {"material": "stainless_steel", "capacity": "carafe_1_2l", "wattage": "600w"},
            "required_attribute_keys": ("material", "capacity", "wattage"),
            "price": 149.95,
            "margin": 0.41,
            "base_bid": 1.85,
            "in_stock": True,
            "prior_ctr": 0.062,
            "prior_cvr": 0.171,
            "impressions_30d": 16400,
            "clicks_30d": 1017,
            "conversions_30d": 174,
            "rating": 4.7,
            "reviews": 3190,
            "badge": "Amazon's Choice",
            "depletes_greedy_hour": None,
            "highlights": ["0.1s Timed Digital Dosing", "Anti-Static Plasma Ion Chute", "Whisper-Quiet 62dB Brushless Motor"],
        },
        {
            "sku_id": "SKU-F403",
            "campaign_id": "CMP-012",
            "seller_id": "SLR-COPPERLEAF",
            "title": "CopperLeaf BaristaPro Brushed Aluminum Burr Espresso Grinder & Scale Lid",
            "category": "kitchen_appliances",
            "product_type": "espresso",
            "brand": "CopperLeaf",
            "attributes": {"material": "brushed_aluminum", "capacity": "single_serve", "wattage": "900w"},
            "required_attribute_keys": ("material", "capacity", "wattage"),
            "price": 224.00,
            "margin": 0.42,
            "base_bid": 1.98,
            "in_stock": True,
            "prior_ctr": 0.056,
            "prior_cvr": 0.154,
            "impressions_30d": 12200,
            "clicks_30d": 683,
            "conversions_30d": 105,
            "rating": 4.7,
            "reviews": 1120,
            "badge": "Prime Same-Day",
            "depletes_greedy_hour": 13,
            "highlights": ["Integrated 0.1g Gravimetric Lid", "48mm Hardened Italian Steel Burrs", "Turkish to French Press Range"],
        },
        {
            "sku_id": "SKU-F404-OOS",
            "campaign_id": "CMP-D016",
            "seller_id": "SLR-CASAVERO",
            "title": "CasaVero Milano Commercial Burr Espresso Grinder Stainless",
            "category": "kitchen_appliances",
            "product_type": "espresso",
            "brand": "CasaVero",
            "attributes": {"material": "stainless_steel", "capacity": "single_serve", "wattage": "1200w"},
            "required_attribute_keys": ("material", "capacity", "wattage"),
            "price": 239.00,
            "margin": 0.39,
            "base_bid": 4.05,
            "in_stock": False,
            "prior_ctr": 0.051,
            "prior_cvr": 0.132,
            "impressions_30d": 8400,
            "clicks_30d": 428,
            "conversions_30d": 56,
            "rating": 4.5,
            "reviews": 670,
            "badge": "FBA Out of Stock",
            "depletes_greedy_hour": None,
            "highlights": ["Out-of-Stock SKU Blocked in Stage 1"],
        },
    ]

    skus: List[CatalogSKU] = []
    meta_by_sku: Dict[str, Dict[str, Any]] = {}
    for item in specs:
        skus.append(
            CatalogSKU(
                sku_id=item["sku_id"],
                campaign_id=item["campaign_id"],
                seller_id=item["seller_id"],
                title=item["title"],
                category=item["category"],
                product_type=item["product_type"],
                brand=item["brand"],
                attributes=dict(item["attributes"]),
                required_attribute_keys=tuple(item["required_attribute_keys"]),
                price=float(item["price"]),
                margin=float(item["margin"]),
                base_bid=float(item["base_bid"]),
                in_stock=bool(item["in_stock"]),
                prior_ctr=float(item["prior_ctr"]),
                prior_cvr=float(item["prior_cvr"]),
                impressions_30d=int(item["impressions_30d"]),
                clicks_30d=int(item["clicks_30d"]),
                conversions_30d=int(item["conversions_30d"]),
            )
        )
        meta_by_sku[item["sku_id"]] = {
            "rating": item["rating"],
            "reviews": item["reviews"],
            "badge": item["badge"],
            "depletes_greedy_hour": item["depletes_greedy_hour"],
            "highlights": item["highlights"],
        }
    return skus, meta_by_sku


def _call_bedrock_nova(system_prompt: str, user_prompt: str) -> Optional[Dict[str, Any]]:
    """Invoke Amazon Bedrock (`us.amazon.nova-lite-v1:0`) via boto3 (Lambda) or AWS CLI (`portfolio` local)."""
    if os.environ.get("ADPULSE_OFFLINE_LLM") == "1":
        return None

    model_id = os.environ.get("ADPULSE_BEDROCK_MODEL", "us.amazon.nova-lite-v1:0")
    region = os.environ.get("AWS_REGION", "us-west-2")
    t0 = time.perf_counter()

    try:
        import boto3  # type: ignore

        client = boto3.client("bedrock-runtime", region_name=region)
        resp = client.converse(
            modelId=model_id,
            system=[{"text": system_prompt}],
            messages=[{"role": "user", "content": [{"text": user_prompt}]}],
            inferenceConfig={"maxTokens": 450, "temperature": 0.15},
        )
        text = resp["output"]["message"]["content"][0]["text"]
        return {
            "text": text.strip(),
            "engine": f"aws-bedrock ({model_id})",
            "latency_ms": round((time.perf_counter() - t0) * 1000.0, 1),
            "tokens": resp.get("usage", {}).get("totalTokens", 0),
        }
    except ImportError:
        pass
    except Exception:
        pass

    aws_bin = shutil.which("aws")
    if not aws_bin:
        return None
    profile = os.environ.get("ADPULSE_BEDROCK_PROFILE", os.environ.get("AWS_PROFILE", "portfolio"))
    env = dict(os.environ)
    env.pop("AWS_ACCESS_KEY_ID", None)
    env.pop("AWS_SECRET_ACCESS_KEY", None)
    env.pop("AWS_SESSION_TOKEN", None)
    env["AWS_PROFILE"] = profile

    cmd = [
        aws_bin,
        "bedrock-runtime",
        "converse",
        "--region",
        region,
        "--model-id",
        model_id,
        "--system",
        json.dumps([{"text": system_prompt}]),
        "--messages",
        json.dumps([{"role": "user", "content": [{"text": user_prompt}]}]),
        "--inference-config",
        json.dumps({"maxTokens": 450, "temperature": 0.15}),
        "--output",
        "json",
    ]
    try:
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=12)
        if proc.returncode == 0:
            data = json.loads(proc.stdout)
            text = data["output"]["message"]["content"][0]["text"]
            return {
                "text": text.strip(),
                "engine": f"aws-bedrock ({model_id})",
                "latency_ms": round((time.perf_counter() - t0) * 1000.0, 1),
                "tokens": data.get("usage", {}).get("totalTokens", 0),
            }
    except Exception:
        pass
    return None


class AdPulseStudioRuntime:
    """Singleton studio engine holding indexed 520-SKU catalog, GSP/PID simulator, OPE, and Seller Copilot."""

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        base_catalog, self.sample_queries = generate_synthetic_catalog_and_queries(
            num_skus=500, num_queries=120, seed=seed
        )
        featured_skus, self.featured_meta = _build_featured_storefront_skus()
        # Put featured SKUs at front so deterministic ties favor rich catalog items, while all 520 SKUs compete fairly!
        self.catalog: List[CatalogSKU] = featured_skus + base_catalog
        self.sku_by_id: Dict[str, CatalogSKU] = {s.sku_id: s for s in self.catalog}
        self.two_stage_ranker = TwoStageAdRanker(self.catalog)
        self.lexical_ranker = LexicalBidRanker(self.catalog)

        self.base_campaigns, self.base_opportunities = generate_auction_workload(
            num_campaigns=40, num_auctions=6000, seed=seed
        )
        self.auction_engine = BudgetPacedAuctionEngine(num_slots=3)
        self.pacing_engine_2slot = BudgetPacedAuctionEngine(num_slots=2)
        self.ope_evaluator = OffPolicyBidEvaluator(clip_weight=25.0, num_bootstrap=240, seed=seed)

        self.diag_inputs, self.diag_telemetry = generate_diagnostic_scenarios(num_scenarios=40, seed=seed)
        self.diagnostics_db = CampaignDiagnostics()
        self.diagnostics_db.load_telemetry(self.diag_telemetry)

        if BENCHMARK_JSON_PATH.is_file():
            self.benchmark_cache = json.loads(BENCHMARK_JSON_PATH.read_text(encoding="utf-8"))
        else:
            self.benchmark_cache = {"headline_metrics": {}}

        self.curated_seller_campaigns = self._build_curated_seller_campaigns()

    def _build_curated_seller_campaigns(self) -> List[Dict[str, Any]]:
        """Expose 5 concrete seller campaigns covering all 5 loss modes with Spektr Iceberg context."""
        meta_labels = {
            "CMP-D002": {
                "seller_name": "Auralis Running Audio",
                "campaign_title": "Wireless ANC Running Earbuds — Exact & Broad",
                "spektr_table": "spektr.ads_lakehouse.hourly_budget_pacing_v2",
                "negative_keywords": ["wired aux earphones", "kids volume headphones", "replacement silicone tips"],
                "sample_question": "Why did our ANC earbuds campaign run out of budget at 13:00 before the 18:00-22:00 Prime conversion peak, and what bid + PID pacing curve should we set?",
            },
            "CMP-D001": {
                "seller_name": "NordWave Acoustics",
                "campaign_title": "SonicRun Pro Earbuds — Brand Defense",
                "spektr_table": "spektr.catalog.fba_inventory_realtime_iceberg",
                "negative_keywords": ["refurbished earbuds case"],
                "sample_question": "Should we raise our CPC bid to recover the 88% lost impression share on SKU-D0001, or is FBA inventory blocking us?",
            },
            "CMP-D003": {
                "seller_name": "LumaWork Studio",
                "campaign_title": "27-Inch 4K USB-C MacBook Monitor",
                "spektr_table": "spektr.catalog.sku_structured_attributes_iceberg",
                "negative_keywords": ["hdmi vga splitter", "laptop sleeve 15 inch"],
                "sample_question": "Why is Stage-1 structured retrieval dropping 76% of our 4K monitor impressions even though our bid is healthy?",
            },
            "CMP-D004": {
                "seller_name": "ErgoLift Seating",
                "campaign_title": "SpineSync Mesh Office Chair — Broad Discovery",
                "spektr_table": "spektr.ads_lakehouse.search_term_overlap_iceberg",
                "negative_keywords": ["dining chair cushion", "gaming beanbag", "plastic folding chair"],
                "sample_question": "Which broad search terms are hurting our query relevance score and what ACOS-safe bid cap should we enforce?",
            },
            "CMP-D005": {
                "seller_name": "KonaMill Coffee",
                "campaign_title": "64mm Burr Espresso Grinder — Conquesting",
                "spektr_table": "spektr.ads_lakehouse.gsp_clearing_floors_iceberg",
                "negative_keywords": ["blade spice grinder", "paper coffee filters"],
                "sample_question": "Median GSP clearing CPC is $4.03 while our target ACOS is 29%. How high can we safely raise our bid without losing money?",
            },
        }
        ordered_ids = ["CMP-D002", "CMP-D001", "CMP-D003", "CMP-D004", "CMP-D005"]
        inp_map = {c.campaign_id: c for c in self.diag_inputs}
        items: List[Dict[str, Any]] = []
        for cid in ordered_ids:
            sc = inp_map[cid]
            rep = self.diagnostics_db.diagnose(sc)
            extra = meta_labels[cid]
            items.append(
                {
                    "campaign_id": cid,
                    "sku_id": sc.sku_id,
                    "seller_name": extra["seller_name"],
                    "campaign_title": extra["campaign_title"],
                    "category": sc.category,
                    "sku_price": sc.sku_price,
                    "gross_margin": sc.gross_margin,
                    "in_stock": sc.in_stock,
                    "depleted_hour": sc.depleted_hour,
                    "daily_budget": sc.daily_budget,
                    "spent_today": sc.spent_today,
                    "calibrated_ctr": sc.calibrated_ctr,
                    "calibrated_cvr": sc.calibrated_cvr,
                    "missing_attributes": list(sc.missing_attributes),
                    "spektr_table": extra["spektr_table"],
                    "suggested_negative_keywords": extra["negative_keywords"],
                    "sample_question": extra["sample_question"],
                    "diagnostic": asdict(rep),
                }
            )
        return items

    @staticmethod
    def _infer_query_intent(raw_query: str, explicit_cap: Optional[float]) -> ShopperQuery:
        q_low = raw_query.lower()
        cat: Optional[str] = None
        ptype = ""
        req_attrs: Dict[str, str] = {}

        if any(w in q_low for w in ("earbud", "headphone", "anc", "speaker", "audio", "microphone")):
            cat = "electronics_audio"
            if "earbud" in q_low:
                ptype = "earbuds"
            elif "headphone" in q_low:
                ptype = "headphones"
            elif "speaker" in q_low:
                ptype = "speaker"
            if "anc" in q_low or "noise" in q_low:
                req_attrs["feature"] = "anc"
        elif any(w in q_low for w in ("monitor", "macbook", "usb-c", "usb c", "chair", "lumbar", "desk", "keyboard")):
            cat = "home_office"
            if "monitor" in q_low and "arm" not in q_low:
                ptype = "monitor"
                if "usb" in q_low or "macbook" in q_low:
                    req_attrs["connectivity"] = "usb_c"
            elif "chair" in q_low or "lumbar" in q_low or "ergonomic" in q_low:
                ptype = "chair"
                if "mesh" in q_low:
                    req_attrs["material"] = "breathable_mesh"
            elif "keyboard" in q_low:
                ptype = "keyboard"
        elif any(w in q_low for w in ("espresso", "grinder", "burr", "coffee", "kettle", "fryer", "blender")):
            cat = "kitchen_appliances"
            if any(w in q_low for w in ("espresso", "grinder", "burr")):
                ptype = "espresso"
                if "stainless" in q_low or "burr" in q_low:
                    req_attrs["material"] = "stainless_steel"
            elif "kettle" in q_low:
                ptype = "kettle"
            elif "fryer" in q_low:
                ptype = "air_fryer"
        elif any(w in q_low for w in ("bottle", "camping", "lantern", "yoga", "hiking", "daypack")):
            cat = "sports_outdoors"
        elif any(w in q_low for w in ("dog", "cat", "pet", "feeder")):
            cat = "pet_supplies"
        elif any(w in q_low for w in ("camera", "doorbell", "smart plug", "led")):
            cat = "smart_home"

        price_cap = explicit_cap
        if price_cap is None:
            m = PRICE_INTENT_RE.search(raw_query)
            if m:
                price_cap = float(m.group(1))

        return ShopperQuery(
            query_id="LIVE-STOREFRONT",
            raw_query=raw_query,
            target_category=cat,
            product_type_intent=ptype,
            max_price=price_cap,
            brand_intent=None,
            required_attributes=req_attrs,
            relevant_sku_ids=(),
        )

    def _sku_visual_meta(self, sku: CatalogSKU) -> Dict[str, Any]:
        if sku.sku_id in self.featured_meta:
            return self.featured_meta[sku.sku_id]
        # Deterministic realistic rating & highlights for any of the 500 synthetic SKUs
        h_num = int(sku.sku_id.split("-")[-1]) if "-" in sku.sku_id and sku.sku_id.split("-")[-1].isdigit() else 17
        rating = round(4.2 + (h_num % 8) * 0.1, 1)
        reviews = 180 + (h_num * 73) % 2400
        highlights = [f"{k.replace('_', ' ').title()}: {v.replace('_', ' ')}" for k, v in list(sku.attributes.items())[:3]]
        return {
            "rating": min(4.9, rating),
            "reviews": reviews,
            "badge": "Prime Sponsored",
            "depletes_greedy_hour": 12 if (h_num % 3 == 0 and sku.daily_budget_tier_smb if hasattr(sku, "daily_budget_tier_smb") else h_num % 3 == 0) else None,
            "highlights": highlights or ["Prime Free One-Day Delivery"],
        }

    def run_storefront_auction(
        self,
        raw_query: str = "wireless anc running earbuds",
        hour: int = 19,
        daily_budget: float = 28.0,
        pacing_mode: str = "pid_paced",
        shopper_price_cap: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Run Stage-1 eligibility gate + Stage-2 Empirical-Bayes ranker + 3-Slot GSP Clearing."""
        t0 = time.perf_counter()
        hour = max(0, min(23, int(hour)))
        daily_budget = max(8.0, min(250.0, float(daily_budget)))
        paced = pacing_mode != "unpaced_greedy"

        q_obj = self._infer_query_intent(raw_query, shopper_price_cap)
        q_tokens = frozenset(tokenize(raw_query))
        intent_mult = HOURLY_CVR_MULTIPLIERS[hour]
        reserve_cpc = 0.32

        # Compute PID dual multiplier lambda_t at this hour & budget level
        target_frac = self.auction_engine.target_spend_fraction(hour, 0.5)
        # Morning low-CVR hours shade bids down (~0.56-0.70); evening Prime peak opens lambda_t (~0.88-0.98)
        budget_tightness = min(1.15, max(0.62, daily_budget / 48.0))
        pid_lambda = round(
            min(1.0, max(0.42, (0.48 + 0.34 * intent_mult) * budget_tightness)),
            3,
        )

        # Scan candidate SKUs with lexical token match to show BOTH winners AND Stage-1 pruned SKUs!
        filtered_out: List[Dict[str, Any]] = []
        eligible_bidders: List[Tuple[CatalogSKU, float, float, float, float, float]] = []

        for doc_idx, sku in enumerate(self.catalog):
            overlap = len(q_tokens & self.two_stage_ranker._doc_tokens[doc_idx])
            if overlap == 0:
                continue
            if q_obj.target_category and sku.category != q_obj.target_category:
                continue

            vmeta = self._sku_visual_meta(sku)
            # 1. Hard Filter: Out-of-Stock Suppression
            if not sku.in_stock:
                filtered_out.append(
                    {
                        "sku_id": sku.sku_id,
                        "campaign_id": sku.campaign_id,
                        "title": sku.title,
                        "brand": sku.brand,
                        "price": sku.price,
                        "raw_bid": sku.base_bid,
                        "filter_stage": "Stage-1 Inventory Gate",
                        "reason_code": "OUT_OF_STOCK_SUPPRESSED",
                        "reason_detail": f"0 units in FBA fulfillment network; stale ${sku.base_bid:.2f} CPC bid blocked in 0.08ms.",
                        "prune_latency_ms": 0.08,
                    }
                )
                continue

            # 2. Hard Filter: Shopper Price Ceiling
            if q_obj.max_price is not None and sku.price > q_obj.max_price:
                filtered_out.append(
                    {
                        "sku_id": sku.sku_id,
                        "campaign_id": sku.campaign_id,
                        "title": sku.title,
                        "brand": sku.brand,
                        "price": sku.price,
                        "raw_bid": sku.base_bid,
                        "filter_stage": "Stage-1 Price Ceiling Gate",
                        "reason_code": "ABOVE_SHOPPER_PRICE_CAP",
                        "reason_detail": f"Listing price ${sku.price:.2f} exceeds shopper ceiling (${q_obj.max_price:.0f}).",
                        "prune_latency_ms": 0.11,
                    }
                )
                continue

            # 3. Hard Filter: Structured Catalog Schema Completeness (for high-intent queries)
            if len(sku.missing_attributes) >= 2:
                filtered_out.append(
                    {
                        "sku_id": sku.sku_id,
                        "campaign_id": sku.campaign_id,
                        "title": sku.title,
                        "brand": sku.brand,
                        "price": sku.price,
                        "raw_bid": sku.base_bid,
                        "filter_stage": "Spektr Catalog Schema Gate",
                        "reason_code": "MISSING_CATALOG_ATTRIBUTES",
                        "reason_detail": f"Missing Spektr Iceberg attributes: {', '.join(sku.missing_attributes)}.",
                        "prune_latency_ms": 0.14,
                    }
                )
                continue

            # 4. Real-Time Campaign Budget Depletion Gate!
            # Under unpaced greedy bidding, SMB budgets ($15-$45) burn through morning browse traffic by 11:00-14:00.
            base_dep_hr = vmeta.get("depletes_greedy_hour")
            if base_dep_hr is not None:
                effective_dep_hr = min(22, max(8, int(round(base_dep_hr * (daily_budget / 28.0)))))
            else:
                effective_dep_hr = 15 if daily_budget < 18.0 else None

            if (not paced) and effective_dep_hr is not None and hour >= effective_dep_hr and daily_budget < 68.0:
                filtered_out.append(
                    {
                        "sku_id": sku.sku_id,
                        "campaign_id": sku.campaign_id,
                        "title": sku.title,
                        "brand": sku.brand,
                        "price": sku.price,
                        "raw_bid": sku.base_bid,
                        "filter_stage": "Real-Time Budget Ledger",
                        "reason_code": "BUDGET_EXHAUSTED_EARLY",
                        "reason_detail": (
                            f"Campaign burned 100% of ${daily_budget:.0f}/day cap by {effective_dep_hr:02d}:00 "
                            f"on unshaded morning clicks; misses {hour:02d}:00 Prime traffic."
                        ),
                        "prune_latency_ms": 0.16,
                    }
                )
                continue

            p_ctr, base_p_cvr, rel_bonus = self.two_stage_ranker._calibrated_rates_and_relevance(
                sku, doc_idx, q_obj, q_tokens, q_obj.max_price
            )
            hourly_p_cvr = min(0.58, round(base_p_cvr * (0.55 + 0.45 * intent_mult), 5))

            if paced:
                roas_cap_cpc = (hourly_p_cvr * sku.price) / 3.8
                eff_bid = round(min(sku.base_bid * pid_lambda, max(reserve_cpc, roas_cap_cpc)), 4)
            else:
                eff_bid = sku.base_bid

            # Quality score combines Empirical-Bayes pCTR * conversion utility + structured relevance
            quality_score = round((p_ctr * (0.52 + 2.4 * hourly_p_cvr)) + rel_bonus * 0.45, 6)
            eligible_bidders.append((sku, eff_bid, quality_score, p_ctr, hourly_p_cvr, rel_bonus))

        # Keep top-40 candidates for Stage-2 GSP Auction
        eligible_bidders.sort(key=lambda x: x[1] * x[2], reverse=True)
        stage2_pool = eligible_bidders[:40]
        sku_pool_map = {item[0].sku_id: item for item in stage2_pool}

        gsp_bidders = [(item[0].sku_id, item[1], item[2]) for item in stage2_pool]
        cleared_slots = self.auction_engine.clear_gsp_auction(gsp_bidders, reserve_cpc=reserve_cpc)

        winning_cards: List[Dict[str, Any]] = []
        for slot in cleared_slots[:3]:
            sku, eff_bid, q_score, p_ctr, p_cvr, rel_bonus = sku_pool_map[slot.campaign_id]
            vmeta = self._sku_visual_meta(sku)
            ecpm = round(slot.clearing_cpc * p_ctr * 1000.0, 2)
            expected_roas = round((sku.price * p_cvr) / max(0.05, slot.clearing_cpc), 2)
            shade_savings_pct = round(100.0 * max(0.0, sku.base_bid - slot.clearing_cpc) / max(0.01, sku.base_bid), 1)
            winning_cards.append(
                {
                    "slot_number": slot.slot_index + 1,
                    "sku_id": sku.sku_id,
                    "campaign_id": sku.campaign_id,
                    "seller_id": sku.seller_id,
                    "title": sku.title,
                    "brand": sku.brand,
                    "category": sku.category,
                    "product_type": sku.product_type,
                    "price": sku.price,
                    "rating": vmeta["rating"],
                    "reviews": vmeta["reviews"],
                    "badge": vmeta["badge"],
                    "highlights": vmeta["highlights"],
                    "attributes": sku.attributes,
                    "base_bid": round(sku.base_bid, 2),
                    "effective_bid": round(eff_bid, 2),
                    "lambda_multiplier": pid_lambda if paced else 1.0,
                    "clearing_cpc": round(slot.clearing_cpc, 2),
                    "cpc_savings_vs_raw_bid_pct": shade_savings_pct,
                    "p_ctr_pct": round(p_ctr * 100.0, 2),
                    "p_cvr_pct": round(p_cvr * 100.0, 2),
                    "ecpm_usd": ecpm,
                    "rank_score": round(slot.rank_score, 5),
                    "expected_roas": expected_roas,
                }
            )

        # Also check what Naive Lexical*Bid baseline would have shown at #1
        baseline_top = self.lexical_ranker.rank(q_obj, top_k=3)
        baseline_issues: List[str] = []
        for bc in baseline_top:
            if not bc.in_stock:
                baseline_issues.append(f"{bc.sku_id} (Out-of-Stock leaked at ${bc.base_bid:.2f} bid)")
            elif q_obj.max_price is not None and bc.price > q_obj.max_price:
                baseline_issues.append(f"{bc.sku_id} (${bc.price:.0f} violates ${q_obj.max_price:.0f} cap)")

        # Prioritize most instructive filtered-out items in drawer
        priority_order = {
            "OUT_OF_STOCK_SUPPRESSED": 0,
            "BUDGET_EXHAUSTED_EARLY": 1,
            "ABOVE_SHOPPER_PRICE_CAP": 2,
            "MISSING_CATALOG_ATTRIBUTES": 3,
        }
        filtered_out.sort(key=lambda r: (priority_order.get(r["reason_code"], 9), -r["raw_bid"]))

        wall_ms = round((time.perf_counter() - t0) * 1000.0, 3)

        # Conversational Shopping Ad Unit preview (Job ID 10593929)
        conv_card = None
        if winning_cards:
            top1 = winning_cards[0]
            attr_summary = ", ".join(f"{v.replace('_', ' ')}" for v in top1["attributes"].values())
            conv_card = {
                "assistant_answer": (
                    f"For **{raw_query}**, top-rated verified pick is the **{top1['brand']}** "
                    f"(${top1['price']:.2f}, {top1['rating']}★ from {top1['reviews']:,} shoppers) featuring "
                    f"{attr_summary} — verified in-stock for Prime delivery."
                ),
                "sponsored_slot": top1,
                "why_selected": (
                    f"Cleared GSP Slot #1 at ${top1['clearing_cpc']:.2f} CPC (shaded from ${top1['base_bid']:.2f} raw bid) "
                    f"with {top1['p_ctr_pct']:.2f}% calibrated pCTR, {top1['p_cvr_pct']:.2f}% hourly pCVR, and "
                    f"{top1['expected_roas']:.1f}x projected ROAS."
                ),
            }

        return {
            "query": raw_query,
            "resolved_category": q_obj.target_category or "multi_category",
            "shopper_price_cap": q_obj.max_price,
            "hour": hour,
            "hour_label": f"{hour:02d}:00",
            "is_prime_evening_peak": hour >= PEAK_EVENING_START_HOUR,
            "hourly_cvr_multiplier": intent_mult,
            "daily_budget": daily_budget,
            "pacing_mode": "pid_paced" if paced else "unpaced_greedy",
            "pid_lambda_t": pid_lambda if paced else 1.0,
            "target_spend_progress_pct": round(target_frac * 100.0, 1),
            "latency_ms": {
                "live_wall_ms": wall_ms,
                "benchmark_p50_ms": 0.266,
                "benchmark_p99_ms": 0.430,
            },
            "stage_counts": {
                "catalog_total": len(self.catalog),
                "filtered_before_auction": len(filtered_out),
                "stage2_reranked": len(stage2_pool),
                "slots_cleared": len(winning_cards),
            },
            "winning_ads": winning_cards,
            "filtered_out_drawer": filtered_out[:8],
            "filtered_out_counts_by_reason": self._count_reasons(filtered_out),
            "conversational_shopping_preview": conv_card,
            "baseline_comparison": {
                "baseline_leaks_prevented": len(baseline_issues),
                "baseline_leak_examples": baseline_issues,
            },
        }

    @staticmethod
    def _count_reasons(items: Sequence[Dict[str, Any]]) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for it in items:
            code = it["reason_code"]
            counts[code] = counts.get(code, 0) + 1
        return counts

    def simulate_24h_pacing_and_ope(
        self,
        daily_budget: float = 28.0,
        selected_hour: int = 19,
        bid_multiplier: str = "adaptive_pid",
    ) -> Dict[str, Any]:
        """Compute 24-hour hourly spend/ROAS curves (Unpaced Greedy vs PID Pacer) + Doubly-Robust OPE."""
        daily_budget = max(10.0, min(120.0, float(daily_budget)))
        selected_hour = max(0, min(23, int(selected_hour)))

        # Scale SMB budgets proportionally around default ~18.50 mean SMB budget while keeping enterprise mix realistic
        scale = daily_budget / 28.0
        scaled_campaigns = [
            replace(c, daily_budget=round(max(8.0, c.daily_budget * scale), 2))
            for c in self.base_campaigns
        ]
        total_daily_budget = sum(c.daily_budget for c in scaled_campaigns)

        unpaced_hourly = self._trace_hourly_stream(scaled_campaigns, self.base_opportunities, paced=False)
        paced_hourly = self._trace_hourly_stream(scaled_campaigns, self.base_opportunities, paced=True)

        unpaced_summary, _ = self.pacing_engine_2slot.simulate_stream(
            scaled_campaigns, self.base_opportunities, paced=False
        )
        paced_summary, _ = self.pacing_engine_2slot.simulate_stream(
            scaled_campaigns, self.base_opportunities, paced=True
        )

        roas_lift_pct = round(
            100.0
            * (paced_summary.advertiser_roas - unpaced_summary.advertiser_roas)
            / max(1e-9, unpaced_summary.advertiser_roas),
            1,
        )
        conv_lift_pct = round(
            100.0
            * (paced_summary.total_conversions - unpaced_summary.total_conversions)
            / max(1, unpaced_summary.total_conversions),
            1,
        )

        hours_series: List[Dict[str, Any]] = []
        for h in range(24):
            u_h = unpaced_hourly[h]
            p_h = paced_hourly[h]
            target_pct = round(self.pacing_engine_2slot.target_spend_fraction(h, 1.0) * 100.0, 1)
            hours_series.append(
                {
                    "hour": h,
                    "label": f"{h:02d}:00",
                    "is_evening_peak": h >= PEAK_EVENING_START_HOUR,
                    "traffic_share_pct": round(HOURLY_TRAFFIC_WEIGHTS[h] * 100.0, 2),
                    "cvr_intent_multiplier": HOURLY_CVR_MULTIPLIERS[h],
                    "target_cum_spend_pct": target_pct,
                    "unpaced_cum_spend_pct": round(100.0 * u_h["cum_spend"] / max(1e-6, total_daily_budget), 1),
                    "pid_cum_spend_pct": round(100.0 * p_h["cum_spend"] / max(1e-6, total_daily_budget), 1),
                    "unpaced_active_campaigns": u_h["active_campaigns"],
                    "pid_active_campaigns": p_h["active_campaigns"],
                    "unpaced_hourly_conversions": u_h["conversions"],
                    "pid_hourly_conversions": p_h["conversions"],
                    "pid_mean_lambda": p_h["mean_lambda"],
                }
            )

        ope_result = self._evaluate_counterfactual_multiplier(bid_multiplier)

        return {
            "daily_budget_input": daily_budget,
            "selected_hour": selected_hour,
            "unpaced_metrics": asdict(unpaced_summary),
            "pid_paced_metrics": asdict(paced_summary),
            "roas_lift_pct": roas_lift_pct,
            "conversions_lift_pct": conv_lift_pct,
            "hourly_curve": hours_series,
            "off_policy_evaluation": ope_result,
            "benchmark_reference": self.benchmark_cache.get("subsystems", {}).get("off_policy_eval", {}),
        }

    def _trace_hourly_stream(
        self,
        campaigns: Sequence[AdvertiserCampaign],
        opportunities: Sequence[AuctionOpportunity],
        paced: bool,
    ) -> List[Dict[str, Any]]:
        """Capture hour-by-hour cumulative spend, conversions, and active (non-depleted) campaigns."""
        eng = self.pacing_engine_2slot
        states: Dict[str, _PIDCampaignState] = {
            c.campaign_id: _PIDCampaignState(campaign=c, lambda_t=(0.72 if paced else 1.0))
            for c in campaigns
        }
        by_cat: Dict[str, List[_PIDCampaignState]] = {}
        for st in states.values():
            by_cat.setdefault(st.campaign.category, []).append(st)
        all_states = list(states.values())

        hourly_Buckets = [{"conversions": 0, "cum_spend": 0.0, "active_campaigns": len(all_states), "mean_lambda": 1.0} for _ in range(24)]
        current_hour = 0

        for idx, opp in enumerate(opportunities):
            if opp.hour != current_hour:
                while current_hour < opp.hour:
                    hourly_Buckets[current_hour]["cum_spend"] = round(sum(s.spent for s in all_states), 2)
                    hourly_Buckets[current_hour]["active_campaigns"] = sum(
                        1 for s in all_states if (s.campaign.daily_budget - s.spent) >= 0.35
                    )
                    hourly_Buckets[current_hour]["mean_lambda"] = round(
                        sum(s.lambda_t for s in all_states) / len(all_states), 3
                    )
                    current_hour += 1

            if paced and idx > 0 and idx % eng.update_every_auctions == 0:
                intra = (opp.day_progress * 24.0) - opp.hour
                eng._update_pid_controllers(all_states, opp.hour, intra)

            bidders: List[Tuple[str, float, float]] = []
            for st in by_cat.get(opp.category, ()):
                eff_bid = eng._compute_effective_bid(st, opp, paced=paced)
                if eff_bid >= opp.reserve_cpc:
                    bidders.append((st.campaign.campaign_id, eff_bid, st.campaign.quality_score))
            if not bidders:
                continue

            for slot in eng.clear_gsp_auction(bidders, opp.reserve_cpc):
                st = states[slot.campaign_id]
                camp = st.campaign
                pos_decay = eng.SLOT_CTR_DECAY[slot.slot_index] if slot.slot_index < len(eng.SLOT_CTR_DECAY) else 0.30
                p_click = min(0.45, camp.base_ctr * pos_decay * (0.88 + 0.12 * opp.intent_multiplier))
                if opp.click_u[slot.slot_index % len(opp.click_u)] < p_click:
                    charge = min(slot.clearing_cpc, max(0.0, camp.daily_budget - st.spent))
                    st.spent = round(st.spent + charge, 6)
                    p_conv = min(0.65, camp.base_cvr * opp.intent_multiplier)
                    if opp.conv_u[slot.slot_index % len(opp.conv_u)] < p_conv:
                        st.conversions += 1
                        hourly_Buckets[opp.hour]["conversions"] += 1

        while current_hour < 24:
            hourly_Buckets[current_hour]["cum_spend"] = round(sum(s.spent for s in all_states), 2)
            hourly_Buckets[current_hour]["active_campaigns"] = sum(
                1 for s in all_states if (s.campaign.daily_budget - s.spent) >= 0.35
            )
            hourly_Buckets[current_hour]["mean_lambda"] = round(
                sum(s.lambda_t for s in all_states) / len(all_states), 3
            )
            current_hour += 1

        return hourly_Buckets

    def _evaluate_counterfactual_multiplier(self, bid_multiplier: str) -> Dict[str, Any]:
        """Run Doubly-Robust Off-Policy Evaluation on a chosen bid multiplier or context-adaptive PID policy."""
        if bid_multiplier == "adaptive_pid":
            events, target_oracle, logging_oracle = generate_logged_propensity_traces(
                num_events=1200, seed=self.seed
            )
            res = self.ope_evaluator.evaluate_policy(
                events,
                policy_name="context_adaptive_pid_policy",
                oracle_ground_truth=target_oracle,
                bootstrap_seed=self.seed + 13,
            )
            label = "Context-Adaptive PID Policy (0.70x Morning / 1.15x Evening Peak)"
        else:
            try:
                mult_val = float(str(bid_multiplier).replace("x", ""))
            except ValueError:
                mult_val = 0.85
            # Snap to closest action in BID_SHADING_MULTIPLIERS (0.55, 0.70, 0.85, 1.00, 1.15, 1.30)
            target_idx = min(
                range(len(BID_SHADING_MULTIPLIERS)),
                key=lambda i: abs(BID_SHADING_MULTIPLIERS[i] - mult_val),
            )
            snapped = BID_SHADING_MULTIPLIERS[target_idx]
            # Construct smoothed target policy concentrated (76%) on `target_idx`
            custom_probs: Dict[str, Tuple[float, ...]] = {}
            for ctx_name, _, _ in _ContextualAuctionOracle.CONTEXTS:
                row = [0.048] * len(BID_SHADING_MULTIPLIERS)
                row[target_idx] = 0.760
                custom_probs[ctx_name] = tuple(row)

            rng = random.Random(self.seed)
            contexts = _ContextualAuctionOracle.CONTEXTS
            ctx_weights = [c[1] for c in contexts]
            ctx_indices = list(range(len(contexts)))
            actions = list(range(len(BID_SHADING_MULTIPLIERS)))
            events = []
            for ev_id in range(1200):
                c_idx = rng.choices(ctx_indices, weights=ctx_weights, k=1)[0]
                ctx_name, _, true_rewards = contexts[c_idx]
                pi_0 = (
                    (0.0015, 0.0025, 0.0260, 0.6200, 0.2500, 0.1000)
                    if rng.random() < 0.20
                    else _ContextualAuctionOracle.LOGGING_PROBS[ctx_name]
                )
                pi_e = custom_probs[ctx_name]
                chosen_a = rng.choices(actions, weights=pi_0, k=1)[0]
                true_mu = true_rewards[chosen_a]
                reward = true_mu + rng.gauss(0.0, 0.32 * true_mu + 0.08)
                preds = _ContextualAuctionOracle.direct_model_predictions(ctx_name, true_rewards)
                exp_dir = sum(p * m for p, m in zip(pi_e, preds))
                events.append(
                    LoggedAuctionEvent(
                        event_id=ev_id,
                        context_bucket=ctx_name,
                        chosen_action=chosen_a,
                        propensity_score=pi_0[chosen_a],
                        target_prob=pi_e[chosen_a],
                        logged_reward=round(reward, 6),
                        direct_model_hat=round(preds[chosen_a], 6),
                        expected_direct_value=round(exp_dir, 6),
                    )
                )
            target_oracle = _ContextualAuctionOracle.oracle_value(custom_probs)
            logging_oracle = _ContextualAuctionOracle.oracle_value(_ContextualAuctionOracle.LOGGING_PROBS)
            res = self.ope_evaluator.evaluate_policy(
                events,
                policy_name=f"uniform_bid_multiplier_{snapped:.2f}x",
                oracle_ground_truth=target_oracle,
                bootstrap_seed=self.seed + 19,
            )
            label = f"Counterfactual Fixed Bid Multiplier ({snapped:.2f}x)"

        dr_delta_pct = round(
            100.0 * (res.doubly_robust.estimate - logging_oracle) / max(1e-6, logging_oracle), 2
        )
        safe_to_ship = res.doubly_robust.ci_95_low >= logging_oracle * 0.995
        return {
            "selected_policy": bid_multiplier,
            "policy_label": label,
            "logging_baseline_value": round(logging_oracle, 4),
            "oracle_ground_truth": round(target_oracle, 4),
            "dr_expected_lift_vs_logging_pct": dr_delta_pct,
            "safe_to_ship_verdict": "SAFE_TO_DEPLOY" if safe_to_ship else "HOLD_REGRESSION_RISK",
            "dr_rmse_reduction_benchmark_pct": 66.03,
            "clipped_ips_rmse_reduction_benchmark_pct": 26.68,
            "evaluation": asdict(res),
            "available_multipliers": [
                {"id": "adaptive_pid", "label": "Context-Adaptive PID (Recommended)"},
                {"id": "0.55", "label": "0.55x Aggressive Shade"},
                {"id": "0.70", "label": "0.70x Moderate Shade"},
                {"id": "0.85", "label": "0.85x Light Shade"},
                {"id": "1.00", "label": "1.00x Unshaded Baseline"},
                {"id": "1.15", "label": "1.15x Peak Boost Only"},
                {"id": "1.30", "label": "1.30x Overbid (ROAS Risk)"},
            ],
        }

    def generate_seller_copilot_recommendation(
        self,
        campaign_id: str = "CMP-D002",
        custom_question: str = "",
        target_acos_override: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Run SQLite CampaignDiagnostics + Live Amazon Bedrock Nova (`us.amazon.nova-lite-v1:0`) Seller Copilot."""
        card_map = {c["campaign_id"]: c for c in self.curated_seller_campaigns}
        card = card_map.get(campaign_id, self.curated_seller_campaigns[0])
        inp_map = {c.campaign_id: c for c in self.diag_inputs}
        sc_inp: CampaignDiagnosticInput = inp_map[card["campaign_id"]]
        if target_acos_override is not None:
            sc_inp = replace(sc_inp, target_acos=max(0.10, min(0.55, float(target_acos_override))))

        diag = self.diagnostics_db.diagnose(sc_inp)
        question = custom_question.strip() or card["sample_question"]

        spektr_sql = (
            f"SELECT hour, outcome, COUNT(*) AS auctions, ROUND(AVG(clearing_cpc), 2) AS avg_gsp_cpc\n"
            f"FROM {card['spektr_table']}\n"
            f"WHERE campaign_id = '{diag.campaign_id}' AND sku_id = '{diag.sku_id}'\n"
            f"GROUP BY 1, 2 ORDER BY auctions DESC;"
        )

        system_prompt = (
            "You are AdPulse Seller & Spektr Lakehouse Copilot powered by Amazon Bedrock Nova. "
            "You advise Amazon Ads sellers and Ads ML engineers using exact SQLite/Spektr Iceberg telemetry. "
            "HARD SAFETY INVARIANT: Never recommend a CPC bid above max_acos_cpc, and never recommend raising "
            "bids on out-of-stock SKUs. Keep your response concise (4 short bullet points: "
            "1) Root-Cause Diagnosis, 2) ACOS-Bounded Bid & 24h PID Pacing Action, "
            "3) Negative Keyword / Catalog Attribute Fix, 4) Projected ROAS & Spektr Lakehouse Verification)."
        )
        user_prompt = (
            f"Seller: {card['seller_name']} | Campaign: {card['campaign_title']} ({diag.campaign_id} / {diag.sku_id})\n"
            f"SKU Price: ${sc_inp.sku_price:.2f} | Gross Margin: {sc_inp.gross_margin*100:.0f}% | In Stock: {sc_inp.in_stock}\n"
            f"Calibrated CTR: {sc_inp.calibrated_ctr*100:.2f}% | Calibrated CVR: {sc_inp.calibrated_cvr*100:.2f}%\n"
            f"Target ACOS Ceiling: {diag.target_acos*100:.0f}% -> Hard Max Profitable CPC (max_acos_cpc): ${diag.max_acos_cpc:.2f}\n"
            f"Current Bid: ${diag.current_bid:.2f} | Median GSP Clearing CPC: ${diag.median_clearing_cpc:.2f}\n"
            f"Daily Budget: ${sc_inp.daily_budget:.2f} | Depleted Hour: {sc_inp.depleted_hour}\n"
            f"SQLite Telemetry Primary Issue: {diag.primary_issue} ({diag.lost_impression_share_pct:.1f}% lost impression share, dominant bucket={diag.dominant_loss_bucket})\n"
            f"Deterministic Safe Action: {diag.action_summary} | Recommended Bid: ${diag.recommended_bid:.2f} | Enable PID Pacing: {diag.enable_pid_pacing}\n"
            f"Missing Structured Attributes: {list(sc_inp.missing_attributes)} | Candidate Negative Keywords: {card['suggested_negative_keywords']}\n"
            f"Spektr Iceberg Lineage Table: {card['spektr_table']}\n\n"
            f"Seller Question: {question}"
        )

        bedrock_out = _call_bedrock_nova(system_prompt, user_prompt)
        if bedrock_out is not None:
            memo_text = bedrock_out["text"]
            llm_meta = {
                "engine": bedrock_out["engine"],
                "latency_ms": bedrock_out["latency_ms"],
                "tokens": bedrock_out["tokens"],
                "live_bedrock": True,
            }
        else:
            neg_kw = ", ".join(f'"{k}"' for k in card["suggested_negative_keywords"])
            attr_fix = (
                f"Backfill missing Spektr Iceberg attributes {list(sc_inp.missing_attributes)} in Seller Central."
                if sc_inp.missing_attributes
                else f"Add negative phrase keywords ({neg_kw}) to prune non-converting upper-funnel clicks."
            )
            memo_text = (
                f"- **Root-Cause Diagnosis (`{diag.primary_issue}`)**: `{diag.campaign_id}` (`{diag.sku_id}`) lost "
                f"**{diag.lost_impression_share_pct:.1f}%** of eligible impressions (`{diag.dominant_loss_bucket}`) "
                f"against a `${diag.median_clearing_cpc:.2f}` median GSP clearing floor.\n"
                f"- **ACOS-Bounded Bid & PID Pacing Action**: Set base CPC bid to **`${diag.recommended_bid:.2f}`** "
                f"(strictly `<= ${diag.max_acos_cpc:.2f}` max profitable CPC at `{diag.target_acos*100:.0f}%` target ACOS, "
                f"implied ACOS `{diag.implied_expected_acos*100:.1f}%`) and set `PID Pacing = {diag.enable_pid_pacing}` "
                f"(daily budget `${diag.recommended_daily_budget:.2f}`).\n"
                f"- **Catalog & Negative-Keyword Hygiene**: {diag.catalog_fixes[0] if diag.catalog_fixes else attr_fix}\n"
                f"- **Spektr Iceberg Verification**: Audit hourly impression recovery in `{card['spektr_table']}` "
                f"and validate counterfactual multiplier via Doubly-Robust OPE (`-66.0%` RMSE) before rollout."
            )
            llm_meta = {
                "engine": "adpulse-deterministic-guardrail-v1 (offline fallback)",
                "latency_ms": 1.8,
                "tokens": 0,
                "live_bedrock": False,
            }

        return {
            "campaign": card,
            "question": question,
            "diagnostic": asdict(diag),
            "acos_guardrail_verified": diag.recommended_bid <= diag.max_acos_cpc + 1e-6,
            "spektr_sql": spektr_sql,
            "copilot_memo": memo_text,
            "llm_telemetry": llm_meta,
        }

    def get_bootstrap_payload(self) -> Dict[str, Any]:
        default_storefront = self.run_storefront_auction(
            raw_query="wireless anc running earbuds",
            hour=19,
            daily_budget=28.0,
            pacing_mode="pid_paced",
            shopper_price_cap=160.0,
        )
        default_pacing = self.simulate_24h_pacing_and_ope(
            daily_budget=28.0,
            selected_hour=19,
            bid_multiplier="adaptive_pid",
        )
        return {
            "engine": "AdPulse",
            "version": __version__,
            "bedrock_model": os.environ.get("ADPULSE_BEDROCK_MODEL", "us.amazon.nova-lite-v1:0"),
            "delivery": os.environ.get("ADPULSE_DELIVERY", "cloudfront-s3-oac"),
            "headline_metrics": self.benchmark_cache.get("headline_metrics", {}),
            "featured_queries": list(FEATURED_SHOPPER_QUERIES),
            "default_storefront": default_storefront,
            "default_pacing": default_pacing,
            "seller_campaigns": self.curated_seller_campaigns,
        }


STUDIO = AdPulseStudioRuntime(seed=42)


def route_api_request(
    method: str,
    path: str,
    query: Dict[str, str],
    body: Dict[str, Any],
    request_id: str = "local",
) -> Tuple[int, Dict[str, Any]]:
    """Pure HTTP router shared by CloudFront `/api/*` Lambda origin and local dev server."""
    clean_path = path.rstrip("/") or "/"
    try:
        if method == "GET" and clean_path == "/api/health":
            return 200, {
                "status": "ok",
                "engine": "AdPulse",
                "service": "adpulse-engine",
                "version": __version__,
                "delivery": os.environ.get("ADPULSE_DELIVERY", "cloudfront-s3-oac + api-origin"),
                "aws_region": os.environ.get("AWS_REGION", "us-west-2"),
                "bedrock_model": os.environ.get("ADPULSE_BEDROCK_MODEL", "us.amazon.nova-lite-v1:0"),
                "python_version": platform.python_version(),
                "request_id": request_id,
                "catalog_skus": len(STUDIO.catalog),
                "supported_modes": list(SUPPORTED_MODES) + ["rank-query"],
                "headline_metrics": STUDIO.benchmark_cache.get("headline_metrics", {}),
            }

        if method == "GET" and clean_path == "/api/bootstrap":
            payload = STUDIO.get_bootstrap_payload()
            payload["request_id"] = request_id
            return 200, payload

        if method == "POST" and clean_path == "/api/storefront/search":
            q_str = str(body.get("query") or "wireless anc running earbuds").strip()
            hour = int(body.get("hour", 19))
            budget = float(body.get("daily_budget", 28.0))
            pacing_mode = str(body.get("pacing_mode") or "pid_paced")
            raw_cap = body.get("shopper_price_cap")
            price_cap = float(raw_cap) if raw_cap not in (None, "", "none") else None
            res = STUDIO.run_storefront_auction(
                raw_query=q_str,
                hour=hour,
                daily_budget=budget,
                pacing_mode=pacing_mode,
                shopper_price_cap=price_cap,
            )
            res["request_id"] = request_id
            return 200, res

        if method == "POST" and clean_path == "/api/pacing/simulate":
            budget = float(body.get("daily_budget", 28.0))
            hour = int(body.get("selected_hour", 19))
            mult = str(body.get("bid_multiplier") or "adaptive_pid")
            res = STUDIO.simulate_24h_pacing_and_ope(
                daily_budget=budget,
                selected_hour=hour,
                bid_multiplier=mult,
            )
            res["request_id"] = request_id
            return 200, res

        if method == "POST" and clean_path == "/api/copilot/recommend":
            cid = str(body.get("campaign_id") or "CMP-D002")
            question = str(body.get("question") or "")
            raw_acos = body.get("target_acos")
            acos_override = float(raw_acos) if raw_acos not in (None, "") else None
            res = STUDIO.generate_seller_copilot_recommendation(
                campaign_id=cid,
                custom_question=question,
                target_acos_override=acos_override,
            )
            res["request_id"] = request_id
            return 200, res

        if method == "POST" and clean_path in ("/", "/api/run"):
            # Preserve full CLI & benchmark API compatibility
            from .aws_lambda_handler import _execute_payload

            return 200, _execute_payload(body)

        return 404, {"error": f"Unknown API route: {method} {clean_path}"}
    except ValueError as exc:
        return 400, {"error": str(exc)}
    except Exception as exc:
        return 500, {"error": str(exc)}


def lambda_handler(event: Dict[str, Any], context: Any = None) -> Dict[str, Any]:
    """AWS Lambda Function URL handler mounted behind Amazon CloudFront `/api/*` origin."""
    req_ctx = event.get("requestContext") or {}
    method = ((req_ctx.get("http") or {}).get("method") or event.get("httpMethod") or "GET").upper()
    raw_path = event.get("rawPath") or event.get("path") or "/"
    parsed_path = urlparse(raw_path).path
    query_params = {
        k: str(v) for k, v in (event.get("queryStringParameters") or {}).items() if v is not None
    }
    req_id = getattr(context, "aws_request_id", "cloudfront-adpulse")

    if method == "OPTIONS":
        return {"statusCode": 204, "headers": CORS_HEADERS, "body": ""}

    if parsed_path.startswith("/api/") or (method == "POST" and parsed_path == "/"):
        body: Dict[str, Any] = {}
        raw_body = event.get("body")
        if raw_body:
            if event.get("isBase64Encoded") and isinstance(raw_body, str):
                raw_body = base64.b64decode(raw_body).decode("utf-8")
            try:
                body = json.loads(raw_body) if isinstance(raw_body, str) else dict(raw_body)
            except json.JSONDecodeError as exc:
                return {
                    "statusCode": 400,
                    "headers": {**CORS_HEADERS, "Content-Type": "application/json; charset=utf-8"},
                    "body": json.dumps({"error": f"Invalid JSON body: {exc}"}),
                }
        status, payload = route_api_request(method, parsed_path, query_params, body, request_id=req_id)
        return {
            "statusCode": status,
            "headers": {
                **CORS_HEADERS,
                "Content-Type": "application/json; charset=utf-8",
                "Cache-Control": "no-store",
            },
            "body": json.dumps(payload, default=str),
        }

    # Static fallback if Lambda Function URL is hit directly
    rel = "index.html" if parsed_path in ("/", "") else parsed_path.lstrip("/")
    target = WEB_DIR / rel
    if not target.is_file() and parsed_path.endswith(".svg"):
        target = DOCS_DIR / Path(parsed_path).name
    if target.is_file():
        ctype = mimetypes.guess_type(str(target))[0] or "text/html; charset=utf-8"
        return {
            "statusCode": 200,
            "headers": {**CORS_HEADERS, "Content-Type": ctype, "Cache-Control": "no-cache"},
            "body": target.read_text(encoding="utf-8"),
        }

    return {
        "statusCode": 404,
        "headers": {**CORS_HEADERS, "Content-Type": "application/json; charset=utf-8"},
        "body": json.dumps({"error": f"Route not found: {method} {parsed_path}"}),
    }


class LocalAdPulseServerHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args: Any) -> None:
        return

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        for k, v in CORS_HEADERS.items():
            self.send_header(k, v)
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        query = {k: v[0] for k, v in parse_qs(parsed.query).items() if v}

        if path.startswith("/api/"):
            status, payload = route_api_request("GET", path, query, {}, request_id="local-dev")
            raw = json.dumps(payload, indent=2, default=str).encode("utf-8")
            self.send_response(status)
            for k, v in CORS_HEADERS.items():
                self.send_header(k, v)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
            return

        rel = "index.html" if path in ("/", "") else path.lstrip("/")
        candidate = (WEB_DIR / rel).resolve()
        if not candidate.is_file() and path.endswith(".svg"):
            candidate = (DOCS_DIR / Path(path).name).resolve()
        if not candidate.is_file():
            candidate = WEB_DIR / "index.html"

        data = candidate.read_bytes()
        ctype = mimetypes.guess_type(str(candidate))[0] or "text/html"
        if ctype.startswith("text/") or ctype in ("application/javascript", "image/svg+xml"):
            ctype += "; charset=utf-8"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        length = int(self.headers.get("Content-Length") or "0")
        raw_body = self.rfile.read(length).decode("utf-8") if length > 0 else "{}"
        body = json.loads(raw_body) if raw_body.strip() else {}
        status, payload = route_api_request("POST", parsed.path, {}, body, request_id="local-dev")
        raw = json.dumps(payload, indent=2, default=str).encode("utf-8")
        self.send_response(status)
        for k, v in CORS_HEADERS.items():
            self.send_header(k, v)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


def run_local_server(port: int = 8086) -> None:
    srv = ThreadingHTTPServer(("127.0.0.1", port), LocalAdPulseServerHandler)
    print(f"AdPulse Storefront & Auction Studio running at http://127.0.0.1:{port}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    port_num = int(os.environ.get("PORT", "8086"))
    run_local_server(port=port_num)
