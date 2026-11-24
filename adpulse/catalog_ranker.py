"""Two-Stage Sponsored Catalog Retrieval and Calibrated eCPM Ranker (`adpulse/catalog_ranker.py`).

Implements:
1. Deterministic 500-SKU multi-category Amazon catalog + 120-query shopper workload (`seed=42`).
2. `LexicalBidRanker` baseline: token-overlap retrieval ordered by raw CPC bid.
3. `TwoStageAdRanker`: inverted-index candidate generation with structured eligibility filtering
   (in-stock enforcement, shopper price ceiling, category alignment) followed by Empirical-Bayes
   calibrated expected-value scoring (`bid * pCTR * pCVR + relevance_bonus`) with p50/p99 latency tracking.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
import heapq
import math
import random
import re
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


TOKEN_RE = re.compile(r"[a-z0-9]+")
PRICE_INTENT_RE = re.compile(r"\b(?:under|below|max)\s*\$?\s*(\d+(?:\.\d+)?)\b", re.IGNORECASE)

STOPWORDS = frozenset(
    {
        "the",
        "and",
        "for",
        "with",
        "under",
        "below",
        "max",
        "inch",
        "pack",
        "new",
        "edition",
    }
)


CATEGORY_SPECS: Dict[str, Dict[str, Any]] = {
    "electronics_audio": {
        "brands": ("SoundCoreX", "Auralis", "NordWave", "KodaAudio", "VeloTone"),
        "product_types": (
            ("headphones", "wireless over ear headphones"),
            ("earbuds", "bluetooth noise cancelling earbuds"),
            ("speaker", "portable waterproof bluetooth speaker"),
            ("microphone", "usb condenser podcast microphone"),
        ),
        "attributes": {
            "connectivity": ("bluetooth_5_3", "usb_c", "multipoint_bt", "wired_aux"),
            "feature": ("anc", "low_latency", "spatial_audio", "waterproof_ipx7"),
            "color": ("matte_black", "arctic_white", "midnight_navy", "graphite"),
        },
        "price_range": (29.0, 279.0),
        "prior_ctr": 0.042,
        "prior_cvr": 0.115,
    },
    "kitchen_appliances": {
        "brands": ("BrewCraft", "ChefPulse", "CasaVero", "KonaMill", "CopperLeaf"),
        "product_types": (
            ("espresso", "stainless burr espresso machine"),
            ("air_fryer", "compact digital basket air fryer"),
            ("blender", "portable usb personal smoothie blender"),
            ("kettle", "gooseneck variable temp electric kettle"),
        ),
        "attributes": {
            "material": ("stainless_steel", "borosilicate_glass", "brushed_aluminum", "bpa_free"),
            "capacity": ("single_serve", "family_6qt", "compact_2qt", "carafe_1_2l"),
            "wattage": ("1200w", "1500w", "900w", "600w"),
        },
        "price_range": (34.0, 320.0),
        "prior_ctr": 0.039,
        "prior_cvr": 0.128,
    },
    "home_office": {
        "brands": ("ErgoLift", "DeskForge", "LumaWork", "AxisMount", "KeyCraft"),
        "product_types": (
            ("chair", "ergonomic mesh lumbar office chair"),
            ("keyboard", "wireless mechanical low profile keyboard"),
            ("monitor_arm", "heavy duty gas spring monitor arm"),
            ("desk_riser", "portable adjustable standing desk converter"),
        ),
        "attributes": {
            "connectivity": ("wireless_2_4g", "bluetooth_5_3", "usb_c", "mechanical_mount"),
            "material": ("breathable_mesh", "anodized_aluminum", "walnut_steel", "pbt_keycaps"),
            "color": ("matte_black", "space_gray", "warm_white", "charcoal"),
        },
        "price_range": (45.0, 360.0),
        "prior_ctr": 0.036,
        "prior_cvr": 0.108,
    },
    "sports_outdoors": {
        "brands": ("TrailPeak", "HydroVault", "SummitStride", "TerraCamp", "AeroFlex"),
        "product_types": (
            ("bottle", "stainless vacuum insulated water bottle"),
            ("lantern", "portable rechargeable led camping lantern"),
            ("yoga_mat", "non slip natural rubber yoga mat"),
            ("daypack", "ultralight waterproof hiking daypack"),
        ),
        "attributes": {
            "capacity": ("24oz", "32oz", "20l_pack", "6mm_cushion"),
            "feature": ("insulated_24h", "waterproof_ipx7", "ultralight", "solar_usb"),
            "color": ("forest_green", "matte_black", "canyon_orange", "alpine_blue"),
        },
        "price_range": (22.0, 185.0),
        "prior_ctr": 0.046,
        "prior_cvr": 0.142,
    },
    "pet_supplies": {
        "brands": ("PawHaven", "BarkWell", "FurNest", "KibbleSync", "PurePaws"),
        "product_types": (
            ("feeder", "smart wifi automaticステンレス pet feeder".replace("ステンレス", "stainless")),
            ("dog_bed", "orthopedic memory foam washable dog bed"),
            ("fountain", "ultra quiet stainless cat water fountain"),
            ("carrier", "airline approved portable soft pet carrier"),
        ),
        "attributes": {
            "size": ("small_breed", "medium_breed", "large_breed", "multi_pet"),
            "material": ("stainless_steel", "memory_foam", "oxford_mesh", "ceramic_top"),
            "feature": ("app_schedule", "washable_cover", "battery_backup", "airline_ready"),
        },
        "price_range": (24.0, 165.0),
        "prior_ctr": 0.048,
        "prior_cvr": 0.152,
    },
    "smart_home": {
        "brands": ("VoltNest", "LumaGrid", "SentryCam", "HomeLinkX", "AuraSense"),
        "product_types": (
            ("camera", "wireless indoor outdoor 2k security camera"),
            ("smart_plug", "compact matter wifi energy monitoring plug"),
            ("doorbell", "battery video doorbell chime kit"),
            ("light_strip", "smart dimmable rgbic led light strip"),
        ),
        "attributes": {
            "connectivity": ("wifi_6", "matter_thread", "zigbee_hub", "bluetooth_mesh"),
            "feature": ("night_vision", "energy_meter", "local_sd", "voice_control"),
            "color": ("arctic_white", "matte_black", "soft_gray", "warm_white"),
        },
        "price_range": (19.0, 210.0),
        "prior_ctr": 0.044,
        "prior_cvr": 0.131,
    },
    "beauty_personal_care": {
        "brands": ("VelvetGlow", "IonicPure", "DermaBotanics", "SonicDent", "SilkWave"),
        "product_types": (
            ("hair_dryer", "high speed ionic brushless hair dryer"),
            ("toothbrush", "rechargeable sonic travel electric toothbrush"),
            ("serum", "vitamin c hyaluronic brightening facial serum"),
            ("trimmer", "waterproof cordless precision beard trimmer"),
        ),
        "attributes": {
            "feature": ("ionic_frizz_free", "pressure_sensor", "fragrance_free", "travel_lock"),
            "skin_hair_type": ("sensitive", "curly_thick", "all_types", "fine_hair"),
            "color": ("rose_quartz", "pearl_white", "matte_black", "sage_green"),
        },
        "price_range": (18.0, 225.0),
        "prior_ctr": 0.045,
        "prior_cvr": 0.146,
    },
    "baby_gear": {
        "brands": ("TinyTrail", "CradleCalm", "CloudTot", "PureSprout", "NestGuard"),
        "product_types": (
            ("monitor", "split screen hd video baby monitor"),
            ("stroller", "compact one hand fold travel stroller"),
            ("sound_machine", "portable white noise nursery sound machine"),
            ("feeding_set", "silicone suction bowl toddler feeding set"),
        ),
        "attributes": {
            "material": ("food_grade_silicone", "organic_cotton", "lightweight_alloy", "bpa_free"),
            "feature": ("no_wifi_fhss", "cabin_fold", "cry_sensor", "dishwasher_safe"),
            "color": ("oatmeal_beige", "sage_green", "cloud_gray", "soft_blush"),
        },
        "price_range": (20.0, 290.0),
        "prior_ctr": 0.047,
        "prior_cvr": 0.149,
    },
}


def tokenize(text: str) -> List[str]:
    """Normalize text into lowercase alphanumeric tokens excluding generic stopwords."""
    normalized = text.lower().replace("_", " ").replace("-", " ")
    return [tok for tok in TOKEN_RE.findall(normalized) if tok not in STOPWORDS and len(tok) > 1]


def percentile(sorted_values: Sequence[float], pct: float) -> float:
    """Compute linear-interpolated percentile on a non-empty sorted sequence."""
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    pos = (pct / 100.0) * (len(sorted_values) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(sorted_values[lo])
    weight = pos - lo
    return float(sorted_values[lo] * (1.0 - weight) + sorted_values[hi] * weight)


@dataclass(frozen=True)
class CatalogSKU:
    sku_id: str
    campaign_id: str
    seller_id: str
    title: str
    category: str
    product_type: str
    brand: str
    attributes: Dict[str, str]
    required_attribute_keys: Tuple[str, ...]
    price: float
    margin: float
    base_bid: float
    in_stock: bool
    prior_ctr: float
    prior_cvr: float
    impressions_30d: int
    clicks_30d: int
    conversions_30d: int

    @property
    def missing_attributes(self) -> Tuple[str, ...]:
        return tuple(k for k in self.required_attribute_keys if not self.attributes.get(k))


@dataclass(frozen=True)
class ShopperQuery:
    query_id: str
    raw_query: str
    target_category: Optional[str]
    product_type_intent: str
    max_price: Optional[float]
    brand_intent: Optional[str]
    required_attributes: Dict[str, str]
    relevant_sku_ids: Tuple[str, ...]


@dataclass(frozen=True)
class RankedAdCandidate:
    sku_id: str
    campaign_id: str
    title: str
    category: str
    brand: str
    price: float
    in_stock: bool
    base_bid: float
    p_ctr: float
    p_cvr: float
    relevance_bonus: float
    score: float


@dataclass(frozen=True)
class RankerMetrics:
    ranker_name: str
    top3_recall: float
    ndcg_at_3: float
    precision_at_3: float
    out_of_stock_impression_rate_pct: float
    price_cap_violation_rate_pct: float
    category_mismatch_rate_pct: float
    expected_conversions_per_1k_queries: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float


@dataclass(frozen=True)
class RankerBenchmarkReport:
    catalog_size: int
    query_count: int
    baseline: RankerMetrics
    two_stage: RankerMetrics
    top3_recall_lift_pct: float
    ndcg_lift_pct: float
    expected_conversions_lift_pct: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def generate_synthetic_catalog_and_queries(
    num_skus: int = 500,
    num_queries: int = 120,
    seed: int = 42,
) -> Tuple[List[CatalogSKU], List[ShopperQuery]]:
    """Generate a deterministic 500-SKU multi-category catalog and 120 shopper queries.

    Includes realistic marketplace failure modes:
    - Out-of-stock SKUs with high stale bids (~16% of SKUs).
    - Premium/overpriced SKUs above shopper price ceilings with aggressive brand-defense bids.
    - Cross-category token collisions ('wireless', 'portable', 'stainless', 'smart', 'video')
      with high clearance bids but poor conversion rates.
    - Incomplete structured catalog attributes on a subset of SKUs.
    """
    rng = random.Random(seed)
    categories = list(CATEGORY_SPECS.keys())
    catalog: List[CatalogSKU] = []

    for idx in range(num_skus):
        sku_id = f"SKU-{idx + 1:04d}"
        campaign_id = f"CMP-{(idx % 40) + 1:03d}"
        seller_id = f"SLR-{(idx % 25) + 1:03d}"
        category = categories[idx % len(categories)]
        spec = CATEGORY_SPECS[category]
        brand = spec["brands"][(idx // len(categories)) % len(spec["brands"])]
        ptype_key, ptype_phrase = spec["product_types"][(idx // 3) % len(spec["product_types"])]

        attr_schema = spec["attributes"]
        req_keys = tuple(attr_schema.keys())
        attrs: Dict[str, str] = {}
        # ~15% of SKUs omit 1-2 structured attributes even when title has generic keywords.
        drop_attrs = (idx % 7 == 0)
        for a_idx, (k, choices) in enumerate(attr_schema.items()):
            val = choices[(idx + a_idx * 3) % len(choices)]
            if drop_attrs and a_idx >= 1:
                continue
            attrs[k] = val

        lo_p, hi_p = spec["price_range"]
        # Create structured price tiers: value, mid-tier, and premium/high-price
        tier = idx % 5
        if tier in (0, 1):
            price_frac = rng.uniform(0.08, 0.42)
        elif tier in (2, 3):
            price_frac = rng.uniform(0.38, 0.68)
        else:
            price_frac = rng.uniform(0.75, 1.00)
        price = round(lo_p + (hi_p - lo_p) * price_frac, 2)
        margin = round(rng.uniform(0.20, 0.50), 3)

        # Stock status: ~16% out-of-stock, deliberately affecting some high-bidding SKUs
        in_stock = not (idx % 6 == 0 and idx % 18 != 0)

        # Archetypes:
        # - High-bid low-relevance / clearance distractors (aggressive bid, weak CVR)
        # - High-converting well-matched SKUs (moderate sensible bid, strong CTR/CVR)
        # - Overpriced luxury SKUs with high raw bids
        is_clearance_bidding_aggressively = (idx % 4 == 1)
        if not in_stock:
            # Stale unpaused campaign with high historical bid
            base_bid = round(rng.uniform(2.40, 4.10), 2)
            true_ctr = round(spec["prior_ctr"] * rng.uniform(0.75, 1.15), 4)
            true_cvr = round(spec["prior_cvr"] * rng.uniform(0.65, 1.05), 4)
        elif tier == 4:
            # Expensive SKU with high AOV-driven bid
            base_bid = round(rng.uniform(2.50, 4.25), 2)
            true_ctr = round(spec["prior_ctr"] * rng.uniform(0.55, 0.85), 4)
            true_cvr = round(spec["prior_cvr"] * rng.uniform(0.35, 0.65), 4)
        elif is_clearance_bidding_aggressively:
            # Aggressive raw bid trying to buy impressions despite low conversion quality
            base_bid = round(rng.uniform(2.20, 3.85), 2)
            true_ctr = round(spec["prior_ctr"] * rng.uniform(0.42, 0.68), 4)
            true_cvr = round(spec["prior_cvr"] * rng.uniform(0.22, 0.44), 4)
        else:
            # Healthy value/mid-tier catalog item with strong organic conversion rate
            base_bid = round(rng.uniform(0.85, 2.05), 2)
            true_ctr = round(spec["prior_ctr"] * rng.uniform(1.05, 1.55), 4)
            true_cvr = round(spec["prior_cvr"] * rng.uniform(1.15, 1.75), 4)

        impressions_30d = rng.randint(1400, 18000)
        clicks_30d = max(1, int(round(impressions_30d * true_ctr * rng.uniform(0.92, 1.08))))
        conversions_30d = max(0, min(clicks_30d, int(round(clicks_30d * true_cvr * rng.uniform(0.90, 1.10)))))

        attr_words = " ".join(v.replace("_", " ") for v in attrs.values())
        series_tag = "pro portable wireless smart" if is_clearance_bidding_aggressively else "pro series"
        title = f"{brand} {ptype_phrase} {attr_words} {series_tag}".strip()

        catalog.append(
            CatalogSKU(
                sku_id=sku_id,
                campaign_id=campaign_id,
                seller_id=seller_id,
                title=title,
                category=category,
                product_type=ptype_key,
                brand=brand,
                attributes=attrs,
                required_attribute_keys=req_keys,
                price=price,
                margin=margin,
                base_bid=base_bid,
                in_stock=in_stock,
                prior_ctr=true_ctr,
                prior_cvr=true_cvr,
                impressions_30d=impressions_30d,
                clicks_30d=clicks_30d,
                conversions_30d=conversions_30d,
            )
        )

    # Build 120 shopper queries with ground-truth relevant in-stock affordable SKUs
    skus_by_cat_ptype: Dict[Tuple[str, str], List[CatalogSKU]] = defaultdict(list)
    for sku in catalog:
        skus_by_cat_ptype[(sku.category, sku.product_type)].append(sku)

    cat_ptype_pairs = sorted(skus_by_cat_ptype.keys())
    queries: List[ShopperQuery] = []

    for q_idx in range(num_queries):
        category, ptype_key = cat_ptype_pairs[q_idx % len(cat_ptype_pairs)]
        spec = CATEGORY_SPECS[category]
        ptype_phrase = dict(spec["product_types"])[ptype_key]
        pool = skus_by_cat_ptype[(category, ptype_key)]

        lo_p, hi_p = spec["price_range"]
        # Price cap around 68%-78% of category range so overpriced luxury SKUs exceed ceiling
        raw_cap = lo_p + (hi_p - lo_p) * rng.uniform(0.66, 0.78)
        max_price = float(int(math.ceil(raw_cap / 5.0) * 5))

        # Eligible high-converting in-stock SKUs under price ceiling
        viable = [
            s
            for s in pool
            if s.in_stock and s.price <= max_price and s.prior_cvr >= spec["prior_cvr"] * 0.85
        ]
        if len(viable) < 3:
            max_price = float(int(math.ceil(hi_p / 5.0) * 5))
            viable = [s for s in pool if s.in_stock and s.price <= max_price]

        anchor = viable[q_idx % len(viable)]
        req_attrs: Dict[str, str] = {}
        first_attr_key = spec[" required_key"] if " required_key" in spec else next(iter(spec["attributes"].keys()))
        if first_attr_key in anchor.attributes and (q_idx % 3 != 2):
            req_attrs[first_attr_key] = anchor.attributes[first_attr_key]

        brand_intent: Optional[str] = anchor.brand if (q_idx % 4 == 0) else None

        # Ground-truth relevant set: in-stock, right category & product_type, <= max_price,
        # above median conversion quality, and matching required attributes if any (with fallback)
        strict_relevant = [
            s
            for s in viable
            if all(s.attributes.get(k) == v for k, v in req_attrs.items())
            and (brand_intent is None or s.brand == brand_intent)
        ]
        if len(strict_relevant) < 3:
            # Relax brand intent into soft preference for ground-truth top-3 set if <3 exact brand SKUs
            strict_relevant = [
                s for s in viable if all(s.attributes.get(k) == v for k, v in req_attrs.items())
            ]
        if len(strict_relevant) < 3:
            strict_relevant = list(viable)

        # Rank ground-truth relevant set by true expected conversion utility
        strict_relevant.sort(
            key=lambda s: (
                1 if (brand_intent and s.brand == brand_intent) else 0,
                sum(1 for k, v in req_attrs.items() if s.attributes.get(k) == v),
                s.prior_ctr * s.prior_cvr,
            ),
            reverse=True,
        )
        relevant_ids = tuple(s.sku_id for s in strict_relevant[:6])

        query_parts: List[str] = []
        if brand_intent:
            query_parts.append(brand_intent.lower())
        query_parts.append(ptype_phrase)
        for val in req_attrs.values():
            query_parts.append(val.replace("_", " "))
        query_parts.append(f"under {int(max_price)}")
        raw_query = " ".join(query_parts)

        queries.append(
            ShopperQuery(
                query_id=f"Q-{q_idx + 1:03d}",
                raw_query=raw_query,
                target_category=category,
                product_type_intent=ptype_key,
                max_price=max_price,
                brand_intent=brand_intent,
                required_attributes=req_attrs,
                relevant_sku_ids=relevant_ids,
            )
        )

    return catalog, queries


class LexicalBidRanker:
    """Baseline ad retriever: lexical token-overlap proportion weighted by raw CPC bid."""

    def __init__(self, catalog: Sequence[CatalogSKU]) -> None:
        self.catalog: List[CatalogSKU] = list(catalog)
        self._sku_tokens: List[frozenset[str]] = [
            frozenset(tokenize(sku.title)) for sku in self.catalog
        ]

    def rank(self, query: ShopperQuery, top_k: int = 3) -> List[RankedAdCandidate]:
        q_tokens = frozenset(t for t in tokenize(query.raw_query) if not t.isdigit())
        if not q_tokens:
            q_tokens = frozenset(tokenize(query.raw_query))
        q_len = max(1, len(q_tokens))

        matches: List[Tuple[float, int, CatalogSKU]] = []
        for idx, sku in enumerate(self.catalog):
            overlap = len(q_tokens & self._sku_tokens[idx])
            if overlap == 0:
                continue
            # Standard uncalibrated lexical*bid baseline: token overlap fraction * raw CPC bid
            overlap_ratio = overlap / q_len
            score = overlap_ratio * sku.base_bid
            matches.append((score, overlap, sku))

        top_matches = heapq.nlargest(top_k, matches, key=lambda item: (item[0], item[1]))
        return [
            RankedAdCandidate(
                sku_id=sku.sku_id,
                campaign_id=sku.campaign_id,
                title=sku.title,
                category=sku.category,
                brand=sku.brand,
                price=sku.price,
                in_stock=sku.in_stock,
                base_bid=sku.base_bid,
                p_ctr=sku.prior_ctr,
                p_cvr=sku.prior_cvr,
                relevance_bonus=0.0,
                score=round(score, 6),
            )
            for score, _, sku in top_matches
        ]


class TwoStageAdRanker:
    """Two-stage inverted-index retrieval + structured attribute filter + calibrated eCPM ranker.

    Stage 1 (Retrieval & Hard Eligibility Filter):
      - Inverted posting list lookup over title, brand, category, product_type, and attribute tokens.
      - Strictly suppresses out-of-stock SKUs (`in_stock == False`).
      - Enforces shopper price ceiling (`price <= max_price` from structured field or query text).
      - Enforces category alignment when shopper query resolves to a target category.
      - Selects top `stage1_limit` candidates via IDF-weighted lexical + structured attribute score.

    Stage 2 (Calibrated Expected-Value Reranker):
      - Computes Empirical-Bayes smoothed `pCTR` and `pCVR` shrunk toward category Beta priors,
        adjusted by exact product-type, brand, and structured attribute alignment.
      - Ranks candidates by `bid * pCTR * pCVR + relevance_bonus`.
    """

    def __init__(
        self,
        catalog: Sequence[CatalogSKU],
        stage1_limit: int = 40,
        relevance_weight: float = 0.0085,
    ) -> None:
        self.catalog: List[CatalogSKU] = list(catalog)
        self.stage1_limit = stage1_limit
        self.relevance_weight = relevance_weight

        self._inverted_index: Dict[str, List[int]] = defaultdict(list)
        self._doc_tokens: List[frozenset[str]] = []
        self._title_tokens: List[frozenset[str]] = []
        self._idf: Dict[str, float] = {}
        self._cat_priors: Dict[str, Tuple[float, float, float, float]] = {}

        self._build_indices_and_priors()

    def _build_indices_and_priors(self) -> None:
        n_docs = max(1, len(self.catalog))
        df_counts: Dict[str, int] = defaultdict(int)

        cat_imp: Dict[str, int] = defaultdict(int)
        cat_clk: Dict[str, int] = defaultdict(int)
        cat_conv: Dict[str, int] = defaultdict(int)

        for idx, sku in enumerate(self.catalog):
            title_toks = frozenset(tokenize(sku.title))
            attr_blob = " ".join(f"{k} {v}" for k, v in sku.attributes.items())
            full_toks = frozenset(
                tokenize(f"{sku.title} {sku.brand} {sku.category} {sku.product_type} {attr_blob}")
            )
            self._title_tokens.append(title_toks)
            self._doc_tokens.append(full_toks)

            for tok in full_toks:
                self._inverted_index[tok].append(idx)
                df_counts[tok] += 1

            cat_imp[sku.category] += sku.impressions_30d
            cat_clk[sku.category] += sku.clicks_30d
            cat_conv[sku.category] += sku.conversions_30d

        for tok, df in df_counts.items():
            self._idf[tok] = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))

        # Empirical-Bayes Beta prior pseudo-counts per category
        ctr_strength = 600.0
        cvr_strength = 45.0
        for cat in set(s.category for s in self.catalog):
            mean_ctr = cat_clk[cat] / max(1, cat_imp[cat])
            mean_cvr = cat_conv[cat] / max(1, cat_clk[cat])
            alpha_ctr = mean_ctr * ctr_strength
            beta_ctr = (1.0 - mean_ctr) * ctr_strength
            alpha_cvr = mean_cvr * cvr_strength
            beta_cvr = (1.0 - mean_cvr) * cvr_strength
            self._cat_priors[cat] = (alpha_ctr, beta_ctr, alpha_cvr, beta_cvr)

    @staticmethod
    def _resolve_price_cap(query: ShopperQuery) -> Optional[float]:
        if query.max_price is not None:
            return query.max_price
        match = PRICE_INTENT_RE.search(query.raw_query)
        if match:
            return float(match.group(1))
        return None

    def retrieve_stage1(self, query: ShopperQuery) -> List[Tuple[float, int]]:
        """Stage 1 inverted-index retrieval + structured eligibility filtering."""
        q_tokens = tokenize(query.raw_query)
        price_cap = self._resolve_price_cap(query)
        accum_scores: Dict[int, float] = defaultdict(float)

        for tok in set(q_tokens):
            idf = self._idf.get(tok, 0.0)
            if idf <= 0.0:
                continue
            for doc_idx in self._inverted_index.get(tok, ()):
                accum_scores[doc_idx] += idf

        eligible: List[Tuple[float, int]] = []
        brand_lower = query.brand_intent.lower() if query.brand_intent else None

        for doc_idx, lex_score in accum_scores.items():
            sku = self.catalog[doc_idx]
            # Hard Filter 1: Out-of-stock suppression
            if not sku.in_stock:
                continue
            # Hard Filter 2: Shopper price ceiling
            if price_cap is not None and sku.price > price_cap:
                continue
            # Hard Filter 3: Structured category match
            if query.target_category and sku.category != query.target_category:
                continue

            struct_boost = 0.0
            if query.product_type_intent and sku.product_type == query.product_type_intent:
                struct_boost += 3.0
            if brand_lower and sku.brand.lower() == brand_lower:
                struct_boost += 2.2
            for req_k, req_v in query.required_attributes.items():
                if sku.attributes.get(req_k) == req_v:
                    struct_boost += 1.8
                elif req_k not in sku.attributes:
                    struct_boost -= 0.6

            eligible.append((lex_score + struct_boost, doc_idx))

        return heapq.nlargest(self.stage1_limit, eligible, key=lambda pair: pair[0])

    def _calibrated_rates_and_relevance(
        self,
        sku: CatalogSKU,
        doc_idx: int,
        query: ShopperQuery,
        q_token_set: frozenset[str],
        price_cap: Optional[float],
    ) -> Tuple[float, float, float]:
        alpha_ctr, beta_ctr, alpha_cvr, beta_cvr = self._cat_priors[sku.category]
        eb_ctr = (sku.clicks_30d + alpha_ctr) / (sku.impressions_30d + alpha_ctr + beta_ctr)
        eb_cvr = (sku.conversions_30d + alpha_cvr) / (sku.clicks_30d + alpha_cvr + beta_cvr)

        # Query-SKU match features
        non_numeric_q = frozenset(t for t in q_token_set if not t.isdigit())
        lex_recall = (
            len(non_numeric_q & self._doc_tokens[doc_idx]) / max(1, len(non_numeric_q))
        )
        ptype_match = 1.0 if (not query.product_type_intent or sku.product_type == query.product_type_intent) else 0.35
        brand_match = (
            1.22
            if (query.brand_intent and sku.brand.lower() == query.brand_intent.lower())
            else (0.94 if query.brand_intent else 1.0)
        )

        if query.required_attributes:
            matched_attrs = sum(
                1 for k, v in query.required_attributes.items() if sku.attributes.get(k) == v
            )
            attr_ratio = matched_attrs / len(query.required_attributes)
            attr_mult = 0.72 + 0.43 * attr_ratio
        else:
            attr_ratio = 1.0 - 0.15 * len(sku.missing_attributes)
            attr_mult = max(0.80, attr_ratio)

        price_fit = 1.0
        if price_cap is not None and price_cap > 0:
            headroom = max(0.0, (price_cap - sku.price) / price_cap)
            price_fit = 0.92 + 0.18 * min(1.0, headroom * 2.0)

        p_ctr = min(0.45, max(0.001, eb_ctr * ptype_match * brand_match * (0.75 + 0.35 * lex_recall)))
        p_cvr = min(0.55, max(0.002, eb_cvr * ptype_match * attr_mult * price_fit))

        structured_quality = (
            0.40 * lex_recall
            + 0.30 * (1.0 if ptype_match >= 1.0 else 0.0)
            + 0.20 * max(0.0, min(1.0, attr_ratio))
            + 0.10 * (1.0 if brand_match > 1.0 else 0.5)
        )
        relevance_bonus = self.relevance_weight * structured_quality
        return p_ctr, p_cvr, relevance_bonus

    def rank(self, query: ShopperQuery, top_k: int = 3) -> List[RankedAdCandidate]:
        """Execute Stage 1 retrieval + Stage 2 calibrated eCPM ranking."""
        stage1 = self.retrieve_stage1(query)
        if not stage1:
            return []

        q_token_set = frozenset(tokenize(query.raw_query))
        price_cap = self._resolve_price_cap(query)
        scored: List[Tuple[float, RankedAdCandidate]] = []

        for _, doc_idx in stage1:
            sku = self.catalog[doc_idx]
            p_ctr, p_cvr, rel_bonus = self._calibrated_rates_and_relevance(
                sku, doc_idx, query, q_token_set, price_cap
            )
            score = (sku.base_bid * p_ctr * p_cvr) + rel_bonus
            scored.append(
                (
                    score,
                    RankedAdCandidate(
                        sku_id=sku.sku_id,
                        campaign_id=sku.campaign_id,
                        title=sku.title,
                        category=sku.category,
                        brand=sku.brand,
                        price=sku.price,
                        in_stock=sku.in_stock,
                        base_bid=sku.base_bid,
                        p_ctr=round(p_ctr, 5),
                        p_cvr=round(p_cvr, 5),
                        relevance_bonus=round(rel_bonus, 6),
                        score=round(score, 6),
                    ),
                )
            )

        top_scored = heapq.nlargest(top_k, scored, key=lambda item: item[0])
        return [cand for _, cand in top_scored]


def _evaluate_ranker(
    ranker_name: str,
    rank_fn: Any,
    catalog_by_id: Dict[str, CatalogSKU],
    queries: Sequence[ShopperQuery],
    top_k: int = 3,
    timing_passes: int = 5,
) -> RankerMetrics:
    latencies_ms: List[float] = []
    results_per_query: List[List[RankedAdCandidate]] = []

    for pass_idx in range(timing_passes):
        for q_idx, query in enumerate(queries):
            t0 = time.perf_counter()
            ranked = rank_fn(query, top_k=top_k)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            latencies_ms.append(elapsed_ms)
            if pass_idx == 0:
                results_per_query.append(ranked)

    recalls: List[float] = []
    ndcgs: List[float] = []
    precisions: List[float] = []
    total_shown = 0
    oos_shown = 0
    price_violations = 0
    cat_mismatches = 0
    expected_conversions = 0.0

    position_discounts = (1.0, 0.72, 0.55)

    for query, ranked in zip(queries, results_per_query):
        rel_set = set(query.relevant_sku_ids)
        denom = min(top_k, len(rel_set))
        hits = sum(1 for cand in ranked[:top_k] if cand.sku_id in rel_set)
        recalls.append(hits / max(1, denom))
        precisions.append(hits / max(1, top_k))

        dcg = 0.0
        for rank_idx, cand in enumerate(ranked[:top_k]):
            if cand.sku_id in rel_set:
                dcg += 1.0 / math.log2(rank_idx + 2.0)
        idcg = sum(1.0 / math.log2(i + 2.0) for i in range(denom))
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)

        for rank_idx, cand in enumerate(ranked[:top_k]):
            total_shown += 1
            sku = catalog_by_id[cand.sku_id]
            pos_mult = position_discounts[rank_idx] if rank_idx < len(position_discounts) else 0.4
            if not sku.in_stock:
                oos_shown += 1
                continue
            if query.max_price is not None and sku.price > query.max_price:
                price_violations += 1
            if query.target_category and sku.category != query.target_category:
                cat_mismatches += 1

            # True expected conversion contribution on ground-truth SKU rates
            relevance_mult = 1.0 if cand.sku_id in rel_set else (0.35 if sku.category == query.target_category else 0.08)
            expected_conversions += pos_mult * sku.prior_ctr * sku.prior_cvr * relevance_mult

    latencies_ms.sort()
    n_q = max(1, len(queries))
    n_imp = max(1, total_shown)

    return RankerMetrics(
        ranker_name=ranker_name,
        top3_recall=round(sum(recalls) / n_q, 4),
        ndcg_at_3=round(sum(ndcgs) / n_q, 4),
        precision_at_3=round(sum(precisions) / n_q, 4),
        out_of_stock_impression_rate_pct=round(100.0 * oos_shown / n_imp, 2),
        price_cap_violation_rate_pct=round(100.0 * price_violations / n_imp, 2),
        category_mismatch_rate_pct=round(100.0 * cat_mismatches / n_imp, 2),
        expected_conversions_per_1k_queries=round(1000.0 * expected_conversions / n_q, 2),
        p50_latency_ms=round(percentile(latencies_ms, 50.0), 3),
        p95_latency_ms=round(percentile(latencies_ms, 95.0), 3),
        p99_latency_ms=round(percentile(latencies_ms, 99.0), 3),
    )


def run_ranker_benchmark(
    num_skus: int = 500,
    num_queries: int = 120,
    seed: int = 42,
) -> RankerBenchmarkReport:
    """Run deterministic comparison between LexicalBidRanker and TwoStageAdRanker."""
    catalog, queries = generate_synthetic_catalog_and_queries(
        num_skus=num_skus, num_queries=num_queries, seed=seed
    )
    catalog_by_id = {s.sku_id: s for s in catalog}

    baseline_ranker = LexicalBidRanker(catalog)
    two_stage_ranker = TwoStageAdRanker(catalog)

    base_metrics = _evaluate_ranker(
        "lexical_raw_bid_baseline", baseline_ranker.rank, catalog_by_id, queries
    )
    two_stage_metrics = _evaluate_ranker(
        "two_stage_calibrated_ecpm", two_stage_ranker.rank, catalog_by_id, queries
    )

    recall_lift = (
        100.0 * (two_stage_metrics.top3_recall - base_metrics.top3_recall) / max(1e-9, base_metrics.top3_recall)
    )
    ndcg_lift = (
        100.0 * (two_stage_metrics.ndcg_at_3 - base_metrics.ndcg_at_3) / max(1e-9, base_metrics.ndcg_at_3)
    )
    conv_lift = (
        100.0
        * (
            two_stage_metrics.expected_conversions_per_1k_queries
            - base_metrics.expected_conversions_per_1k_queries
        )
        / max(1e-9, base_metrics.expected_conversions_per_1k_queries)
    )

    return RankerBenchmarkReport(
        catalog_size=len(catalog),
        query_count=len(queries),
        baseline=base_metrics,
        two_stage=two_stage_metrics,
        top3_recall_lift_pct=round(recall_lift, 2),
        ndcg_lift_pct=round(ndcg_lift, 2),
        expected_conversions_lift_pct=round(conv_lift, 2),
    )
