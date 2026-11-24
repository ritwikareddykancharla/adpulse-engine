"""Seller Campaign Diagnostic & ACOS-Bounded Recommendation Engine (`adpulse/campaign_diagnostics.py`).

Analyzes auction loss logs in SQLite alongside SKU catalog completeness, inventory status,
budget depletion hour, and seller target ACOS/ROAS ceilings to classify root-cause loss modes:
  - `OUT_OF_STOCK_SUPPRESSED`
  - `BUDGET_EXHAUSTED_EARLY`
  - `MISSING_CATALOG_ATTRIBUTES`
  - `LOW_QUERY_RELEVANCE`
  - `BID_BELOW_CLEARING_FLOOR`

Every bid recommendation is strictly bounded by `max_acos_cpc = price * calibrated_cvr * target_acos`
so sellers never receive unprofitable bid hike advice that violates their target ACOS/ROAS.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import random
import sqlite3
from typing import Any, Dict, List, Optional, Sequence, Tuple


class IssueCode(str, Enum):
    OUT_OF_STOCK_SUPPRESSED = "OUT_OF_STOCK_SUPPRESSED"
    BUDGET_EXHAUSTED_EARLY = "BUDGET_EXHAUSTED_EARLY"
    MISSING_CATALOG_ATTRIBUTES = "MISSING_CATALOG_ATTRIBUTES"
    LOW_QUERY_RELEVANCE = "LOW_QUERY_RELEVANCE"
    BID_BELOW_CLEARING_FLOOR = "BID_BELOW_CLEARING_FLOOR"
    HEALTHY_OPTIMIZED = "HEALTHY_OPTIMIZED"


@dataclass(frozen=True)
class AuctionLossTelemetryRow:
    campaign_id: str
    sku_id: str
    hour: int
    outcome: str  # WON, LOST_OOS, LOST_BUDGET, LOST_ATTR_FILTER, LOST_LOW_RELEVANCE, LOST_BID_FLOOR
    clearing_cpc: float
    query_token_overlap: float


@dataclass(frozen=True)
class CampaignDiagnosticInput:
    campaign_id: str
    sku_id: str
    category: str
    sku_price: float
    gross_margin: float
    current_bid: float
    daily_budget: float
    spent_today: float
    depleted_hour: Optional[int]
    in_stock: bool
    missing_attributes: Tuple[str, ...]
    calibrated_ctr: float
    calibrated_cvr: float
    target_acos: float  # e.g. 0.25 => 25% ACOS ceiling (4.0x minimum ROAS)
    expected_primary_issue: Optional[str] = None

    @property
    def target_roas_floor(self) -> float:
        return round(1.0 / max(1e-6, self.target_acos), 3)

    @property
    def max_acos_cpc(self) -> float:
        """Maximum CPC bid compatible with seller's target ACOS ceiling."""
        effective_acos_cap = min(self.target_acos, self.gross_margin)
        return round(self.sku_price * self.calibrated_cvr * effective_acos_cap, 2)


@dataclass(frozen=True)
class CampaignDiagnosticReport:
    campaign_id: str
    sku_id: str
    primary_issue: str
    secondary_issues: Tuple[str, ...]
    lost_impression_share_pct: float
    dominant_loss_bucket: str
    median_clearing_cpc: float
    current_bid: float
    recommended_bid: float
    max_acos_cpc: float
    target_acos: float
    implied_expected_acos: float
    recommended_daily_budget: float
    enable_pid_pacing: bool
    catalog_fixes: Tuple[str, ...]
    action_summary: str


@dataclass(frozen=True)
class DiagnosticsBenchmarkReport:
    scenarios_evaluated: int
    primary_issue_accuracy_pct: float
    acos_ceiling_compliance_pct: float
    oos_zero_bid_hike_compliance_pct: float
    issue_distribution: Dict[str, int]
    sample_reports: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CampaignDiagnostics:
    """SQLite-backed auction loss aggregator and ACOS-bounded seller diagnostic advisor."""

    def __init__(self) -> None:
        self._conn: Optional[sqlite3.Connection] = sqlite3.connect(":memory:")
        self._init_schema()

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> "CampaignDiagnostics":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()

    def _require_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("CampaignDiagnostics SQLite connection is closed")
        return self._conn

    def _init_schema(self) -> None:
        conn = self._require_conn()
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS auction_telemetry (
                campaign_id TEXT NOT NULL,
                sku_id TEXT NOT NULL,
                hour INTEGER NOT NULL,
                outcome TEXT NOT NULL,
                clearing_cpc REAL NOT NULL,
                query_token_overlap REAL NOT NULL
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_telemetry_cmp_sku ON auction_telemetry (campaign_id, sku_id)"
        )

    def load_telemetry(self, rows: Sequence[AuctionLossTelemetryRow]) -> None:
        conn = self._require_conn()
        conn.execute("DELETE FROM auction_telemetry")
        conn.executemany(
            """
            INSERT INTO auction_telemetry (
                campaign_id, sku_id, hour, outcome, clearing_cpc, query_token_overlap
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    r.campaign_id,
                    r.sku_id,
                    r.hour,
                    r.outcome,
                    r.clearing_cpc,
                    r.query_token_overlap,
                )
                for r in rows
            ],
        )
        conn.commit()

    def _query_loss_breakdown(
        self, campaign_id: str, sku_id: str
    ) -> Tuple[Dict[str, int], float, float]:
        conn = self._require_conn()
        cur = conn.execute(
            """
            SELECT outcome, COUNT(*), AVG(clearing_cpc), AVG(query_token_overlap)
            FROM auction_telemetry
            WHERE campaign_id = ? AND sku_id = ?
            GROUP BY outcome
            """,
            (campaign_id, sku_id),
        )
        counts: Dict[str, int] = {}
        weighted_cpc = 0.0
        weighted_overlap = 0.0
        total_n = 0
        for outcome, cnt, avg_cpc, avg_ov in cur.fetchall():
            counts[str(outcome)] = int(cnt)
            weighted_cpc += float(avg_cpc or 0.0) * int(cnt)
            weighted_overlap += float(avg_ov or 0.0) * int(cnt)
            total_n += int(cnt)

        if total_n == 0:
            return {}, 0.0, 1.0
        return counts, weighted_cpc / total_n, weighted_overlap / total_n

    def diagnose(self, item: CampaignDiagnosticInput) -> CampaignDiagnosticReport:
        """Diagnose primary/secondary auction loss causes and compute ACOS-bounded actions."""
        counts, mean_clearing_cpc, mean_overlap = self._query_loss_breakdown(
            item.campaign_id, item.sku_id
        )
        total_auctions = max(1, sum(counts.values()))
        won = counts.get("WON", 0)
        lost_total = total_auctions - won
        lost_share_pct = round(100.0 * lost_total / total_auctions, 2)

        loss_only = {k: v for k, v in counts.items() if k != "WON" and v > 0}
        dominant_bucket = (
            max(loss_only.items(), key=lambda kv: kv[1])[0] if loss_only else "NONE"
        )

        max_cpc = item.max_acos_cpc
        detected: List[IssueCode] = []

        # Priority 1: Out-of-stock suppression (never raise bids on unavailable inventory)
        if not item.in_stock or counts.get("LOST_OOS", 0) >= max(5, int(0.25 * total_auctions)):
            detected.append(IssueCode.OUT_OF_STOCK_SUPPRESSED)

        # Priority 2: Structured catalog attribute gaps causing Stage-1 filter drops
        if item.missing_attributes and (
            counts.get("LOST_ATTR_FILTER", 0) >= max(5, int(0.20 * total_auctions))
            or len(item.missing_attributes) >= 2
        ):
            detected.append(IssueCode.MISSING_CATALOG_ATTRIBUTES)

        # Priority 3: Early budget exhaustion before prime evening conversion window (18:00)
        if (
            item.depleted_hour is not None and item.depleted_hour < 18
        ) or counts.get("LOST_BUDGET", 0) >= max(8, int(0.24 * total_auctions)):
            detected.append(IssueCode.BUDGET_EXHAUSTED_EARLY)

        # Priority 4: Low lexical/intent relevance
        if (
            counts.get("LOST_LOW_RELEVANCE", 0) >= max(8, int(0.24 * total_auctions))
            or mean_overlap < 0.34
        ):
            detected.append(IssueCode.LOW_QUERY_RELEVANCE)

        # Priority 5: Bid below clearing floor
        if (
            counts.get("LOST_BID_FLOOR", 0) >= max(8, int(0.24 * total_auctions))
            or (mean_clearing_cpc > 0.0 and item.current_bid < mean_clearing_cpc * 0.90)
        ):
            detected.append(IssueCode.BID_BELOW_CLEARING_FLOOR)

        if not detected:
            detected.append(IssueCode.HEALTHY_OPTIMIZED)

        primary = detected[0]
        secondary = tuple(code.value for code in detected[1:])

        # Compute ACOS-bounded bid & catalog/pacing recommendations
        recommended_bid = min(item.current_bid, max_cpc)
        recommended_budget = item.daily_budget
        enable_pid_pacing = False
        catalog_fixes: List[str] = []

        if primary == IssueCode.OUT_OF_STOCK_SUPPRESSED:
            # Never increase bid on out-of-stock SKUs; cap at current or max_cpc
            recommended_bid = round(min(item.current_bid, max_cpc), 2)
            catalog_fixes.append(
                f"Restock FBA inventory for {item.sku_id} before resuming bid adjustments."
            )
            action_summary = (
                f"Hold bid at ${recommended_bid:.2f}; SKU is out-of-stock suppressed "
                f"({lost_share_pct:.1f}% impressions blocked). Replenish inventory first."
            )
        elif primary == IssueCode.MISSING_CATALOG_ATTRIBUTES:
            missing_list = ", ".join(item.missing_attributes)
            catalog_fixes.append(
                f"Backfill structured catalog attributes [{missing_list}] in Seller Central."
            )
            recommended_bid = round(min(item.current_bid, max_cpc), 2)
            action_summary = (
                f"Populate missing attributes ({missing_list}) to recover structured filter "
                f"eligibility while keeping bid <= ${max_cpc:.2f} ACOS cap."
            )
        elif primary == IssueCode.BUDGET_EXHAUSTED_EARLY:
            enable_pid_pacing = True
            shaded = min(item.current_bid * 0.82, max_cpc)
            recommended_bid = round(max(0.15, shaded), 2)
            # Only recommend modest budget expansion if unit economics comfortably beat ACOS cap
            if item.current_bid <= max_cpc * 0.95:
                recommended_budget = round(item.daily_budget * 1.25, 2)
            dep_str = f"{item.depleted_hour:02d}:00" if item.depleted_hour is not None else "pre-peak"
            action_summary = (
                f"Enable Lagrangian PID bid-shading (base bid ${recommended_bid:.2f} <= ${max_cpc:.2f} "
                f"ACOS cap) to eliminate {dep_str} budget depletion and capture 18:00-22:00 peak CVR."
            )
        elif primary == IssueCode.LOW_QUERY_RELEVANCE:
            recommended_bid = round(min(item.current_bid * 0.90, max_cpc), 2)
            catalog_fixes.append(
                "Add negative exact/phrase keywords for non-converting broad terms and align product_type title tokens."
            )
            action_summary = (
                f"Prune low-overlap broad queries (avg overlap {mean_overlap:.2f}) and cap CPC bid at "
                f"${recommended_bid:.2f} (ACOS ceiling ${max_cpc:.2f})."
            )
        elif primary == IssueCode.BID_BELOW_CLEARING_FLOOR:
            target_clearing_bid = round(mean_clearing_cpc * 1.06, 2)
            if target_clearing_bid <= max_cpc:
                recommended_bid = target_clearing_bid
                action_summary = (
                    f"Raise bid from ${item.current_bid:.2f} to ${recommended_bid:.2f} to clear "
                    f"median auction floor (${mean_clearing_cpc:.2f}) within ${max_cpc:.2f} ACOS cap."
                )
            else:
                recommended_bid = max_cpc
                catalog_fixes.append(
                    "Clearing CPC exceeds ACOS ceiling; improve detail-page CVR (images/A+ content) instead of overbidding."
                )
                action_summary = (
                    f"Cap bid at ${recommended_bid:.2f} (ACOS ceiling) rather than chasing ${mean_clearing_cpc:.2f} "
                    f"clearing floor; lift SKU CVR to unlock higher profitable CPC headroom."
                )
        else:
            recommended_bid = round(min(item.current_bid, max_cpc), 2)
            enable_pid_pacing = True
            action_summary = (
                f"Campaign healthy; maintain PID pacing and ${recommended_bid:.2f} bid within ${max_cpc:.2f} ACOS ceiling."
            )

        # Final hard safety invariant: recommended_bid <= max_acos_cpc
        recommended_bid = round(min(recommended_bid, max_cpc), 2)
        expected_sales_per_click = max(1e-6, item.sku_price * item.calibrated_cvr)
        implied_acos = round(recommended_bid / expected_sales_per_click, 4)

        return CampaignDiagnosticReport(
            campaign_id=item.campaign_id,
            sku_id=item.sku_id,
            primary_issue=primary.value,
            secondary_issues=secondary,
            lost_impression_share_pct=lost_share_pct,
            dominant_loss_bucket=dominant_bucket,
            median_clearing_cpc=round(mean_clearing_cpc, 2),
            current_bid=item.current_bid,
            recommended_bid=recommended_bid,
            max_acos_cpc=max_cpc,
            target_acos=item.target_acos,
            implied_expected_acos=implied_acos,
            recommended_daily_budget=recommended_budget,
            enable_pid_pacing=enable_pid_pacing,
            catalog_fixes=tuple(catalog_fixes),
            action_summary=action_summary,
        )


def generate_diagnostic_scenarios(
    num_scenarios: int = 40,
    seed: int = 42,
) -> Tuple[List[CampaignDiagnosticInput], List[AuctionLossTelemetryRow]]:
    """Generate 40 deterministic seller campaign scenarios across all 5 root-cause loss modes."""
    rng = random.Random(seed)
    archetypes = (
        IssueCode.OUT_OF_STOCK_SUPPRESSED,
        IssueCode.BUDGET_EXHAUSTED_EARLY,
        IssueCode.MISSING_CATALOG_ATTRIBUTES,
        IssueCode.LOW_QUERY_RELEVANCE,
        IssueCode.BID_BELOW_CLEARING_FLOOR,
    )
    categories = ("electronics_audio", "kitchen_appliances", "home_office", "smart_home")

    inputs: List[CampaignDiagnosticInput] = []
    telemetry: List[AuctionLossTelemetryRow] = []

    for idx in range(num_scenarios):
        issue = archetypes[idx % len(archetypes)]
        cmp_id = f"CMP-D{idx + 1:03d}"
        sku_id = f"SKU-D{idx + 1:04d}"
        cat = categories[idx % len(categories)]

        price = round(rng.uniform(42.0, 150.0), 2)
        margin = round(rng.uniform(0.32, 0.52), 3)
        cvr = round(rng.uniform(0.09, 0.16), 4)
        ctr = round(rng.uniform(0.038, 0.068), 4)
        target_acos = round(rng.uniform(0.20, 0.30), 2)
        max_cpc = round(price * cvr * min(target_acos, margin), 2)

        in_stock = True
        missing_attrs: Tuple[str, ...] = ()
        depleted_hour: Optional[int] = None
        daily_budget = round(rng.uniform(25.0, 55.0), 2)
        spent_today = round(daily_budget * 0.82, 2)
        current_bid = round(max(0.35, max_cpc * 0.82), 2)
        clearing_cpc = round(max(0.45, max_cpc * 0.88), 2)

        if issue == IssueCode.OUT_OF_STOCK_SUPPRESSED:
            in_stock = False
            spent_today = round(daily_budget * 0.12, 2)
            outcomes = [("LOST_OOS", 42), ("WON", 6), ("LOST_BID_FLOOR", 2)]
            overlap_range = (0.65, 0.88)
        elif issue == IssueCode.MISSING_CATALOG_ATTRIBUTES:
            missing_attrs = ("connectivity", "material")
            outcomes = [("LOST_ATTR_FILTER", 34), ("WON", 12), ("LOST_BID_FLOOR", 4)]
            overlap_range = (0.58, 0.82)
        elif issue == IssueCode.BUDGET_EXHAUSTED_EARLY:
            depleted_hour = rng.randint(9, 14)
            spent_today = daily_budget
            current_bid = round(max_cpc * 1.15, 2)
            outcomes = [("LOST_BUDGET", 32), ("WON", 15), ("LOST_BID_FLOOR", 3)]
            overlap_range = (0.62, 0.86)
        elif issue == IssueCode.LOW_QUERY_RELEVANCE:
            outcomes = [("LOST_LOW_RELEVANCE", 35), ("WON", 10), ("LOST_BID_FLOOR", 5)]
            overlap_range = (0.12, 0.28)
        else:  # BID_BELOW_CLEARING_FLOOR
            clearing_cpc = round(max_cpc * rng.uniform(0.82, 1.28), 2)
            current_bid = round(max(0.20, clearing_cpc * 0.58), 2)
            outcomes = [("LOST_BID_FLOOR", 36), ("WON", 11), ("LOST_LOW_RELEVANCE", 3)]
            overlap_range = (0.66, 0.90)

        inputs.append(
            CampaignDiagnosticInput(
                campaign_id=cmp_id,
                sku_id=sku_id,
                category=cat,
                sku_price=price,
                gross_margin=margin,
                current_bid=current_bid,
                daily_budget=daily_budget,
                spent_today=spent_today,
                depleted_hour=depleted_hour,
                in_stock=in_stock,
                missing_attributes=missing_attrs,
                calibrated_ctr=ctr,
                calibrated_cvr=cvr,
                target_acos=target_acos,
                expected_primary_issue=issue.value,
            )
        )

        for outcome_label, count in outcomes:
            for step in range(count):
                telemetry.append(
                    AuctionLossTelemetryRow(
                        campaign_id=cmp_id,
                        sku_id=sku_id,
                        hour=(step * 5) % 24,
                        outcome=outcome_label,
                        clearing_cpc=round(clearing_cpc * rng.uniform(0.92, 1.08), 3),
                        query_token_overlap=round(rng.uniform(*overlap_range), 3),
                    )
                )

    return inputs, telemetry


def run_diagnostics_benchmark(
    num_scenarios: int = 40,
    seed: int = 42,
) -> DiagnosticsBenchmarkReport:
    """Evaluate Seller Campaign Diagnostics accuracy and ACOS-ceiling safety bounds."""
    scenarios, telemetry = generate_diagnostic_scenarios(num_scenarios=num_scenarios, seed=seed)
    with CampaignDiagnostics() as engine:
        engine.load_telemetry(telemetry)

        correct_primary = 0
        acos_compliant = 0
        oos_total = 0
        oos_safe = 0
        dist: Dict[str, int] = {}
        reports: List[CampaignDiagnosticReport] = []

        for sc in scenarios:
            rep = engine.diagnose(sc)
            reports.append(rep)
            dist[rep.primary_issue] = dist.get(rep.primary_issue, 0) + 1

            if rep.primary_issue == sc.expected_primary_issue:
                correct_primary += 1
            if rep.recommended_bid <= rep.max_acos_cpc + 1e-6:
                acos_compliant += 1
            if not sc.in_stock:
                oos_total += 1
                if rep.recommended_bid <= sc.current_bid + 1e-6:
                    oos_safe += 1

    n = max(1, len(scenarios))
    return DiagnosticsBenchmarkReport(
        scenarios_evaluated=len(scenarios),
        primary_issue_accuracy_pct=round(100.0 * correct_primary / n, 2),
        acos_ceiling_compliance_pct=round(100.0 * acos_compliant / n, 2),
        oos_zero_bid_hike_compliance_pct=round(100.0 * oos_safe / max(1, oos_total), 2),
        issue_distribution=dist,
        sample_reports=[asdict(r) for r in reports[:5]],
    )
