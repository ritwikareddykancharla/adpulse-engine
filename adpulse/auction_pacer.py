"""GSP / Second-Price Auction & Lagrangian PID Budget Pacer (`adpulse/auction_pacer.py`).

Simulates a 24-hour Sponsored Products impression stream (`6,000` auctions across `40`
advertiser campaigns with hourly traffic & conversion peaks) comparing:
1. Unpaced Greedy Bidding (`effective_bid = base_bid` until daily budget hits zero).
2. Lagrangian PID Bid-Shading Pacer (`effective_bid = base_bid * lambda_t`, tracking an
   hourly target spend curve while enforcing a campaign ROAS floor and strict daily budget cap).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple


# Realistic 24-hour retail search traffic curve (morning browse spike + prime evening peak)
HOURLY_TRAFFIC_WEIGHTS: Tuple[float, ...] = (
    0.014, 0.010, 0.008, 0.007, 0.009, 0.018,  # 00:00 - 05:59
    0.038, 0.056, 0.068, 0.066, 0.058, 0.050,  # 06:00 - 11:59 (morning browse spike)
    0.046, 0.043, 0.041, 0.042, 0.046, 0.054,  # 12:00 - 17:59 (afternoon steady)
    0.068, 0.078, 0.076, 0.058, 0.030, 0.016,  # 18:00 - 23:59 (evening purchase peak)
)

# Relative shopper purchase-intent multiplier by hour (window-shopping morning vs high-CVR evening)
HOURLY_CVR_MULTIPLIERS: Tuple[float, ...] = (
    0.72, 0.68, 0.65, 0.65, 0.68, 0.72,        # 00:00 - 05:59
    0.58, 0.54, 0.52, 0.56, 0.64, 0.74,        # 06:00 - 11:59 (low-CVR upper-funnel browsing)
    0.86, 0.90, 0.92, 0.96, 1.05, 1.24,        # 12:00 - 17:59
    1.58, 1.78, 1.82, 1.62, 1.18, 0.88,        # 18:00 - 23:59 (high-intent prime shopping)
)

PEAK_EVENING_START_HOUR = 18


@dataclass(frozen=True)
class AdvertiserCampaign:
    campaign_id: str
    category: str
    daily_budget: float
    base_bid: float
    product_price: float
    quality_score: float
    base_ctr: float
    base_cvr: float
    target_roas: float


@dataclass(frozen=True)
class AuctionOpportunity:
    auction_id: int
    hour: int
    day_progress: float
    category: str
    reserve_cpc: float
    intent_multiplier: float
    click_u: Tuple[float, ...]
    conv_u: Tuple[float, ...]


@dataclass(frozen=True)
class AuctionSlotCleared:
    slot_index: int
    campaign_id: str
    effective_bid: float
    rank_score: float
    clearing_cpc: float


@dataclass
class _PIDCampaignState:
    campaign: AdvertiserCampaign
    spent: float = 0.0
    impressions: int = 0
    clicks: int = 0
    conversions: int = 0
    attributed_sales: float = 0.0
    lambda_t: float = 0.55
    integral_err: float = 0.0
    prev_err: float = 0.0
    depleted_hour: Optional[int] = None


@dataclass(frozen=True)
class SimulationMetrics:
    policy_name: str
    total_auctions: int
    total_impressions: int
    total_clicks: int
    total_conversions: int
    total_spend: float
    total_attributed_sales: float
    platform_clearing_revenue: float
    advertiser_roas: float
    mean_clearing_cpc: float
    early_depletion_rate_pct: float
    campaigns_depleted_early: int
    total_campaigns: int
    budget_utilization_pct: float
    budget_cap_violations: int
    evening_peak_impression_share_pct: float


@dataclass(frozen=True)
class PacingComparisonReport:
    num_campaigns: int
    num_auctions: int
    unpaced_greedy: SimulationMetrics
    pid_paced: SimulationMetrics
    early_depletion_reduction_pct: float
    roas_lift_pct: float
    conversions_lift_pct: float
    platform_revenue_change_pct: float
    budget_utilization_accuracy_pct: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def generate_auction_workload(
    num_campaigns: int = 40,
    num_auctions: int = 6000,
    seed: int = 42,
) -> Tuple[List[AdvertiserCampaign], List[AuctionOpportunity]]:
    """Generate 40 multi-category advertiser campaigns and 6,000 24-hour auctions."""
    rng = random.Random(seed)
    categories = ("electronics_audio", "kitchen_appliances", "home_office", "smart_home")
    campaigns: List[AdvertiserCampaign] = []

    for idx in range(num_campaigns):
        cat = categories[idx % len(categories)]
        product_price = round(rng.uniform(42.0, 155.0), 2)
        base_ctr = round(rng.uniform(0.080, 0.125), 4)
        base_cvr = round(rng.uniform(0.095, 0.160), 4)
        quality_score = round(base_ctr * rng.uniform(0.92, 1.12), 5)
        base_bid = round(rng.uniform(1.60, 2.95), 2)
        # Marketplace mix: 30% ample-budget enterprise campaigns + 70% budget-constrained SMB campaigns
        is_enterprise = (idx % 10 in (0, 4, 8))
        if is_enterprise:
            daily_budget = round(rng.uniform(62.0, 92.0), 2)
        else:
            daily_budget = round(rng.uniform(13.0, 24.0), 2)
        nominal_roas = (product_price * base_cvr) / base_bid
        target_roas = round(max(2.0, nominal_roas * rng.uniform(0.66, 0.84)), 2)

        campaigns.append(
            AdvertiserCampaign(
                campaign_id=f"CMP-{idx + 1:03d}",
                category=cat,
                daily_budget=daily_budget,
                base_bid=base_bid,
                product_price=product_price,
                quality_score=quality_score,
                base_ctr=base_ctr,
                base_cvr=base_cvr,
                target_roas=target_roas,
            )
        )

    # Allocate the 6,000 auctions across 24 hours proportional to HOURLY_TRAFFIC_WEIGHTS
    weight_sum = sum(HOURLY_TRAFFIC_WEIGHTS)
    hourly_counts = [int(round(num_auctions * w / weight_sum)) for w in HOURLY_TRAFFIC_WEIGHTS]
    diff = num_auctions - sum(hourly_counts)
    hourly_counts[19] += diff  # ensure exact total == num_auctions

    opportunities: List[AuctionOpportunity] = []
    auction_id = 0
    for hour, count in enumerate(hourly_counts):
        for step in range(count):
            intra_hour = (step + 0.5) / max(1, count)
            day_progress = (hour + intra_hour) / 24.0
            cat = categories[(auction_id + rng.randint(0, 3)) % len(categories)]
            reserve_cpc = round(rng.uniform(0.22, 0.42), 2)
            # Within each hour, individual queries also vary in purchase intent (exact vs broad)
            query_intent = rng.choice((0.62, 0.85, 1.05, 1.32))
            intent_multiplier = round(HOURLY_CVR_MULTIPLIERS[hour] * query_intent, 4)

            opportunities.append(
                AuctionOpportunity(
                    auction_id=auction_id,
                    hour=hour,
                    day_progress=day_progress,
                    category=cat,
                    reserve_cpc=reserve_cpc,
                    intent_multiplier=intent_multiplier,
                    click_u=(rng.random(), rng.random(), rng.random()),
                    conv_u=(rng.random(), rng.random(), rng.random()),
                )
            )
            auction_id += 1

    return campaigns, opportunities


class BudgetPacedAuctionEngine:
    """GSP / Second-Price Auction Engine with Lagrangian PID Bid-Shading Pacer.

    - Supports truthful single-slot Second-Price (`num_slots=1`) and multi-slot GSP (`num_slots>=2`).
    - Guarantees strict budget feasibility (`spent <= daily_budget` for every campaign).
    - In `pid_paced` mode, updates per-campaign dual multiplier `lambda_t` against a
      conversion-weighted cumulative target spend curve while gating/shading bids that
      violate the campaign's target ROAS floor.
    """

    SLOT_CTR_DECAY: Tuple[float, ...] = (1.0, 0.64, 0.42)

    def __init__(
        self,
        num_slots: int = 2,
        kp: float = 1.85,
        ki: float = 0.45,
        kd: float = 0.22,
        lambda_min: float = 0.10,
        update_every_auctions: int = 30,
    ) -> None:
        if num_slots < 1:
            raise ValueError("num_slots must be >= 1")
        self.num_slots = num_slots
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.lambda_min = lambda_min
        self.update_every_auctions = update_every_auctions
        self._cumulative_target_curve = self._build_target_spend_curve()

    @staticmethod
    def _build_target_spend_curve() -> Tuple[float, ...]:
        """Build cumulative hourly spend curve weighted by expected conversion volume."""
        hourly_value_weights = [
            w * (0.35 + 0.65 * cvr_m)
            for w, cvr_m in zip(HOURLY_TRAFFIC_WEIGHTS, HOURLY_CVR_MULTIPLIERS)
        ]
        total = sum(hourly_value_weights)
        cum = 0.0
        curve = [0.0]
        for val in hourly_value_weights:
            cum += val / total
            curve.append(min(1.0, cum))
        curve[-1] = 1.0
        return tuple(curve)

    def target_spend_fraction(self, hour: int, intra_hour_frac: float = 0.5) -> float:
        h = max(0, min(23, hour))
        lo = self._cumulative_target_curve[h]
        hi = self._cumulative_target_curve[h + 1]
        return lo + max(0.0, min(1.0, intra_hour_frac)) * (hi - lo)

    def clear_gsp_auction(
        self,
        bidders: Sequence[Tuple[str, float, float]],
        reserve_cpc: float,
    ) -> List[AuctionSlotCleared]:
        """Clear a GSP / Second-Price auction.

        Args:
            bidders: Sequence of `(campaign_id, effective_bid, quality_score)`.
            reserve_cpc: Minimum CPC floor.

        Returns:
            Cleared slots sorted from top position (`slot_index=0`) downward.
            For `num_slots=1`, this reduces to exact Vickrey second-price clearing:
            `clearing_cpc = max(reserve_cpc, second_rank_score / winner_quality)`.
        """
        eligible: List[Tuple[float, float, float, str]] = []
        for cid, eff_bid, quality in bidders:
            if quality <= 0.0 or eff_bid < reserve_cpc:
                continue
            rank_score = eff_bid * quality
            eligible.append((rank_score, eff_bid, quality, cid))

        # Deterministic ordering: rank_score desc, tie-broken by campaign_id asc
        eligible.sort(key=lambda item: (-item[0], item[3]))

        cleared: List[AuctionSlotCleared] = []
        k = min(self.num_slots, len(eligible))
        for slot_idx in range(k):
            rank_score, eff_bid, quality, cid = eligible[slot_idx]
            if slot_idx + 1 < len(eligible):
                next_score = eligible[slot_idx + 1][0]
                gsp_cpc = next_score / quality
                clearing_cpc = max(reserve_cpc, min(eff_bid, gsp_cpc))
            else:
                clearing_cpc = reserve_cpc

            cleared.append(
                AuctionSlotCleared(
                    slot_index=slot_idx,
                    campaign_id=cid,
                    effective_bid=round(eff_bid, 6),
                    rank_score=round(rank_score, 6),
                    clearing_cpc=round(clearing_cpc, 6),
                )
            )
        return cleared

    def _compute_effective_bid(
        self,
        state: _PIDCampaignState,
        opp: AuctionOpportunity,
        paced: bool,
    ) -> float:
        camp = state.campaign
        remaining = camp.daily_budget - state.spent
        if remaining < opp.reserve_cpc or remaining <= 1e-6:
            return 0.0

        if not paced:
            return min(camp.base_bid, remaining)

        # Calibrated expected conversion value per click at current hour & query intent
        expected_cvr = min(0.65, camp.base_cvr * opp.intent_multiplier)
        roas_ceiling_cpc = (expected_cvr * camp.product_price) / max(0.5, camp.target_roas)

        # Suppress low-intent auctions that cannot meet 75% of the campaign ROAS floor at reserve
        if roas_ceiling_cpc < opp.reserve_cpc * 0.75:
            return 0.0

        # Lagrangian PID bid shading: effective_bid = base_bid * lambda_t bounded by ROAS ceiling
        shaded_bid = min(camp.base_bid * state.lambda_t, roas_ceiling_cpc * (0.75 + 0.35 * state.lambda_t))
        return min(shaded_bid, remaining)

    def _update_pid_controllers(
        self,
        states: Sequence[_PIDCampaignState],
        hour: int,
        intra_hour_frac: float,
    ) -> None:
        target_frac = self.target_spend_fraction(hour, intra_hour_frac)
        for st in states:
            budget = st.campaign.daily_budget
            if st.spent >= budget * 0.995:
                st.lambda_t = self.lambda_min
                continue

            target_spend = budget * target_frac
            err = (target_spend - st.spent) / budget  # positive => underspent, negative => overspent
            st.integral_err = max(-0.35, min(0.35, st.integral_err + err * 0.12))
            deriv = err - st.prev_err
            st.prev_err = err

            delta = self.kp * err + self.ki * st.integral_err + self.kd * deriv
            st.lambda_t = max(self.lambda_min, min(1.0, st.lambda_t + 0.35 * delta))

    def simulate_stream(
        self,
        campaigns: Sequence[AdvertiserCampaign],
        opportunities: Sequence[AuctionOpportunity],
        paced: bool,
    ) -> Tuple[SimulationMetrics, Dict[str, Dict[str, Any]]]:
        """Simulate 24-hour auction stream under either unpaced greedy or PID paced bidding."""
        states: Dict[str, _PIDCampaignState] = {
            c.campaign_id: _PIDCampaignState(campaign=c, lambda_t=(0.72 if paced else 1.0))
            for c in campaigns
        }
        by_category: Dict[str, List[_PIDCampaignState]] = {}
        for st in states.values():
            by_category.setdefault(st.campaign.category, []).append(st)

        all_states = list(states.values())
        evening_impressions = 0

        for idx, opp in enumerate(opportunities):
            if paced and idx > 0 and idx % self.update_every_auctions == 0:
                intra = (opp.day_progress * 24.0) - opp.hour
                self._update_pid_controllers(all_states, opp.hour, intra)

            cat_states = by_category.get(opp.category, ())
            bidders: List[Tuple[str, float, float]] = []
            for st in cat_states:
                eff_bid = self._compute_effective_bid(st, opp, paced=paced)
                if eff_bid >= opp.reserve_cpc:
                    bidders.append((st.campaign.campaign_id, eff_bid, st.campaign.quality_score))

            if not bidders:
                continue

            cleared_slots = self.clear_gsp_auction(bidders, opp.reserve_cpc)
            for slot in cleared_slots:
                st = states[slot.campaign_id]
                camp = st.campaign
                st.impressions += 1
                if opp.hour >= PEAK_EVENING_START_HOUR:
                    evening_impressions += 1

                pos_decay = (
                    self.SLOT_CTR_DECAY[slot.slot_index]
                    if slot.slot_index < len(self.SLOT_CTR_DECAY)
                    else 0.30
                )
                p_click = min(0.45, camp.base_ctr * pos_decay * (0.88 + 0.12 * opp.intent_multiplier))
                u_click = opp.click_u[slot.slot_index % len(opp.click_u)]

                if u_click < p_click:
                    charge = min(slot.clearing_cpc, max(0.0, camp.daily_budget - st.spent))
                    st.spent = round(st.spent + charge, 6)
                    st.clicks += 1

                    p_conv = min(0.65, camp.base_cvr * opp.intent_multiplier)
                    u_conv = opp.conv_u[slot.slot_index % len(opp.conv_u)]
                    if u_conv < p_conv:
                        st.conversions += 1
                        st.attributed_sales = round(st.attributed_sales + camp.product_price, 4)

                    if st.depleted_hour is None and (camp.daily_budget - st.spent) < 0.25:
                        st.depleted_hour = opp.hour

        total_impressions = sum(s.impressions for s in all_states)
        total_clicks = sum(s.clicks for s in all_states)
        total_conversions = sum(s.conversions for s in all_states)
        total_spend = sum(s.spent for s in all_states)
        total_sales = sum(s.attributed_sales for s in all_states)
        total_budget = sum(s.campaign.daily_budget for s in all_states)

        depleted_early = sum(
            1
            for s in all_states
            if s.depleted_hour is not None and s.depleted_hour < PEAK_EVENING_START_HOUR
        )
        cap_violations = sum(
            1 for s in all_states if s.spent > s.campaign.daily_budget + 1e-5
        )

        metrics = SimulationMetrics(
            policy_name="pid_bid_shaded_pacer" if paced else "unpaced_greedy_bidding",
            total_auctions=len(opportunities),
            total_impressions=total_impressions,
            total_clicks=total_clicks,
            total_conversions=total_conversions,
            total_spend=round(total_spend, 2),
            total_attributed_sales=round(total_sales, 2),
            platform_clearing_revenue=round(total_spend, 2),
            advertiser_roas=round(total_sales / max(1e-9, total_spend), 3),
            mean_clearing_cpc=round(total_spend / max(1, total_clicks), 3),
            early_depletion_rate_pct=round(100.0 * depleted_early / max(1, len(all_states)), 2),
            campaigns_depleted_early=depleted_early,
            total_campaigns=len(all_states),
            budget_utilization_pct=round(100.0 * total_spend / max(1e-9, total_budget), 2),
            budget_cap_violations=cap_violations,
            evening_peak_impression_share_pct=round(
                100.0 * evening_impressions / max(1, total_impressions), 2
            ),
        )

        per_campaign = {
            s.campaign.campaign_id: {
                "daily_budget": s.campaign.daily_budget,
                "spent": round(s.spent, 2),
                "clicks": s.clicks,
                "conversions": s.conversions,
                "attributed_sales": round(s.attributed_sales, 2),
                "roas": round(s.attributed_sales / s.spent, 3) if s.spent > 0 else 0.0,
                "final_lambda": round(s.lambda_t, 4),
                "depleted_hour": s.depleted_hour,
            }
            for s in all_states
        }
        return metrics, per_campaign


def run_pacing_benchmark(
    num_campaigns: int = 40,
    num_auctions: int = 6000,
    seed: int = 42,
) -> PacingComparisonReport:
    """Compare Unpaced Greedy Bidding vs. Lagrangian PID Bid-Shading Pacer."""
    campaigns, opportunities = generate_auction_workload(
        num_campaigns=num_campaigns, num_auctions=num_auctions, seed=seed
    )
    engine = BudgetPacedAuctionEngine(num_slots=2)

    unpaced_metrics, _ = engine.simulate_stream(campaigns, opportunities, paced=False)
    paced_metrics, _ = engine.simulate_stream(campaigns, opportunities, paced=True)

    early_depletion_reduction = (
        100.0
        * (unpaced_metrics.early_depletion_rate_pct - paced_metrics.early_depletion_rate_pct)
        / max(1e-9, unpaced_metrics.early_depletion_rate_pct)
    )
    roas_lift = (
        100.0
        * (paced_metrics.advertiser_roas - unpaced_metrics.advertiser_roas)
        / max(1e-9, unpaced_metrics.advertiser_roas)
    )
    conv_lift = (
        100.0
        * (paced_metrics.total_conversions - unpaced_metrics.total_conversions)
        / max(1, unpaced_metrics.total_conversions)
    )
    rev_change = (
        100.0
        * (paced_metrics.platform_clearing_revenue - unpaced_metrics.platform_clearing_revenue)
        / max(1e-9, unpaced_metrics.platform_clearing_revenue)
    )

    return PacingComparisonReport(
        num_campaigns=num_campaigns,
        num_auctions=num_auctions,
        unpaced_greedy=unpaced_metrics,
        pid_paced=paced_metrics,
        early_depletion_reduction_pct=round(early_depletion_reduction, 2),
        roas_lift_pct=round(roas_lift, 2),
        conversions_lift_pct=round(conv_lift, 2),
        platform_revenue_change_pct=round(rev_change, 2),
        budget_utilization_accuracy_pct=paced_metrics.budget_utilization_pct,
    )
