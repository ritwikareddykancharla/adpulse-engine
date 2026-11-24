"""Clipped IPS & Doubly-Robust Off-Policy Evaluator (`adpulse/off_policy_eval.py`).

Evaluates candidate bid-shading and ranking policies from logged auction propensity traces
(`propensity_score`, `target_prob`, `logged_reward`, `direct_model_hat`) with paired
bootstrap 95% confidence intervals, comparing:
  1. Naive Inverse Propensity Scoring (IPS)
  2. Clipped IPS (truncated importance weights `min(M, pi_e / pi_0)`)
  3. Doubly-Robust (DR) estimation (`E_pi_e[mu_hat] + (pi_e / pi_0) * (r - mu_hat)`)
against exact ground-truth oracle policy rollouts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple


BID_SHADING_MULTIPLIERS: Tuple[float, ...] = (0.55, 0.70, 0.85, 1.00, 1.15, 1.30)


def _pct(sorted_vals: Sequence[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    idx = (p / 100.0) * (len(sorted_vals) - 1)
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return float(sorted_vals[lo])
    frac = idx - lo
    return float(sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac)


@dataclass(frozen=True)
class LoggedAuctionEvent:
    event_id: int
    context_bucket: str
    chosen_action: int
    propensity_score: float
    target_prob: float
    logged_reward: float
    direct_model_hat: float
    expected_direct_value: float


@dataclass(frozen=True)
class BootstrapInterval:
    estimate: float
    ci_95_low: float
    ci_95_high: float
    bootstrap_std: float

    @property
    def ci_width(self) -> float:
        return round(self.ci_95_high - self.ci_95_low, 6)


@dataclass(frozen=True)
class PolicyEvaluationResult:
    policy_name: str
    sample_size: int
    effective_sample_size: float
    max_importance_weight: float
    oracle_ground_truth: Optional[float]
    naive_ips: BootstrapInterval
    clipped_ips: BootstrapInterval
    doubly_robust: BootstrapInterval


@dataclass(frozen=True)
class EstimatorAccuracySummary:
    estimator_name: str
    rmse: float
    mae: float
    mean_signed_bias: float
    ci_95_coverage_pct: float
    mean_ci_width: float


@dataclass(frozen=True)
class OPEBenchmarkReport:
    num_trials: int
    events_per_trial: int
    clip_threshold: float
    oracle_target_policy_value: float
    oracle_logging_policy_value: float
    single_trace_demo: PolicyEvaluationResult
    naive_ips_summary: EstimatorAccuracySummary
    clipped_ips_summary: EstimatorAccuracySummary
    doubly_robust_summary: EstimatorAccuracySummary
    clipped_ips_rmse_reduction_pct: float
    dr_rmse_reduction_pct: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class OffPolicyBidEvaluator:
    """Off-policy evaluator supporting Naive IPS, Clipped IPS, and Doubly-Robust with paired bootstrap CIs."""

    def __init__(
        self,
        clip_weight: float = 25.0,
        dr_clip_weight: Optional[float] = None,
        num_bootstrap: int = 240,
        seed: int = 42,
    ) -> None:
        if clip_weight <= 0.0:
            raise ValueError("clip_weight must be positive")
        self.clip_weight = clip_weight
        self.dr_clip_weight = dr_clip_weight
        self.num_bootstrap = num_bootstrap
        self.seed = seed

    def compute_point_estimates(
        self,
        events: Sequence[LoggedAuctionEvent],
    ) -> Tuple[float, float, float, List[float], List[float], List[float]]:
        """Return `(naive_ips, clipped_ips, doubly_robust)` point estimates and per-event terms."""
        if not events:
            raise ValueError("events cannot be empty")

        ips_terms: List[float] = []
        cips_terms: List[float] = []
        dr_terms: List[float] = []

        clip_m = self.clip_weight
        dr_m = self.dr_clip_weight

        for ev in events:
            if ev.propensity_score <= 0.0:
                raise ValueError(f"Invalid non-positive propensity score: {ev.propensity_score}")
            w = ev.target_prob / ev.propensity_score
            w_clip = w if w <= clip_m else clip_m
            w_dr = w if (dr_m is None or w <= dr_m) else dr_m

            ips_terms.append(w * ev.logged_reward)
            cips_terms.append(w_clip * ev.logged_reward)
            dr_terms.append(ev.expected_direct_value + w_dr * (ev.logged_reward - ev.direct_model_hat))

        n = float(len(events))
        return (
            sum(ips_terms) / n,
            sum(cips_terms) / n,
            sum(dr_terms) / n,
            ips_terms,
            cips_terms,
            dr_terms,
        )

    def evaluate_policy(
        self,
        events: Sequence[LoggedAuctionEvent],
        policy_name: str = "candidate_shaded_policy",
        oracle_ground_truth: Optional[float] = None,
        bootstrap_seed: Optional[int] = None,
    ) -> PolicyEvaluationResult:
        """Evaluate a candidate policy on logged propensity traces with paired 95% bootstrap CIs."""
        naive_est, cips_est, dr_est, ips_terms, cips_terms, dr_terms = self.compute_point_estimates(
            events
        )
        n = len(events)
        weights = [ev.target_prob / ev.propensity_score for ev in events]
        sum_w = sum(weights)
        sum_w_sq = sum(w * w for w in weights)
        ess = (sum_w * sum_w) / max(1e-12, sum_w_sq) if sum_w > 0 else 0.0
        max_w = max(weights) if weights else 0.0

        rng = random.Random(self.seed if bootstrap_seed is None else bootstrap_seed)
        boot_ips: List[float] = []
        boot_cips: List[float] = []
        boot_dr: List[float] = []
        indices_range = range(n)
        inv_n = 1.0 / float(n)

        for _ in range(self.num_bootstrap):
            sample_idxs = rng.choices(indices_range, k=n)
            s_ips = 0.0
            s_cips = 0.0
            s_dr = 0.0
            for idx in sample_idxs:
                s_ips += ips_terms[idx]
                s_cips += cips_terms[idx]
                s_dr += dr_terms[idx]
            boot_ips.append(s_ips * inv_n)
            boot_cips.append(s_cips * inv_n)
            boot_dr.append(s_dr * inv_n)

        def _make_interval(est: float, samples: List[float]) -> BootstrapInterval:
            samples.sort()
            mean_b = sum(samples) / max(1, len(samples))
            var_b = sum((x - mean_b) ** 2 for x in samples) / max(1, len(samples) - 1)
            return BootstrapInterval(
                estimate=round(est, 6),
                ci_95_low=round(_pct(samples, 2.5), 6),
                ci_95_high=round(_pct(samples, 97.5), 6),
                bootstrap_std=round(math.sqrt(var_b), 6),
            )

        return PolicyEvaluationResult(
            policy_name=policy_name,
            sample_size=n,
            effective_sample_size=round(ess, 2),
            max_importance_weight=round(max_w, 3),
            oracle_ground_truth=round(oracle_ground_truth, 6) if oracle_ground_truth is not None else None,
            naive_ips=_make_interval(naive_est, boot_ips),
            clipped_ips=_make_interval(cips_est, boot_cips),
            doubly_robust=_make_interval(dr_est, boot_dr),
        )


class _ContextualAuctionOracle:
    """Synthetic ground-truth ad auction environment for verifying off-policy estimators.

    Models 8 query-category-hour segments across 6 bid-shading multipliers (0.55x..1.30x)
    with realistic SKU value heterogeneity ($0.35 to $5.80 expected conversion value per auction)
    and epsilon-explored production logging propensities including rare tail explorations.
    """

    CONTEXTS: Tuple[Tuple[str, float, Tuple[float, ...]], ...] = (
        # (context_name, context_prob, true_expected_value_per_action for multipliers 0.55..1.30)
        ("morning_broad_consumables", 0.18, (0.42, 0.48, 0.40, 0.28, 0.16, 0.08)),
        ("morning_brand_electronics", 0.10, (2.10, 2.65, 3.15, 2.95, 2.55, 2.05)),
        ("midday_category_home", 0.18, (0.95, 1.28, 1.42, 1.18, 0.88, 0.58)),
        ("afternoon_comparison_appliances", 0.14, (1.85, 2.40, 2.88, 2.72, 2.25, 1.70)),
        ("evening_exact_high_aov_audio", 0.16, (2.90, 3.75, 4.65, 5.55, 5.80, 5.10)),
        ("evening_competitor_smart_home", 0.11, (1.55, 2.05, 2.55, 2.78, 2.52, 1.95)),
        ("late_night_mobile_browse", 0.08, (0.38, 0.49, 0.44, 0.31, 0.19, 0.10)),
        ("weekend_high_intent_bundle", 0.05, (2.45, 3.25, 3.60, 3.10, 2.45, 1.80)),
    )

    # Production logging policy concentrates on legacy unshaded bids (actions 3, 4) with
    # epsilon-exploration tail propensities (0.0025 - 0.015) on shaded multipliers across high-AOV contexts.
    LOGGING_PROBS: Dict[str, Tuple[float, ...]] = {
        "morning_broad_consumables": (0.004, 0.036, 0.110, 0.500, 0.240, 0.110),
        "morning_brand_electronics": (0.003, 0.012, 0.125, 0.490, 0.250, 0.120),
        "midday_category_home": (0.004, 0.036, 0.130, 0.490, 0.240, 0.100),
        "afternoon_comparison_appliances": (0.003, 0.012, 0.135, 0.490, 0.240, 0.120),
        "evening_exact_high_aov_audio": (0.003, 0.007, 0.090, 0.480, 0.300, 0.120),
        "evening_competitor_smart_home": (0.003, 0.012, 0.125, 0.490, 0.250, 0.120),
        "late_night_mobile_browse": (0.004, 0.036, 0.110, 0.510, 0.230, 0.110),
        "weekend_high_intent_bundle": (0.003, 0.010, 0.127, 0.490, 0.250, 0.120),
    }

    # Candidate policy shades bids in low-CVR morning/night and bids up in evening peak.
    TARGET_PROBS: Dict[str, Tuple[float, ...]] = {
        "morning_broad_consumables": (0.10, 0.44, 0.34, 0.08, 0.03, 0.01),
        "morning_brand_electronics": (0.04, 0.18, 0.52, 0.18, 0.06, 0.02),
        "midday_category_home": (0.06, 0.34, 0.46, 0.10, 0.03, 0.01),
        "afternoon_comparison_appliances": (0.04, 0.20, 0.52, 0.17, 0.05, 0.02),
        "evening_exact_high_aov_audio": (0.01, 0.05, 0.12, 0.32, 0.40, 0.10),
        "evening_competitor_smart_home": (0.02, 0.10, 0.30, 0.40, 0.14, 0.04),
        "late_night_mobile_browse": (0.10, 0.44, 0.34, 0.08, 0.03, 0.01),
        "weekend_high_intent_bundle": (0.05, 0.26, 0.50, 0.13, 0.05, 0.01),
    }

    @classmethod
    def oracle_value(cls, policy_probs: Dict[str, Tuple[float, ...]]) -> float:
        total = 0.0
        for ctx_name, ctx_weight, true_rewards in cls.CONTEXTS:
            probs = policy_probs[ctx_name]
            ctx_val = sum(p * r for p, r in zip(probs, true_rewards))
            total += ctx_weight * ctx_val
        return total

    @classmethod
    def direct_model_predictions(cls, ctx_name: str, true_rewards: Tuple[float, ...]) -> Tuple[float, ...]:
        """Fitted direct value model with realistic regularization shrinkage & +5% drift."""
        mean_r = sum(true_rewards) / len(true_rewards)
        ctx_bias = 0.08 if "morning" in ctx_name else -0.06
        return tuple(
            max(0.02, 0.86 * r + 0.14 * mean_r + ctx_bias)
            for r in true_rewards
        )


def generate_logged_propensity_traces(
    num_events: int = 1200,
    seed: int = 42,
    direct_model_bias_override: Optional[float] = None,
) -> Tuple[List[LoggedAuctionEvent], float, float]:
    """Generate logged auction propensity traces + exact oracle values `(target_oracle, logging_oracle)`."""
    rng = random.Random(seed)
    contexts = _ContextualAuctionOracle.CONTEXTS
    ctx_weights = [c[1] for c in contexts]
    ctx_indices = list(range(len(contexts)))
    actions = list(range(len(BID_SHADING_MULTIPLIERS)))

    events: List[LoggedAuctionEvent] = []
    for event_id in range(num_events):
        c_idx = rng.choices(ctx_indices, weights=ctx_weights, k=1)[0]
        ctx_name, _, true_rewards = contexts[c_idx]
        base_pi_0 = _ContextualAuctionOracle.LOGGING_PROBS[ctx_name]
        pi_e = _ContextualAuctionOracle.TARGET_PROBS[ctx_name]

        # 20% of auctions come from high-confidence exploitation traffic with tiny exploration floor
        # (creating heavy-tailed propensity ratios w = pi_e / pi_0 up to ~120 while preserving exact pi_0).
        if rng.random() < 0.20:
            pi_0: Tuple[float, ...] = (0.0015, 0.0025, 0.0260, 0.6200, 0.2500, 0.1000)
        else:
            pi_0 = base_pi_0

        chosen_a = rng.choices(actions, weights=pi_0, k=1)[0]
        true_mu = true_rewards[chosen_a]

        # Zero-mean conditional auction realization noise around true_mu (E[r_t | x, a] == true_mu)
        noise = rng.gauss(0.0, 0.32 * true_mu + 0.08)
        logged_reward = true_mu + noise

        if direct_model_bias_override is not None:
            model_preds = tuple(r + direct_model_bias_override for r in true_rewards)
        else:
            model_preds = _ContextualAuctionOracle.direct_model_predictions(ctx_name, true_rewards)

        expected_direct = sum(p_e * m_hat for p_e, m_hat in zip(pi_e, model_preds))
        events.append(
            LoggedAuctionEvent(
                event_id=event_id,
                context_bucket=ctx_name,
                chosen_action=chosen_a,
                propensity_score=pi_0[chosen_a],
                target_prob=pi_e[chosen_a],
                logged_reward=round(logged_reward, 6),
                direct_model_hat=round(model_preds[chosen_a], 6),
                expected_direct_value=round(expected_direct, 6),
            )
        )

    target_oracle = _ContextualAuctionOracle.oracle_value(_ContextualAuctionOracle.TARGET_PROBS)
    logging_oracle = _ContextualAuctionOracle.oracle_value(_ContextualAuctionOracle.LOGGING_PROBS)
    return events, target_oracle, logging_oracle


def run_ope_benchmark(
    num_trials: int = 40,
    events_per_trial: int = 1200,
    clip_weight: float = 25.0,
    seed: int = 42,
) -> OPEBenchmarkReport:
    """Evaluate Naive IPS, Clipped IPS, and Doubly-Robust across repeated propensity trace trials."""
    evaluator = OffPolicyBidEvaluator(
        clip_weight=clip_weight,
        dr_clip_weight=None,
        num_bootstrap=240,
        seed=seed,
    )

    ips_errors: List[float] = []
    cips_errors: List[float] = []
    dr_errors: List[float] = []

    ips_covered = 0
    cips_covered = 0
    dr_covered = 0

    ips_widths: List[float] = []
    cips_widths: List[float] = []
    dr_widths: List[float] = []

    demo_result: Optional[PolicyEvaluationResult] = None
    target_oracle = 0.0
    logging_oracle = 0.0

    for trial in range(num_trials):
        trial_seed = seed + trial * 97
        events, target_oracle, logging_oracle = generate_logged_propensity_traces(
            num_events=events_per_trial,
            seed=trial_seed,
        )
        res = evaluator.evaluate_policy(
            events,
            policy_name="context_shaded_pid_policy",
            oracle_ground_truth=target_oracle,
            bootstrap_seed=trial_seed + 13,
        )
        if trial == 0:
            demo_result = res

        ips_errors.append(res.naive_ips.estimate - target_oracle)
        cips_errors.append(res.clipped_ips.estimate - target_oracle)
        dr_errors.append(res.doubly_robust.estimate - target_oracle)

        if res.naive_ips.ci_95_low <= target_oracle <= res.naive_ips.ci_95_high:
            ips_covered += 1
        if res.clipped_ips.ci_95_low <= target_oracle <= res.clipped_ips.ci_95_high:
            cips_covered += 1
        if res.doubly_robust.ci_95_low <= target_oracle <= res.doubly_robust.ci_95_high:
            dr_covered += 1

        ips_widths.append(res.naive_ips.ci_width)
        cips_widths.append(res.clipped_ips.ci_width)
        dr_widths.append(res.doubly_robust.ci_width)

    def _summarize(
        name: str, errs: List[float], covered: int, widths: List[float]
    ) -> EstimatorAccuracySummary:
        n_t = max(1, len(errs))
        rmse = math.sqrt(sum(e * e for e in errs) / n_t)
        mae = sum(abs(e) for e in errs) / n_t
        bias = sum(errs) / n_t
        return EstimatorAccuracySummary(
            estimator_name=name,
            rmse=round(rmse, 5),
            mae=round(mae, 5),
            mean_signed_bias=round(bias, 5),
            ci_95_coverage_pct=round(100.0 * covered / n_t, 2),
            mean_ci_width=round(sum(widths) / n_t, 5),
        )

    ips_sum = _summarize("naive_ips", ips_errors, ips_covered, ips_widths)
    cips_sum = _summarize("clipped_ips", cips_errors, cips_covered, cips_widths)
    dr_sum = _summarize("doubly_robust", dr_errors, dr_covered, dr_widths)

    cips_reduction = 100.0 * (ips_sum.rmse - cips_sum.rmse) / max(1e-12, ips_sum.rmse)
    dr_reduction = 100.0 * (ips_sum.rmse - dr_sum.rmse) / max(1e-12, ips_sum.rmse)

    assert demo_result is not None
    return OPEBenchmarkReport(
        num_trials=num_trials,
        events_per_trial=events_per_trial,
        clip_threshold=clip_weight,
        oracle_target_policy_value=round(target_oracle, 5),
        oracle_logging_policy_value=round(logging_oracle, 5),
        single_trace_demo=demo_result,
        naive_ips_summary=ips_sum,
        clipped_ips_summary=cips_sum,
        doubly_robust_summary=dr_sum,
        clipped_ips_rmse_reduction_pct=round(cips_reduction, 2),
        dr_rmse_reduction_pct=round(dr_reduction, 2),
    )
