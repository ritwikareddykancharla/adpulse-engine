#!/usr/bin/env python3
"""Deterministic Benchmark Runner (`benchmarks/run_benchmarks.py`).

Executes all seven AdPulse v2 subsystems on seeded (`random.Random(42)`) workloads,
prints a summary table to stdout, and writes exact measured metrics to
`benchmarks/latest_results.json`.
"""

from __future__ import annotations

import json
from pathlib import Path
import platform
import sys
import time
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adpulse import (  # noqa: E402
    __version__,
    generate_synthetic_catalog_and_queries,
    run_cax_benchmark,
    run_diagnostics_benchmark,
    run_offsearch_benchmark,
    run_ope_benchmark,
    run_pacing_benchmark,
    run_pal_and_ha_benchmark,
    run_ranker_benchmark,
)
from adpulse.web_api import _build_featured_storefront_skus  # noqa: E402


def run_all_benchmarks(seed: int = 42) -> Dict[str, Any]:
    t0 = time.perf_counter()

    ranker_report = run_ranker_benchmark(num_skus=500, num_queries=120, seed=seed)
    pacing_report = run_pacing_benchmark(num_campaigns=40, num_auctions=6000, seed=seed)
    ope_report = run_ope_benchmark(seed=seed)
    diag_report = run_diagnostics_benchmark(num_scenarios=40, seed=seed)

    base_catalog, _ = generate_synthetic_catalog_and_queries(num_skus=500, num_queries=120, seed=seed)
    featured_skus, featured_meta = _build_featured_storefront_skus()
    cax_report = run_cax_benchmark(
        featured_skus + base_catalog, featured_meta=featured_meta, seed=seed
    )
    offsearch_report = run_offsearch_benchmark(seed=seed)
    pal_ha_report = run_pal_and_ha_benchmark(seed=seed)

    wall_ms = round((time.perf_counter() - t0) * 1000.0, 2)

    headline_metrics = {
        "top3_recall_baseline": ranker_report.baseline.top3_recall,
        "top3_recall_two_stage": ranker_report.two_stage.top3_recall,
        "top3_recall_lift_pct": ranker_report.top3_recall_lift_pct,
        "ndcg_at_3_baseline": ranker_report.baseline.ndcg_at_3,
        "ndcg_at_3_two_stage": ranker_report.two_stage.ndcg_at_3,
        "ndcg_lift_pct": ranker_report.ndcg_lift_pct,
        "expected_conversions_lift_pct": ranker_report.expected_conversions_lift_pct,
        "oos_impression_rate_baseline_pct": ranker_report.baseline.out_of_stock_impression_rate_pct,
        "oos_impression_rate_two_stage_pct": ranker_report.two_stage.out_of_stock_impression_rate_pct,
        "price_cap_violation_baseline_pct": ranker_report.baseline.price_cap_violation_rate_pct,
        "price_cap_violation_two_stage_pct": ranker_report.two_stage.price_cap_violation_rate_pct,
        "ranker_p50_ms": ranker_report.two_stage.p50_latency_ms,
        "ranker_p99_ms": ranker_report.two_stage.p99_latency_ms,
        "early_depletion_unpaced_pct": pacing_report.unpaced_greedy.early_depletion_rate_pct,
        "early_depletion_paced_pct": pacing_report.pid_paced.early_depletion_rate_pct,
        "early_depletion_reduction_pct": pacing_report.early_depletion_reduction_pct,
        "roas_unpaced": pacing_report.unpaced_greedy.advertiser_roas,
        "roas_paced": pacing_report.pid_paced.advertiser_roas,
        "roas_lift_pct": pacing_report.roas_lift_pct,
        "pacing_conversions_lift_pct": pacing_report.conversions_lift_pct,
        "platform_revenue_change_pct": pacing_report.platform_revenue_change_pct,
        "budget_utilization_pct": pacing_report.pid_paced.budget_utilization_pct,
        "budget_cap_violations": pacing_report.pid_paced.budget_cap_violations,
        "naive_ips_rmse": ope_report.naive_ips_summary.rmse,
        "clipped_ips_rmse": ope_report.clipped_ips_summary.rmse,
        "doubly_robust_rmse": ope_report.doubly_robust_summary.rmse,
        "clipped_ips_rmse_reduction_pct": ope_report.clipped_ips_rmse_reduction_pct,
        "dr_rmse_reduction_pct": ope_report.dr_rmse_reduction_pct,
        "dr_ci95_coverage_pct": ope_report.doubly_robust_summary.ci_95_coverage_pct,
        "diagnostics_primary_accuracy_pct": diag_report.primary_issue_accuracy_pct,
        "diagnostics_acos_compliance_pct": diag_report.acos_ceiling_compliance_pct,
        "cax_unfloored_pollution_pct": cax_report.unfloored_pollution_rate_pct,
        "cax_floored_pollution_pct": cax_report.cax_pollution_rate_pct,
        "cax_constraint_satisfaction_pct": cax_report.constraint_satisfaction_pct,
        "cax_policy_pass_rate_pct": cax_report.ad_policy_pass_rate_pct,
        "cax_linucb_lift_pct": cax_report.linucb_acceptance_lift_pct,
        "offsearch_naive_redundant_pct": offsearch_report.naive_redundant_durable_rate_pct,
        "offsearch_graph_redundant_pct": offsearch_report.graph_redundant_durable_rate_pct,
        "offsearch_naive_incompat_pct": offsearch_report.naive_incompatible_accessory_rate_pct,
        "offsearch_graph_incompat_pct": offsearch_report.graph_incompatible_accessory_rate_pct,
        "cuped_theta": offsearch_report.cuped_theta,
        "cuped_variance_reduction_pct": offsearch_report.cuped_variance_reduction_pct,
        "cuped_rpm_lift_pct": offsearch_report.cuped_rpm_lift_pct,
        "pal_naive_logloss": pal_ha_report.naive_log_loss,
        "pal_isotonic_logloss": pal_ha_report.pal_isotonic_log_loss,
        "pal_naive_rig_pct": pal_ha_report.naive_rig_pct,
        "pal_isotonic_rig_pct": pal_ha_report.pal_isotonic_rig_pct,
        "pal_naive_ece": pal_ha_report.naive_ece,
        "pal_isotonic_ece": pal_ha_report.pal_isotonic_ece,
        "pal_naive_copc": pal_ha_report.naive_copc,
        "pal_isotonic_copc": pal_ha_report.pal_isotonic_copc,
        "shuffle_virtual_shards": pal_ha_report.virtual_shards,
        "shuffle_blast_radius_pct": pal_ha_report.shuffle_sharding_blast_radius_pct,
    }

    return {
        "engine": "AdPulse",
        "version": __version__,
        "python_version": platform.python_version(),
        "seed": seed,
        "benchmark_wall_time_ms": wall_ms,
        "headline_metrics": headline_metrics,
        "subsystems": {
            "catalog_ranker": ranker_report.to_dict(),
            "auction_pacer": pacing_report.to_dict(),
            "off_policy_eval": ope_report.to_dict(),
            "campaign_diagnostics": diag_report.to_dict(),
            "conversational_cax": cax_report.to_dict(),
            "offsearch_pdp_cuped": offsearch_report.to_dict(),
            "pal_calibration_shuffle_ha": pal_ha_report.to_dict(),
        },
    }


def _print_summary_table(results: Dict[str, Any]) -> None:
    hm = results["headline_metrics"]
    rows = [
        (
            "1. Catalog Ranker - Top-3 Recall (500 SKUs / 120 Queries)",
            f"{hm['top3_recall_two_stage']:.4f} (+{hm['top3_recall_lift_pct']:.2f}%)",
            f"Baseline: {hm['top3_recall_baseline']:.4f}",
        ),
        (
            "1. Catalog Ranker - NDCG@3 & Expected Conversions Lift",
            f"NDCG {hm['ndcg_at_3_two_stage']:.4f} (+{hm['ndcg_lift_pct']:.2f}%)",
            f"+{hm['expected_conversions_lift_pct']:.2f}% expected conversions",
        ),
        (
            "1. Catalog Ranker - Query Latency (p50 / p99)",
            f"{hm['ranker_p50_ms']:.3f} ms / {hm['ranker_p99_ms']:.3f} ms",
            f"{hm['oos_impression_rate_two_stage_pct']:.1f}% OOS / {hm['price_cap_violation_two_stage_pct']:.1f}% cap",
        ),
        (
            "2. Auction Pacer - Advertiser ROAS & Burnout",
            f"{hm['roas_paced']:.2f}x (+{hm['roas_lift_pct']:.2f}% ROAS)",
            f"{hm['early_depletion_unpaced_pct']:.1f}% -> {hm['early_depletion_paced_pct']:.1f}% early burnout",
        ),
        (
            "3. Off-Policy Eval - Doubly-Robust RMSE vs Naive IPS",
            f"{hm['doubly_robust_rmse']:.5f} (-{hm['dr_rmse_reduction_pct']:.2f}% RMSE)",
            f"Naive IPS: {hm['naive_ips_rmse']:.5f} | 95% CI cov: {hm['dr_ci95_coverage_pct']:.1f}%",
        ),
        (
            "4. Seller Diagnostics - Root-Cause & ACOS Safety",
            f"{hm['diagnostics_primary_accuracy_pct']:.1f}% accuracy",
            f"{hm['diagnostics_acos_compliance_pct']:.1f}% ACOS ceiling compliance",
        ),
        (
            "5. Conversational Ads (CAX) - Ad Floor & LinUCB",
            f"{hm['cax_floored_pollution_pct']:.1f}% irrelevant ad leaks",
            f"Unfloored: {hm['cax_unfloored_pollution_pct']:.1f}% | LinUCB +{hm['cax_linucb_lift_pct']:.2f}%",
        ),
        (
            "6. Off-Search PDP & CUPED A/B - Variance & Compat",
            f"-{hm['cuped_variance_reduction_pct']:.2f}% CUPED var (theta={hm['cuped_theta']:.3f})",
            f"Redundant: {hm['offsearch_naive_redundant_pct']:.1f}%->{hm['offsearch_graph_redundant_pct']:.1f}% | Incompat: {hm['offsearch_naive_incompat_pct']:.1f}%->{hm['offsearch_graph_incompat_pct']:.1f}%",
        ),
        (
            "7. PAL pCTR Calibration & C(8,2)=28 Shuffle Sharding",
            f"COPC {hm['pal_isotonic_copc']:.4f} | RIG {hm['pal_isotonic_rig_pct']:.2f}%",
            f"Naive COPC {hm['pal_naive_copc']:.4f}, ECE {hm['pal_naive_ece']:.4f}->{hm['pal_isotonic_ece']:.4f} | {hm['shuffle_blast_radius_pct']:.2f}% blast",
        ),
    ]

    print("=" * 118)
    print(f"AdPulse v2 Deterministic Benchmark Suite (seed={results['seed']}, wall={results['benchmark_wall_time_ms']} ms)")
    print("=" * 118)
    print(f"{'Subsystem & Metric':<56} | {'Measured Result':<28} | {'Baseline / Context'}")
    print("-" * 118)
    for label, measured, context in rows:
        print(f"{label:<56} | {measured:<28} | {context}")
    print("=" * 118)


def main() -> int:
    results = run_all_benchmarks(seed=42)
    out_path = Path(__file__).resolve().parent / "latest_results.json"
    out_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    _print_summary_table(results)
    print(f"Saved full telemetry to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
