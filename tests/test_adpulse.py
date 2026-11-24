"""Unit & Integration Test Suite for AdPulse (`tests/test_adpulse.py`)."""

from __future__ import annotations

import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from adpulse.auction_pacer import (
    AdvertiserCampaign,
    AuctionOpportunity,
    BudgetPacedAuctionEngine,
    generate_auction_workload,
    run_pacing_benchmark,
)
from adpulse.aws_lambda_handler import handler as lambda_handler
from adpulse.campaign_diagnostics import (
    AuctionLossTelemetryRow,
    CampaignDiagnosticInput,
    CampaignDiagnostics,
    IssueCode,
    run_diagnostics_benchmark,
)
from adpulse.catalog_ranker import (
    CatalogSKU,
    LexicalBidRanker,
    ShopperQuery,
    TwoStageAdRanker,
    generate_synthetic_catalog_and_queries,
    run_ranker_benchmark,
)
from adpulse.cli import execute_cli_mode, main as cli_main
from adpulse.off_policy_eval import (
    LoggedAuctionEvent,
    OffPolicyBidEvaluator,
    generate_logged_propensity_traces,
    run_ope_benchmark,
)


class TestCatalogRanker(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.catalog, cls.queries = generate_synthetic_catalog_and_queries(
            num_skus=500, num_queries=120, seed=42
        )
        cls.ranker = TwoStageAdRanker(cls.catalog)
        cls.baseline = LexicalBidRanker(cls.catalog)

    def test_catalog_and_query_determinism(self) -> None:
        cat2, q2 = generate_synthetic_catalog_and_queries(num_skus=500, num_queries=120, seed=42)
        self.assertEqual(len(self.catalog), 500)
        self.assertEqual(len(self.queries), 120)
        self.assertEqual([s.sku_id for s in self.catalog], [s.sku_id for s in cat2])
        self.assertEqual([s.price for s in self.catalog], [s.price for s in cat2])
        self.assertEqual([q.raw_query for q in self.queries], [q.raw_query for q in q2])

    def test_two_stage_never_shows_out_of_stock_skus(self) -> None:
        baseline_oos = 0
        for q in self.queries:
            res = self.ranker.rank(q, top_k=5)
            for cand in res:
                self.assertTrue(cand.in_stock, f"OOS SKU {cand.sku_id} leaked for {q.query_id}")
            for b_cand in self.baseline.rank(q, top_k=3):
                if not b_cand.in_stock:
                    baseline_oos += 1
        self.assertGreater(baseline_oos, 0)

    def test_two_stage_honors_explicit_and_textual_price_caps(self) -> None:
        for q in self.queries:
            self.assertIsNotNone(q.max_price)
            for cand in self.ranker.rank(q, top_k=5):
                self.assertLessEqual(cand.price, float(q.max_price))

        # Natural-language "under $X" fallback when structured max_price is None
        nl_query = ShopperQuery(
            query_id="Q-TEXT-CAP",
            raw_query="portable waterproof bluetooth speaker under 75",
            target_category="electronics_audio",
            product_type_intent="speaker",
            max_price=None,
            brand_intent=None,
            required_attributes={},
            relevant_sku_ids=(),
        )
        nl_results = self.ranker.rank(nl_query, top_k=5)
        self.assertGreater(len(nl_results), 0)
        for cand in nl_results:
            self.assertLessEqual(cand.price, 75.0)

    def test_two_stage_honors_category_alignment(self) -> None:
        for q in self.queries:
            for cand in self.ranker.rank(q, top_k=5):
                self.assertEqual(cand.category, q.target_category)

    def test_calibrated_ecpm_orders_high_converting_sku_above_high_bid_low_cvr(self) -> None:
        sku_high_bid_poor_cvr = CatalogSKU(
            sku_id="SKU-A",
            campaign_id="CMP-A",
            seller_id="SLR-A",
            title="Auralis wireless over ear headphones anc matte black",
            category="electronics_audio",
            product_type="headphones",
            brand="Auralis",
            attributes={"connectivity": "bluetooth_5_3", "feature": "anc", "color": "matte_black"},
            required_attribute_keys=("connectivity", "feature", "color"),
            price=120.0,
            margin=0.35,
            base_bid=3.60,
            in_stock=True,
            prior_ctr=0.018,
            prior_cvr=0.015,
            impressions_30d=10000,
            clicks_30d=180,
            conversions_30d=2,
        )
        sku_moderate_bid_strong_cvr = CatalogSKU(
            sku_id="SKU-B",
            campaign_id="CMP-B",
            seller_id="SLR-B",
            title="Auralis wireless over ear headphones anc matte black",
            category="electronics_audio",
            product_type="headphones",
            brand="Auralis",
            attributes={"connectivity": "bluetooth_5_3", "feature": "anc", "color": "matte_black"},
            required_attribute_keys=("connectivity", "feature", "color"),
            price=115.0,
            margin=0.35,
            base_bid=1.75,
            in_stock=True,
            prior_ctr=0.075,
            prior_cvr=0.210,
            impressions_30d=10000,
            clicks_30d=750,
            conversions_30d=158,
        )
        mini_ranker = TwoStageAdRanker([sku_high_bid_poor_cvr, sku_moderate_bid_strong_cvr])
        q = ShopperQuery(
            query_id="Q-UNIT",
            raw_query="auralis wireless over ear headphones anc under 150",
            target_category="electronics_audio",
            product_type_intent="headphones",
            max_price=150.0,
            brand_intent="Auralis",
            required_attributes={"feature": "anc"},
            relevant_sku_ids=("SKU-B",),
        )
        ranked = mini_ranker.rank(q, top_k=2)
        self.assertEqual(ranked[0].sku_id, "SKU-B")
        self.assertGreater(ranked[0].score, ranked[1].score)

    def test_two_stage_beats_lexical_baseline_recall_and_ndcg(self) -> None:
        report = run_ranker_benchmark(seed=42)
        self.assertGreater(report.two_stage.top3_recall, report.baseline.top3_recall)
        self.assertGreater(report.top3_recall_lift_pct, 100.0)
        self.assertEqual(report.two_stage.out_of_stock_impression_rate_pct, 0.0)
        self.assertEqual(report.two_stage.price_cap_violation_rate_pct, 0.0)
        self.assertGreater(report.two_stage.p50_latency_ms, 0.0)
        self.assertGreaterEqual(report.two_stage.p99_latency_ms, report.two_stage.p50_latency_ms)

    def test_impossible_price_cap_returns_empty_cleanly(self) -> None:
        impossible_q = ShopperQuery(
            query_id="Q-ZERO",
            raw_query="wireless headphones under 1",
            target_category="electronics_audio",
            product_type_intent="headphones",
            max_price=0.05,
            brand_intent=None,
            required_attributes={},
            relevant_sku_ids=(),
        )
        self.assertEqual(self.ranker.rank(impossible_q, top_k=3), [])


class TestAuctionPacer(unittest.TestCase):
    def test_single_slot_second_price_truthfulness(self) -> None:
        engine = BudgetPacedAuctionEngine(num_slots=1)
        bidders_low = [("CMP-1", 3.00, 0.08), ("CMP-2", 2.10, 0.07), ("CMP-3", 1.50, 0.09)]
        bidders_high = [("CMP-1", 9.50, 0.08), ("CMP-2", 2.10, 0.07), ("CMP-3", 1.50, 0.09)]

        cleared_low = engine.clear_gsp_auction(bidders_low, reserve_cpc=0.30)
        cleared_high = engine.clear_gsp_auction(bidders_high, reserve_cpc=0.30)

        self.assertEqual(len(cleared_low), 1)
        self.assertEqual(cleared_low[0].campaign_id, "CMP-1")
        self.assertEqual(cleared_high[0].campaign_id, "CMP-1")
        # Truthful Vickrey property: winner's payment depends only on runner-up score (2.10 * 0.07 = 0.147 -> / 0.08 = 1.8375)
        self.assertAlmostEqual(cleared_low[0].clearing_cpc, 1.8375, places=5)
        self.assertAlmostEqual(cleared_high[0].clearing_cpc, cleared_low[0].clearing_cpc, places=6)

    def test_multi_slot_gsp_monotonic_clearing_and_reserve_floor(self) -> None:
        engine = BudgetPacedAuctionEngine(num_slots=2)
        bidders = [("CMP-A", 2.80, 0.10), ("CMP-B", 2.00, 0.10), ("CMP-C", 1.20, 0.10)]
        cleared = engine.clear_gsp_auction(bidders, reserve_cpc=0.35)
        self.assertEqual([c.campaign_id for c in cleared], ["CMP-A", "CMP-B"])
        self.assertAlmostEqual(cleared[0].clearing_cpc, 2.00, places=5)
        self.assertAlmostEqual(cleared[1].clearing_cpc, 1.20, places=5)

        lone = engine.clear_gsp_auction([("CMP-SOLO", 2.50, 0.10)], reserve_cpc=0.40)
        self.assertEqual(len(lone), 1)
        self.assertAlmostEqual(lone[0].clearing_cpc, 0.40, places=5)

    def test_invalid_num_slots_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            BudgetPacedAuctionEngine(num_slots=0)

    def test_pid_pacer_never_exceeds_daily_budget_cap(self) -> None:
        campaigns, opportunities = generate_auction_workload(num_campaigns=40, num_auctions=6000, seed=42)
        engine = BudgetPacedAuctionEngine(num_slots=2)

        for paced_flag in (False, True):
            metrics, per_cmp = engine.simulate_stream(campaigns, opportunities, paced=paced_flag)
            self.assertEqual(metrics.budget_cap_violations, 0)
            for cid, stats in per_cmp.items():
                self.assertLessEqual(
                    stats["spent"],
                    stats["daily_budget"] + 1e-4,
                    f"Budget cap exceeded for {cid} (paced={paced_flag})",
                )

    def test_pid_pacer_reduces_early_depletion_and_lifts_roas(self) -> None:
        report = run_pacing_benchmark(seed=42)
        self.assertLess(
            report.pid_paced.early_depletion_rate_pct,
            report.unpaced_greedy.early_depletion_rate_pct,
        )
        self.assertGreater(report.pid_paced.advertiser_roas, report.unpaced_greedy.advertiser_roas)
        self.assertGreater(report.pid_paced.total_conversions, report.unpaced_greedy.total_conversions)

    def test_roas_floor_suppresses_low_intent_unprofitable_auctions(self) -> None:
        engine = BudgetPacedAuctionEngine(num_slots=1)
        camp = AdvertiserCampaign(
            campaign_id="CMP-STRICT",
            category="electronics_audio",
            daily_budget=50.0,
            base_bid=2.00,
            product_price=30.0,
            quality_score=0.08,
            base_ctr=0.08,
            base_cvr=0.05,
            target_roas=8.0,  # requires high expected CVR to clear reserve
        )
        low_opp = AuctionOpportunity(
            auction_id=1,
            hour=7,
            day_progress=7.0 / 24.0,
            category="electronics_audio",
            reserve_cpc=0.45,
            intent_multiplier=0.35,  # expected_cvr = 0.0175 => roas_ceiling = $0.0656 << 0.75 * $0.45
            click_u=(0.01,),
            conv_u=(0.01,),
        )
        high_opp = AuctionOpportunity(
            auction_id=2,
            hour=20,
            day_progress=20.0 / 24.0,
            category="electronics_audio",
            reserve_cpc=0.35,
            intent_multiplier=3.80,  # expected_cvr = 0.19 => roas_ceiling = $0.7125 >= reserve
            click_u=(0.01,),
            conv_u=(0.01,),
        )
        m_low, _ = engine.simulate_stream([camp], [low_opp], paced=True)
        m_high, _ = engine.simulate_stream([camp], [high_opp], paced=True)
        self.assertEqual(m_low.total_impressions, 0)
        self.assertEqual(m_high.total_impressions, 1)

    def test_target_spend_curve_monotonic_and_bounded(self) -> None:
        engine = BudgetPacedAuctionEngine()
        vals = [engine.target_spend_fraction(h, 0.5) for h in range(24)]
        for prev, curr in zip(vals, vals[1:]):
            self.assertLessEqual(prev, curr)
        self.assertGreaterEqual(vals[0], 0.0)
        self.assertLessEqual(vals[-1], 1.0)


class TestOffPolicyEval(unittest.TestCase):
    def test_doubly_robust_unbiasedness_under_biased_direct_model(self) -> None:
        evaluator = OffPolicyBidEvaluator(clip_weight=25.0, num_bootstrap=40, seed=42)
        dr_estimates = []
        oracle_val = 0.0
        for trial in range(60):
            events, oracle_val, _ = generate_logged_propensity_traces(
                num_events=1000,
                seed=100 + trial * 31,
                direct_model_bias_override=0.80,  # +$0.80 systematic bias in direct model!
            )
            _, _, dr_est, _, _, _ = evaluator.compute_point_estimates(events)
            dr_estimates.append(dr_est)

        mean_dr = sum(dr_estimates) / len(dr_estimates)
        self.assertAlmostEqual(mean_dr, oracle_val, delta=0.04)

    def test_doubly_robust_achieves_lower_rmse_than_naive_ips(self) -> None:
        report = run_ope_benchmark(seed=42)
        self.assertLess(report.doubly_robust_summary.rmse, report.clipped_ips_summary.rmse)
        self.assertLess(report.clipped_ips_summary.rmse, report.naive_ips_summary.rmse)
        self.assertGreater(report.dr_rmse_reduction_pct, 50.0)

    def test_clipped_ips_truncates_extreme_propensity_weights(self) -> None:
        ev = LoggedAuctionEvent(
            event_id=0,
            context_bucket="tail",
            chosen_action=0,
            propensity_score=0.001,
            target_prob=0.50,  # raw importance weight = 500
            logged_reward=2.0,
            direct_model_hat=1.8,
            expected_direct_value=1.8,
        )
        evaluator = OffPolicyBidEvaluator(clip_weight=10.0, num_bootstrap=20, seed=42)
        naive_est, cips_est, _, _, _, _ = evaluator.compute_point_estimates([ev])
        self.assertAlmostEqual(naive_est, 1000.0, places=5)
        self.assertAlmostEqual(cips_est, 20.0, places=5)

    def test_paired_bootstrap_confidence_interval_ordering_and_coverage(self) -> None:
        events, oracle_val, _ = generate_logged_propensity_traces(num_events=1200, seed=42)
        evaluator = OffPolicyBidEvaluator(clip_weight=25.0, num_bootstrap=200, seed=42)
        res = evaluator.evaluate_policy(events, oracle_ground_truth=oracle_val)
        self.assertLess(res.doubly_robust.ci_95_low, res.doubly_robust.estimate)
        self.assertLess(res.doubly_robust.estimate, res.doubly_robust.ci_95_high)
        self.assertGreater(res.doubly_robust.bootstrap_std, 0.0)

    def test_ope_rejects_empty_events_or_non_positive_propensity(self) -> None:
        evaluator = OffPolicyBidEvaluator()
        with self.assertRaises(ValueError):
            evaluator.compute_point_estimates([])

        bad_ev = LoggedAuctionEvent(
            event_id=1,
            context_bucket="ctx",
            chosen_action=0,
            propensity_score=0.0,
            target_prob=0.5,
            logged_reward=1.0,
            direct_model_hat=1.0,
            expected_direct_value=1.0,
        )
        with self.assertRaises(ValueError):
            evaluator.compute_point_estimates([bad_ev])

    def test_dr_reduces_to_exact_direct_expectation_when_residuals_are_zero(self) -> None:
        events = [
            LoggedAuctionEvent(
                event_id=i,
                context_bucket="ctx",
                chosen_action=0,
                propensity_score=0.05,
                target_prob=0.40,
                logged_reward=2.5,
                direct_model_hat=2.5,
                expected_direct_value=3.1,
            )
            for i in range(10)
        ]
        evaluator = OffPolicyBidEvaluator()
        _, _, dr_est, _, _, _ = evaluator.compute_point_estimates(events)
        self.assertAlmostEqual(dr_est, 3.1, places=6)


class TestCampaignDiagnostics(unittest.TestCase):
    def test_diagnostics_identifies_all_five_loss_archetypes(self) -> None:
        report = run_diagnostics_benchmark(num_scenarios=40, seed=42)
        self.assertEqual(report.primary_issue_accuracy_pct, 100.0)
        self.assertEqual(report.acos_ceiling_compliance_pct, 100.0)
        self.assertEqual(report.oos_zero_bid_hike_compliance_pct, 100.0)
        for code in (
            IssueCode.OUT_OF_STOCK_SUPPRESSED,
            IssueCode.BUDGET_EXHAUSTED_EARLY,
            IssueCode.MISSING_CATALOG_ATTRIBUTES,
            IssueCode.LOW_QUERY_RELEVANCE,
            IssueCode.BID_BELOW_CLEARING_FLOOR,
        ):
            self.assertEqual(report.issue_distribution.get(code.value), 8)

    def test_diagnostics_strictly_respects_seller_target_acos_ceiling(self) -> None:
        with CampaignDiagnostics() as diag:
            diag.load_telemetry(
                [
                    AuctionLossTelemetryRow(
                        campaign_id="CMP-CAP",
                        sku_id="SKU-CAP",
                        hour=14,
                        outcome="LOST_BID_FLOOR",
                        clearing_cpc=4.80,  # Far above seller's profitable ACOS cap ($1.25)
                        query_token_overlap=0.80,
                    )
                    for _ in range(30)
                ]
            )
            inp = CampaignDiagnosticInput(
                campaign_id="CMP-CAP",
                sku_id="SKU-CAP",
                category="kitchen_appliances",
                sku_price=50.0,
                gross_margin=0.40,
                current_bid=0.70,
                daily_budget=40.0,
                spent_today=12.0,
                depleted_hour=None,
                in_stock=True,
                missing_attributes=(),
                calibrated_ctr=0.05,
                calibrated_cvr=0.10,
                target_acos=0.25,  # max_acos_cpc = 50 * 0.10 * 0.25 = $1.25
            )
            rep = diag.diagnose(inp)
        self.assertEqual(rep.primary_issue, IssueCode.BID_BELOW_CLEARING_FLOOR.value)
        self.assertEqual(rep.max_acos_cpc, 1.25)
        self.assertLessEqual(rep.recommended_bid, 1.25)
        self.assertLessEqual(rep.implied_expected_acos, 0.25 + 1e-4)

    def test_out_of_stock_sku_never_receives_bid_increase(self) -> None:
        with CampaignDiagnostics() as diag:
            diag.load_telemetry(
                [
                    AuctionLossTelemetryRow("CMP-OOS", "SKU-OOS", 10, "LOST_OOS", 1.80, 0.85)
                    for _ in range(25)
                ]
            )
            inp = CampaignDiagnosticInput(
                campaign_id="CMP-OOS",
                sku_id="SKU-OOS",
                category="pet_supplies",
                sku_price=60.0,
                gross_margin=0.45,
                current_bid=1.10,
                daily_budget=35.0,
                spent_today=2.0,
                depleted_hour=None,
                in_stock=False,
                missing_attributes=(),
                calibrated_ctr=0.05,
                calibrated_cvr=0.14,
                target_acos=0.28,
            )
            rep = diag.diagnose(inp)
        self.assertEqual(rep.primary_issue, IssueCode.OUT_OF_STOCK_SUPPRESSED.value)
        self.assertLessEqual(rep.recommended_bid, inp.current_bid)
        self.assertTrue(any("Restock" in fix for fix in rep.catalog_fixes))

    def test_early_budget_depletion_enables_pid_pacing(self) -> None:
        with CampaignDiagnostics() as diag:
            diag.load_telemetry(
                [
                    AuctionLossTelemetryRow("CMP-DEP", "SKU-DEP", 11, "LOST_BUDGET", 1.60, 0.78)
                    for _ in range(25)
                ]
            )
            inp = CampaignDiagnosticInput(
                campaign_id="CMP-DEP",
                sku_id="SKU-DEP",
                category="home_office",
                sku_price=100.0,
                gross_margin=0.40,
                current_bid=2.90,
                daily_budget=22.0,
                spent_today=22.0,
                depleted_hour=11,
                in_stock=True,
                missing_attributes=(),
                calibrated_ctr=0.045,
                calibrated_cvr=0.10,
                target_acos=0.22,  # max_acos_cpc = $2.20
            )
            rep = diag.diagnose(inp)
        self.assertEqual(rep.primary_issue, IssueCode.BUDGET_EXHAUSTED_EARLY.value)
        self.assertTrue(rep.enable_pid_pacing)
        self.assertLessEqual(rep.recommended_bid, 2.20)

    def test_missing_catalog_attributes_surfaces_exact_schema_keys(self) -> None:
        with CampaignDiagnostics() as diag:
            diag.load_telemetry(
                [
                    AuctionLossTelemetryRow("CMP-ATTR", "SKU-ATTR", 15, "LOST_ATTR_FILTER", 1.40, 0.72)
                    for _ in range(20)
                ]
            )
            inp = CampaignDiagnosticInput(
                campaign_id="CMP-ATTR",
                sku_id="SKU-ATTR",
                category="smart_home",
                sku_price=80.0,
                gross_margin=0.42,
                current_bid=1.40,
                daily_budget=40.0,
                spent_today=18.0,
                depleted_hour=None,
                in_stock=True,
                missing_attributes=("connectivity", "wattage"),
                calibrated_ctr=0.04,
                calibrated_cvr=0.12,
                target_acos=0.25,
            )
            rep = diag.diagnose(inp)
        self.assertEqual(rep.primary_issue, IssueCode.MISSING_CATALOG_ATTRIBUTES.value)
        self.assertIn("connectivity", rep.action_summary)
        self.assertIn("wattage", rep.action_summary)


class TestCLIAndLambdaHandler(unittest.TestCase):
    def test_cli_all_supported_modes_return_valid_json(self) -> None:
        for mode in ("ranker", "pacing", "ope", "diagnostics", "demo-all"):
            buf = io.StringIO()
            with patch("sys.stdout", buf):
                rc = cli_main(["--mode", mode, "--compact"])
            self.assertEqual(rc, 0)
            parsed = json.loads(buf.getvalue())
            self.assertEqual(parsed["engine"], "AdPulse")
            self.assertEqual(parsed["mode"], mode)
            self.assertIn("telemetry", parsed)

    def test_cli_unsupported_mode_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            execute_cli_mode("nonexistent-mode")

    def test_lambda_handler_health_and_run_endpoints(self) -> None:
        health = lambda_handler({"httpMethod": "GET", "path": "/api/health"})
        self.assertEqual(health["statusCode"], 200)
        self.assertEqual(json.loads(health["body"])["status"], "ok")

        run_diag = lambda_handler(
            {
                "httpMethod": "POST",
                "path": "/api/run",
                "body": json.dumps({"mode": "diagnostics"}),
            }
        )
        self.assertEqual(run_diag["statusCode"], 200)
        self.assertEqual(
            json.loads(run_diag["body"])["telemetry"]["primary_issue_accuracy_pct"],
            100.0,
        )

        run_q = lambda_handler(
            {
                "httpMethod": "POST",
                "path": "/api/run",
                "body": json.dumps(
                    {
                        "mode": "rank-query",
                        "query": "wireless over ear headphones anc under 160",
                        "category": "electronics_audio",
                        "top_k": 3,
                    }
                ),
            }
        )
        self.assertEqual(run_q["statusCode"], 200)
        cands = json.loads(run_q["body"])["candidates"]
        self.assertEqual(len(cands), 3)
        for c in cands:
            self.assertTrue(c["in_stock"])
            self.assertLessEqual(c["price"], 160.0)

        bad_req = lambda_handler(
            {"httpMethod": "POST", "path": "/api/run", "body": json.dumps({"mode": "bad"})}
        )
        self.assertEqual(bad_req["statusCode"], 400)

        not_found = lambda_handler({"httpMethod": "GET", "path": "/unknown"})
        self.assertEqual(not_found["statusCode"], 404)

    def test_benchmark_results_json_exists_and_matches_live_deterministic_run(self) -> None:
        results_path = Path(__file__).resolve().parent.parent / "benchmarks" / "latest_results.json"
        self.assertTrue(results_path.is_file())
        saved = json.loads(results_path.read_text(encoding="utf-8"))
        hm = saved["headline_metrics"]
        live_ranker = run_ranker_benchmark(seed=42)
        self.assertEqual(hm["top3_recall_lift_pct"], live_ranker.top3_recall_lift_pct)
        self.assertEqual(hm["budget_cap_violations"], 0)


if __name__ == "__main__":
    unittest.main()
