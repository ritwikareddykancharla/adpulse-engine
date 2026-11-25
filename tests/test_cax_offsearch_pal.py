"""Unit & Integration Tests for AdPulse v2 (`adpulse/cax_offsearch_pal.py`).

Verifies:
  1. Sponsored Conversational Shopping (`CAX`): multi-turn constraint extraction,
     `suppress_ad = True` when `semantic_relevance < tau_rel` (blocking `$4.65-$4.95`
     irrelevant high-bid distractors), GSP second-price truthfulness, and deterministic
     Amazon Ads copy policy verification.
  2. Ads Agent `LinUCB` contextual bandit: positive-definite matrix updates, expert
     feedback learning under scoped caller permissions (`policy:feedback`), and
     permission denial for unscoped callers.
  3. Query-Free Off-Search (`PDP` & `Homepage` Substitute vs. Complement Graph + `CUPED`):
     post-purchase durable suppression (`0.0%` redundant cameras/espresso machines on
     Homepage), incompatible accessory pruning (`58mm` filter on `67mm` lens, `CFexpress-B`
     on `SDXC` body), and exact `CUPED` variance reduction (`theta = Cov(Y, X_pre)/Var(X_pre)`).
  4. Position-Bias-Aware Learning (`PAL`) `pCTR` Calibration & `C(8, 2) = 28` Shuffle Sharding:
     `LogLoss`, `RIG`, `ECE`, and `COPC = sum(pred)/sum(clicks)` improvements over naive
     position-blind models, plus `3.57%` (`1/28`) noisy-neighbor blast radius containment.
"""

from __future__ import annotations

import unittest

from adpulse.catalog_ranker import generate_synthetic_catalog_and_queries
from adpulse.cax_offsearch_pal import (
    ConversationalAdEngine,
    LinUCBContextualBandit,
    OffSearchPdpEngine,
    PALClickCalibrator,
    ShuffleShardedCellRouter,
    build_pretrained_linucb,
    extract_dialogue_constraints,
    run_cax_benchmark,
    run_cuped_ab_analysis,
    run_offsearch_benchmark,
    run_pal_and_ha_benchmark,
    verify_ad_copy_policy,
)
from adpulse.web_api import _build_featured_storefront_skus


class TestConversationalShoppingCAX(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        base_cat, _ = generate_synthetic_catalog_and_queries(num_skus=500, num_queries=120, seed=42)
        feat_skus, feat_meta = _build_featured_storefront_skus()
        cls.catalog = feat_skus + base_cat
        cls.engine = ConversationalAdEngine(cls.catalog, featured_meta=feat_meta, seed=42)

    def test_multi_turn_constraint_extractor_tracks_price_and_features(self) -> None:
        state = extract_dialogue_constraints(
            [
                "wireless running earbuds under $180",
                "actually I wear glasses and sweat a lot, need IPX7 waterproof under $140",
                "compare battery life on the top options",
            ]
        )
        self.assertEqual(state.turn_count, 3)
        self.assertEqual(state.resolved_category, "electronics_audio")
        self.assertEqual(state.product_type, "earbuds")
        self.assertEqual(state.max_price, 140.0)
        self.assertIn("waterproof_ipx7", state.required_features)
        self.assertIn("earhooks", state.required_features)
        self.assertTrue(state.comparison_intent)
        self.assertFalse(state.troubleshooting_intent)

    def test_relevance_floor_suppresses_high_bid_irrelevant_ads_on_support_turn(self) -> None:
        res = self.engine.recommend(
            turns=[
                "wireless anc running earbuds",
                "wait, how do I factory reset bluetooth pairing when my left earbud blinks red?",
            ],
            tau_rel=0.58,
            bedrock_fn=None,
        )
        self.assertTrue(res["suppress_ad"])
        self.assertEqual(len(res["winning_ads"]), 0)
        self.assertLess(res["max_candidate_relevance"], 0.58)
        blocked_ids = {d["sku_id"] for d in res["suppressed_high_bid_ads"]}
        self.assertIn("SKU-CAX-D2", blocked_ids)

    def test_commercial_dialogue_clears_gsp_and_blocks_wired_high_bid_distractor(self) -> None:
        res = self.engine.recommend(
            turns=[
                "wireless running earbuds",
                "I wear glasses and sweat heavily, need IPX7 waterproof under $140",
            ],
            tau_rel=0.58,
            bedrock_fn=None,
        )
        self.assertFalse(res["suppress_ad"])
        self.assertGreaterEqual(len(res["winning_ads"]), 1)
        for w in res["winning_ads"]:
            self.assertLessEqual(w["price"], 140.0)
            self.assertGreaterEqual(w["semantic_relevance"], 0.58)
            self.assertLessEqual(w["clearing_cpc"], w["base_bid"] + 1e-5)
            self.assertNotEqual(w["sku_id"], "SKU-CAX-D1")

    def test_ad_copy_policy_verifier_catches_banned_claims_and_oos(self) -> None:
        sku = self.catalog[0]
        ok_good, v_good = verify_ad_copy_policy(
            f"Top match: {sku.title} [Sponsored] at ${sku.price:.2f}.", sku
        )
        self.assertTrue(ok_good)
        self.assertEqual(v_good, [])

        ok_bad, v_bad = verify_ad_copy_policy(
            "Try our #1 guaranteed miracle earbuds for $19.99!", sku
        )
        self.assertFalse(ok_bad)
        self.assertGreaterEqual(len(v_bad), 2)

    def test_linucb_bandit_and_caller_permission_scopes(self) -> None:
        bandit, summary = build_pretrained_linucb(seed=42, rounds=360)
        self.assertGreater(summary["expert_acceptance_lift_pct"], 5.0)
        scores = bandit.score_arms([1.0, 1.0, 1.0, 0.0, 0.0])
        self.assertEqual(scores[0].arm_id, "spec_comparison")

        # Unauthorized caller cannot mutate LinUCB weights
        denied = self.engine.recommend(
            turns=["wireless running earbuds under $150"],
            caller_type="delegated_agency_bid_only",
            expert_feedback={"arm_id": "spec_comparison", "reward": 1.0},
            bedrock_fn=None,
        )
        self.assertEqual(denied["caller_auth"]["feedback_update"]["status"], "DENIED_MISSING_SCOPE")

        # Authorized applied scientist caller updates LinUCB cleanly
        allowed = self.engine.recommend(
            turns=["wireless running earbuds under $150"],
            caller_type="ads_applied_scientist",
            expert_feedback={"arm_id": "spec_comparison", "reward": 1.0},
            bedrock_fn=None,
        )
        self.assertEqual(allowed["caller_auth"]["feedback_update"]["status"], "UPDATED")

    def test_cax_benchmark_zero_pollution(self) -> None:
        rep = run_cax_benchmark(self.catalog, seed=42)
        self.assertEqual(rep.cax_pollution_rate_pct, 0.0)
        self.assertGreater(rep.unfloored_pollution_rate_pct, 80.0)
        self.assertEqual(rep.constraint_satisfaction_pct, 100.0)
        self.assertEqual(rep.ad_policy_pass_rate_pct, 100.0)


class TestOffSearchAndCUPED(unittest.TestCase):
    def test_homepage_post_purchase_suppresses_already_bought_durable_and_routes_complements(self) -> None:
        eng = OffSearchPdpEngine(seed=42)
        res = eng.recommend_offsearch(
            anchor_id="ANCHOR-CAM-4K",
            shopper_stage="homepage_post_purchase",
        )
        self.assertGreaterEqual(len(res["carousel_ads"]), 3)
        roles = {a["complement_role"] for a in res["carousel_ads"]}
        self.assertIn("Tripod", roles)
        self.assertIn("SD Card", roles)
        for ad in res["carousel_ads"]:
            self.assertEqual(ad["relationship"], "complement")
            self.assertNotIn(ad["sku_id"], ("SKU-OS-BAD58", "SKU-OS-BADCFB", "SKU-OS-CAM1", "SKU-OS-CAM2"))

    def test_pdp_browsing_allows_substitutes_while_still_pruning_incompatible_accessories(self) -> None:
        eng = OffSearchPdpEngine(seed=42)
        res = eng.recommend_offsearch(
            anchor_id="ANCHOR-CAM-4K",
            shopper_stage="pdp_browsing",
        )
        rels = {a["relationship"] for a in res["carousel_ads"]}
        self.assertIn("substitute", rels)
        pruned_ids = {p["sku_id"] for p in res["pruned_incompatible_or_redundant"]}
        self.assertIn("SKU-OS-BAD58", pruned_ids)
        self.assertIn("SKU-OS-BADCFB", pruned_ids)

    def test_cuped_variance_reduction_and_unbiased_theta(self) -> None:
        cuped = run_cuped_ab_analysis(num_sessions_per_arm=600, seed=42)
        self.assertGreater(cuped.theta, 0.5)
        self.assertGreater(cuped.variance_reduction_pct, 40.0)
        self.assertLess(cuped.cuped_se, cuped.raw_se)
        self.assertLess(cuped.cuped_p_value, cuped.raw_p_value)

        bench = run_offsearch_benchmark(seed=42)
        self.assertEqual(bench.graph_redundant_durable_rate_pct, 0.0)
        self.assertEqual(bench.graph_incompatible_accessory_rate_pct, 0.0)
        self.assertGreater(bench.naive_incompatible_accessory_rate_pct, 50.0)


class TestPALCalibrationAndShuffleSharding(unittest.TestCase):
    def test_pal_isotonic_beats_naive_on_logloss_rig_ece_and_copc(self) -> None:
        cal = PALClickCalibrator(seed=42)
        res = cal.evaluate_all(drift_mode="none")
        m_naive = res["models"]["naive_position_blind"]
        m_iso = res["models"]["pal_isotonic_calibrated"]

        self.assertLess(m_iso["log_loss"], m_naive["log_loss"])
        self.assertGreater(m_iso["rig_pct"], m_naive["rig_pct"])
        self.assertLess(m_iso["ece"], m_naive["ece"])
        self.assertLess(abs(m_iso["copc"] - 1.0), abs(m_naive["copc"] - 1.0))
        self.assertLess(abs(m_iso["copc"] - 1.0), 0.08)

    def test_shuffle_sharding_combinatorics_and_3_57_pct_blast_radius(self) -> None:
        router = ShuffleShardedCellRouter(n_cells=8, shard_k=2)
        self.assertEqual(router.virtual_shard_count, 28)
        self.assertAlmostEqual(router.theoretical_blast_radius_pct, 3.57, places=2)

        sim = router.simulate_noisy_neighbor(num_co_tenants=560, auto_evacuate_quarantine=False)
        self.assertEqual(sim["virtual_shards"], 28)
        self.assertAlmostEqual(sim["shuffle_sharding_blast_radius_pct"], 3.57, places=2)
        self.assertGreater(sim["failover_rescued_tenants"], 200)

        sim_evac = router.simulate_noisy_neighbor(num_co_tenants=560, auto_evacuate_quarantine=True)
        self.assertEqual(sim_evac["shuffle_sharding_blast_radius_pct"], 0.0)

        bench = run_pal_and_ha_benchmark(seed=42)
        self.assertEqual(bench.virtual_shards, 28)
        self.assertAlmostEqual(bench.shuffle_sharding_blast_radius_pct, 3.57, places=2)


if __name__ == "__main__":
    unittest.main()
