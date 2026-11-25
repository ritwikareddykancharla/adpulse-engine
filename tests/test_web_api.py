"""Integration tests for AdPulse Storefront, Pacing, OPE & Copilot Web API (`tests/test_web_api.py`)."""

from __future__ import annotations

import json
import os
import unittest
from unittest.mock import patch

from adpulse.server import lambda_handler as server_lambda_handler
from adpulse.web_api import lambda_handler


class TestAdPulseWebApi(unittest.TestCase):
    def test_health_and_bootstrap_endpoints(self) -> None:
        h_resp = lambda_handler({"httpMethod": "GET", "rawPath": "/api/health"})
        self.assertEqual(h_resp["statusCode"], 200)
        h_body = json.loads(h_resp["body"])
        self.assertEqual(h_body["status"], "ok")
        self.assertEqual(h_body["bedrock_model"], "us.amazon.nova-lite-v1:0")

        b_resp = lambda_handler({"httpMethod": "GET", "rawPath": "/api/bootstrap"})
        self.assertEqual(b_resp["statusCode"], 200)
        b_body = json.loads(b_resp["body"])
        self.assertEqual(len(b_body["featured_queries"]), 4)
        self.assertEqual(len(b_body["default_storefront"]["winning_ads"]), 3)
        self.assertGreater(len(b_body["default_storefront"]["filtered_out_drawer"]), 0)
        self.assertEqual(len(b_body["default_pacing"]["hourly_curve"]), 24)
        self.assertEqual(len(b_body["seller_campaigns"]), 5)

    def test_storefront_search_prunes_oos_and_budget_depleted_skus(self) -> None:
        resp_greedy = lambda_handler(
            {
                "httpMethod": "POST",
                "rawPath": "/api/storefront/search",
                "body": json.dumps(
                    {
                        "query": "wireless anc running earbuds",
                        "hour": 20,
                        "daily_budget": 24,
                        "pacing_mode": "unpaced_greedy",
                        "shopper_price_cap": 160,
                    }
                ),
            }
        )
        self.assertEqual(resp_greedy["statusCode"], 200)
        greedy_body = json.loads(resp_greedy["body"])
        reasons = greedy_body["filtered_out_counts_by_reason"]
        self.assertGreater(reasons.get("OUT_OF_STOCK_SUPPRESSED", 0), 0)
        self.assertGreater(reasons.get("BUDGET_EXHAUSTED_EARLY", 0), 0)
        for winner in greedy_body["winning_ads"]:
            self.assertLessEqual(winner["price"], 160.0)
            self.assertLessEqual(winner["clearing_cpc"], winner["effective_bid"] + 1e-4)

        resp_pid = lambda_handler(
            {
                "httpMethod": "POST",
                "rawPath": "/api/storefront/search",
                "body": json.dumps(
                    {
                        "query": "4k usb-c monitor macbook",
                        "hour": 20,
                        "daily_budget": 28,
                        "pacing_mode": "pid_paced",
                        "shopper_price_cap": 340,
                    }
                ),
            }
        )
        self.assertEqual(resp_pid["statusCode"], 200)
        pid_body = json.loads(resp_pid["body"])
        self.assertEqual(pid_body["filtered_out_counts_by_reason"].get("BUDGET_EXHAUSTED_EARLY", 0), 0)
        self.assertEqual(len(pid_body["winning_ads"]), 3)

    def test_pacing_simulator_and_offline_copilot_guardrail(self) -> None:
        sim_resp = lambda_handler(
            {
                "httpMethod": "POST",
                "rawPath": "/api/pacing/simulate",
                "body": json.dumps(
                    {
                        "daily_budget": 26,
                        "selected_hour": 19,
                        "bid_multiplier": "0.85",
                    }
                ),
            }
        )
        self.assertEqual(sim_resp["statusCode"], 200)
        sim_body = json.loads(sim_resp["body"])
        self.assertGreater(sim_body["roas_lift_pct"], 40.0)
        self.assertIn("doubly_robust", sim_body["off_policy_evaluation"]["evaluation"])

        with patch.dict(os.environ, {"ADPULSE_OFFLINE_LLM": "1"}):
            cop_resp = lambda_handler(
                {
                    "httpMethod": "POST",
                    "rawPath": "/api/copilot/recommend",
                    "body": json.dumps({"campaign_id": "CMP-D005", "target_acos": 0.25}),
                }
            )
        self.assertEqual(cop_resp["statusCode"], 200)
        cop_body = json.loads(cop_resp["body"])
        self.assertTrue(cop_body["acos_guardrail_verified"])
        self.assertLessEqual(
            cop_body["diagnostic"]["recommended_bid"],
            cop_body["diagnostic"]["max_acos_cpc"] + 1e-6,
        )

    def test_v2_cax_offsearch_and_pal_endpoints(self) -> None:
        with patch.dict(os.environ, {"ADPULSE_OFFLINE_LLM": "1"}):
            cax_resp = server_lambda_handler(
                {
                    "httpMethod": "POST",
                    "rawPath": "/api/cax/recommend",
                    "body": json.dumps(
                        {
                            "turns": [
                                "wireless running earbuds",
                                "need ipx7 waterproof under $140",
                            ],
                            "tau_rel": 0.58,
                        }
                    ),
                }
            )
            self.assertEqual(cax_resp["statusCode"], 200)
            cax_body = json.loads(cax_resp["body"])
            self.assertFalse(cax_body["suppress_ad"])
            self.assertGreaterEqual(len(cax_body["winning_ads"]), 1)

            cax_sup = server_lambda_handler(
                {
                    "httpMethod": "POST",
                    "rawPath": "/api/cax/recommend",
                    "body": json.dumps(
                        {
                            "turns": ["how do I factory reset bluetooth pairing when one earbud blinks red?"],
                            "tau_rel": 0.58,
                        }
                    ),
                }
            )
            self.assertEqual(cax_sup["statusCode"], 200)
            self.assertTrue(json.loads(cax_sup["body"])["suppress_ad"])

        pdp_resp = server_lambda_handler(
            {
                "httpMethod": "POST",
                "rawPath": "/api/offsearch/pdp",
                "body": json.dumps(
                    {
                        "anchor_id": "ANCHOR-CAM-4K",
                        "shopper_stage": "homepage_post_purchase",
                    }
                ),
            }
        )
        self.assertEqual(pdp_resp["statusCode"], 200)
        pdp_body = json.loads(pdp_resp["body"])
        self.assertGreater(pdp_body["cuped_experiment"]["variance_reduction_pct"], 40.0)
        for ad in pdp_body["carousel_ads"]:
            self.assertEqual(ad["relationship"], "complement")

        pal_resp = server_lambda_handler(
            {
                "httpMethod": "POST",
                "rawPath": "/api/pal/calibrate",
                "body": json.dumps({"drift_mode": "none", "n_cells": 8, "shard_k": 2}),
            }
        )
        self.assertEqual(pal_resp["statusCode"], 200)
        pal_body = json.loads(pal_resp["body"])
        self.assertEqual(pal_body["shuffle_sharding"]["virtual_shards"], 28)
        self.assertAlmostEqual(
            pal_body["shuffle_sharding"]["shuffle_sharding_blast_radius_pct"], 3.57, places=2
        )

    def test_ui_and_readme_strict_guardrails(self) -> None:
        from pathlib import Path

        root = Path(__file__).resolve().parent.parent
        html_text = (root / "web" / "index.html").read_text(encoding="utf-8")
        readme_text = (root / "README.md").read_text(encoding="utf-8")

        for fname, text in (("web/index.html", html_text), ("README.md", readme_text)):
            self.assertNotIn("—", text, f"Em-dash found in {fname}")
            self.assertNotIn("\\u2014", text, f"Escaped em-dash found in {fname}")
            self.assertNotIn("&mdash;", text, f"HTML em-dash found in {fname}")
            self.assertNotIn("localStorage", text, f"localStorage found in {fname}")
            self.assertNotIn("sessionStorage", text, f"sessionStorage found in {fname}")
            for banned in ("in plain english", "under the hood", "stdlib", "zero-dependency"):
                self.assertNotIn(banned, text.lower(), f"Banned phrase '{banned}' in {fname}")
            self.assertNotRegex(text.lower(), r"why\s+\w+\s+fails", f"Banned 'Why X Fails' in {fname}")

        self.assertIn("/api/cax/recommend", html_text)
        self.assertIn("/api/offsearch/pdp", html_text)
        self.assertIn("/api/pal/calibrate", html_text)


if __name__ == "__main__":
    unittest.main()
