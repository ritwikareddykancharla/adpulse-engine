# AdPulse — Sponsored Product Ranker, Hourly Budget Pacer & Seller Diagnostics

> **Live Demo:** [https://rggupypb65mdvo3uld5vfebzwy0hppos.lambda-url.us-west-2.on.aws/](https://rggupypb65mdvo3uld5vfebzwy0hppos.lambda-url.us-west-2.on.aws/)  
> **Health Check:** `https://rggupypb65mdvo3uld5vfebzwy0hppos.lambda-url.us-west-2.on.aws/api/health`  
> **Live API Call:** `curl -s -X POST https://rggupypb65mdvo3uld5vfebzwy0hppos.lambda-url.us-west-2.on.aws/api/run -H 'Content-Type: application/json' -d '{}'`

**AdPulse** ranks sponsored product ads, paces daily seller ad budgets across 24-hour traffic curves, evaluates bidding changes offline on historical logs, and diagnoses why seller campaigns lose impressions — preventing out-of-stock or overpriced ads from surfacing in search, stopping seller budgets from burning out before evening conversion peaks, and keeping bid recommendations below each seller's profitable ACOS cap.

---

## What Problem This Solves

- **Raw lexical + CPC bid ranking surfaces out-of-stock and overpriced SKUs:** Ranking keyword-matching ads primarily by raw advertiser bid leaks out-of-stock items (**15.28%** of baseline impressions) and violates shopper price ceilings (**59.72%** of impressions) while letting low-converting clearance bids crowd out relevant catalog items. Filtering structured catalog attributes in Stage 1 and reranking top-40 candidates by calibrated expected value (`bid * pCTR * pCVR + relevance_bonus`) eliminates **100%** of out-of-stock and price-cap violations and raises Top-3 Recall from **0.1861 to 0.8056** (**+332.89%** lift) at **0.266 ms p50 / 0.430 ms p99** latency.
- **Unpaced greedy bidding exhausts SMB daily budgets during low-intent morning traffic:** Retail search volume spikes during morning browse hours (`06:00–11:59`), while purchase conversion rates peak in the evening (`18:00–22:00`). Under unpaced greedy bidding across 6,000 auctions and 40 campaigns, **55.00%** of campaigns exhaust their daily budget before `18:00`. Shading bids via a Lagrangian PID controller (`effective_bid = base_bid * lambda_t`) plus a campaign ROAS floor eliminates early pre-18:00 budget depletion (**55.00% → 0.00%**, **-100.00%** reduction), lifts advertiser ROAS from **5.01x to 9.33x** (**+86.30%**), and increases 24-hour conversions by **+68.00%** with **0** budget cap violations (**89.01%** total budget utilization; **-7.56%** gross click spend).
- **Naive Inverse Propensity Scoring (IPS) suffers high variance on rare bid-shading explorations:** Production logging policies assign small propensities (`0.0015–0.015`) to shaded bid multipliers, producing heavy-tailed importance weights up to `~120x`. Across 40 logged propensity trace trials (`1,200` auctions each), Clipped IPS (`M=25`) lowers estimation RMSE by **26.68%** (`0.16606 → 0.12175`), while Doubly-Robust (DR) estimation lowers RMSE by **66.03%** (`0.16606 → 0.05641`) while remaining unbiased under direct-model calibration drift.
- **Unbounded bid recommendations push sellers past profitable ACOS targets:** Advising a seller to match median auction clearing CPC when their SKU has low conversion rate destroys unit economics, and recommending bid increases on out-of-stock SKUs wastes seller trust. `CampaignDiagnostics` aggregates SQLite auction loss logs across five root-cause codes (`100.0%` classification accuracy across 40 scenarios) and caps every bid recommendation at `price * calibrated_cvr * min(target_acos, gross_margin)` (**100.0%** ACOS ceiling compliance).

---

## System Architecture

> **Excalidraw Source:** [`docs/architecture.excalidraw`](docs/architecture.excalidraw)

![AdPulse Architecture Diagram](docs/architecture.svg)

---

## How It Works

1. **Two-Stage Catalog Retrieval & Calibrated eCPM Ranker (`adpulse/catalog_ranker.py`)**  
   - **Stage 1 (`retrieve_stage1`):** Builds an inverted posting index with BM25-style IDF weights over normalized title, brand, category, product-type, and structured attribute tokens (`500` SKUs across 8 retail categories). Enforces hard eligibility gates (`in_stock == True`, `price <= max_price` from structured fields or natural-language `"under $X"` intent, and `category == target_category`) before selecting top-40 candidates via `heapq.nlargest`.
   - **Stage 2 (`rank`):** Shrinks 30-day SKU click and conversion counts toward category-level Beta priors (`Empirical-Bayes` smoothing), applies brand, product-type, and structured attribute alignment multipliers, and orders candidates by `bid * pCTR * pCVR + relevance_bonus`.

2. **GSP / Second-Price Auction & Lagrangian PID Budget Pacer (`adpulse/auction_pacer.py`)**  
   - **Truthful Slot Clearing (`clear_gsp_auction`):** Ranks eligible bidders by `effective_bid * quality_score` above `reserve_cpc`. Each winning slot `j` pays per click `max(reserve_cpc, min(effective_bid, next_rank_score / quality_i))`, reducing to exact Vickrey second-price clearing when `num_slots=1`.
   - **PID Bid-Shading (`BudgetPacedAuctionEngine`):** Tracks a 24-hour conversion-weighted cumulative spend curve across `6,000` auctions (`40` enterprise + SMB campaigns). Updates per-campaign dual multiplier `lambda_t` (`Kp=1.85, Ki=0.45, Kd=0.22` with anti-windup clamping), suppresses low-intent auctions that violate the campaign ROAS floor at reserve, and caps per-click charges so `spent <= daily_budget` holds strictly.

3. **Clipped IPS & Doubly-Robust Off-Policy Evaluator (`adpulse/off_policy_eval.py`)**  
   - Evaluates candidate context-aware bid-shading policies on logged auction propensity traces (`propensity_score`, `target_prob`, `logged_reward`, `direct_model_hat`, `expected_direct_value`).
   - Computes point estimates, effective sample size (`ESS`), and **240-resample paired bootstrap 95% confidence intervals** across Naive IPS, Clipped IPS (`min(M, pi_e / pi_0)`), and Doubly-Robust (`E_pi_e[mu_hat] + (pi_e / pi_0) * (r - mu_hat)`), benchmarked against analytical ground-truth oracle policy values.

4. **SQLite Seller Campaign Diagnostic Engine (`adpulse/campaign_diagnostics.py`)**  
   - Ingests per-auction win/loss telemetry into an in-memory SQLite table (`auction_telemetry`) and classifies campaigns across `OUT_OF_STOCK_SUPPRESSED`, `MISSING_CATALOG_ATTRIBUTES`, `BUDGET_EXHAUSTED_EARLY`, `LOW_QUERY_RELEVANCE`, and `BID_BELOW_CLEARING_FLOOR`.
   - Enforces `recommended_bid <= price * calibrated_cvr * min(target_acos, gross_margin)`, holding bids flat on out-of-stock SKUs and recommending detail-page CVR / attribute fixes whenever clearing floors exceed profitable ACOS caps.

5. **CLI & HTTP / Lambda Handler (`adpulse/cli.py`, `adpulse/aws_lambda_handler.py`)**  
   - Exposes JSON telemetry over CLI (`--mode demo-all | ranker | pacing | ope | diagnostics`) and an AWS Lambda Function URL v2.0 / local HTTP server handler (`GET /api/health`, `POST /api/run`).

---

## Benchmark Results

Reproducible on Python 3.10+ (stdlib only, `seed=42`) via `python3 benchmarks/run_benchmarks.py` (saved in [`benchmarks/latest_results.json`](benchmarks/latest_results.json)):

| Subsystem & Metric | Measured Result | Baseline / Comparison |
| :--- | :---: | :--- |
| **Catalog Ranker — Top-3 Recall (`500` SKUs, `120` Queries)** | **0.8056** (**+332.89%** lift) | `0.1861` lexical + raw CPC bid baseline |
| **Catalog Ranker — NDCG@3 & Expected Conversions** | **0.8331 NDCG@3** (**+312.43%**) | `0.2020` baseline (`20.53` vs `1.90` conv / 1k queries) |
| **Catalog Ranker — OOS & Price-Cap Violation Rate** | **0.0% OOS** / **0.0% price-cap** | `15.28%` OOS / `59.72%` price-cap violations in baseline |
| **Catalog Ranker — Query Latency (`p50` / `p99`)** | **0.266 ms p50** / **0.430 ms p99** | Inverted posting index + top-40 Stage-2 eCPM rerank |
| **Auction Pacer — Early Budget Depletion (`< 18:00`)** | **0.00%** (`0 / 40` campaigns; **-100.00%**) | `55.00%` (`22 / 40` campaigns) under unpaced greedy bidding |
| **Auction Pacer — Advertiser ROAS & 24h Conversions** | **9.33x ROAS** (**+86.30%** lift) | `5.01x` unpaced ROAS; **+68.00%** conversions (`126` vs `75`) |
| **Auction Pacer — Budget Utilization & Clearing Yield** | **89.01% utilization** (**0** cap violations) | `-7.56%` gross click spend (`$1,309.03` vs `$1,416.14`); `+72.2%` GMV |
| **Off-Policy Eval — Doubly-Robust RMSE vs Naive IPS** | **0.05641 RMSE** (**-66.03%** vs IPS) | Naive IPS: `0.16606` • Clipped IPS (`M=25`): `0.12175` (**-26.68%**) |
| **Seller Diagnostics — Root-Cause & ACOS Safety (`40` Cases)** | **100.0% root-cause accuracy** | **100.0%** ACOS ceiling compliance & **100.0%** OOS hold safety |

---

## Quickstart

Requires Python 3.10+ with zero external dependencies:

```bash
# 1. Run the 29-test unit and integration test suite
python3 -m unittest discover -s tests -v

# 2. Reproduce all deterministic benchmark metrics in benchmarks/latest_results.json
python3 benchmarks/run_benchmarks.py

# 3. Run subsystem JSON telemetry via CLI
python3 -m adpulse.cli --mode demo-all
python3 -m adpulse.cli --mode ranker
python3 -m adpulse.cli --mode pacing
python3 -m adpulse.cli --mode ope
python3 -m adpulse.cli --mode diagnostics

# 4. Run local HTTP server matching the AWS Lambda Function URL contract
python3 -m adpulse.aws_lambda_handler --port 8086
```
