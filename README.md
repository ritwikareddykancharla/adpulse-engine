# AdPulse v2: Sponsored Search, Conversational Ads (CAX), Off-Search PDP, PAL Calibration & GSP Pacer

**Live Production App (Amazon CloudFront + Private S3 OAC + `/api/*` + Bedrock Nova):** https://d1g4bbd5r2rls1.cloudfront.net  
**Health Check:** `https://d1g4bbd5r2rls1.cloudfront.net/api/health`  
**Live Storefront Auction API:** `curl -s -X POST https://d1g4bbd5r2rls1.cloudfront.net/api/storefront/search -H 'Content-Type: application/json' -d '{"query":"wireless anc running earbuds","hour":19,"daily_budget":28,"pacing_mode":"pid_paced"}'`  
**Live Conversational Shopping (CAX) API:** `curl -s -X POST https://d1g4bbd5r2rls1.cloudfront.net/api/cax/recommend -H 'Content-Type: application/json' -d '{"turns":["wireless running earbuds","I wear glasses and sweat heavily, need IPX7 waterproof under $140"],"tau_rel":0.58}'`  
**Live Off-Search PDP & CUPED API:** `curl -s -X POST https://d1g4bbd5r2rls1.cloudfront.net/api/offsearch/pdp -H 'Content-Type: application/json' -d '{"anchor_id":"ANCHOR-CAM-4K","shopper_stage":"homepage_post_purchase"}'`  
**Live PAL Calibration & Shuffle-Sharding API:** `curl -s -X POST https://d1g4bbd5r2rls1.cloudfront.net/api/pal/calibrate -H 'Content-Type: application/json' -d '{"drift_mode":"none","n_cells":8,"shard_k":2}'`

**AdPulse v2** ranks sponsored product ads across a 520-SKU retail catalog, runs multi-turn Sponsored Conversational Shopping (`CAX`) with a semantic relevance threshold floor (`tau_rel`) and `LinUCB` contextual bandit framing, sources query-free Off-Search (`PDP` and `Homepage`) substitutes vs. compatible accessories evaluated via `CUPED` variance reduction, calibrates position-debiased `PAL` click models alongside `C(8, 2) = 28` multi-AZ shuffle sharding, paces 24-hour seller budgets via a Lagrangian PID controller, evaluates counterfactual bid multipliers via Doubly-Robust estimation, and synthesizes ACOS-bounded campaign guidance via **Amazon Bedrock (`us.amazon.nova-lite-v1:0`)**.

![AdPulse CloudFront Architecture](docs/architecture.svg)

---

## What You Can Do Live in the Web Studio

1. **Live Storefront & 3-Slot GSP Clearinghouse (`/api/storefront/search`)**:
   - Search with 1-click shopper queries (*"wireless anc running earbuds"*, *"4k usb-c monitor macbook"*, *"ergonomic mesh office chair"*, *"espresso grinder burr"*) or enter any custom query, and scrub **Hour of Day (`00:00` to `23:00`)** and **Daily Campaign Budget Cap**.
   - Inspect **Winning Sponsored Slots `#1`, `#2`, `#3`** with calibrated Empirical-Bayes `pCTR`, hourly `pCVR`, `eCPM`, truthful **Second-Price GSP Clearing CPC**, `0.158 ms p50 / 0.217 ms p99` rank latency, and the Stage-1 rejection drawer blocking out-of-stock, price-violating, or early-depleted SKUs.
2. **24-Hour PID Budget Pacing Curve & Doubly-Robust Bid Simulator (`/api/pacing/simulate`)**:
   - Compare **Unpaced Greedy Bidding** (`55.0%` of campaigns exhaust daily budget before the `18:00-22:00` Prime conversion peak) against **Lagrangian PID Bid Shading** (`+86.30%` ROAS, `0` pre-18:00 depletions, `0` budget cap violations), and test counterfactual bid multipliers (`0.55x` to `1.30x`) via **Doubly-Robust Off-Policy Evaluation** (`-66.03%` RMSE vs. Naive IPS) with 240x paired bootstrap 95% CIs.
3. **Amazon Bedrock (`us.amazon.nova-lite-v1:0`) Seller Campaign & Spektr Iceberg Copilot (`/api/copilot/recommend`)**:
   - Diagnose campaign impression losses in SQLite across 5 loss archetypes and generate concrete bid, negative-keyword, hourly pacing, and Spektr Iceberg SQL recommendations strictly bounded by `recommended_bid <= price * calibrated_cvr * min(target_acos, gross_margin)` (`100.0%` ACOS compliance).
4. **Sponsored Conversational Shopping (`CAX`), Off-Search `PDP` Compatibility & `PAL` Calibration (`/api/cax/recommend`, `/api/offsearch/pdp`, `/api/pal/calibrate`)**:
   - **Multi-Turn `CAX` + Relevance Floor (`tau_rel`) + `LinUCB`**: Extract cumulative shopper constraints across turns, suppress high-bid (`$4.95`) irrelevant ads (`suppress_ad = True`) whenever `semantic_relevance < tau_rel`, verify Amazon Ads copy policy (`100.0%` pass rate), select framing via a ridge-regression `LinUCB` contextual bandit (`+61.06%` expert alignment lift), and synthesize grounded answers on **Amazon Bedrock Nova**.
   - **Query-Free Off-Search (`PDP` & `Homepage`) + `CUPED` A/B Testing**: Suppress already-purchased durable categories (`100.0% -> 0.0%` redundant camera/espresso impressions on Homepage), prune physically incompatible accessories (`58mm` filter on `67mm` lens, `CFexpress-B` on `SDXC` body), route compatible `Tripod` and `SD Card` complements vs. pre-purchase substitutes, and accelerate online A/B tests with `CUPED` (`theta = 0.8107`, **`-58.27%` variance reduction**).
   - **Position-Bias-Aware Learning (`PAL`) `pCTR` Calibration & `C(8, 2) = 28` Shuffle Sharding**: Factorize `P(click | x, pos) = P(seen | pos) * P(click | x, seen)` with Platt and PAV Isotonic calibration (`COPC` tightened from `1.1844` to **`1.0016`**, `RIG` lifted from `1.48%` to **`8.01%`**, `ECE` cut from `0.04103` to **`0.01788`**), while isolating multi-AZ noisy-neighbor retry storms from `100.0%` shared-fleet impact down to **`3.57%`** (`1/28` virtual shards).

---

## Measured Benchmark Results (`benchmarks/run_benchmarks.py`)

| Subsystem & Metric | Measured Result | Baseline / Comparison |
| :--- | :---: | :--- |
| **Catalog Ranker: Top-3 Recall (`500` SKUs, `120` Queries)** | **0.8056** (**+332.89%** lift) | `0.1861` lexical + raw CPC bid baseline |
| **Catalog Ranker: NDCG@3 & Expected Conversions** | **0.8331 NDCG@3** (**+312.43%**) | `0.2020` baseline (`20.53` vs `1.90` conv / 1k queries, **+980.53%**) |
| **Catalog Ranker: OOS & Price-Cap Violation Rate** | **0.0% OOS** / **0.0% price-cap** | `15.28%` OOS / `59.72%` price-cap violations in baseline |
| **Catalog Ranker: Query Latency (`p50` / `p99`)** | **0.158 ms p50** / **0.217 ms p99** | Inverted posting index + top-40 Stage-2 eCPM rerank |
| **Auction Pacer: Early Budget Depletion (`< 18:00`)** | **0.00%** (`0 / 40` campaigns; **-100.00%**) | `55.00%` (`22 / 40` campaigns) under unpaced greedy bidding |
| **Auction Pacer: Advertiser ROAS & 24h Conversions** | **9.33x ROAS** (**+86.30%** lift) | `5.01x` unpaced ROAS; **+68.00%** conversions (`126` vs `75`) |
| **Off-Policy Eval: Doubly-Robust RMSE vs Naive IPS** | **0.05641 RMSE** (**-66.03%** vs IPS) | Naive IPS: `0.16606` · Clipped IPS (`M=25`): `0.12175` (**-26.68%**) |
| **Seller Diagnostics: Root-Cause & ACOS Safety (`40` Cases)** | **100.0% root-cause accuracy** | **100.0%** ACOS ceiling compliance & **100.0%** OOS hold safety |
| **Conversational Ads (`CAX`): Ad Floor & `LinUCB` (`60` Dialogues)** | **0.0% irrelevant ad leaks** | `100.0%` unfloored leak rate · **+61.06%** `LinUCB` expert lift · `100.0%` policy pass |
| **Off-Search `PDP`/`Homepage` & `CUPED` (`240` + `1,200` Sessions)** | **-58.27% CUPED variance** (`theta = 0.8107`) | `0.0%` redundant durables (`100.0%` naive) · `0.0%` incompatible specs (`+6.41%` RPM) |
| **`PAL` `pCTR` Calibration & `C(8, 2) = 28` Shuffle Sharding** | **1.0016 COPC** · **8.01% RIG** · **3.57% blast** | Naive `1.1844` COPC, `1.48%` RIG, `0.04103 -> 0.01788` ECE · `100.0%` shared-fleet blast |

---

## Run Locally & Deploy to Amazon CloudFront

```bash
# 1. Run all 44 unit & integration tests + deterministic benchmarks
PYTHONPATH=. python3 -m unittest discover -s tests -v
PYTHONPATH=. python3 benchmarks/run_benchmarks.py

# 2. Start the interactive studio locally
PYTHONPATH=. python3 -m adpulse.server
# -> Open http://127.0.0.1:8086

# 3. Deploy to Amazon CloudFront + Private S3 (OAC) + /api/* + Amazon Bedrock (us-west-2)
chmod +x infra/deploy_cloudfront.sh
./infra/deploy_cloudfront.sh
```
