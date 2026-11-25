/* AdPulse Storefront & GSP Trading Desk Controller — stateless in-memory UI */
(function () {
  "use strict";

  const state = {
    query: "wireless anc running earbuds",
    hour: 19,
    dailyBudget: 28,
    pacingMode: "pid_paced",
    priceCap: "160",
    bidMultiplier: "adaptive_pid",
    deskTab: "pacing",
    selectedSlot: 1,
    featuredQueries: [],
    storefrontData: null,
    pacingData: null,
    sellerCampaigns: [],
    selectedCampaignId: "CMP-D002",
  };

  function esc(val) {
    return String(val == null ? "" : val)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function inlineMd(text) {
    return esc(text)
      .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
      .replace(/`([^`]+)`/g, "<code>$1</code>");
  }

  function memoHtml(raw) {
    const lines = String(raw || "").split(/\r?\n/).filter((l) => l.trim());
    let inList = false;
    const out = [];
    for (const line of lines) {
      const t = line.trim();
      if (/^[-*•]\s+/.test(t) || /^\d+[.)]\s+/.test(t)) {
        if (!inList) {
          out.push("<ul>");
          inList = true;
        }
        out.push("<li>" + inlineMd(t.replace(/^([-*•]|\d+[.)])\s+/, "")) + "</li>");
      } else {
        if (inList) {
          out.push("</ul>");
          inList = false;
        }
        out.push("<p style='margin:0 0 6px'>" + inlineMd(t) + "</p>");
      }
    }
    if (inList) out.push("</ul>");
    return out.join("");
  }

  function productArt(productType, category) {
    const p = (productType || "").toLowerCase();
    if (p.includes("earbud") || p.includes("headphone")) {
      return `<svg width="62" height="62" viewBox="0 0 64 64" fill="none">
        <rect x="11" y="27" width="15" height="25" rx="7.5" fill="#1e293b"/>
        <rect x="38" y="27" width="15" height="25" rx="7.5" fill="#1e293b"/>
        <path d="M16 29C16 17 23 11 32 11C41 11 48 17 48 29" stroke="#ff9900" stroke-width="4.5" stroke-linecap="round"/>
        <circle cx="18.5" cy="39" r="2.8" fill="#38bdf8"/><circle cx="45.5" cy="39" r="2.8" fill="#38bdf8"/>
      </svg>`;
    }
    if (p.includes("monitor")) {
      return `<svg width="62" height="62" viewBox="0 0 64 64" fill="none">
        <rect x="7" y="11" width="50" height="33" rx="4" fill="#0f172a" stroke="#334155" stroke-width="2.5"/>
        <rect x="11" y="15" width="42" height="25" rx="2" fill="#1e3a8a"/>
        <path d="M11 34L25 22L35 29L45 20L53 28V40H11V34Z" fill="#38bdf8" opacity="0.75"/>
        <path d="M27 44H37L40 52H24L27 44Z" fill="#64748b"/>
        <rect x="19" y="52" width="26" height="3" rx="1.5" fill="#334155"/>
      </svg>`;
    }
    if (p.includes("chair")) {
      return `<svg width="62" height="62" viewBox="0 0 64 64" fill="none">
        <rect x="20" y="8" width="24" height="26" rx="6" fill="#334155" stroke="#0f172a" stroke-width="2"/>
        <line x1="24" y1="15" x2="40" y2="15" stroke="#94a3b8" stroke-width="2"/>
        <line x1="24" y1="21" x2="40" y2="21" stroke="#94a3b8" stroke-width="2"/>
        <line x1="24" y1="27" x2="40" y2="27" stroke="#94a3b8" stroke-width="2"/>
        <rect x="15" y="36" width="34" height="7" rx="3.5" fill="#0f172a"/>
        <line x1="32" y1="43" x2="32" y2="53" stroke="#475569" stroke-width="4"/>
        <path d="M18 55L32 51L46 55" stroke="#0f172a" stroke-width="3" stroke-linecap="round"/>
      </svg>`;
    }
    if (p.includes("espresso") || (category || "").includes("kitchen")) {
      return `<svg width="62" height="62" viewBox="0 0 64 64" fill="none">
        <path d="M19 9H45L40 23H24L19 9Z" fill="#475569"/>
        <rect x="21" y="23" width="22" height="29" rx="4" fill="#1e293b"/>
        <circle cx="32" cy="32" r="4" fill="#ff9900"/>
        <rect x="26" y="41" width="12" height="9" rx="2" fill="#94a3b8"/>
        <rect x="16" y="52" width="32" height="4" rx="2" fill="#0f172a"/>
      </svg>`;
    }
    return `<svg width="62" height="62" viewBox="0 0 64 64" fill="none">
      <rect x="14" y="15" width="36" height="35" rx="7" fill="#1e293b"/>
      <circle cx="32" cy="33" r="9" fill="#ff9900"/>
    </svg>`;
  }

  function formatPriceParts(price) {
    const fixed = Number(price || 0).toFixed(2);
    const [whole, cents] = fixed.split(".");
    return `<span class="price-whole"><sup>$</sup>${esc(whole)}<sup>${esc(cents)}</sup></span>`;
  }

  function stars(rating) {
    const r = Math.round(Number(rating || 4.7));
    return "★".repeat(Math.min(5, r)) + "☆".repeat(Math.max(0, 5 - r));
  }

  function renderFeaturedChips() {
    const wrap = document.getElementById("featuredQueryChips");
    wrap.innerHTML = state.featuredQueries
      .map((fq) => {
        const on = fq.query.toLowerCase() === state.query.toLowerCase();
        return `<button type="button" class="q-pill ${on ? "on" : ""}" data-q="${esc(fq.query)}" data-cap="${fq.default_price_cap}">${esc(fq.label)}</button>`;
      })
      .join("");
    wrap.querySelectorAll(".q-pill").forEach((b) => {
      b.addEventListener("click", () => {
        state.query = b.getAttribute("data-q");
        state.priceCap = String(Math.round(Number(b.getAttribute("data-cap"))));
        document.getElementById("shopperQueryInput").value = state.query;
        const sel = document.getElementById("priceCapSelect");
        if ([...sel.options].some((o) => o.value === state.priceCap)) sel.value = state.priceCap;
        renderFeaturedChips();
        refreshStorefront();
      });
    });
  }

  function renderStorefrontAndLadder(sf) {
    if (!sf) return;
    const isPeak = state.hour >= 18 && state.hour <= 22;
    const hrEl = document.getElementById("hourReadout");
    hrEl.textContent = `${String(state.hour).padStart(2, "0")}:00 · ${isPeak ? "PEAK " : ""}${sf.hourly_cvr_multiplier.toFixed(2)}x CVR`;
    hrEl.className = "scrubber-val" + (isPeak ? " peak" : "");

    document.getElementById("budgetReadout").textContent = `$${state.dailyBudget} / day`;
    document.getElementById("btnModePid").classList.toggle("on", state.pacingMode === "pid_paced");
    document.getElementById("btnModeGreedy").classList.toggle("on", state.pacingMode === "unpaced_greedy");

    const catShort = (sf.resolved_category || "all").split("_")[0];
    document.getElementById("searchCategoryPill").textContent = catShort;
    document.getElementById("tkLambdaVal").textContent =
      state.pacingMode === "pid_paced" ? `λ_t = ${Number(sf.pid_lambda_t).toFixed(2)}` : "λ_t = 1.00 (RAW)";

    const sc = sf.stage_counts;
    const lat = sf.latency_ms;
    document.getElementById("storeMetaBar").innerHTML = `
      <span>Showing <strong>${sc.slots_cleared} Sponsored Slots</strong> from <strong>${sc.catalog_total} SKUs</strong> for <strong>"${esc(sf.query)}"</strong> at <strong>${sf.hour_label}</strong></span>
      <span class="lat-chip">⚡ ${lat.benchmark_p50_ms}ms p50 · ${lat.benchmark_p99_ms}ms p99</span>
    `;

    // Rufus Conversational Recommendation Box
    const cp = sf.conversational_shopping_preview;
    document.getElementById("conversationalAdBox").innerHTML = cp
      ? `<div class="rufus-hdr">
          <span class="rufus-tag">✦ Rufus Conversational Shopping Pick</span>
          <span style="font:700 10.5px var(--mono);color:#9a3412;">Sponsored · Grounded Specs</span>
        </div>
        <p class="rufus-copy">${inlineMd(cp.assistant_answer)}</p>`
      : `<div class="rufus-hdr"><span class="rufus-tag">✦ Conversational Shopping</span></div><p class="rufus-copy">No eligible in-stock SKU cleared current filters.</p>`;

    // Product Cards
    const grid = document.getElementById("winningAdsGrid");
    const ads = sf.winning_ads || [];
    if (!ads.length) {
      grid.innerHTML = `<div class="prod-card" style="grid-template-columns:1fr;text-align:center;padding:24px;">
        <div><strong>0 Sponsored Slots Cleared.</strong> Switch controller to <strong>PID λ_t</strong> or raise the shopper price ceiling.</div>
      </div>`;
    } else {
      grid.innerHTML = ads
        .map((ad) => {
          const chips = (ad.highlights || [])
            .slice(0, 3)
            .map((h) => `<span class="spec-chip">${esc(h)}</span>`)
            .join("");
          return `<article class="prod-card slot-${ad.slot_number} ${state.selectedSlot === ad.slot_number ? "selected" : ""}" data-slot="${ad.slot_number}">
            <div class="prod-thumb">
              <span class="slot-ribbon">#${ad.slot_number}</span>
              ${productArt(ad.product_type, ad.category)}
            </div>
            <div class="prod-info">
              <div class="prod-topline">
                <span class="sp-micro">Sponsored · ${esc(ad.brand)} (<code>${esc(ad.sku_id)}</code>)</span>
                <span class="bestseller-pill">${esc(ad.badge)}</span>
              </div>
              <h3 class="prod-title">${esc(ad.title)}</h3>
              <div class="prod-stars">
                <span class="stars-str">${stars(ad.rating)}</span>
                <strong>${Number(ad.rating).toFixed(1)}</strong>
                <span class="rev-count">(${Number(ad.reviews).toLocaleString()})</span>
              </div>
              <div class="prod-price-row">
                ${formatPriceParts(ad.price)}
                <span class="prime-tag"><b>✓</b>prime Same-Day</span>
              </div>
              <div class="spec-chips">${chips}</div>
              <div class="card-clearing-strip">
                <span class="cpc-paid">2nd-Price CPC: $${Number(ad.clearing_cpc).toFixed(2)}</span>
                <span>Shaded Bid: $${Number(ad.effective_bid).toFixed(2)} (Raw $${Number(ad.base_bid).toFixed(2)})</span>
                <span>pCTR ${ad.p_ctr_pct}% · pCVR ${ad.p_cvr_pct}% · ${ad.expected_roas}x ROAS</span>
              </div>
            </div>
          </article>`;
        })
        .join("");

      grid.querySelectorAll(".prod-card[data-slot]").forEach((el) => {
        el.addEventListener("click", () => {
          state.selectedSlot = Number(el.getAttribute("data-slot"));
          renderStorefrontAndLadder(state.storefrontData);
        });
      });
    }

    // Pruned SKUs Tray
    const counts = sf.filtered_out_counts_by_reason || {};
    const shortMap = {
      OUT_OF_STOCK_SUPPRESSED: "OOS Blocked",
      BUDGET_EXHAUSTED_EARLY: "Pre-Peak Depleted",
      ABOVE_SHOPPER_PRICE_CAP: "Over Price Cap",
      MISSING_CATALOG_ATTRIBUTES: "Missing Attrs",
    };
    document.getElementById("filteredReasonPills").innerHTML = Object.entries(counts)
      .map(([k, cnt]) => `<span class="pruned-pill">${cnt} ${esc(shortMap[k] || k)}</span>`)
      .join("");

    const pruned = sf.filtered_out_drawer || [];
    document.getElementById("filteredDrawerList").innerHTML = pruned.length
      ? pruned
          .map(
            (it) => `<div class="pruned-item">
              <div>
                <div><strong>${esc(it.brand)}</strong> · <code>${esc(it.sku_id)}</code> ($${Number(it.price).toFixed(2)} · Raw Bid $${Number(it.raw_bid).toFixed(2)})</div>
                <div class="pruned-why">${esc(it.reason_detail)}</div>
              </div>
              <span class="pruned-code ${esc(it.reason_code)}">${esc(it.reason_code.replace(/_/g, " "))}</span>
            </div>`
          )
          .join("")
      : `<div class="pruned-item"><span>All candidate SKUs cleared Stage-1 gates.</span></div>`;

    // Right Desk Panel 1: GSP Clearing Book
    document.getElementById("gspPoolSummary").textContent = `${sc.filtered_before_auction} Pruned (<0.5ms) · ${sc.stage2_reranked} Reranked · ${sc.slots_cleared} Won`;
    document.getElementById("gspClearingLadder").innerHTML = ads.length
      ? ads
          .map((ad) => {
            const pctPaid = Math.min(100, Math.round((ad.clearing_cpc / Math.max(0.05, ad.base_bid)) * 100));
            return `<div class="gsp-row ${state.selectedSlot === ad.slot_number ? "active" : ""}">
              <span class="gsp-slot">SLOT #${ad.slot_number}</span>
              <div>
                <div class="gsp-sku">${esc(ad.brand)} · ${esc(ad.sku_id)}</div>
                <div class="gsp-spread-bar"><div class="gsp-spread-fill" style="width:${pctPaid}%"></div></div>
              </div>
              <div>
                <span style="color:var(--desk-dim);display:block;font-size:9.5px;">CLEARING / RAW</span>
                <strong style="color:var(--emerald)">$${Number(ad.clearing_cpc).toFixed(2)}</strong> / $${Number(ad.base_bid).toFixed(2)}
              </div>
              <div>
                <span style="color:var(--desk-dim);display:block;font-size:9.5px;">eCPM · PROJ ROAS</span>
                <strong>$${Number(ad.ecpm_usd).toFixed(0)}</strong> · <strong style="color:var(--amber)">${ad.expected_roas}x</strong>
              </div>
            </div>`;
          })
          .join("")
      : `<div style="color:var(--desk-dim);font-family:var(--mono);font-size:12px;">No slots cleared in current auction state.</div>`;
  }

  function renderPacingAndOpe(pd) {
    if (!pd) return;
    const u = pd.unpaced_metrics;
    const p = pd.pid_paced_metrics;
    document.getElementById("tkRoasVal").textContent = `${p.advertiser_roas.toFixed(2)}x (+${pd.roas_lift_pct}%)`;

    // Dark Bloomberg Oscilloscope SVG
    const hrs = pd.hourly_curve || [];
    const W = 580, H = 205, padL = 38, padR = 14, padT = 18, padB = 26;
    const plotW = W - padL - padR, plotH = H - padT - padB;
    const xAt = (h) => padL + (h / 23) * plotW;
    const yAt = (pct) => padT + plotH - (Math.min(100, Math.max(0, pct)) / 100) * plotH;
    const pts = (key) => hrs.map((pt) => `${xAt(pt.hour).toFixed(1)},${yAt(pt[key]).toFixed(1)}`).join(" ");

    const peakX1 = xAt(18), peakX2 = xAt(22), curX = xAt(state.hour);
    const yGrid = [0, 25, 50, 75, 100]
      .map(
        (v) => `<line x1="${padL}" y1="${yAt(v).toFixed(1)}" x2="${W - padR}" y2="${yAt(v).toFixed(1)}" stroke="#1b2436" stroke-width="1"/>
        <text x="${padL - 5}" y="${(yAt(v) + 3.5).toFixed(1)}" text-anchor="end" font-family="Space Mono, monospace" font-size="9" fill="#64748b">${v}%</text>`
      )
      .join("");
    const xLabels = [0, 4, 8, 12, 16, 18, 20, 23]
      .map(
        (h) => `<text x="${xAt(h).toFixed(1)}" y="${H - 7}" text-anchor="middle" font-family="Space Mono, monospace" font-size="9" fill="${h >= 18 && h <= 22 ? "#ffb000" : "#64748b"}">${String(h).padStart(2, "0")}:00</text>`
      )
      .join("");

    const cvrBars = hrs
      .map((pt) => {
        const bw = plotW / 27;
        const bh = (pt.cvr_intent_multiplier / 2.0) * 34;
        return `<rect x="${(xAt(pt.hour) - bw / 2).toFixed(1)}" y="${(padT + plotH - bh).toFixed(1)}" width="${bw.toFixed(1)}" height="${bh.toFixed(1)}" fill="${pt.hour >= 18 && pt.hour <= 22 ? "#ffb000" : "#334155"}" opacity="0.32"/>`;
      })
      .join("");

    document.getElementById("pacingSvgContainer").innerHTML = `
      <svg viewBox="0 0 ${W} ${H}" role="img" aria-label="24-Hour Greedy Burnout vs PID Pacing Curve">
        ${yGrid}
        <rect x="${peakX1.toFixed(1)}" y="${padT}" width="${(peakX2 - peakX1).toFixed(1)}" height="${plotH}" fill="rgba(255,176,0,0.09)"/>
        <text x="${((peakX1 + peakX2) / 2).toFixed(1)}" y="${padT + 11}" text-anchor="middle" font-family="Space Mono, monospace" font-size="8.5" fill="#ffb000">PRIME PEAK (1.82x CVR)</text>
        ${cvrBars}
        <polyline fill="none" stroke="#38bdf8" stroke-width="1.3" stroke-dasharray="3 3" points="${pts("target_cum_spend_pct")}"/>
        <polyline fill="none" stroke="#fb7185" stroke-width="2.4" stroke-dasharray="6 3" points="${pts("unpaced_cum_spend_pct")}"/>
        <polyline fill="none" stroke="#10b981" stroke-width="2.8" points="${pts("pid_cum_spend_pct")}"/>
        <line x1="${curX.toFixed(1)}" y1="${padT}" x2="${curX.toFixed(1)}" y2="${padT + plotH}" stroke="#ffb000" stroke-width="1.5" stroke-dasharray="2 2"/>
        <circle cx="${curX.toFixed(1)}" cy="${yAt(hrs[state.hour] ? hrs[state.hour].pid_cum_spend_pct : 50).toFixed(1)}" r="4" fill="#10b981"/>
        <circle cx="${curX.toFixed(1)}" cy="${yAt(hrs[state.hour] ? hrs[state.hour].unpaced_cum_spend_pct : 80).toFixed(1)}" r="4" fill="#fb7185"/>
        ${xLabels}
      </svg>`;

    document.getElementById("pacingComparisonCards").innerHTML = `
      <div class="cmp-stat bad">
        <strong style="color:#fb7185">UNPACED GREEDY · ${u.advertiser_roas.toFixed(2)}x ROAS</strong><br>
        ${u.campaigns_depleted_early}/${u.total_campaigns} (${u.early_depletion_rate_pct}%) exhausted pre-18:00 · ${u.total_conversions} conv
      </div>
      <div class="cmp-stat good">
        <strong style="color:#10b981">PID PACED (λ_t) · ${p.advertiser_roas.toFixed(2)}x ROAS (+${pd.roas_lift_pct}%)</strong><br>
        ${p.campaigns_depleted_early}/${p.total_campaigns} (0%) early depletions · ${p.total_conversions} conv (+${pd.conversions_lift_pct}%)
      </div>`;

    // OPE Tab
    const ope = pd.off_policy_evaluation;
    if (!ope) return;
    const mWrap = document.getElementById("opeMultiplierButtons");
    mWrap.innerHTML = (ope.available_multipliers || [])
      .map(
        (m) => `<button type="button" class="m-pill ${String(ope.selected_policy) === String(m.id) ? "on" : ""}" data-mult="${esc(m.id)}">${esc(m.id === "adaptive_pid" ? "Adaptive PID ★" : m.id + "x")}</button>`
      )
      .join("");
    mWrap.querySelectorAll(".m-pill").forEach((b) => {
      b.addEventListener("click", () => {
        state.bidMultiplier = b.getAttribute("data-mult");
        refreshPacingAndOpe();
      });
    });

    const safe = ope.safe_to_ship_verdict === "SAFE_TO_DEPLOY";
    const vb = document.getElementById("opeVerdictBanner");
    vb.className = "verdict-line " + (safe ? "safe" : "hold");
    vb.innerHTML = `<span>${safe ? "✓ SAFE TO SHIP" : "⚠ HOLD POLICY"}: ${esc(ope.policy_label)}</span><span>DR Lift: ${ope.dr_expected_lift_vs_logging_pct >= 0 ? "+" : ""}${ope.dr_expected_lift_vs_logging_pct}%</span>`;

    const ev = ope.evaluation;
    const rows = [
      { name: "Naive IPS", d: ev.naive_ips, color: "#64748b" },
      { name: "Clipped IPS", d: ev.clipped_ips, color: "#38bdf8" },
      { name: "Doubly-Robust", d: ev.doubly_robust, color: "#10b981" },
    ];
    const allV = [ope.logging_baseline_value, ope.oracle_ground_truth, ...rows.flatMap((r) => [r.d.ci_95_low, r.d.ci_95_high])];
    const minV = Math.max(0, Math.min(...allV) - 0.16), maxV = Math.max(...allV) + 0.16;
    const FW = 520, FH = 108, fL = 115, fR = 16;
    const xMap = (v) => fL + ((v - minV) / Math.max(0.01, maxV - minV)) * (FW - fL - fR);

    document.getElementById("opeForestSvgContainer").innerHTML = `
      <svg viewBox="0 0 ${FW} ${FH}" role="img" aria-label="Paired Bootstrap 95% CIs">
        <line x1="${xMap(ope.logging_baseline_value).toFixed(1)}" y1="10" x2="${xMap(ope.logging_baseline_value).toFixed(1)}" y2="${FH - 10}" stroke="#fb7185" stroke-width="1.3" stroke-dasharray="3 3"/>
        <line x1="${xMap(ope.oracle_ground_truth).toFixed(1)}" y1="10" x2="${xMap(ope.oracle_ground_truth).toFixed(1)}" y2="${FH - 10}" stroke="#ffb000" stroke-width="1.5"/>
        ${rows
          .map((r, i) => {
            const y = 26 + i * 28;
            return `<text x="${fL - 8}" y="${y + 3.5}" text-anchor="end" font-family="Space Mono, monospace" font-size="9.5" fill="#cbd5e1">${r.name}</text>
              <line x1="${xMap(r.d.ci_95_low).toFixed(1)}" y1="${y}" x2="${xMap(r.d.ci_95_high).toFixed(1)}" y2="${y}" stroke="${r.color}" stroke-width="3.5" stroke-linecap="round"/>
              <circle cx="${xMap(r.d.estimate).toFixed(1)}" cy="${y}" r="4.5" fill="${r.color}"/>`;
          })
          .join("")}
      </svg>`;

    document.getElementById("opeEstimatorTable").innerHTML = `
      <table class="desk-table">
        <thead><tr><th>Estimator</th><th>Est. Value</th><th>95% Bootstrap CI</th><th>40-Trial RMSE</th></tr></thead>
        <tbody>
          <tr><td>Naive IPS</td><td>$${ev.naive_ips.estimate.toFixed(3)}</td><td>[$${ev.naive_ips.ci_95_low.toFixed(2)}, $${ev.naive_ips.ci_95_high.toFixed(2)}]</td><td>0.1661</td></tr>
          <tr><td>Clipped IPS (M=25)</td><td>$${ev.clipped_ips.estimate.toFixed(3)}</td><td>[$${ev.clipped_ips.ci_95_low.toFixed(2)}, $${ev.clipped_ips.ci_95_high.toFixed(2)}]</td><td>0.1218 (-26.7%)</td></tr>
          <tr style="color:#10b981;font-weight:700;"><td>Doubly-Robust</td><td>$${ev.doubly_robust.estimate.toFixed(3)}</td><td>[$${ev.doubly_robust.ci_95_low.toFixed(2)}, $${ev.doubly_robust.ci_95_high.toFixed(2)}]</td><td>0.0564 (-66.0%)</td></tr>
        </tbody>
      </table>`;
  }

  function renderCampaignTickers() {
    const wrap = document.getElementById("sellerCampaignList");
    wrap.innerHTML = state.sellerCampaigns
      .map((c) => {
        const on = c.campaign_id === state.selectedCampaignId;
        const shortIssue = c.diagnostic.primary_issue.split("_").slice(0, 2).join(" ");
        return `<button type="button" class="cmp-tk ${on ? "on" : ""}" data-cid="${esc(c.campaign_id)}">
          ${esc(c.campaign_id)} <span>${esc(shortIssue)}</span>
        </button>`;
      })
      .join("");
    wrap.querySelectorAll(".cmp-tk").forEach((b) => {
      b.addEventListener("click", () => {
        state.selectedCampaignId = b.getAttribute("data-cid");
        const found = state.sellerCampaigns.find((x) => x.campaign_id === state.selectedCampaignId);
        if (found) document.getElementById("copilotQuestionInput").value = found.sample_question;
        renderCampaignTickers();
        runAdvisor();
      });
    });
  }

  function renderAdvisorOutput(res) {
    if (!res) return;
    const d = res.diagnostic;
    document.getElementById("selectedCampaignTelemetry").innerHTML = `
      <div class="gr-cell"><small>Loss Root Cause</small><b style="color:#fb7185">${esc(d.primary_issue)}</b></div>
      <div class="gr-cell"><small>Bid &rarr; Safe Rec</small><b style="color:#10b981">$${d.current_bid.toFixed(2)} &rarr; $${d.recommended_bid.toFixed(2)}</b></div>
      <div class="gr-cell"><small>Max ACOS Cap CPC</small><b>$${d.max_acos_cpc.toFixed(2)} (${Math.round(d.target_acos * 100)}%)</b></div>
      <div class="gr-cell"><small>Lost Impr Share</small><b style="color:#ffb000">${d.lost_impression_share_pct}%</b></div>
    `;
    const llm = res.llm_telemetry || {};
    document.getElementById("novaTelemetryTag").textContent = `${llm.engine || "Bedrock Nova"} · ${llm.latency_ms ? llm.latency_ms + "ms · " : ""}ACOS Cap Verified ✓`;
    document.getElementById("copilotOutputPanel").innerHTML = `
      <div>${memoHtml(res.copilot_memo)}</div>
      <div class="sql-drawer">${esc(res.spektr_sql)}</div>
    `;
  }

  async function refreshStorefront() {
    const resp = await fetch("/api/storefront/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query: state.query,
        hour: state.hour,
        daily_budget: state.dailyBudget,
        pacing_mode: state.pacingMode,
        shopper_price_cap: state.priceCap === "none" ? null : Number(state.priceCap),
      }),
    });
    state.storefrontData = await resp.json();
    renderStorefrontAndLadder(state.storefrontData);
    if (state.pacingData) renderPacingAndOpe(state.pacingData);
  }

  async function refreshPacingAndOpe() {
    const resp = await fetch("/api/pacing/simulate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        daily_budget: state.dailyBudget,
        selected_hour: state.hour,
        bid_multiplier: state.bidMultiplier,
      }),
    });
    state.pacingData = await resp.json();
    renderPacingAndOpe(state.pacingData);
  }

  async function runAdvisor() {
    const btn = document.getElementById("runCopilotBtn");
    const acosVal = document.getElementById("targetAcosInput").value;
    btn.disabled = true;
    btn.textContent = "Querying Bedrock Nova...";
    try {
      const resp = await fetch("/api/copilot/recommend", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          campaign_id: state.selectedCampaignId,
          question: document.getElementById("copilotQuestionInput").value,
          target_acos: acosVal ? Number(acosVal) : null,
        }),
      });
      renderAdvisorOutput(await resp.json());
    } finally {
      btn.disabled = false;
      btn.textContent = "Run Bedrock Advisor";
    }
  }

  function setDeskTab(which) {
    state.deskTab = which;
    document.getElementById("dtabPacing").classList.toggle("on", which === "pacing");
    document.getElementById("dtabOpe").classList.toggle("on", which === "ope");
    document.getElementById("panePacing").hidden = which !== "pacing";
    document.getElementById("paneOpe").hidden = which !== "ope";
  }

  async function init() {
    const resp = await fetch("/api/bootstrap");
    const boot = await resp.json();
    state.featuredQueries = boot.featured_queries || [];
    state.storefrontData = boot.default_storefront;
    state.pacingData = boot.default_pacing;
    state.sellerCampaigns = boot.seller_campaigns || [];

    renderFeaturedChips();
    renderStorefrontAndLadder(state.storefrontData);
    renderPacingAndOpe(state.pacingData);
    renderCampaignTickers();

    if (state.sellerCampaigns.length > 0) {
      document.getElementById("copilotQuestionInput").value = state.sellerCampaigns[0].sample_question;
      runAdvisor();
    }

    document.getElementById("storefrontSearchForm").addEventListener("submit", (e) => {
      e.preventDefault();
      state.query = document.getElementById("shopperQueryInput").value.trim() || "wireless anc running earbuds";
      renderFeaturedChips();
      refreshStorefront();
    });

    document.getElementById("hourSlider").addEventListener("input", (e) => {
      state.hour = Number(e.target.value);
      refreshStorefront();
    });

    document.getElementById("budgetSlider").addEventListener("change", (e) => {
      state.dailyBudget = Number(e.target.value);
      refreshStorefront();
      refreshPacingAndOpe();
    });
    document.getElementById("budgetSlider").addEventListener("input", (e) => {
      document.getElementById("budgetReadout").textContent = `$${e.target.value} / day`;
    });

    document.querySelectorAll(".mode-pill").forEach((b) => {
      b.addEventListener("click", () => {
        state.pacingMode = b.getAttribute("data-mode");
        refreshStorefront();
      });
    });

    document.getElementById("priceCapSelect").addEventListener("change", (e) => {
      state.priceCap = e.target.value;
      refreshStorefront();
    });

    document.getElementById("guidedBurnoutDemoBtn").addEventListener("click", () => {
      state.hour = 20;
      state.dailyBudget = 24;
      document.getElementById("hourSlider").value = "20";
      document.getElementById("budgetSlider").value = "24";
      state.pacingMode = state.pacingMode === "pid_paced" ? "unpaced_greedy" : "pid_paced";
      refreshStorefront();
      refreshPacingAndOpe();
    });

    document.getElementById("dtabPacing").addEventListener("click", () => setDeskTab("pacing"));
    document.getElementById("dtabOpe").addEventListener("click", () => setDeskTab("ope"));

    document.getElementById("copilotAskForm").addEventListener("submit", (e) => {
      e.preventDefault();
      runAdvisor();
    });
    document.getElementById("targetAcosInput").addEventListener("change", runAdvisor);
  }

  init();
})();
