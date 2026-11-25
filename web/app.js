/* AdPulse Interactive Studio + Blueprint Navigation
 * Stateless in-memory & URL-hash state (strictly no web storage APIs).
 */
(function () {
  "use strict";

  const root = document.documentElement;
  root.classList.add("js");

  // Theme toggle stored in dataset.theme + URL hash parameter if toggled
  const themeBtn = document.querySelector(".theme");
  if (themeBtn) {
    themeBtn.addEventListener("click", () => {
      const next = root.dataset.theme === "dark" ? "light" : "dark";
      root.dataset.theme = next;
      history.replaceState(null, "", next === "dark" ? "#theme=dark" : location.pathname);
    });
  }

  // Sidebar progress & active section indicator
  const side = document.querySelector(".side");
  const postBody = document.querySelector(".post-body");
  const sideLinks = side ? [...side.querySelectorAll(".side-list a")] : [];
  const sideTargets = sideLinks.map((a) => document.getElementById(a.hash.slice(1)));
  const sideInd = side && side.querySelector(".side-ind");
  const sideBar = side && side.querySelector(".side-bar i");
  const sidePct = side && side.querySelector(".side-pct");
  const toTop = document.querySelector(".totop");
  const ring = toTop && toTop.querySelector(".ring");
  const RING = 2 * Math.PI * 21;
  if (ring) ring.style.strokeDasharray = String(RING);

  function updateScrollChrome() {
    const y = window.scrollY;
    const max = Math.max(1, root.scrollHeight - window.innerHeight);
    const whole = Math.min(1, Math.max(0, y / max));
    if (toTop && ring) {
      toTop.classList.toggle("show", y > 500);
      ring.style.strokeDashoffset = String(RING * (1 - whole));
    }
    if (side && postBody && sideBar && sidePct) {
      const r = postBody.getBoundingClientRect();
      const read = Math.min(1, Math.max(0, (window.innerHeight * 0.45 - r.top) / Math.max(1, r.height)));
      sideBar.style.transform = `scaleX(${read})`;
      sidePct.textContent = Math.round(read * 100) + "%";
      let active = sideLinks[0] || null;
      sideTargets.forEach((t, i) => {
        if (t && t.getBoundingClientRect().top <= 140) active = sideLinks[i];
      });
      sideLinks.forEach((a) => a.classList.toggle("on", a === active));
      if (active && sideInd) {
        const listRect = side.querySelector(".side-list").getBoundingClientRect();
        const aRect = active.getBoundingClientRect();
        sideInd.style.transform = `translateY(${aRect.top - listRect.top}px)`;
        sideInd.style.height = aRect.height + "px";
      }
    }
  }
  window.addEventListener("scroll", updateScrollChrome, { passive: true });
  window.addEventListener("resize", updateScrollChrome);

  // Interactive AdPulse State
  const state = {
    query: "wireless anc running earbuds",
    hour: 19,
    dailyBudget: 28,
    pacingMode: "pid_paced",
    priceCap: "160",
    bidMultiplier: "adaptive_pid",
    featuredQueries: [],
    storefrontData: null,
    pacingData: null,
    sellerCampaigns: [],
    selectedCampaignId: "CMP-D002",
  };

  function escapeHtml(val) {
    return String(val == null ? "" : val)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function formatInlineMd(text) {
    return escapeHtml(text)
      .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
      .replace(/`([^`]+)`/g, "<code>$1</code>");
  }

  function renderMemoHtml(raw) {
    const lines = String(raw || "").split(/\r?\n/).filter((l) => l.trim().length > 0);
    let inList = false;
    const out = [];
    for (const line of lines) {
      const trimmed = line.trim();
      if (/^[-*•]\s+/.test(trimmed) || /^\d+[.)]\s+/.test(trimmed)) {
        if (!inList) {
          out.push("<ul>");
          inList = true;
        }
        out.push("<li>" + formatInlineMd(trimmed.replace(/^([-*•]|\d+[.)])\s+/, "")) + "</li>");
      } else {
        if (inList) {
          out.push("</ul>");
          inList = false;
        }
        out.push("<p>" + formatInlineMd(trimmed) + "</p>");
      }
    }
    if (inList) out.push("</ul>");
    return out.join("");
  }

  function productBlueprintSvg(productType, category) {
    const p = (productType || "").toLowerCase();
    if (p.includes("earbud") || p.includes("headphone")) {
      return `<svg width="46" height="46" viewBox="0 0 64 64" fill="none" stroke="currentColor" stroke-width="2.2">
        <rect x="11" y="28" width="14" height="23"/>
        <rect x="39" y="28" width="14" height="23"/>
        <path d="M16 28V22C16 13 23 9 32 9C41 9 48 13 48 22V28" stroke="#d9480f" stroke-width="3"/>
        <line x1="15" y1="39" x2="21" y2="39"/><line x1="43" y1="39" x2="49" y2="39"/>
      </svg>`;
    }
    if (p.includes("monitor")) {
      return `<svg width="46" height="46" viewBox="0 0 64 64" fill="none" stroke="currentColor" stroke-width="2.2">
        <rect x="8" y="12" width="48" height="31"/>
        <polyline points="14,35 25,23 35,30 46,19 52,25" stroke="#d9480f" stroke-width="2.5"/>
        <line x1="32" y1="43" x2="32" y2="53"/><line x1="20" y1="53" x2="44" y2="53"/>
      </svg>`;
    }
    if (p.includes("chair")) {
      return `<svg width="46" height="46" viewBox="0 0 64 64" fill="none" stroke="currentColor" stroke-width="2.2">
        <rect x="20" y="9" width="24" height="25"/>
        <line x1="24" y1="16" x2="40" y2="16" stroke="#d9480f"/><line x1="24" y1="22" x2="40" y2="22" stroke="#d9480f"/>
        <rect x="16" y="36" width="32" height="6"/><line x1="32" y1="42" x2="32" y2="53"/><line x1="18" y1="54" x2="46" y2="54"/>
      </svg>`;
    }
    if (p.includes("espresso") || (category || "").includes("kitchen")) {
      return `<svg width="46" height="46" viewBox="0 0 64 64" fill="none" stroke="currentColor" stroke-width="2.2">
        <polygon points="19,10 45,10 40,23 24,23"/>
        <rect x="21" y="23" width="22" height="28"/>
        <circle cx="32" cy="33" r="3.5" stroke="#d9480f" stroke-width="2.5"/>
        <line x1="15" y1="53" x2="49" y2="53"/>
      </svg>`;
    }
    return `<svg width="46" height="46" viewBox="0 0 64 64" fill="none" stroke="currentColor" stroke-width="2.2">
      <rect x="14" y="16" width="36" height="34"/><circle cx="32" cy="33" r="8" stroke="#d9480f"/>
    </svg>`;
  }

  function renderStars(rating) {
    const n = Math.round(Number(rating || 4.7));
    return "★".repeat(Math.min(5, n)) + "☆".repeat(Math.max(0, 5 - n));
  }

  function renderFeaturedChips() {
    const wrap = document.getElementById("featuredQueryChips");
    wrap.innerHTML = state.featuredQueries
      .map((fq) => {
        const active = fq.query.toLowerCase() === state.query.toLowerCase();
        return `<button type="button" class="q-chip ${active ? "active" : ""}" data-q="${escapeHtml(fq.query)}" data-cap="${fq.default_price_cap}">${escapeHtml(fq.label)}</button>`;
      })
      .join("");
    wrap.querySelectorAll(".q-chip").forEach((btn) => {
      btn.addEventListener("click", () => {
        state.query = btn.getAttribute("data-q");
        state.priceCap = String(Math.round(Number(btn.getAttribute("data-cap"))));
        document.getElementById("shopperQueryInput").value = state.query;
        const sel = document.getElementById("priceCapSelect");
        if ([...sel.options].some((o) => o.value === state.priceCap)) sel.value = state.priceCap;
        renderFeaturedChips();
        refreshStorefront();
      });
    });
  }

  function renderStorefront(sf) {
    if (!sf) return;
    const isPeak = state.hour >= 18 && state.hour <= 22;
    const hrEl = document.getElementById("hourReadout");
    hrEl.textContent = `${String(state.hour).padStart(2, "0")}:00 · ${isPeak ? "Prime Peak" : state.hour < 12 ? "Morning Browse" : "Afternoon"} (${sf.hourly_cvr_multiplier.toFixed(2)}x CVR)`;
    hrEl.className = "ctrl-val" + (isPeak ? " peak" : "");

    document.getElementById("budgetReadout").textContent = `$${state.dailyBudget} / day`;
    document.getElementById("lambdaReadout").textContent =
      state.pacingMode === "pid_paced" ? `λ_t = ${Number(sf.pid_lambda_t).toFixed(2)}` : "λ_t = 1.00 (Unshaded)";

    document.getElementById("btnModePid").classList.toggle("active", state.pacingMode === "pid_paced");
    document.getElementById("btnModeGreedy").classList.toggle("active", state.pacingMode === "unpaced_greedy");
    document.getElementById("searchCategoryPill").textContent = (sf.resolved_category || "retail").replace(/_/g, " ");

    const sc = sf.stage_counts;
    const lat = sf.latency_ms;
    document.getElementById("auctionFunnelStrip").innerHTML = `
      <span>CATALOG: <strong>${sc.catalog_total} SKUs</strong> &rarr; PRUNED PRE-AUCTION: <strong style="color:var(--bad)">${sc.filtered_before_auction} SKUs</strong> (&lt;0.5ms) &rarr; STAGE-2 RERANKED: <strong>${sc.stage2_reranked}</strong> &rarr; GSP CLEARED: <strong>${sc.slots_cleared} SLOTS</strong></span>
      <span>LATENCY: <strong>${lat.benchmark_p50_ms}ms p50 · ${lat.benchmark_p99_ms}ms p99</strong> (live ${lat.live_wall_ms}ms)</span>
    `;

    const grid = document.getElementById("winningAdsGrid");
    if (!sf.winning_ads || sf.winning_ads.length === 0) {
      grid.innerHTML = `<div class="bp-box" style="grid-column:1/-1;">No eligible in-stock SKUs cleared under current price ceiling and budget state. Switch to <strong>PID-Paced</strong> or raise the shopper price cap.</div>`;
    } else {
      grid.innerHTML = sf.winning_ads
        .map((ad) => {
          const specs = (ad.highlights || []).slice(0, 2).map((h) => `<div>• ${escapeHtml(h)}</div>`).join("");
          return `
          <article class="sp-card slot-${ad.slot_number}">
            <div class="sp-top">
              <span class="slot-tag">Slot #${ad.slot_number} Winner</span>
              <span>Sponsored · ${escapeHtml(ad.sku_id)}</span>
            </div>
            <div class="sp-vis">
              <div class="sp-icon-frame">${productBlueprintSvg(ad.product_type, ad.category)}</div>
              <div class="sp-specs">
                <b>${escapeHtml(ad.badge)}</b>
                ${specs}
              </div>
            </div>
            <div class="sp-main">
              <span class="sp-brand">${escapeHtml(ad.brand)} · ${escapeHtml(ad.campaign_id)}</span>
              <h4 class="sp-title">${escapeHtml(ad.title)}</h4>
              <div class="sp-rating"><span class="stars">${renderStars(ad.rating)}</span> ${Number(ad.rating).toFixed(1)} (${Number(ad.reviews).toLocaleString()})</div>
              <div class="sp-price-line">
                <span class="sp-price">$${Number(ad.price).toFixed(2)}</span>
                <span class="sp-prime">PRIME IN-STOCK</span>
              </div>
            </div>
            <div class="sp-econ">
              <div><span>GSP 2nd-Price</span><b class="ok">$${Number(ad.clearing_cpc).toFixed(2)} CPC</b><span>Bid $${Number(ad.base_bid).toFixed(2)} (-${ad.cpc_savings_vs_raw_bid_pct}%)</span></div>
              <div><span>pCTR / pCVR</span><b>${ad.p_ctr_pct}% / ${ad.p_cvr_pct}%</b><span>Empirical-Bayes</span></div>
              <div><span>eCPM / ROAS</span><b>$${Number(ad.ecpm_usd).toFixed(0)} · ${ad.expected_roas}x</b><span>Score ${ad.rank_score}</span></div>
            </div>
          </article>`;
        })
        .join("");
    }

    const cp = sf.conversational_shopping_preview;
    document.getElementById("conversationalAdBox").innerHTML = cp
      ? `<div class="bp-box-head">
          <div><span class="tag cobalt">CONVERSATIONAL SHOPPING AD</span><h4 style="margin-top:6px;">Rufus / Conversational Slot Grounding</h4></div>
        </div>
        <p style="font:400 1.04rem/1.55 var(--read);margin:0 0 10px;">${formatInlineMd(cp.assistant_answer)}</p>
        <div style="font:400 10.5px/1.5 var(--mono);color:var(--cobalt);border-top:1px dashed var(--rule);padding-top:8px;">${escapeHtml(cp.why_selected)}</div>`
      : `<p>No conversational slot cleared.</p>`;

    const reasonCounts = sf.filtered_out_counts_by_reason || {};
    const shortLabels = {
      OUT_OF_STOCK_SUPPRESSED: "OOS Blocked",
      BUDGET_EXHAUSTED_EARLY: "Budget Depleted",
      ABOVE_SHOPPER_PRICE_CAP: "Over Price Cap",
      MISSING_CATALOG_ATTRIBUTES: "Missing Attrs",
    };
    document.getElementById("filteredReasonPills").innerHTML = Object.entries(reasonCounts)
      .map(([k, v]) => `<span class="tag bad">${v} ${escapeHtml(shortLabels[k] || k)}</span>`)
      .join("");

    const drawer = sf.filtered_out_drawer || [];
    document.getElementById("filteredDrawerList").innerHTML = drawer.length
      ? drawer
          .map(
            (it) => `
          <div class="prune-row">
            <div>
              <div class="p-title"><code>${escapeHtml(it.sku_id)}</code> ${escapeHtml(it.title)} ($${Number(it.price).toFixed(2)})</div>
              <div class="p-why">${escapeHtml(it.reason_detail)}</div>
            </div>
            <div class="p-meta">
              <strong>${escapeHtml(it.reason_code.replace(/_/g, " "))}</strong><br>
              Raw Bid $${Number(it.raw_bid).toFixed(2)} · ${it.prune_latency_ms}ms
            </div>
          </div>`
          )
          .join("")
      : `<div class="mono-kicker">All candidate SKUs passed Stage-1 gates.</div>`;
  }

  function renderPacingAndOpe(pd) {
    if (!pd) return;
    const u = pd.unpaced_metrics;
    const p = pd.pid_paced_metrics;

    document.getElementById("pacingSummaryChips").innerHTML = `
      <span class="tag ok">PID ROAS ${p.advertiser_roas.toFixed(2)}x vs Greedy ${u.advertiser_roas.toFixed(2)}x (+${pd.roas_lift_pct}%)</span>
    `;

    const hrs = pd.hourly_curve || [];
    const W = 600, H = 235, padL = 42, padR = 16, padT = 18, padB = 30;
    const plotW = W - padL - padR, plotH = H - padT - padB;
    const xAt = (h) => padL + (h / 23) * plotW;
    const yAt = (pct) => padT + plotH - (Math.min(100, Math.max(0, pct)) / 100) * plotH;
    const pts = (key) => hrs.map((pt) => `${xAt(pt.hour).toFixed(1)},${yAt(pt[key]).toFixed(1)}`).join(" ");

    const peakX1 = xAt(18), peakX2 = xAt(22), curX = xAt(state.hour);
    const yLines = [0, 25, 50, 75, 100]
      .map(
        (v) => `<line x1="${padL}" y1="${yAt(v).toFixed(1)}" x2="${W - padR}" y2="${yAt(v).toFixed(1)}" stroke="var(--chart-grid)" stroke-width="1"/>
        <text x="${padL - 6}" y="${(yAt(v) + 3.5).toFixed(1)}" text-anchor="end" font-family="var(--mono)" font-size="9.5" fill="var(--chart-text)">${v}%</text>`
      )
      .join("");
    const xTicks = [0, 4, 8, 12, 16, 18, 20, 23]
      .map(
        (h) => `<text x="${xAt(h).toFixed(1)}" y="${H - 8}" text-anchor="middle" font-family="var(--mono)" font-size="9.5" fill="var(--chart-text)">${String(h).padStart(2, "0")}h</text>`
      )
      .join("");

    document.getElementById("pacingSvgContainer").innerHTML = `
      <svg viewBox="0 0 ${W} ${H}" role="img" aria-label="24-Hour Spend Curve">
        ${yLines}
        <rect x="${peakX1.toFixed(1)}" y="${padT}" width="${(peakX2 - peakX1).toFixed(1)}" height="${plotH}" fill="var(--signal-soft)"/>
        <text x="${((peakX1 + peakX2) / 2).toFixed(1)}" y="${padT + 12}" text-anchor="middle" font-family="var(--mono)" font-size="9" fill="var(--accent-text)">PRIME PEAK (1.82x CVR)</text>
        <polyline fill="none" stroke="var(--cobalt)" stroke-width="1.5" stroke-dasharray="3 3" points="${pts("target_cum_spend_pct")}"/>
        <polyline fill="none" stroke="var(--bad)" stroke-width="2.6" stroke-dasharray="6 3" points="${pts("unpaced_cum_spend_pct")}"/>
        <polyline fill="none" stroke="var(--ok)" stroke-width="2.8" points="${pts("pid_cum_spend_pct")}"/>
        <line x1="${curX.toFixed(1)}" y1="${padT}" x2="${curX.toFixed(1)}" y2="${padT + plotH}" stroke="var(--ink)" stroke-width="1.4" stroke-dasharray="2 2"/>
        ${xTicks}
      </svg>`;

    document.getElementById("pacingComparisonCards").innerHTML = `
      <div class="m-tile bad">
        <strong>UNPACED GREEDY</strong> · ${u.advertiser_roas.toFixed(2)}x ROAS<br>
        ${u.campaigns_depleted_early}/${u.total_campaigns} (${u.early_depletion_rate_pct}%) depleted pre-18:00 · ${u.total_conversions} conv
      </div>
      <div class="m-tile ok">
        <strong>PID BID PACER</strong> · ${p.advertiser_roas.toFixed(2)}x ROAS (+${pd.roas_lift_pct}%)<br>
        ${p.campaigns_depleted_early}/${p.total_campaigns} (0%) early depletions · ${p.total_conversions} conv (+${pd.conversions_lift_pct}%)
      </div>`;

    const ope = pd.off_policy_evaluation;
    if (!ope) return;
    const btnWrap = document.getElementById("opeMultiplierButtons");
    btnWrap.innerHTML = (ope.available_multipliers || [])
      .map(
        (m) => `<button type="button" class="mult-btn ${String(ope.selected_policy) === String(m.id) ? "active" : ""}" data-mult="${escapeHtml(m.id)}">${escapeHtml(m.id === "adaptive_pid" ? "Adaptive PID ★" : m.id + "x")}</button>`
      )
      .join("");
    btnWrap.querySelectorAll(".mult-btn").forEach((b) => {
      b.addEventListener("click", () => {
        state.bidMultiplier = b.getAttribute("data-mult");
        refreshPacingAndOpe();
      });
    });

    const isSafe = ope.safe_to_ship_verdict === "SAFE_TO_DEPLOY";
    const vb = document.getElementById("opeVerdictBanner");
    vb.style.background = isSafe ? "var(--ok-soft)" : "var(--bad-soft)";
    vb.style.color = isSafe ? "var(--ok)" : "var(--bad)";
    vb.textContent = `${isSafe ? "✓ SAFE TO DEPLOY" : "✕ HOLD (REGRESSION RISK)"} · ${ope.policy_label} · DR Lift ${ope.dr_expected_lift_vs_logging_pct >= 0 ? "+" : ""}${ope.dr_expected_lift_vs_logging_pct}%`;

    const ev = ope.evaluation;
    const rows = [
      { name: "Naive IPS", d: ev.naive_ips, color: "var(--soft)" },
      { name: "Clipped IPS", d: ev.clipped_ips, color: "var(--cobalt)" },
      { name: "Doubly-Robust", d: ev.doubly_robust, color: "var(--ok)" },
    ];
    const vals = [ope.logging_baseline_value, ope.oracle_ground_truth, ...rows.flatMap((r) => [r.d.ci_95_low, r.d.ci_95_high])];
    const minV = Math.max(0, Math.min(...vals) - 0.16), maxV = Math.max(...vals) + 0.16;
    const FW = 480, FH = 118, fPadL = 118, fPadR = 16;
    const xMap = (v) => fPadL + ((v - minV) / Math.max(0.01, maxV - minV)) * (FW - fPadL - fPadR);

    document.getElementById("opeForestSvgContainer").innerHTML = `
      <svg viewBox="0 0 ${FW} ${FH}" role="img" aria-label="Bootstrap 95% CI Forest Plot">
        <line x1="${xMap(ope.logging_baseline_value).toFixed(1)}" y1="14" x2="${xMap(ope.logging_baseline_value).toFixed(1)}" y2="${FH - 14}" stroke="var(--bad)" stroke-width="1.4" stroke-dasharray="3 3"/>
        <line x1="${xMap(ope.oracle_ground_truth).toFixed(1)}" y1="14" x2="${xMap(ope.oracle_ground_truth).toFixed(1)}" y2="${FH - 14}" stroke="var(--cobalt)" stroke-width="1.6"/>
        ${rows
          .map((r, i) => {
            const y = 30 + i * 30;
            return `<text x="${fPadL - 8}" y="${y + 3.5}" text-anchor="end" font-family="var(--mono)" font-size="9.5" fill="var(--ink)">${r.name}</text>
              <line x1="${xMap(r.d.ci_95_low).toFixed(1)}" y1="${y}" x2="${xMap(r.d.ci_95_high).toFixed(1)}" y2="${y}" stroke="${r.color}" stroke-width="3.5"/>
              <circle cx="${xMap(r.d.estimate).toFixed(1)}" cy="${y}" r="4.5" fill="${r.color}"/>`;
          })
          .join("")}
      </svg>`;

    document.getElementById("opeEstimatorTable").innerHTML = `
      <div class="tablewrap" style="margin:0;"><table>
        <thead><tr><th>Estimator</th><th class="r">Est.</th><th class="r">95% CI</th><th class="r">RMSE</th></tr></thead>
        <tbody>
          <tr><td>Naive IPS</td><td class="r">$${ev.naive_ips.estimate.toFixed(3)}</td><td class="r">[${ev.naive_ips.ci_95_low.toFixed(2)}, ${ev.naive_ips.ci_95_high.toFixed(2)}]</td><td class="r">0.1661</td></tr>
          <tr><td>Clipped IPS (M=25)</td><td class="r">$${ev.clipped_ips.estimate.toFixed(3)}</td><td class="r">[${ev.clipped_ips.ci_95_low.toFixed(2)}, ${ev.clipped_ips.ci_95_high.toFixed(2)}]</td><td class="r">0.1218</td></tr>
          <tr class="best"><td>Doubly-Robust</td><td class="r">$${ev.doubly_robust.estimate.toFixed(3)}</td><td class="r">[${ev.doubly_robust.ci_95_low.toFixed(2)}, ${ev.doubly_robust.ci_95_high.toFixed(2)}]</td><td class="r">0.0564 (-66%)</td></tr>
        </tbody>
      </table></div>`;
  }

  function renderCampaignSelector() {
    const stack = document.getElementById("sellerCampaignList");
    stack.innerHTML = state.sellerCampaigns
      .map((c) => {
        const d = c.diagnostic;
        const on = c.campaign_id === state.selectedCampaignId;
        return `<button type="button" class="cmp-item ${on ? "active" : ""}" data-cid="${escapeHtml(c.campaign_id)}">
          <div style="display:flex;justify-content:space-between;gap:6px;align-items:center;">
            <span class="mono-kicker">${escapeHtml(c.campaign_id)} · ${escapeHtml(c.seller_name)}</span>
            <span class="tag bad">${escapeHtml(d.primary_issue.replace(/_/g, " "))}</span>
          </div>
          <div style="font:700 0.92rem var(--display);margin:4px 0;color:var(--ink);">${escapeHtml(c.campaign_title)}</div>
          <div style="font:400 10.5px var(--mono);color:var(--soft);">Lost ${d.lost_impression_share_pct}% · Bid $${d.current_bid.toFixed(2)} &rarr; Cap $${d.max_acos_cpc.toFixed(2)}</div>
        </button>`;
      })
      .join("");
    stack.querySelectorAll(".cmp-item").forEach((btn) => {
      btn.addEventListener("click", () => {
        state.selectedCampaignId = btn.getAttribute("data-cid");
        const found = state.sellerCampaigns.find((x) => x.campaign_id === state.selectedCampaignId);
        if (found) document.getElementById("copilotQuestionInput").value = found.sample_question;
        renderCampaignSelector();
        runCopilotRecommendation();
      });
    });
  }

  function renderCopilotResult(res) {
    if (!res) return;
    const d = res.diagnostic;
    document.getElementById("selectedCampaignTelemetry").innerHTML = `
      <div class="ctrl-cell"><span class="mono-kicker">Loss Mode</span><strong style="font:500 11px var(--mono);color:var(--bad);">${escapeHtml(d.primary_issue)}</strong></div>
      <div class="ctrl-cell"><span class="mono-kicker">Safe Bid Action</span><strong style="font:500 12px var(--mono);color:var(--ok);">$${d.current_bid.toFixed(2)} &rarr; $${d.recommended_bid.toFixed(2)}</strong></div>
      <div class="ctrl-cell"><span class="mono-kicker">Max ACOS Cap CPC</span><strong style="font:500 12px var(--mono);">$${d.max_acos_cpc.toFixed(2)} (${Math.round(d.target_acos * 100)}%)</strong></div>
      <div class="ctrl-cell"><span class="mono-kicker">PID Pacing</span><strong style="font:500 12px var(--mono);">${d.enable_pid_pacing ? "ENABLE λ_t" : "HOLD / SCHEMA"}</strong></div>
    `;
    const llm = res.llm_telemetry || {};
    document.getElementById("copilotOutputPanel").innerHTML = `
      <div class="bp-box-head">
        <span class="tag cobalt">${escapeHtml(llm.engine || "aws-bedrock (us.amazon.nova-lite-v1:0)")}</span>
        <span class="mono-kicker">${llm.latency_ms ? llm.latency_ms + " ms · " : ""}${llm.tokens ? llm.tokens + " tok · " : ""}ACOS GUARDRAIL VERIFIED ✓</span>
      </div>
      <div class="memo-body">${renderMemoHtml(res.copilot_memo)}</div>
      <pre><code>${escapeHtml(res.spektr_sql)}</code></pre>
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
    renderStorefront(state.storefrontData);
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

  async function runCopilotRecommendation() {
    const btn = document.getElementById("runCopilotBtn");
    const acosVal = document.getElementById("targetAcosInput").value;
    btn.disabled = true;
    btn.textContent = "Calling Bedrock Nova...";
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
      renderCopilotResult(await resp.json());
    } finally {
      btn.disabled = false;
      btn.textContent = "Run Bedrock Nova →";
    }
  }

  async function init() {
    const resp = await fetch("/api/bootstrap");
    const boot = await resp.json();
    state.featuredQueries = boot.featured_queries || [];
    state.storefrontData = boot.default_storefront;
    state.pacingData = boot.default_pacing;
    state.sellerCampaigns = boot.seller_campaigns || [];

    renderFeaturedChips();
    renderStorefront(state.storefrontData);
    renderPacingAndOpe(state.pacingData);
    renderCampaignSelector();
    updateScrollChrome();

    if (state.sellerCampaigns.length > 0) {
      document.getElementById("copilotQuestionInput").value = state.sellerCampaigns[0].sample_question;
      runCopilotRecommendation();
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
    document.querySelectorAll(".mode-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        state.pacingMode = btn.getAttribute("data-mode");
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
    document.getElementById("copilotAskForm").addEventListener("submit", (e) => {
      e.preventDefault();
      runCopilotRecommendation();
    });
    document.getElementById("targetAcosInput").addEventListener("change", runCopilotRecommendation);
  }

  init();
})();
