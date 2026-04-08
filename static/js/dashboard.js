/* dashboard.js — Admin dashboard: animated SVG chart + live refresh */
"use strict";

/* ── Animated SVG hourly bar chart ─────────────────────────── */

function renderHourlyChart(data, containerId) {
  const container = document.getElementById(containerId);
  if (!container || !data || !data.length) return;

  const W      = container.offsetWidth || 640;
  const H      = 188;
  const padL   = 28, padR = 6, padT = 22, padB = 34;
  const chartW = W - padL - padR;
  const chartH = H - padT - padB;
  const maxVal = Math.max(...data, 1);
  const barW   = chartW / data.length;
  const gap    = Math.max(1, barW * 0.22);
  const bw     = barW - gap;
  const peakVal = Math.max(...data);

  /* Area-fill path behind bars */
  let areaPath = `M${padL},${padT + chartH}`;
  data.forEach((val, i) => {
    const cx = padL + i * barW + barW / 2;
    const bh = (val / maxVal) * chartH;
    areaPath += ` L${cx.toFixed(1)},${(padT + chartH - bh).toFixed(1)}`;
  });
  areaPath += ` L${(padL + chartW).toFixed(1)},${padT + chartH} Z`;

  /* SVG elements */
  let gridLines = "", bars = "", labels = "", xAxis = "";

  for (let t = 0; t <= 4; t++) {
    const yPos = padT + chartH - (t / 4) * chartH;
    const tick = Math.round((t / 4) * maxVal);
    gridLines += `
      <line x1="${padL}" y1="${yPos.toFixed(1)}"
            x2="${(padL + chartW).toFixed(1)}" y2="${yPos.toFixed(1)}"
            stroke="hsl(216 34% 13%)" stroke-width="1"/>
      <text x="${(padL - 5).toFixed(1)}" y="${(yPos + 3).toFixed(1)}"
            text-anchor="end" font-size="8" fill="hsl(218 11% 28%)"
            font-family="JetBrains Mono,monospace">${tick}</text>`;
  }

  data.forEach((val, i) => {
    const x      = padL + i * barW + gap / 2;
    const bh     = val === 0 ? 0 : Math.max(3, (val / maxVal) * chartH);
    const y      = padT + chartH - bh;
    const delay  = 60 + i * 28;
    const isPeak = val === peakVal && val > 0;

    bars += `
      <rect class="hc-bar"
            x="${x.toFixed(1)}" y="${y.toFixed(1)}"
            width="${bw.toFixed(1)}" height="${bh.toFixed(1)}"
            fill="url(#hcBarGrad)" rx="3"
            opacity="${val === 0 ? 0.1 : 1}"
            ${isPeak ? 'filter="url(#hcPeakGlow)"' : ''}
            data-val="${val}" data-hour="${i}"
            style="transform-box:fill-box;transform-origin:50% 100%;
                   animation:hcBarRise 620ms cubic-bezier(.22,.61,.36,1) ${delay}ms both"/>`;

    if (val > 0) {
      labels += `
        <text x="${(x + bw / 2).toFixed(1)}" y="${(y - 5).toFixed(1)}"
              text-anchor="middle" font-size="8.5"
              fill="hsl(38 92% 72%)"
              font-family="JetBrains Mono,monospace"
              style="animation:hcFadeIn 300ms ease ${delay + 520}ms both">${val}</text>`;
    }

    if (i % 3 === 0) {
      xAxis += `
        <text x="${(x + bw / 2).toFixed(1)}" y="${(H - 10).toFixed(1)}"
              text-anchor="middle" font-size="9" fill="hsl(218 11% 30%)"
              font-family="JetBrains Mono,monospace">${String(i).padStart(2, '0')}</text>`;
    }
  });

  container.innerHTML = `
    <style>
      @keyframes hcBarRise { from { transform:scaleY(0); } to { transform:scaleY(1); } }
      @keyframes hcFadeIn  { from { opacity:0; }           to { opacity:1; }           }
      @keyframes hcAreaIn  { from { opacity:0; }           to { opacity:1; }           }
      .hc-bar { cursor:pointer; transition: filter 120ms ease; }
    </style>
    <div style="position:relative;user-select:none;">
      <svg id="hc-svg-${containerId}" width="100%" height="${H}" viewBox="0 0 ${W} ${H}"
           style="overflow:visible;display:block;">
        <defs>
          <linearGradient id="hcBarGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%"   stop-color="hsl(38 92% 54%)" stop-opacity="1"/>
            <stop offset="100%" stop-color="hsl(32 90% 36%)" stop-opacity="0.6"/>
          </linearGradient>
          <linearGradient id="hcAreaGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%"   stop-color="hsl(38 92% 53%)" stop-opacity="0.15"/>
            <stop offset="100%" stop-color="hsl(38 92% 53%)" stop-opacity="0"/>
          </linearGradient>
          <filter id="hcPeakGlow" x="-60%" y="-60%" width="220%" height="220%">
            <feGaussianBlur stdDeviation="3.5" result="blur"/>
            <feFlood flood-color="hsl(38 92% 53%)" flood-opacity="0.55" result="color"/>
            <feComposite in="color" in2="blur" operator="in" result="glow"/>
            <feMerge><feMergeNode in="glow"/><feMergeNode in="SourceGraphic"/></feMerge>
          </filter>
        </defs>
        ${gridLines}
        <path d="${areaPath}" fill="url(#hcAreaGrad)"
              style="animation:hcAreaIn 700ms ease 80ms both"/>
        ${bars}
        ${labels}
        ${xAxis}
      </svg>
      <div id="hc-tip-${containerId}"
           style="position:absolute;pointer-events:none;display:none;
                  background:hsl(224 71% 7%);border:1px solid hsl(216 34% 22%);
                  border-radius:7px;padding:6px 12px;
                  font-family:'JetBrains Mono',monospace;font-size:11px;
                  color:hsl(213 31% 91%);white-space:nowrap;
                  box-shadow:0 4px 20px rgba(0,0,0,0.55);z-index:20;"></div>
    </div>`;

  /* Hover interactions */
  const svgEl = container.querySelector(`#hc-svg-${containerId}`);
  const tip   = container.querySelector(`#hc-tip-${containerId}`);
  if (!svgEl || !tip) return;

  svgEl.querySelectorAll(".hc-bar").forEach(bar => {
    bar.addEventListener("mouseenter", () => {
      if (parseFloat(bar.getAttribute("opacity")) < 0.5) return;
      bar.style.filter = "brightness(1.3) drop-shadow(0 0 6px hsl(38 92% 53% / 0.5))";
      const v = bar.dataset.val;
      const h = parseInt(bar.dataset.hour, 10);
      tip.innerHTML = `<span style="color:hsl(38 92% 65%)">${String(h).padStart(2,'0')}:00</span>`
                    + ` &nbsp;·&nbsp; `
                    + `<strong style="color:hsl(213 31% 91%)">${v}</strong>`
                    + `<span style="color:hsl(218 11% 40%)"> violation${v=='1'?'':'s'}</span>`;
      tip.style.display = "block";
    });
    bar.addEventListener("mousemove", e => {
      const cr = container.getBoundingClientRect();
      let left = e.clientX - cr.left + 14;
      if (left + 200 > cr.width) left = e.clientX - cr.left - 210;
      tip.style.left = left + "px";
      tip.style.top  = Math.max(0, e.clientY - cr.top - 36) + "px";
    });
    bar.addEventListener("mouseleave", () => {
      bar.style.filter = "";
      tip.style.display = "none";
    });
  });
}

/* ── Animate mini bars from 0 → target on page load ────────── */

function animateMiniBarFills() {
  document.querySelectorAll(".mini-bar-fill").forEach((bar, i) => {
    const target = bar.style.width || "100%";
    bar.style.width      = "0";
    bar.style.transition = `width 750ms cubic-bezier(.22,.61,.36,1) ${120 + i * 90}ms`;
    requestAnimationFrame(() => requestAnimationFrame(() => { bar.style.width = target; }));
  });
}

/* ── Live stat refresh ─────────────────────────────────────── */

function refreshStats() {
  fetch("/admin/dashboard/data")
    .then(r => r.json())
    .then(d => {
      const totalEl = document.getElementById("stat-total");
      const todayEl = document.getElementById("stat-today");
      if (totalEl) totalEl.textContent = d.total_violations.toLocaleString();
      if (todayEl) todayEl.textContent = d.violations_today.toLocaleString();
    })
    .catch(() => {});
}

/* ── Init ──────────────────────────────────────────────────── */

document.addEventListener("DOMContentLoaded", () => {
  const hourlyData = window.HOURLY_DATA;
  if (hourlyData) {
    renderHourlyChart(hourlyData, "hourly-chart");
    window.addEventListener("resize", () => renderHourlyChart(hourlyData, "hourly-chart"));
  }

  animateMiniBarFills();
  setInterval(refreshStats, 60_000);

  document.querySelectorAll("tr[data-href]").forEach(row => {
    row.addEventListener("click", () => { window.location.href = row.dataset.href; });
  });
});
