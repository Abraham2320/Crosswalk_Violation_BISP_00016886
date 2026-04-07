/* dashboard.js — Admin dashboard: SVG bar chart + live stat refresh */

"use strict";

/* ── SVG bar chart renderer ────────────────────────────────── */

function renderHourlyChart(data, containerId) {
  const container = document.getElementById(containerId);
  if (!container || !data || !data.length) return;

  const W = container.offsetWidth || 640;
  const H = 160;
  const padL = 32, padR = 8, padT = 16, padB = 32;
  const chartW = W - padL - padR;
  const chartH = H - padT - padB;

  const maxVal = Math.max(...data, 1);
  const barW   = chartW / data.length;
  const barGap = Math.max(1, barW * 0.15);

  let bars = "";
  let xLabels = "";
  let yLabels = "";

  data.forEach((val, i) => {
    const x      = padL + i * barW + barGap / 2;
    const bw     = barW - barGap;
    const barH   = (val / maxVal) * chartH;
    const y      = padT + chartH - barH;
    const opacity = val === 0 ? 0.15 : 0.85;

    bars += `<rect x="${x.toFixed(1)}" y="${y.toFixed(1)}"
             width="${bw.toFixed(1)}" height="${barH.toFixed(1)}"
             fill="#F59E0B" opacity="${opacity}" rx="2"/>`;

    if (val > 0) {
      bars += `<text x="${(x + bw / 2).toFixed(1)}" y="${(y - 3).toFixed(1)}"
               text-anchor="middle" font-size="9" fill="#F59E0B"
               font-family="'JetBrains Mono',monospace">${val}</text>`;
    }

    // x-axis: label every 3 hours
    if (i % 3 === 0) {
      xLabels += `<text x="${(x + bw / 2).toFixed(1)}"
                  y="${(H - 6).toFixed(1)}"
                  text-anchor="middle" font-size="9" fill="#4B5563"
                  font-family="'JetBrains Mono',monospace">${String(i).padStart(2,'0')}</text>`;
    }
  });

  // Y-axis tick lines
  for (let t = 0; t <= 4; t++) {
    const yPos = padT + chartH - (t / 4) * chartH;
    const val  = Math.round((t / 4) * maxVal);
    yLabels += `<line x1="${padL}" y1="${yPos.toFixed(1)}"
                x2="${(padL + chartW).toFixed(1)}" y2="${yPos.toFixed(1)}"
                stroke="#1F2937" stroke-width="1"/>`;
    yLabels += `<text x="${(padL - 4).toFixed(1)}" y="${(yPos + 3).toFixed(1)}"
               text-anchor="end" font-size="9" fill="#4B5563"
               font-family="'JetBrains Mono',monospace">${val}</text>`;
  }

  container.innerHTML = `
    <svg width="100%" height="${H}" viewBox="0 0 ${W} ${H}" preserveAspectRatio="none">
      ${yLabels}
      ${bars}
      ${xLabels}
    </svg>`;
}

/* ── Live stat refresh ─────────────────────────────────────── */

function refreshStats() {
  fetch("/admin/dashboard/data")
    .then(r => r.json())
    .then(data => {
      const totalEl = document.getElementById("stat-total");
      const todayEl = document.getElementById("stat-today");
      if (totalEl) totalEl.textContent = data.total_violations.toLocaleString();
      if (todayEl) todayEl.textContent = data.violations_today.toLocaleString();
    })
    .catch(() => {/* silent fail */});
}

/* ── Init ──────────────────────────────────────────────────── */

document.addEventListener("DOMContentLoaded", () => {
  // Render hourly chart
  const hourlyData = window.HOURLY_DATA;
  if (hourlyData) {
    renderHourlyChart(hourlyData, "hourly-chart");
    // Re-render on window resize
    window.addEventListener("resize", () => {
      renderHourlyChart(hourlyData, "hourly-chart");
    });
  }

  // Live refresh every 60 seconds
  setInterval(refreshStats, 60_000);

  // Clickable rows
  document.querySelectorAll("tr[data-href]").forEach(row => {
    row.addEventListener("click", () => {
      window.location.href = row.dataset.href;
    });
  });
});
