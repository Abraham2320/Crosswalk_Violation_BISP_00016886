/* portal.js — Violator portal: plate input + staggered card animation */

"use strict";

document.addEventListener("DOMContentLoaded", () => {

  /* ── Auto-uppercase plate input ──────────────────────────── */
  const plateInput = document.getElementById("plate-input");
  if (plateInput) {
    plateInput.addEventListener("input", () => {
      const cur = plateInput.selectionStart;
      plateInput.value = plateInput.value.toUpperCase().replace(/\s/g, "");
      plateInput.setSelectionRange(cur, cur);
    });
    plateInput.addEventListener("keyup", () => {
      plateInput.value = plateInput.value.toUpperCase().replace(/\s/g, "");
    });
  }

  /* ── Loading button state ────────────────────────────────── */
  const searchForm = document.getElementById("portal-form");
  const searchBtn  = document.getElementById("search-btn");
  if (searchForm && searchBtn) {
    searchForm.addEventListener("submit", (e) => {
      const val = plateInput ? plateInput.value.trim() : "";
      if (!val) {
        e.preventDefault();
        return;
      }
      searchBtn.classList.add("loading");
    });
  }

  /* ── Staggered card fade-in ──────────────────────────────── */
  const cards = document.querySelectorAll(".violation-card");
  cards.forEach((card, i) => {
    setTimeout(() => {
      card.classList.add("visible");
    }, 80 + i * 150);
  });

});
