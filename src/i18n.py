"""
src/i18n.py — Lightweight JSON-based internationalisation helper.

Usage in Flask:
    from src.i18n import get_locale, t_for, SUPPORTED_LANGS

    @app.context_processor
    def inject_i18n():
        lang = get_locale()
        return {"t": t_for(lang), "lang": lang, "SUPPORTED_LANGS": SUPPORTED_LANGS}

Usage in Jinja2:
    {{ t('nav.dashboard') }}
    {{ t('violations.records_found', count=total_count) }}
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SUPPORTED_LANGS: dict[str, str] = {
    "en": "English",
    "uz": "O'zbek",
    "ru": "Русский",
    "ja": "日本語",
}
DEFAULT_LANG = "en"

_TRANSLATIONS_DIR = Path(__file__).resolve().parent.parent / "translations"


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------
@lru_cache(maxsize=8)
def _load(lang: str) -> dict:
    """Load translation JSON for *lang*; fall back to English on missing keys."""
    path = _TRANSLATIONS_DIR / f"{lang}.json"
    if not path.exists():
        if lang != DEFAULT_LANG:
            return _load(DEFAULT_LANG)
        return {}
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _en() -> dict:
    return _load(DEFAULT_LANG)


# ---------------------------------------------------------------------------
# Key resolver
# ---------------------------------------------------------------------------
def _resolve(data: dict, key: str, fallback: str) -> str:
    """
    Resolve a dot-separated *key* inside *data*.
    Returns *fallback* if the key is missing at any level.
    """
    parts = key.split(".")
    node: Any = data
    for part in parts:
        if isinstance(node, dict):
            node = node.get(part)
        else:
            return fallback
        if node is None:
            return fallback
    return str(node) if not isinstance(node, (dict, list)) else fallback


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def get_locale() -> str:
    """
    Return the active language code from the Flask session.
    Falls back to DEFAULT_LANG if the session is unavailable or invalid.
    Call this inside a Flask request context.
    """
    try:
        from flask import session  # noqa: PLC0415
        lang = session.get("lang", DEFAULT_LANG)
    except RuntimeError:
        lang = DEFAULT_LANG
    if lang not in SUPPORTED_LANGS:
        lang = DEFAULT_LANG
    return lang


def t_for(lang: str):
    """
    Return a translation function bound to *lang*.

    The returned callable accepts a dot-separated key and optional keyword
    arguments for ``str.format_map`` interpolation.
    """
    data = _load(lang)
    en_data = _en()

    def _t(key: str, **kwargs: Any) -> str:
        # Try requested language, fall back to English, then the raw key.
        text = _resolve(data, key, "")
        if not text:
            text = _resolve(en_data, key, key)
        if kwargs:
            try:
                text = text.format_map(kwargs)
            except (KeyError, ValueError):
                pass
        return text

    return _t
