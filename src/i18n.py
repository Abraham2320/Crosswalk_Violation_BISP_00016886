from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

try:
    from flask import request, session
    _FLASK = True
except ImportError:
    _FLASK = False

SUPPORTED_LANGS: dict[str, str] = {}
_TRANSLATIONS: dict[str, dict[str, str]] = {}

_TRANSLATIONS_DIR = Path(__file__).resolve().parent.parent / "translations"


def _flatten(d: dict, prefix: str = "") -> dict[str, str]:
    out: dict[str, str] = {}
    for k, v in d.items():
        full = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten(v, full))
        else:
            out[full] = str(v)
    return out


def _load_translations() -> None:
    for path in sorted(_TRANSLATIONS_DIR.glob("*.json")):
        code = path.stem
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[i18n] Failed to load {path.name}: {exc}")
            continue
        flat = _flatten(data)
        _TRANSLATIONS[code] = flat
        lang_name = flat.get(f"lang.{code}") or flat.get("lang.name") or code.upper()
        SUPPORTED_LANGS[code] = lang_name

    if not _TRANSLATIONS:
        _TRANSLATIONS["en"] = {}
        SUPPORTED_LANGS["en"] = "English"


_load_translations()


def get_locale() -> str:
    if _FLASK:
        try:
            lang = session.get("lang") or request.accept_languages.best_match(list(SUPPORTED_LANGS))
            if lang and lang in SUPPORTED_LANGS:
                return lang
        except RuntimeError:
            pass
    return "en"


def t_for(lang: str) -> Callable[..., str]:
    table = _TRANSLATIONS.get(lang) or _TRANSLATIONS.get("en", {})

    def t(key: str, **kwargs) -> str:
        text = table.get(key, key)
        try:
            return text.format(**kwargs) if kwargs else text
        except (KeyError, IndexError):
            return text

    return t
