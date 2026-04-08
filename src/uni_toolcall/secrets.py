
from __future__ import annotations

import os
from typing import Iterable, Optional


def _first_env(names: Iterable[str]) -> Optional[str]:
    for name in names:
        v = os.environ.get(name, "").strip()
        if v:
            return v
    return None


def get_openai_compatible_key() -> Optional[str]:
    """SiliconFlow or other OpenAI-compatible endpoints (e.g. api1 SiliconFlow URL)."""
    return _first_env(("SILICONFLOW_API_KEY", "OPENAI_API_KEY"))


def get_openai_official_key() -> Optional[str]:
    """OpenAI platform Chat Completions (api2)."""
    return _first_env(("OPENAI_API_KEY",))


def get_anthropic_key() -> Optional[str]:
    """Anthropic Messages API (api3 in inference scripts)."""
    return _first_env(("ANTHROPIC_API_KEY",))


def get_google_api_key() -> Optional[str]:
    """Gemini / Google Generative Language API."""
    return _first_env(("GOOGLE_API_KEY",))
