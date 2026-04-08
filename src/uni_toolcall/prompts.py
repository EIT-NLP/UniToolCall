
from __future__ import annotations

from pathlib import Path
from typing import Optional

from uni_toolcall.paths import DEFAULT_SYSTEM_PROMPT_MD


def read_system_prompt(path: Optional[Path] = None) -> str:
    """Return trimmed system prompt text from the given file (default: ``DEFAULT_SYSTEM_PROMPT_MD``)."""
    p = path or DEFAULT_SYSTEM_PROMPT_MD
    return p.read_text(encoding="utf-8").strip()
