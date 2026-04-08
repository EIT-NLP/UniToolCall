
from pathlib import Path

# src/uni_toolcall/paths.py -> parents: uni_toolcall, src, repo root
REPO_ROOT: Path = Path(__file__).resolve().parent.parent.parent

DEFAULT_TOOLSET_JSON: Path = REPO_ROOT / "tool_set" / "apis" / "toolset.json"

# Single canonical system prompt for inference, pipelines, and dataset conversion scripts
DEFAULT_SYSTEM_PROMPT_MD: Path = REPO_ROOT / "test_set" / "prompt_and_format" / "system_prompt.md"
