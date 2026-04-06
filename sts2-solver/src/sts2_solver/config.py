"""Config router — selects between Profile A (champion) and Profile B (challenger).

This file re-exports every public name from whichever config profile is active,
so all existing imports like `from .config import EVALUATOR` keep working unchanged.

Profile selection (checked in order):
  1. Environment variable STS2_CONFIG_PROFILE=a or STS2_CONFIG_PROFILE=b
  2. The "config_profile" field in sts2_config.json (in the repo root)
  3. Defaults to "a" (the champion) if neither is set

To switch profiles:
  - Set the env var:  STS2_CONFIG_PROFILE=b bash play.sh
  - Or edit sts2_config.json:  "config_profile": "b"
"""

from __future__ import annotations

import json
import os
from pathlib import Path


def _resolve_profile() -> str:
    """Determine which config profile to load: 'a' (champion) or 'b' (challenger).

    Checks the STS2_CONFIG_PROFILE environment variable first, then falls
    back to the config_profile field in sts2_config.json. Defaults to 'a'.
    """
    # 1. Check environment variable (highest priority — set by play.sh or launcher)
    env_profile = os.environ.get("STS2_CONFIG_PROFILE", "").strip().lower()
    if env_profile in ("a", "b"):
        return env_profile

    # 2. Check sts2_config.json (repo root, three directories up from this file)
    config_path = Path(__file__).resolve().parents[3] / "sts2_config.json"
    try:
        with open(config_path) as f:
            data = json.load(f)
        file_profile = str(data.get("config_profile", "")).strip().lower()
        if file_profile in ("a", "b"):
            return file_profile
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        pass  # Config file missing or malformed — use default

    # 3. Default to champion
    return "a"


# ---------------------------------------------------------------------------
# Load the selected profile and re-export all its public names.
# This makes `from .config import EVALUATOR` (etc.) work transparently.
# ---------------------------------------------------------------------------

_profile = _resolve_profile()

if _profile == "b":
    from .config_b import *  # noqa: F401,F403 — intentional wildcard re-export
    _ACTIVE_PROFILE = "b"
else:
    from .config_a import *  # noqa: F401,F403 — intentional wildcard re-export
    _ACTIVE_PROFILE = "a"


def get_active_profile() -> str:
    """Return which config profile is currently loaded: 'a' or 'b'."""
    return _ACTIVE_PROFILE
