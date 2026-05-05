#!/usr/bin/env python3
"""
JY Agent — Entry point.

Usage:
    python -m jyagent          # Run the agent
    jy-agent                   # Via CLI entry point
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .llm import LLMOwner


def create_runtime_owner() -> LLMOwner:
    """Create the default runtime owner from environment configuration."""
    from .config import get_active_model_spec
    from .llm import LLMOwner

    return LLMOwner(get_active_model_spec())


def _load_dotenv() -> None:
    """Load .env before importing env-backed config."""
    try:
        from dotenv import load_dotenv

        # First try project dir .env, then cwd .env.
        project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        env_file = os.path.join(project_dir, ".env")
        if os.path.exists(env_file):
            load_dotenv(env_file)
        elif os.path.exists(".env"):
            load_dotenv(".env")
    except ImportError:
        pass  # python-dotenv not installed, rely on shell env


def main():
    """Main entry point."""
    _load_dotenv()

    # ─── Minimal CLI flag parsing ───────────────────────────────────────
    # We don't use argparse to keep startup lean; the agent reads its
    # options from environment variables.  --ask sets JYAGENT_ASK=1 so
    # config.ASK_BEFORE_TOOLS picks it up at import time (see below).
    import sys as _sys
    argv = _sys.argv[1:]
    if "--ask" in argv:
        os.environ["JYAGENT_ASK"] = "1"
        _sys.argv = [_sys.argv[0]] + [a for a in argv if a != "--ask"]

    # LAUNCH_DIR is set by run.sh *before* it cd's to the project root.
    # os.getcwd() here already points to the project dir, NOT the user's dir.
    # Fall back to cwd only for direct `python -m jyagent` invocations (no run.sh).
    import jyagent.config as _cfg
    _cfg.LAUNCH_DIR = os.environ.get("LAUNCH_DIR") or os.getcwd()
    # Re-read JYAGENT_ASK in case --ask was passed AFTER dotenv load above.
    _cfg.ASK_BEFORE_TOOLS = (
        os.environ.get("JYAGENT_ASK", "0").lower() in ("1", "true", "yes")
    )

    # Change CWD to the user's launch directory so all tools (run_shell,
    # read_file, etc.) operate in the user's project by default.
    # Internal data paths (memory, sessions, skills, traces) are absolute
    # (anchored to PROJECT_ROOT), so they are unaffected.
    if _cfg.LAUNCH_DIR and os.path.isdir(_cfg.LAUNCH_DIR):
        os.chdir(_cfg.LAUNCH_DIR)

    runtime_owner = create_runtime_owner()

    from .agent import run
    run(runtime_owner)


if __name__ == "__main__":
    main()
