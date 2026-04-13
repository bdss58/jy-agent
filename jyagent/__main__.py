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
    from .runtime import RuntimeOwner


def create_runtime_owner() -> RuntimeOwner:
    """Create the default runtime owner from environment configuration."""
    from .config import get_active_model_spec
    from .runtime import RuntimeOwner

    return RuntimeOwner(get_active_model_spec())


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

    # LAUNCH_DIR is set by run.sh *before* it cd's to the project root.
    # os.getcwd() here already points to the project dir, NOT the user's dir.
    # Fall back to cwd only for direct `python -m jyagent` invocations (no run.sh).
    import jyagent.config as _cfg
    _cfg.LAUNCH_DIR = os.environ.get("LAUNCH_DIR") or os.getcwd()

    runtime_owner = create_runtime_owner()

    from .agent import run
    run(runtime_owner)


if __name__ == "__main__":
    main()
