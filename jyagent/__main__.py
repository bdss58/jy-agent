#!/usr/bin/env python3
"""
JY Agent — Entry point.

Usage:
    python -m jyagent          # Run the agent
    jy-agent                   # Via CLI entry point
"""

import os
import sys

import httpx
import anthropic


def create_client() -> anthropic.Anthropic:
    """Create an Anthropic client with optional base_url and auth_token from env."""
    kwargs = {}
    base_url = os.environ.get("ANTHROPIC_BASE_URL")
    auth_token = os.environ.get("ANTHROPIC_AUTH_TOKEN")
    if base_url:
        kwargs["base_url"] = base_url
    if auth_token:
        kwargs["api_key"] = auth_token
    kwargs["http_client"] = httpx.Client(verify=False)
    return anthropic.Anthropic(**kwargs)


def main():
    """Main entry point."""
    # Load .env if present (from project dir or current dir)
    try:
        from dotenv import load_dotenv
        # First try project dir .env, then cwd .env
        project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        env_file = os.path.join(project_dir, ".env")
        if os.path.exists(env_file):
            load_dotenv(env_file)
        elif os.path.exists(".env"):
            load_dotenv(".env")
    except ImportError:
        pass  # python-dotenv not installed, rely on shell env

    # Capture launch directory
    if "LAUNCH_DIR" not in os.environ:
        os.environ["LAUNCH_DIR"] = os.getcwd()

    client = create_client()

    from .agent import run
    run(client)


if __name__ == "__main__":
    main()
