# Central configuration — All env-loaded constants in one place.
#
# Every module reads from here instead of doing its own os.environ.get().
# Override any value via environment variables.

import json
import os
import sys


# ─── Env-var coercion helpers ────────────────────────────────────────────────
#
# Goals (post-2026-05 codex review):
#   * Surface a useful error at import time: instead of a raw
#     ``ValueError: invalid literal for int() with base 10: 'banana'`` traceback,
#     report which env var was bad and the value seen.
#   * Validate ranges (``min``, ``max``) for numeric knobs that would silently
#     break downstream if 0 / negative / oversized.
#   * Provide ONE canonical boolean parse so ``ANTHROPIC_PROMPT_CACHE=banana``
#     and ``JYAGENT_ASK=banana`` agree on what an unknown value means
#     (previously: prompt-cache treated unknown as enabled, ASK treated unknown
#     as disabled — observable behaviour drift).
#
# These helpers run at import time, so they MUST NOT import any provider /
# heavy module — keep them dep-free.


class ConfigError(RuntimeError):
    """Raised when an env-var-backed config value is invalid.

    Surfaces the env var NAME and the offending value, so import-time
    failures point the user directly at what to fix instead of a raw
    Python traceback.  Caught in ``__main__`` for a friendlier message.
    """


def _int_env(name: str, default: int, *,
             min: int | None = None, max: int | None = None) -> int:
    """Parse an int env var with bounds, raising ``ConfigError`` on failure."""
    raw = os.environ.get(name)
    if raw is None or raw == "":
        value = default
    else:
        try:
            value = int(raw.strip())
        except (ValueError, AttributeError) as e:
            raise ConfigError(
                f"{name}={raw!r} is not a valid integer: {e}. "
                f"Default is {default}."
            ) from e
    if min is not None and value < min:
        raise ConfigError(
            f"{name}={value} is below the minimum {min}. "
            f"Set a value >= {min} or unset to use default {default}."
        )
    if max is not None and value > max:
        raise ConfigError(
            f"{name}={value} is above the maximum {max}. "
            f"Set a value <= {max} or unset to use default {default}."
        )
    return value


# Canonical boolean tokens.  ``_bool_env`` falls back to ``default`` (with a
# stderr warning) when the env var is set to an unrecognized value, so a typo
# never silently flips a critical flag in either direction.
_BOOL_TRUE = frozenset({"1", "true", "yes", "on", "y", "t"})
_BOOL_FALSE = frozenset({"0", "false", "no", "off", "n", "f", ""})


def _bool_env(name: str, default: bool) -> bool:
    """Parse a bool env var with a SINGLE canonical mapping.

    Previously two ad-hoc parses lived in this file:
      * ``ANTHROPIC_PROMPT_CACHE`` — "negative list": unknown values
        treated as TRUE (any non-falsy → enabled).
      * ``JYAGENT_ASK`` — "positive list": unknown values treated as FALSE.

    That meant ``ANTHROPIC_PROMPT_CACHE=banana`` enabled caching and
    ``JYAGENT_ASK=banana`` disabled ASK — same input, opposite semantics.

    The unified rule: known TRUE token → True, known FALSE token → False,
    anything else → ``default`` plus a stderr warning so the user sees
    the typo instead of inheriting a silent surprise.
    """
    raw = os.environ.get(name)
    if raw is None:
        return default
    token = raw.strip().lower()
    if token in _BOOL_TRUE:
        return True
    if token in _BOOL_FALSE:
        return False
    print(
        f"WARNING: {name}={raw!r} is not a recognized boolean "
        f"({sorted(_BOOL_TRUE | _BOOL_FALSE - {''})}). "
        f"Falling back to default {default}.",
        file=sys.stderr,
    )
    return default


# ─── Project root (absolute, from __file__ — immune to CWD changes) ──────────

PROJECT_ROOT: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ─── Launch context ───────────────────────────────────────────────────────────

LAUNCH_DIR: str = ""  # Set once at startup by __main__.main(); do NOT read from env.

# ─── API & Model ──────────────────────────────────────────────────────────────

DEFAULT_ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")
AGENT_PROVIDER = (os.environ.get("AGENT_PROVIDER") or "anthropic").strip() or "anthropic"
AGENT_MODEL = os.environ.get("AGENT_MODEL", DEFAULT_ANTHROPIC_MODEL)
DEFAULT_MAX_TOKENS = _int_env("AGENT_MAX_TOKENS", 16384, min=1)
MAX_TOKENS_CAP = _int_env("AGENT_MAX_TOKENS_CAP", 128000, min=1)
SUPPORTED_RUNTIME_PROVIDERS: set[str] = {"anthropic"}

# ─── Anthropic prompt caching ─────────────────────────────────────────────────
# Anthropic Messages API requires an explicit ``cache_control`` field to enable
# prompt caching. Setting it at the top level of the request (vs. on individual
# content blocks) tells the API to auto-place the cache breakpoint at the last
# cacheable block and slide it forward as the conversation grows — perfect for
# multi-turn agents like jy-agent. See:
#   https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching
#
# Pricing impact (per Anthropic docs):
#   - cache write tokens cost 1.25× base input (5m TTL) or 2× (1h TTL)
#   - cache read tokens cost 0.10× base input
# So as long as a cached prefix is read at least ~3 times before expiring,
# caching is a net win. For a typical agent session this is essentially always.
#
# Set ANTHROPIC_PROMPT_CACHE=0 to disable (default: enabled).
# Set ANTHROPIC_PROMPT_CACHE_TTL=1h for 1-hour cache (default: 5m).
ANTHROPIC_PROMPT_CACHE_ENABLED = _bool_env("ANTHROPIC_PROMPT_CACHE", True)
# TTL is validated lazily at provider-call time (only "5m" / "1h" are
# accepted today, but the set may grow — keep authoritative validation
# at the boundary rather than mirror it here).
ANTHROPIC_PROMPT_CACHE_TTL = os.environ.get("ANTHROPIC_PROMPT_CACHE_TTL", "5m").strip()


def register_provider(name: str) -> None:
    """Register an additional runtime provider name as valid."""
    SUPPORTED_RUNTIME_PROVIDERS.add(name)


def get_extra_headers_from_env(env_var: str) -> dict[str, str]:
    """Parse provider-specific extra HTTP headers from a JSON env var.

    Security note: the env var is a trust boundary already (any code that
    can set it can do worse), so we accept arbitrary header names/values.
    Callers that ship the result to a provider should be aware that this
    can override auth or routing headers if mis-set; that is intentional
    (it's how proxy / custom-Anthropic-deployments are supported).
    """
    raw = (os.environ.get(env_var) or "").strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as err:
        raise ValueError(
            f"{env_var} must be a JSON object of string header names to string values: {err.msg}."
        ) from err
    if not isinstance(parsed, dict):
        raise ValueError(
            f"{env_var} must be a JSON object of string header names to string values; "
            f"got {type(parsed).__name__}."
        )
    for key, value in parsed.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise ValueError(
                f"{env_var} must be a JSON object of string header names to string values; "
                f"invalid entry {key!r}: {type(value).__name__}."
            )
    return parsed

# ─── Planner / Tool dispatch ─────────────────────────────────────────────────

DEFAULT_MAX_STEPS = _int_env("AGENT_MAX_STEPS", 100, min=1)
MAX_TOOL_RESULT_CHARS = _int_env("AGENT_MAX_TOOL_RESULT_CHARS", 8000, min=100)
MAX_TOOL_USE_INPUT_CHARS = _int_env("AGENT_MAX_TOOL_USE_INPUT_CHARS", 4000, min=100)
MAX_WORKING_TOKENS = _int_env("AGENT_MAX_WORKING_TOKENS", 180000, min=1000)  # Layer 1 safety net (cheap truncation within a turn)
DEFAULT_TOOL_TIMEOUT = _int_env("AGENT_TOOL_TIMEOUT", 120, min=1)
STREAM_TIMEOUT = _int_env("AGENT_STREAM_TIMEOUT", 300, min=1)
COMPACT_TOOL_RESULT_CHARS = 2000  # aggressive limit when compacting old tool results
OBSERVATION_MASK_DISTANCE = _int_env("AGENT_OBSERVATION_MASK_DISTANCE", 5, min=0)  # fully clear tool results older than N messages from the end

# ─── Tool approval gate ──────────────────────────────────────────────────────
# When True, the CLI prompts before executing any *mutating* tool call.
# Read-only / idempotent tools (read_file, list_directory, grep_files,
# glob_files, web_search, web_fetch, manage_memory query, etc.) are still
# auto-approved — same policy as the ``mutating`` flag in tools/__init__.py.
# Toggled by the ``--ask`` CLI flag (see __main__.py) or the env var below.
ASK_BEFORE_TOOLS = _bool_env("JYAGENT_ASK", False)

# ─── Reasoning preview ──────────────────────────────────────────────────────
#
# Tier-A reasoning UX (see ui/terminal.py::_ReasoningStreamer).  When
# enabled, the first ``REASONING_PREVIEW_LINES`` lines of each thinking
# block stream inline in dim italic; the rest folds behind a `[/think]`
# marker.  Set ``REASONING_SHOW=0`` to disable entirely (spinner-only
# behaviour, same as before tier-A).  Setting ``REASONING_PREVIEW_LINES=0``
# folds everything (only the marker is shown); a large value
# effectively disables folding for short blocks.
REASONING_SHOW = _bool_env("JYAGENT_REASONING_SHOW", True)
REASONING_PREVIEW_LINES = _int_env("JYAGENT_REASONING_PREVIEW_LINES", 5, min=0)

# ─── Memory ───────────────────────────────────────────────────────────────────

MEMORY_DIR = os.path.join(PROJECT_ROOT, "data", "memory")
TOPICS_DIR = os.path.join(MEMORY_DIR, "topics")
JOURNAL_DIR = os.path.join(MEMORY_DIR, "journal")  # Tier 3: never auto-loaded
MEMORY_MD_FILE = os.path.join(MEMORY_DIR, "MEMORY.md")

COMPACT_TOKEN_THRESHOLD = _int_env("AGENT_COMPACT_TOKEN_THRESHOLD", 150000, min=1000)  # Layer 2 (LLM summary between turns); matches Anthropic API default & 75% of 200K window
SUMMARIZE_KEEP_RECENT = _int_env("AGENT_SUMMARIZE_KEEP_RECENT", 6, min=0)
# (Dropped 2026-05: SUMMARIZE_THRESHOLD, MAX_SESSIONS, WEB_FETCH_MIN_CONTENT_LENGTH
#  were never referenced anywhere in the codebase.  Removed per codex review.)
FILE_REINJECTION_COUNT = _int_env("AGENT_FILE_REINJECTION_COUNT", 5, min=0)  # re-inject N most recent files after compaction
FILE_REINJECTION_MAX_TOKENS = _int_env("AGENT_FILE_REINJECTION_MAX_TOKENS", 50000, min=0)  # cap total re-injected content
MAX_MEMORY_INDEX_LINES = 200
MAX_MEMORY_INDEX_BYTES = 25 * 1024
# Soft warning thresholds (matches Letta-style "approaching cap" guidance,
# Anthropic CLAUDE.md best practice: target <200 lines, warn at 75%).
MEMORY_INDEX_WARN_LINES = _int_env("AGENT_MEMORY_WARN_LINES", 150, min=0)
MEMORY_INDEX_WARN_BYTES = _int_env("AGENT_MEMORY_WARN_BYTES", 18 * 1024, min=0)
MAX_MEMORY_PROMPT_CHARS = _int_env("AGENT_MAX_MEMORY_PROMPT_CHARS", 10000, min=0)
CHARS_PER_TOKEN = 4

# Session persistence
SESSIONS_DIR = os.path.join(PROJECT_ROOT, "data", "sessions")

# ─── Skills ───────────────────────────────────────────────────────────────────

MAX_INSTRUCTIONS_CHARS = _int_env("AGENT_MAX_SKILL_CHARS", 8000, min=100)
MAX_RESOURCE_CHARS = _int_env("AGENT_MAX_RESOURCE_CHARS", 10000, min=100)
# NOTE: there is NO skill router — not per-turn, not elsewhere.  The main
# model sees the skill catalog in the system prompt and self-activates via
# the manage_skills tool (progressive disclosure, same pattern Claude Code
# uses).  The former SKILL_PRE_ROUTER + SKILL_ROUTER_* surface was removed
# 2026-05 (see data/memory/journal/).  Eval-only routing — "would query X
# trigger skill Y?" — lives self-contained in
# skills/create-skill/scripts/test_trigger.py.

# ─── Web Fetch ────────────────────────────────────────────────────────────────

WEB_FETCH_DEFAULT_MAX_LENGTH = _int_env("WEB_FETCH_MAX_LENGTH", 8000, min=100)

# ─── File constants ───────────────────────────────────────────────────────────

SKIP_DIRS = {
    '.git', 'node_modules', '__pycache__', '.venv', 'venv', 'env',
    '.mypy_cache', '.pytest_cache', '.tox', '.eggs', '*.egg-info',
    'dist', 'build', '.next', '.nuxt', 'coverage', '.coverage',
    '.idea', '.vscode', '.DS_Store',
}

BINARY_EXTS = {
    '.pyc', '.pyo', '.so', '.dylib', '.dll', '.exe', '.o', '.a',
    '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.ico', '.svg', '.webp',
    '.mp3', '.mp4', '.avi', '.mov', '.wav', '.flac',
    '.zip', '.tar', '.gz', '.bz2', '.xz', '.rar', '.7z',
    '.woff', '.woff2', '.ttf', '.eot',
    '.pdf', '.doc', '.docx', '.xls', '.xlsx',
    '.db', '.sqlite', '.sqlite3',
}


def get_active_model_spec():
    return build_model_spec(AGENT_PROVIDER, AGENT_MODEL, source="AGENT_PROVIDER")


def validate_runtime_provider(provider: str, *, source: str) -> str:
    resolved = (provider or "").strip()
    if not resolved:
        raise ValueError(
            f"{source} is empty. Set a supported provider: {sorted(SUPPORTED_RUNTIME_PROVIDERS)}."
        )
    if resolved not in SUPPORTED_RUNTIME_PROVIDERS:
        raise ValueError(
            f"{source} has unsupported provider '{resolved}'. "
            f"Available providers: {sorted(SUPPORTED_RUNTIME_PROVIDERS)}."
        )
    return resolved


def build_model_spec(provider: str, model: str, *, source: str):
    from .llm.types import ModelSpec

    return ModelSpec(provider=validate_runtime_provider(provider, source=source), model=model)


def get_reasoning_config_for_provider(
    provider: str,
    *,
    max_output_tokens: int | None = None,
    model: str | None = None,
):
    from .llm.providers._anthropic_reasoning import validate_anthropic_reasoning

    provider = validate_runtime_provider(provider, source="reasoning provider")

    if provider == "anthropic":
        config = {}
        thinking_type = (os.environ.get("ANTHROPIC_THINKING_TYPE") or "").strip()
        display = (os.environ.get("ANTHROPIC_THINKING_DISPLAY") or "").strip()
        effort = (os.environ.get("ANTHROPIC_REASONING_EFFORT") or "").strip()
        budget_tokens = (os.environ.get("ANTHROPIC_THINKING_BUDGET_TOKENS") or "").strip()
        if thinking_type:
            config["type"] = thinking_type
        if display:
            config["display"] = display
        if effort:
            config["effort"] = effort
        if budget_tokens:
            config["budget_tokens"] = budget_tokens
        if not config:
            return None
        resolved_model = (model or "").strip()
        if not resolved_model:
            resolved_model = AGENT_MODEL if AGENT_PROVIDER == "anthropic" else DEFAULT_ANTHROPIC_MODEL
        return validate_anthropic_reasoning(config, model=resolved_model)

    elif provider == "openai":
        from .llm.providers._openai_helpers import supports_openai_reasoning_effort

        effort = (os.environ.get("OPENAI_REASONING_EFFORT") or "").strip().lower()
        if not effort:
            return None
        resolved_model = (model or "").strip()
        if not resolved_model and AGENT_PROVIDER == "openai":
            resolved_model = AGENT_MODEL
        if not supports_openai_reasoning_effort(resolved_model):
            # Globally-set env var should not block runs with other models.
            return None
        valid_efforts = {"minimal", "none", "low", "medium", "high", "xhigh"}
        if effort not in valid_efforts:
            raise ValueError(
                f"OPENAI_REASONING_EFFORT must be one of: {sorted(valid_efforts)}. Got '{effort}'."
            )
        return {"effort": effort}

    return None
