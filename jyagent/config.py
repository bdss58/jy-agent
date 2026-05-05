# Central configuration — All env-loaded constants in one place.
#
# Every module reads from here instead of doing its own os.environ.get().
# Override any value via environment variables.

import json
import os

# ─── Project root (absolute, from __file__ — immune to CWD changes) ──────────

PROJECT_ROOT: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ─── Launch context ───────────────────────────────────────────────────────────

LAUNCH_DIR: str = ""  # Set once at startup by __main__.main(); do NOT read from env.

# ─── API & Model ──────────────────────────────────────────────────────────────

DEFAULT_ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")
AGENT_PROVIDER = (os.environ.get("AGENT_PROVIDER") or "anthropic").strip() or "anthropic"
AGENT_MODEL = os.environ.get("AGENT_MODEL", DEFAULT_ANTHROPIC_MODEL)
DEFAULT_MAX_TOKENS = int(os.environ.get("AGENT_MAX_TOKENS", "16384"))
MAX_TOKENS_CAP = int(os.environ.get("AGENT_MAX_TOKENS_CAP", "128000"))
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
ANTHROPIC_PROMPT_CACHE_ENABLED = (
    os.environ.get("ANTHROPIC_PROMPT_CACHE", "1").lower() not in ("0", "false", "no", "off")
)
ANTHROPIC_PROMPT_CACHE_TTL = os.environ.get("ANTHROPIC_PROMPT_CACHE_TTL", "5m").strip()


def register_provider(name: str) -> None:
    """Register an additional runtime provider name as valid."""
    SUPPORTED_RUNTIME_PROVIDERS.add(name)


def get_extra_headers_from_env(env_var: str) -> dict[str, str]:
    """Parse provider-specific extra HTTP headers from a JSON env var."""
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

DEFAULT_MAX_STEPS = int(os.environ.get("AGENT_MAX_STEPS", "100"))
MAX_TOOL_RESULT_CHARS = int(os.environ.get("AGENT_MAX_TOOL_RESULT_CHARS", "8000"))
MAX_TOOL_USE_INPUT_CHARS = int(os.environ.get("AGENT_MAX_TOOL_USE_INPUT_CHARS", "4000"))
MAX_WORKING_TOKENS = int(os.environ.get("AGENT_MAX_WORKING_TOKENS", "180000"))  # Layer 1 safety net (cheap truncation within a turn)
DEFAULT_TOOL_TIMEOUT = int(os.environ.get("AGENT_TOOL_TIMEOUT", "120"))
STREAM_TIMEOUT = int(os.environ.get("AGENT_STREAM_TIMEOUT", "300"))
COMPACT_TOOL_RESULT_CHARS = 2000  # aggressive limit when compacting old tool results
OBSERVATION_MASK_DISTANCE = int(os.environ.get("AGENT_OBSERVATION_MASK_DISTANCE", "5"))  # fully clear tool results older than N messages from the end

# ─── Tool approval gate ──────────────────────────────────────────────────────
# When True, the CLI prompts before executing any *mutating* tool call.
# Read-only / idempotent tools (read_file, list_directory, grep_files,
# glob_files, web_search, web_fetch, manage_memory query, etc.) are still
# auto-approved — same policy as the ``mutating`` flag in tools/__init__.py.
# Toggled by the ``--ask`` CLI flag (see __main__.py) or the env var below.
ASK_BEFORE_TOOLS = (os.environ.get("JYAGENT_ASK", "0").lower() in ("1", "true", "yes"))

# ─── Memory ───────────────────────────────────────────────────────────────────

MEMORY_DIR = os.path.join(PROJECT_ROOT, "data", "memory")
TOPICS_DIR = os.path.join(MEMORY_DIR, "topics")
JOURNAL_DIR = os.path.join(MEMORY_DIR, "journal")  # Tier 3: never auto-loaded
MEMORY_MD_FILE = os.path.join(MEMORY_DIR, "MEMORY.md")

COMPACT_TOKEN_THRESHOLD = int(os.environ.get("AGENT_COMPACT_TOKEN_THRESHOLD", "150000"))  # Layer 2 (LLM summary between turns); matches Anthropic API default & 75% of 200K window
SUMMARIZE_KEEP_RECENT = int(os.environ.get("AGENT_SUMMARIZE_KEEP_RECENT", "6"))
SUMMARIZE_THRESHOLD = int(os.environ.get("AGENT_SUMMARIZE_THRESHOLD", "20"))
FILE_REINJECTION_COUNT = int(os.environ.get("AGENT_FILE_REINJECTION_COUNT", "5"))  # re-inject N most recent files after compaction
FILE_REINJECTION_MAX_TOKENS = int(os.environ.get("AGENT_FILE_REINJECTION_MAX_TOKENS", "50000"))  # cap total re-injected content
MAX_SESSIONS = 50
MAX_MEMORY_INDEX_LINES = 200
MAX_MEMORY_INDEX_BYTES = 25 * 1024
# Soft warning thresholds (matches Letta-style "approaching cap" guidance,
# Anthropic CLAUDE.md best practice: target <200 lines, warn at 75%).
MEMORY_INDEX_WARN_LINES = int(os.environ.get("AGENT_MEMORY_WARN_LINES", "150"))
MEMORY_INDEX_WARN_BYTES = int(os.environ.get("AGENT_MEMORY_WARN_BYTES", str(18 * 1024)))
MAX_MEMORY_PROMPT_CHARS = int(os.environ.get("AGENT_MAX_MEMORY_PROMPT_CHARS", "10000"))
CHARS_PER_TOKEN = 4

# Session persistence
SESSIONS_DIR = os.path.join(PROJECT_ROOT, "data", "sessions")
LATEST_SESSION_FILE = os.path.join(SESSIONS_DIR, "latest.json")

# ─── Skills ───────────────────────────────────────────────────────────────────

MAX_INSTRUCTIONS_CHARS = int(os.environ.get("AGENT_MAX_SKILL_CHARS", "8000"))
MAX_RESOURCE_CHARS = int(os.environ.get("AGENT_MAX_RESOURCE_CHARS", "10000"))
# NOTE: there is NO skill router — not per-turn, not elsewhere.  The main
# model sees the skill catalog in the system prompt and self-activates via
# the manage_skills tool (progressive disclosure, same pattern Claude Code
# uses).  The former SKILL_PRE_ROUTER + SKILL_ROUTER_* surface was removed
# 2026-05 (see data/memory/journal/).  Eval-only routing — "would query X
# trigger skill Y?" — lives self-contained in
# skills/create-skill/scripts/test_trigger.py.

# ─── Web Fetch ────────────────────────────────────────────────────────────────

WEB_FETCH_DEFAULT_MAX_LENGTH = int(os.environ.get("WEB_FETCH_MAX_LENGTH", "8000"))
WEB_FETCH_MIN_CONTENT_LENGTH = 50

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
