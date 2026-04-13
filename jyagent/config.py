# Central configuration — All env-loaded constants in one place.
#
# Every module reads from here instead of doing its own os.environ.get().
# Override any value via environment variables.

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


def register_provider(name: str) -> None:
    """Register an additional runtime provider name as valid."""
    SUPPORTED_RUNTIME_PROVIDERS.add(name)

# ─── Planner / Tool dispatch ─────────────────────────────────────────────────

DEFAULT_MAX_STEPS = int(os.environ.get("AGENT_MAX_STEPS", "100"))
MAX_TOOL_RESULT_CHARS = int(os.environ.get("AGENT_MAX_TOOL_RESULT_CHARS", "8000"))
MAX_TOOL_USE_INPUT_CHARS = int(os.environ.get("AGENT_MAX_TOOL_USE_INPUT_CHARS", "4000"))
MAX_WORKING_TOKENS = int(os.environ.get("AGENT_MAX_WORKING_TOKENS", "180000"))  # Layer 1 safety net (cheap truncation within a turn)
DEFAULT_TOOL_TIMEOUT = int(os.environ.get("AGENT_TOOL_TIMEOUT", "120"))
STREAM_TIMEOUT = int(os.environ.get("AGENT_STREAM_TIMEOUT", "300"))
COMPACT_TOOL_RESULT_CHARS = 2000  # aggressive limit when compacting old tool results
OBSERVATION_MASK_DISTANCE = int(os.environ.get("AGENT_OBSERVATION_MASK_DISTANCE", "5"))  # fully clear tool results older than N messages from the end

# ─── Memory ───────────────────────────────────────────────────────────────────

MEMORY_DIR = os.path.join(PROJECT_ROOT, "data", "memory")
TOPICS_DIR = os.path.join(MEMORY_DIR, "topics")
MEMORY_MD_FILE = os.path.join(MEMORY_DIR, "MEMORY.md")

COMPACT_TOKEN_THRESHOLD = int(os.environ.get("AGENT_COMPACT_TOKEN_THRESHOLD", "150000"))  # Layer 2 (LLM summary between turns); matches Anthropic API default & 75% of 200K window
SUMMARIZE_KEEP_RECENT = int(os.environ.get("AGENT_SUMMARIZE_KEEP_RECENT", "6"))
SUMMARIZE_THRESHOLD = int(os.environ.get("AGENT_SUMMARIZE_THRESHOLD", "20"))
FILE_REINJECTION_COUNT = int(os.environ.get("AGENT_FILE_REINJECTION_COUNT", "5"))  # re-inject N most recent files after compaction
FILE_REINJECTION_MAX_TOKENS = int(os.environ.get("AGENT_FILE_REINJECTION_MAX_TOKENS", "50000"))  # cap total re-injected content
MAX_SESSIONS = 50
MAX_MEMORY_INDEX_LINES = 200
MAX_MEMORY_INDEX_BYTES = 25 * 1024
MAX_MEMORY_PROMPT_CHARS = int(os.environ.get("AGENT_MAX_MEMORY_PROMPT_CHARS", "10000"))
CHARS_PER_TOKEN = 4

# Session persistence
SESSIONS_DIR = os.path.join(PROJECT_ROOT, "data", "sessions")
LATEST_SESSION_FILE = os.path.join(SESSIONS_DIR, "latest.json")

# ─── Skills ───────────────────────────────────────────────────────────────────

MAX_INSTRUCTIONS_CHARS = int(os.environ.get("AGENT_MAX_SKILL_CHARS", "8000"))
MAX_RESOURCE_CHARS = int(os.environ.get("AGENT_MAX_RESOURCE_CHARS", "10000"))
SKILL_ROUTER_PROVIDER = os.environ.get("SKILL_ROUTER_PROVIDER", AGENT_PROVIDER)
SKILL_ROUTER_MODEL = os.environ.get("SKILL_ROUTER_MODEL", AGENT_MODEL)
SKILL_ROUTER_TIMEOUT = int(os.environ.get("SKILL_ROUTER_TIMEOUT", "5"))

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
    from .runtime.types import ModelSpec

    return ModelSpec(provider=validate_runtime_provider(provider, source=source), model=model)


def get_skill_router_model_spec(active_spec=None):
    active_spec = active_spec or get_active_model_spec()
    provider = (os.environ.get("SKILL_ROUTER_PROVIDER") or active_spec.provider).strip() or active_spec.provider
    model = (os.environ.get("SKILL_ROUTER_MODEL") or active_spec.model).strip() or active_spec.model
    return build_model_spec(provider, model, source="SKILL_ROUTER_PROVIDER")


def get_subagent_model_spec(tier: str, active_spec=None):
    active_spec = active_spec or get_active_model_spec()
    tier_key = tier.upper()
    provider = (os.environ.get(f"SUBAGENT_{tier_key}_PROVIDER") or active_spec.provider).strip() or active_spec.provider
    model = (os.environ.get(f"SUBAGENT_{tier_key}_MODEL") or active_spec.model).strip() or active_spec.model
    return build_model_spec(provider, model, source=f"SUBAGENT_{tier_key}_PROVIDER")


def get_reasoning_config_for_provider(
    provider: str,
    *,
    max_output_tokens: int | None = None,
    model: str | None = None,
):
    from .runtime.providers._anthropic_reasoning import validate_anthropic_reasoning

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
        effort = (os.environ.get("OPENAI_REASONING_EFFORT") or "").strip().lower()
        if not effort:
            return None
        resolved_model = (model or "").strip()
        if not resolved_model and AGENT_PROVIDER == "openai":
            resolved_model = AGENT_MODEL
        normalized_model = resolved_model.lower()
        if not (
            normalized_model == "gpt-5.4"
            or normalized_model.startswith("gpt-5.4-")
        ):
            raise ValueError(
                "OPENAI_REASONING_EFFORT is only supported for OpenAI model "
                f"'gpt-5.4' and its variants, not '{resolved_model or '<unset>'}'."
            )
        valid_efforts = {"none", "low", "medium", "high", "xhigh"}
        if effort not in valid_efforts:
            raise ValueError(
                f"OPENAI_REASONING_EFFORT must be one of: {sorted(valid_efforts)}. Got '{effort}'."
            )
        return {"effort": effort}

    return None
