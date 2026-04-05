# Central configuration — All env-loaded constants in one place.
#
# Every module reads from here instead of doing its own os.environ.get().
# Override any value via environment variables.

import os


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    text = value.strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default

# ─── API & Model ──────────────────────────────────────────────────────────────

DEFAULT_ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
AGENT_PROVIDER = (os.environ.get("AGENT_PROVIDER") or "anthropic").strip() or "anthropic"
AGENT_MODEL = os.environ.get("AGENT_MODEL", DEFAULT_ANTHROPIC_MODEL)
DEFAULT_MAX_TOKENS = int(os.environ.get("AGENT_MAX_TOKENS", "16384"))
MAX_TOKENS_CAP = int(os.environ.get("AGENT_MAX_TOKENS_CAP", "128000"))

# ─── Planner / Tool dispatch ─────────────────────────────────────────────────

DEFAULT_MAX_STEPS = int(os.environ.get("AGENT_MAX_STEPS", "100"))
MAX_TOOL_RESULT_CHARS = int(os.environ.get("AGENT_MAX_TOOL_RESULT_CHARS", "8000"))
MAX_TOOL_USE_INPUT_CHARS = int(os.environ.get("AGENT_MAX_TOOL_USE_INPUT_CHARS", "4000"))
MAX_WORKING_TOKENS = int(os.environ.get("AGENT_MAX_WORKING_TOKENS", "100000"))
DEFAULT_TOOL_TIMEOUT = int(os.environ.get("AGENT_TOOL_TIMEOUT", "120"))
STREAM_TIMEOUT = int(os.environ.get("AGENT_STREAM_TIMEOUT", "300"))
COMPACT_TOOL_RESULT_CHARS = 2000  # aggressive limit when compacting old tool results

# ─── Logging / Observability ──────────────────────────────────────────────────

AGENT_LOG_LEVEL = (os.environ.get("AGENT_LOG_LEVEL") or "INFO").strip().upper() or "INFO"
AGENT_LOG_FILE = (os.environ.get("AGENT_LOG_FILE") or os.path.join("data", "logs", "jyagent.jsonl")).strip() or os.path.join("data", "logs", "jyagent.jsonl")
AGENT_LOG_LLM_FAILURE_PAYLOADS = _env_bool("AGENT_LOG_LLM_FAILURE_PAYLOADS", True)
AGENT_LOG_MAX_TEXT_CHARS = int(os.environ.get("AGENT_LOG_MAX_TEXT_CHARS", "4000"))

# ─── Memory ───────────────────────────────────────────────────────────────────

MEMORY_DIR = os.path.join("data", "memory")
TOPICS_DIR = os.path.join(MEMORY_DIR, "topics")
MEMORY_MD_FILE = os.path.join(MEMORY_DIR, "MEMORY.md")

COMPACT_TOKEN_THRESHOLD = int(os.environ.get("AGENT_COMPACT_TOKEN_THRESHOLD", "80000"))
SUMMARIZE_KEEP_RECENT = int(os.environ.get("AGENT_SUMMARIZE_KEEP_RECENT", "6"))
SUMMARIZE_THRESHOLD = int(os.environ.get("AGENT_SUMMARIZE_THRESHOLD", "20"))
MAX_SESSIONS = 50
MAX_MEMORY_INDEX_LINES = 200
MAX_MEMORY_INDEX_BYTES = 25 * 1024
MAX_MEMORY_PROMPT_CHARS = 5000
CHARS_PER_TOKEN = 4

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
    from .runtime.types import ModelSpec

    return ModelSpec(provider=AGENT_PROVIDER, model=AGENT_MODEL)


def get_skill_router_model_spec(active_spec=None):
    from .runtime.types import ModelSpec

    active_spec = active_spec or get_active_model_spec()
    provider = (os.environ.get("SKILL_ROUTER_PROVIDER") or active_spec.provider).strip() or active_spec.provider
    model = (os.environ.get("SKILL_ROUTER_MODEL") or active_spec.model).strip() or active_spec.model
    return ModelSpec(provider=provider, model=model)


def get_subagent_model_spec(tier: str, active_spec=None):
    from .runtime.types import ModelSpec

    active_spec = active_spec or get_active_model_spec()
    tier_key = tier.upper()
    provider = (os.environ.get(f"SUBAGENT_{tier_key}_PROVIDER") or active_spec.provider).strip() or active_spec.provider
    model = (os.environ.get(f"SUBAGENT_{tier_key}_MODEL") or active_spec.model).strip() or active_spec.model
    return ModelSpec(provider=provider, model=model)


def get_reasoning_config_for_provider(provider: str, *, max_output_tokens: int | None = None):
    from .runtime.reasoning import validate_anthropic_thinking, validate_openai_reasoning

    provider = (provider or "").strip()

    if provider == "openai":
        config = {}
        effort = (os.environ.get("OPENAI_REASONING_EFFORT") or "").strip()
        summary = (os.environ.get("OPENAI_REASONING_SUMMARY") or "").strip()
        if effort:
            config["effort"] = effort
        if summary:
            config["summary"] = summary
        if not config:
            return None
        return validate_openai_reasoning(config)

    if provider == "anthropic":
        config = {}
        thinking_type = (os.environ.get("ANTHROPIC_THINKING_TYPE") or "").strip()
        display = (os.environ.get("ANTHROPIC_THINKING_DISPLAY") or "").strip()
        budget_tokens = (os.environ.get("ANTHROPIC_THINKING_BUDGET_TOKENS") or "").strip()
        if thinking_type:
            config["type"] = thinking_type
        if display:
            config["display"] = display
        if budget_tokens:
            try:
                config["budget_tokens"] = int(budget_tokens)
            except ValueError as exc:
                raise ValueError("ANTHROPIC_THINKING_BUDGET_TOKENS must be an integer.") from exc
        if not config:
            return None
        return validate_anthropic_thinking(config, max_output_tokens=max_output_tokens)

    return None
