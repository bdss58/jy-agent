# Central configuration — All env-loaded constants in one place.
#
# Every module reads from here instead of doing its own os.environ.get().
# Override any value via environment variables.

import os

# ─── API & Model ──────────────────────────────────────────────────────────────

DEFAULT_MAX_TOKENS = int(os.environ.get("ANTHROPIC_MAX_TOKENS", "16384"))
MAX_TOKENS_CAP = int(os.environ.get("ANTHROPIC_MAX_TOKENS_CAP", "128000"))
AGENT_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")

# ─── Planner / Tool dispatch ─────────────────────────────────────────────────

DEFAULT_MAX_STEPS = int(os.environ.get("AGENT_MAX_STEPS", "100"))
MAX_TOOL_RESULT_CHARS = int(os.environ.get("AGENT_MAX_TOOL_RESULT_CHARS", "8000"))
MAX_TOOL_USE_INPUT_CHARS = int(os.environ.get("AGENT_MAX_TOOL_USE_INPUT_CHARS", "4000"))
MAX_WORKING_TOKENS = int(os.environ.get("AGENT_MAX_WORKING_TOKENS", "100000"))
DEFAULT_TOOL_TIMEOUT = int(os.environ.get("AGENT_TOOL_TIMEOUT", "120"))
STREAM_TIMEOUT = int(os.environ.get("AGENT_STREAM_TIMEOUT", "300"))
COMPACT_TOOL_RESULT_CHARS = 2000  # aggressive limit when compacting old tool results

# ─── Memory ───────────────────────────────────────────────────────────────────

MEMORY_DIR = os.path.join("data", "memory")
TOPICS_DIR = os.path.join(MEMORY_DIR, "topics")
MEMORY_MD_FILE = os.path.join(MEMORY_DIR, "MEMORY.md")
SESSIONS_FILE = os.path.join(MEMORY_DIR, "session_summaries.json")

COMPACT_TOKEN_THRESHOLD = int(os.environ.get("AGENT_COMPACT_TOKEN_THRESHOLD", "80000"))
SUMMARIZE_KEEP_RECENT = int(os.environ.get("AGENT_SUMMARIZE_KEEP_RECENT", "6"))
SUMMARIZE_THRESHOLD = int(os.environ.get("AGENT_SUMMARIZE_THRESHOLD", "20"))
MAX_SESSIONS = 50
MAX_MEMORY_INDEX_LINES = 200
MAX_MEMORY_INDEX_BYTES = 25 * 1024
MAX_MEMORY_PROMPT_CHARS = 5000
CHARS_PER_TOKEN = 4

# ─── Skills ───────────────────────────────────────────────────────────────────

SKILLS_DIR = os.environ.get("AGENT_SKILLS_DIR", "skills")
MAX_INSTRUCTIONS_CHARS = int(os.environ.get("AGENT_MAX_SKILL_CHARS", "8000"))
MAX_RESOURCE_CHARS = int(os.environ.get("AGENT_MAX_RESOURCE_CHARS", "10000"))
SKILL_ROUTER_MODEL = os.environ.get("SKILL_ROUTER_MODEL", "claude-sonnet-4-20250514")
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
