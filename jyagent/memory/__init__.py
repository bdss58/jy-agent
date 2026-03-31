# memory/ package — Self-use memory system (MEMORY.md, topics, sessions, compaction).
#
# Replaces the monolithic self_memory.py (912 lines) with focused modules:
#   utils.py        — atomic writes, JSON loading, token estimation
#   conversation.py — ConversationMemory (in-memory chat history)
#   persistent.py   — PersistentMemory (file-backed KV store)
#   sessions.py     — SessionSummaries + session lifecycle
#   compaction.py   — Conversation compaction (Claude Code /compact)
#   operations.py   — MEMORY.md + topic file CRUD + remember/forget/show
#   context.py      — build_memory_context (system prompt injection)

# Re-export all public API for backward compatibility
from .conversation import ConversationMemory
from .persistent import PersistentMemory
from .sessions import SessionSummaries, on_session_start, on_session_end
from .compaction import compact_conversation, summarize_if_needed
from .operations import (
    read_memory_md, read_memory_index, write_memory_md, append_memory_md,
    forget_from_memory_md,
    list_topics, read_topic, write_topic, delete_topic,
    remember, forget, show_memory,
)
from .context import build_memory_context
from .utils import estimate_tokens, estimate_conversation_tokens, estimate_message_tokens
