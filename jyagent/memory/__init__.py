# memory/ package — Self-use memory system (conversation, MEMORY.md, topics, compaction, sessions).
#
# Tier-aligned implementation (see data/memory/topics/memory-design.md):
#   _paths.py         — shared dir bootstrap (ensure_dirs)
#   _index.py         — Tier 1: MEMORY.md (always-loaded index)
#   _topics.py        — Tier 2: curated topic files (on-demand)
#   _journal.py       — Tier 3: chronological journal (on-demand)
#   operations.py     — cross-tier verbs (remember / forget / show_memory /
#                       replace_memory_entry / consolidate_memory)
#
# This package surface re-exports the **production** public API only. The set
# of facade-exported names was trimmed 2026-05-17 to match what production
# callers and tests actually import via ``from jyagent.memory import …``;
# anything implementation-shaped or used only from one direct caller now
# lives at its submodule path (``from jyagent.memory.search import
# search_memory``, ``from jyagent.memory.compaction import
# FileAccessTracker``, etc.). See verdict 3.2 in
# ``data/memory/topics/simplification-audit-2026-05.md``.

# Conversation / compaction (compact_conversation, FileAccessTracker etc.
# live at .compaction submodule path)
from .conversation import ConversationMemory
from .compaction import summarize_if_needed

# Shared filesystem bootstrap (creates MEMORY/topics/journal dirs)
from ._paths import ensure_dirs

# Tier 1 — MEMORY.md (always-loaded index)
from ._index import (
    read_memory_md,
    write_memory_md,
    append_memory_md,
    memory_index_size_warning,
)

# Tier 2 — topic files (on-demand)
from ._topics import (
    list_topics, read_topic, read_topic_body, read_topic_meta,
    read_topic_section, list_topic_sections,
    write_topic, delete_topic,
)

# Tier 3 — journal (on-demand)
from ._journal import (
    list_journals, read_journal, append_journal,
)

# Cross-tier verbs (used by manage_memory tool)
from .operations import (
    remember, forget, show_memory,
    consolidate_memory,
)

# Memory-context builder + session persistence
from .context import build_memory_context
from .session import (
    checkpoint_session,
    end_session,
    load_session,
    has_saved_session,
    list_sessions,
    find_session,
    replay_from_events,
)
from .event_log import EventLog, event_log_path

# Proactive extraction trigger (used by agent loop)
from .extraction import should_extract, extract_and_remember, extract_text
