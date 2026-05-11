# memory/ package — Self-use memory system (conversation, MEMORY.md, topics, compaction, sessions).
#
# Tier-aligned implementation (see data/memory/topics/memory-design.md):
#   _paths.py         — shared dir bootstrap (ensure_dirs)
#   _index.py         — Tier 1: MEMORY.md (always-loaded index)
#   _topics.py        — Tier 2: curated topic files (on-demand)
#   _journal.py       — Tier 3: chronological journal (on-demand)
#   _consolidation.py — read-only dedup/size analysis
#   operations.py     — cross-tier verbs (remember / forget / show_memory)
#
# This package surface re-exports the public API. Importers should use
# ``from jyagent.memory import …`` rather than reaching into submodules,
# except for tests / advanced callers that need module-private helpers
# (those should import from the owning ``_index`` / ``_topics`` / ``_journal``).

from .conversation import ConversationMemory
from .compaction import (
    compact_conversation, summarize_if_needed,
    record_file_access, get_file_tracker, FileAccessTracker,
)

# Shared filesystem bootstrap (creates MEMORY/topics/journal dirs)
from ._paths import ensure_dirs

# Tier 1 — MEMORY.md (always-loaded index)
from ._index import (
    read_memory_md, read_memory_index,
    write_memory_md, append_memory_md,
    forget_from_memory_md,
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

# Read-only analyzer
from ._consolidation import consolidate_memory

# Cross-tier verbs (used by manage_memory tool)
from .operations import remember, forget, show_memory, replace_memory_entry

from .context import build_memory_context
from .conversation import estimate_tokens, estimate_conversation_tokens, estimate_message_tokens
from .session import (
    checkpoint_session,
    end_session,
    load_session,
    has_saved_session,
    delete_session,
    list_sessions,
    find_session,
    replay_from_events,
)
from .event_log import EventLog, open_event_log, event_log_path
from .extraction import should_extract, extract_and_remember, extract_text
from .search import search_memory, render_hits, SearchHit, SearchChunk
