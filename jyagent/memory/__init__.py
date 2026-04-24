# memory/ package — Self-use memory system (conversation, MEMORY.md, topics, compaction, sessions).

# Re-export the live memory API.
from .conversation import ConversationMemory
from .compaction import (
    compact_conversation, summarize_if_needed,
    record_file_access, get_file_tracker, FileAccessTracker,
)
from .operations import (
    read_memory_md, read_memory_index, write_memory_md, append_memory_md,
    forget_from_memory_md,
    list_topics, read_topic, read_topic_body, read_topic_meta, write_topic, delete_topic,
    list_journals, read_journal, append_journal,
    memory_index_size_warning, consolidate_memory,
    remember, forget, show_memory,
)
from .context import build_memory_context
from .conversation import estimate_tokens, estimate_conversation_tokens, estimate_message_tokens
from .session import save_session, archive_session, load_session, has_saved_session, delete_session
from .extraction import should_extract, extract_and_remember
