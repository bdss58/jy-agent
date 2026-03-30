"""
edit_file — Precise file editing tool (inspired by Claude Code Edit + Anthropic EditTool).

Modes:
  1. str_replace: old_text → new_text (exact match, single occurrence)
  2. insert: insert_at_line=N inserts new_text before line N
  3. append: old_text="" appends new_text to end of file
  4. create: file doesn't exist + create_if_missing=True creates with new_text

Safety:
  - Exact match required for replace (no fuzzy matching)
  - Fails if old_text appears multiple times (ambiguous) with line positions
  - Whitespace/indentation diagnostics on mismatch
  - Dry-run mode for previewing changes
  - Shows context lines around the edit in result
"""

import os


def edit_file(
    path: str,
    new_text: str,
    old_text: str = "",
    insert_at_line: int = 0,
    create_if_missing: bool = False,
    dry_run: bool = False,
) -> str:
    """Edit a file by replacing old_text with new_text, or insert at a line number.

    Modes:
    - Replace: provide both old_text and new_text to replace a specific block
    - Insert at line: set insert_at_line=N to insert new_text before line N (1-based)
    - Append: provide only new_text (old_text="" and insert_at_line=0) to append to file
    - Create: set create_if_missing=True to create new files with new_text as content

    Args:
        path: File path to edit
        new_text: The new text to insert (replacement, insertion, or file content)
        old_text: The existing text to find and replace (exact match required).
                  Leave empty for append/insert/create modes.
        insert_at_line: Insert new_text before this line number (1-based). 0 = disabled.
        create_if_missing: If True and file doesn't exist, create it with new_text
        dry_run: If True, show what would change without actually writing
    """
    # --- Create mode ---
    if not os.path.exists(path):
        if create_if_missing:
            if dry_run:
                return f"[DRY RUN] Would create {path} ({len(new_text)} chars)"
            dirname = os.path.dirname(path)
            if dirname:
                os.makedirs(dirname, exist_ok=True)
            with open(path, 'w', encoding='utf-8') as f:
                f.write(new_text)
            lines = new_text.count('\n') + (1 if new_text and not new_text.endswith('\n') else 0)
            return f"Created {path} ({lines} lines, {len(new_text)} chars)"
        else:
            return f"Error: File not found: {path} (set create_if_missing=True to create)"

    # --- Read existing file ---
    try:
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        return f"Error reading {path}: {e}"

    # --- Insert at line mode ---
    if insert_at_line > 0:
        all_lines = content.split('\n')
        # insert_at_line is 1-based; inserting before that line
        idx = max(0, min(insert_at_line - 1, len(all_lines)))

        # Ensure new_text ends with newline for clean insertion
        insert_text = new_text if new_text.endswith('\n') else new_text + '\n'
        insert_lines = insert_text.split('\n')
        # split produces trailing empty element for trailing \n
        if insert_lines and insert_lines[-1] == '':
            insert_lines = insert_lines[:-1]
        n_inserted = len(insert_lines)

        if dry_run:
            return (
                f"[DRY RUN] Would insert {n_inserted} lines before L{insert_at_line} in {path}\n"
                f"  File currently has {len(all_lines)} lines"
            )

        new_lines = all_lines[:idx] + insert_lines + all_lines[idx:]
        new_content = '\n'.join(new_lines)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(new_content)

        # Show context
        ctx_start = max(0, idx - 1)
        ctx_end = min(len(new_lines), idx + n_inserted + 1)
        context = '\n'.join(
            f"  {'>' if idx <= j < idx + n_inserted else ' '} "
            f"L{j + 1}: {new_lines[j]}"
            for j in range(ctx_start, ctx_end)
        )

        return (
            f"Inserted {n_inserted} lines before L{insert_at_line} in {path} "
            f"(now {len(new_lines)} lines)\n{context}"
        )

    # --- Append mode ---
    if old_text == "":
        if dry_run:
            return f"[DRY RUN] Would append {len(new_text)} chars to {path}"
        new_content = content + new_text
        with open(path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        added_lines = new_text.count('\n') + (1 if new_text and not new_text.endswith('\n') else 0)
        total_lines = new_content.count('\n') + 1
        return f"Appended to {path} (+{added_lines} lines, total {total_lines} lines)"

    # --- Replace mode (str_replace) ---
    count = content.count(old_text)
    if count == 0:
        return _diagnose_no_match(path, content, old_text)

    if count > 1:
        return _diagnose_multi_match(path, content, old_text, count)

    # Exactly one match — perform the replacement
    new_content = content.replace(old_text, new_text, 1)

    # Calculate edit stats
    old_line_count = old_text.count('\n') + 1
    new_line_count = new_text.count('\n') + 1
    delta = new_line_count - old_line_count

    # Find the line number of the edit
    edit_pos = content.index(old_text)
    edit_line = content[:edit_pos].count('\n') + 1

    if dry_run:
        return (
            f"[DRY RUN] Would edit {path} at L{edit_line}:\n"
            f"  - Remove {old_line_count} lines\n"
            f"  + Add {new_line_count} lines\n"
            f"  Net: {'+' if delta > 0 else ''}{delta} lines"
        )

    with open(path, 'w', encoding='utf-8') as f:
        f.write(new_content)

    total_lines = new_content.count('\n') + 1
    delta_str = f"+{delta}" if delta > 0 else str(delta)

    # Show context around the edit
    result_lines = new_content.split('\n')
    ctx_start = max(0, edit_line - 2)
    ctx_end = min(len(result_lines), edit_line + new_line_count + 1)
    context = '\n'.join(
        f"  {'>' if edit_line <= (j + 1) <= edit_line + new_line_count - 1 else ' '} "
        f"L{j + 1}: {result_lines[j]}"
        for j in range(ctx_start, ctx_end)
    )

    return (
        f"Edited {path}: replaced {old_line_count} lines with {new_line_count} lines "
        f"at L{edit_line} ({delta_str}), total {total_lines} lines\n{context}"
    )


def _diagnose_no_match(path: str, content: str, old_text: str) -> str:
    """Provide helpful diagnostics when old_text is not found."""
    old_stripped = old_text.strip()

    # Check for whitespace/indentation mismatch
    if old_stripped and old_stripped in content:
        all_lines = content.split('\n')
        for i, line in enumerate(all_lines):
            if old_stripped[:40] in line:
                start = max(0, i - 1)
                end = min(len(all_lines), i + 4)
                snippet = '\n'.join(
                    f"  L{start + 1 + j}: {all_lines[start + j]}"
                    for j in range(end - start)
                )
                return (
                    f"Error: Exact match not found in {path}.\n"
                    f"Content exists but with different whitespace/indentation.\n"
                    f"Nearby lines:\n{snippet}\n\n"
                    f"Hint: Copy the exact text including indentation. "
                    f"Use read_file with line_numbers=True to see exact content."
                )

    # Check for partial first-line match
    old_lines = old_text.strip().split('\n')
    if len(old_lines) > 1:
        first_line = old_lines[0].strip()
        if first_line and first_line in content:
            return (
                f"Error: Exact match not found in {path}.\n"
                f"First line found ('{first_line[:60]}') but full block doesn't match.\n"
                f"Hint: Use read_file with line_numbers=True to check exact content."
            )

    return (
        f"Error: old_text not found in {path}.\n"
        f"Searched for ({len(old_text)} chars): {repr(old_text[:100])}{'...' if len(old_text) > 100 else ''}\n"
        f"Hint: Use read_file with line_numbers=True to verify the current content."
    )


def _diagnose_multi_match(path: str, content: str, old_text: str, count: int) -> str:
    """Provide helpful diagnostics when old_text appears multiple times."""
    positions = []
    search_from = 0
    for _ in range(count):
        idx = content.index(old_text, search_from)
        line_num = content[:idx].count('\n') + 1
        positions.append(line_num)
        search_from = idx + 1

    return (
        f"Error: old_text appears {count} times in {path} (at lines {positions}).\n"
        f"Include more surrounding context in old_text to make it unique.\n"
        f"Hint: Add a few lines before/after the target to disambiguate."
    )


TOOL_SCHEMA = {
    "name": "edit_file",
    "description": "Edit a file by replacing old_text with new_text (exact match), inserting at a line number, appending, or creating. "
                   "More token-efficient than write_file for surgical edits. Use read_file with line_numbers=True first to see exact content.",
    "input_schema": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "File path to edit"
            },
            "new_text": {
                "type": "string",
                "description": "The new text to insert (replacement, insertion, or file content)"
            },
            "old_text": {
                "type": "string",
                "description": "The existing text to find and replace (exact match). Leave empty for append/insert/create modes."
            },
            "insert_at_line": {
                "type": "integer",
                "description": "Insert new_text before this line number (1-based). 0=disabled. Use read_file with line_numbers=True to find line numbers."
            },
            "create_if_missing": {
                "type": "boolean",
                "description": "If true and file doesn't exist, create it with new_text (default false)"
            },
            "dry_run": {
                "type": "boolean",
                "description": "If true, show what would change without writing (default false)"
            }
        },
        "required": ["path", "new_text"]
    }
}
