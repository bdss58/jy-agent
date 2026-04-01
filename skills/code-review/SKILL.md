---
name: code-review
description: >-
  Review code for correctness, bugs, security vulnerabilities, performance issues, and 
  maintainability. Use this skill whenever the user asks to review code, audit a file, 
  find bugs, check for security issues, suggest refactors, review a PR/diff, or assess 
  code quality. TRIGGER on: "review this", "find bugs", "is this code safe", "refactor",
  "code quality", "what's wrong with", PR review, security audit, code smell.
  DO NOT TRIGGER on: writing new code from scratch, debugging runtime errors (use 
  python-debugging skill), or explaining code concepts.
metadata:
  author: jy-agent
  version: "2.0"
---

# Code Review

Systematic code review for quality, correctness, and security.

## Decision Tree: Choose Your Approach

```
What's being reviewed?
├─ Single file → Focused Review (below)
├─ Multiple files / directory → Structural Review
│   1. list_directory(depth=2) → understand project layout
│   2. Identify entry points and hot paths
│   3. Review critical files first, then supporting files
├─ PR / Diff → Delta Review
│   1. Focus ONLY on changed code
│   2. Check: does the change break existing behavior?
│   3. Check: are there missing test updates?
│   4. Check: is the commit message / PR description accurate?
└─ "Find bugs" / specific concern → Targeted Review
    1. Focus on the specific concern
    2. Search for patterns of that bug type across the codebase
    3. grep_files to find all instances
```

## Focused Review Process

### Step 1: Understand Before Judging

```
1. read_file(path, line_numbers=True) → see the full file
2. Check imports → what dependencies does it use?
3. Check the surrounding project:
   - list_directory → sibling files, tests
   - grep_files("import <this_module>") → who uses this code?
4. Understand the PURPOSE before criticizing the implementation
```

**Why context first?** Code that looks "wrong" in isolation is often correct for its context. A bare `except:` might be justified in a top-level error handler. An O(n²) loop might be fine for n < 100.

### Step 2: Review (in priority order)

1. **🔴 Correctness** — Does it actually work?
   - Logic errors, off-by-one, wrong return values
   - Unhandled edge cases (None, empty, boundary)
   - Race conditions in concurrent code

2. **🔴 Security** — Can it be exploited?
   - See [📋 Security Checklist](references/security-checklist.md)

3. **🟡 Error Handling** — Does it fail gracefully?
   - Bare `except:` catching too broadly
   - Swallowed exceptions (catch + pass)
   - Missing cleanup (no `finally`, no `with`)

4. **🟡 Performance** — Is it fast enough?
   - O(n²) where O(n) is possible
   - DB queries in loops (N+1 problem)
   - Missing caches for expensive repeated ops

5. **🔵 Maintainability** — Can someone else work on this?
   - Unclear naming, magic numbers
   - Functions doing too many things
   - Missing type hints (Python) or types (TS)

### Step 3: Output

```markdown
## Summary
One-paragraph assessment of overall code quality.

## Issues Found

### 🔴 Critical (must fix before merge)
- **[file:line]** Description of the issue
  ```python
  # Current (problematic)
  ...
  # Suggested fix
  ...
  ```

### 🟡 Warning (should fix)
- **[file:line]** Description with rationale

### 🔵 Suggestion (nice to have)
- **[file:line]** Suggestion with rationale

## What's Good
- Specific positive observations (not just filler)
```

## Anti-Patterns

❌ **Don't** review without reading the file first — never invent issues from memory
✅ **Do** `read_file(path, line_numbers=True)` before every review

❌ **Don't** dump a generic checklist — "consider adding type hints" on every review
✅ **Do** give specific, actionable feedback tied to actual lines

❌ **Don't** flag style issues as critical — inconsistent quotes won't cause a bug
✅ **Do** prioritize by impact: correctness > security > performance > style

❌ **Don't** suggest rewrites without understanding constraints
✅ **Do** ask "why was it done this way?" before proposing alternatives

❌ **Don't** review test files the same way as production code — tests have different norms
✅ **Do** check tests for: coverage of edge cases, assertion quality, test isolation

## Reference Files

- [🔐 Security Checklist](references/security-checklist.md) — Injection, auth, secrets, dependencies
- [🐍 Language Patterns](references/language-patterns.md) — Python, JS/TS, Bash common bugs & fixes
