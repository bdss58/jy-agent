# Agent Skills Specification Summary

**Sources** (verified 2026-05):
- Home: https://agentskills.io
- Spec: https://agentskills.io/specification.md
- llms.txt index: https://agentskills.io/llms.txt
- Reference impl: https://github.com/agentskills/agentskills
- Example skills: https://github.com/anthropics/skills

The Agent Skills format was originally developed by Anthropic and released as
an open standard. It is now community-maintained and adopted across many
agent products (Claude Code, OpenAI Codex, VS Code + Copilot, Cursor, …).

## Directory Structure

```
my-skill/
├── SKILL.md       # Required — YAML frontmatter + Markdown instructions
├── scripts/       # Optional — executable code (run as black boxes)
├── references/    # Optional — detailed docs loaded on demand
├── assets/        # Optional — templates, images, data files
└── evals/         # Convention — test cases (excluded from packaging)
```

VS Code looks for skills in `.agents/skills/` by default. Other clients
have their own conventions but the format is portable.

## SKILL.md Format

### Frontmatter (YAML between `---`)

| Field           | Required | Constraints                                                          |
|-----------------|----------|----------------------------------------------------------------------|
| `name`          | Yes      | 1-64 chars; lowercase `a-z`, digits, hyphens; no `--`; no leading/trailing hyphen; must match dir name |
| `description`   | Yes      | 1-1024 chars; *what it does* AND *when to use it*. Be pushy.        |
| `license`       | No       | License name or path to bundled license file                         |
| `compatibility` | No       | 1-500 chars. Only set if the skill has real env requirements         |
| `metadata`      | No       | Arbitrary string→string map (author, version, tags, …)               |
| `allowed-tools` | No       | Space-separated pre-approved tool list (experimental, client-specific) |

Most skills only need `name` and `description`. Add `compatibility` only
when the skill genuinely depends on a runtime (e.g. "Requires Python 3.14+
and uv", "Designed for Claude Code").

### Body (Markdown after frontmatter)

- Keep under **500 lines / ~5000 tokens** — anything more belongs in references/.
- Open with a decision tree that routes the agent to the right approach.
- Step-by-step core process.
- Anti-patterns (❌/✅ pairs).
- Links to reference files with explicit "load X when Y" guidance.

## Progressive Disclosure (3-Tier Loading)

| Stage     | Budget                 | What loads                            | When                                  |
|-----------|------------------------|---------------------------------------|---------------------------------------|
| Discovery | ~100 tokens/skill      | `name` + `description`                | Startup, for every installed skill    |
| Activate  | < 5000 tokens          | Full `SKILL.md` body                  | When the description matches the task |
| Read      | As needed              | `references/`, `scripts/`, `assets/`  | Agent reads/runs on demand            |

**Tell the agent WHEN to load each file.** "Read `references/api-errors.md`
if the API returns a non-200 status code" is more useful than "see
references/ for details."

## Triggering Nuance

Agents typically only consult skills for tasks that need knowledge or
capabilities beyond what they can already handle. A simple one-step request
("read this PDF") may not trigger a PDF skill even if the description
matches — because the agent can already do it. Skills win on:

- Unfamiliar APIs and domain-specific workflows
- Uncommon formats or fragile procedures
- Multi-step pipelines where consistency matters
- Project-specific conventions the model doesn't know

## Name Rules (enforced by spec)

- Lowercase letters (`a-z`), digits, and hyphens only
- 1-64 characters
- No consecutive hyphens (`--`)
- No leading or trailing hyphen
- Must match the parent directory name

## Validation

The official reference validator is **`skills-ref`**:
https://github.com/agentskills/agentskills/tree/main/skills-ref

```bash
skills-ref validate ./my-skill
```

It checks frontmatter, name rules, and structural conventions. Our local
`scripts/validate_skill.py` adds opinionated quality checks (description
quality, decision-tree presence, broken refs, ALL-CAPS heuristics).

## Key Patterns from Production Skills

1. **Decision trees over checklists** — Route first, execute second.
2. **Pushy descriptions** — Explicitly enumerate TRIGGER / DO NOT TRIGGER cases.
3. **Scripts as black boxes** — "Run with `--help` first. Do not read the source."
4. **Anti-patterns with pairs** — ❌ Don't / ✅ Do, side-by-side.
5. **Explain the why** — Smart models generalize from reasoning, not ALL-CAPS rules.
6. **Progressive disclosure** — `SKILL.md` is the router; details live in references/.
7. **Keep it lean** — If the model ignores an instruction, it's noise. Cut it.
8. **Start from real expertise** — Extract from a real task, not generic LLM output.
9. **Compare with-skill vs without-skill** — Without a baseline, you can't tell
   if the skill is actually helping.
