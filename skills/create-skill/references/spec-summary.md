# Agent Skills Specification Summary

Source: https://agentskills.io/specification
Reference: https://github.com/anthropics/skills

## Directory Structure
```
my-skill/
├── SKILL.md          # Required — YAML frontmatter + markdown instructions
├── scripts/          # Optional — executable code (run as black boxes)
├── references/       # Optional — detailed docs loaded on demand
└── assets/           # Optional — templates, images, data files
```

## SKILL.md Format

### Frontmatter (YAML between `---`)

| Field | Required | Constraints |
|-------|----------|-------------|
| name | Yes | 1-64 chars, lowercase alphanumeric + hyphens, no `--`, must match dir name |
| description | Yes | 1-1024 chars, what it does + when to trigger (be pushy!) |
| license | No | License name or reference |
| compatibility | No | 1-500 chars, environment requirements |
| metadata | No | Arbitrary key-value map (author, version, etc.) |
| allowed-tools | No | Space-delimited pre-approved tool list (experimental) |

### Body (Markdown after frontmatter)
- Keep under 500 lines (~5000 tokens)
- Decision tree first (route the task)
- Step-by-step core process
- Anti-patterns (❌/✅ pairs)
- Links to reference files

## Progressive Disclosure (3-Tier Loading)

| Stage | Budget | What Loads | Example |
|-------|--------|-----------|---------|
| Advertise | ~100 tokens/skill | name + description | Shown in available_skills list |
| Load | < 5000 tokens | Full SKILL.md body | Injected when skill activates |
| Read | As needed | references/, scripts/, assets/ | Agent reads with read_file on demand |

## Name Rules
- Lowercase letters, numbers, and hyphens only
- 1-64 characters
- No consecutive hyphens (`--`)
- No starting/ending with hyphens
- Must match the directory name

## Key Patterns from Anthropic's Production Skills

1. **Decision trees over checklists** — Route to the right approach first
2. **Pushy descriptions** — Explicitly state TRIGGER/DON'T TRIGGER conditions
3. **Scripts as black boxes** — "Run with --help first. DO NOT read the source."
4. **Anti-patterns with pairs** — ❌ Don't / ✅ Do format
5. **Explain the why** — Smart models work better with reasoning than rigid rules
6. **Progressive disclosure** — SKILL.md is the router, references/ has the details
7. **Keep it lean** — Remove instructions that aren't pulling their weight
