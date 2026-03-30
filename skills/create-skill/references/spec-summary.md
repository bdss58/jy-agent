# Agent Skills Specification Summary

Source: https://agentskills.io/specification

## Directory Structure
```
my-skill/
├── SKILL.md          # Required — YAML frontmatter + markdown
├── scripts/          # Optional — executable code
├── references/       # Optional — additional docs
└── assets/           # Optional — templates, images, data
```

## SKILL.md Format

### Frontmatter (YAML between `---`)

| Field | Required | Constraints |
|-------|----------|-------------|
| name | Yes | 1-64 chars, lowercase alphanumeric + hyphens, no `--`, must match dir name |
| description | Yes | 1-1024 chars, describes what + when to use |
| license | No | License name or reference |
| compatibility | No | 1-500 chars, environment requirements |
| metadata | No | Arbitrary key-value map |
| allowed-tools | No | Space-delimited pre-approved tool list (experimental) |

### Body (Markdown after frontmatter)
- No format restrictions
- Recommended: step-by-step instructions, examples, edge cases
- Keep under 500 lines (~5000 tokens)
- Move detailed content to references/

## Progressive Disclosure (3-Tier Loading)

| Stage | Budget | What Loads |
|-------|--------|-----------|
| Advertise | ~100 tokens/skill | Only name + description |
| Load | < 5000 tokens | Full SKILL.md body |
| Read | As needed | Files from scripts/, references/, assets/ |

## File References
- Use relative paths from skill root
- Example: `references/API.md`, `scripts/validate.py`
- Avoid deeply nested reference chains
