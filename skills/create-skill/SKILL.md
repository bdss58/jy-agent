---
name: create-skill
description: >-
  Create new Agent Skills, modify existing skills, and improve skill quality following
  the open standard (agentskills.io). Use this skill whenever the user wants to create
  a skill from scratch, edit or improve an existing skill, turn a workflow into a reusable
  skill, or package domain expertise. TRIGGER on: "create a skill", "make a skill for",
  "turn this into a skill", "improve this skill", "skill for X", packaging expertise,
  creating reusable agent instructions. DO NOT TRIGGER on: using an existing skill
  (that skill should trigger instead), or general coding tasks.
metadata:
  author: jy-agent
  version: "2.0"
---

# Skill Creator

Create and improve Agent Skills. Based on patterns from [Anthropic's skill-creator](https://github.com/anthropics/skills).

## Decision Tree: What's the User Doing?

```
User request →
├─ "Create a skill for X" → Full Creation Flow (below)
├─ "Turn this conversation into a skill" → Extract from Context
│   1. Identify the workflow from conversation history
│   2. Extract: tools used, step sequence, corrections made, patterns learned
│   3. Skip interview — infer from context, confirm gaps with user
│   4. Continue to Write phase
├─ "Improve/fix this skill" → Improvement Flow
│   1. manage_skills(action="info", name="skill-name")
│   2. Read current SKILL.md and resources
│   3. Identify specific issues (triggering? quality? missing cases?)
│   4. Apply targeted fixes
└─ "What skills do I have?" → Inventory
    1. manage_skills(action="list")
    2. Summarize what each does and quality assessment
```

## Full Creation Flow

### Phase 1: Capture Intent

Ask the user (but infer from context first — don't ask what you already know):

1. **What** should this skill enable the agent to do?
2. **When** should it trigger? (specific phrases, contexts, tool use patterns)
3. **Output format** — what does success look like?
4. **Edge cases** — what could go wrong?

### Phase 2: Write the SKILL.md

#### Anatomy of a Skill
```
skill-name/
├── SKILL.md              # Required — YAML frontmatter + instructions
├── references/           # Detailed docs, checklists, examples
├── scripts/              # Executable helpers (use as black boxes)
└── assets/               # Templates, icons, data files
```

#### Template
```markdown
---
name: my-skill-name
description: >-
  Clear description of what AND when to use. Be "pushy" — describe
  specific trigger phrases. Include TRIGGER and DO NOT TRIGGER guidance.
  Max 1024 chars.
metadata:
  author: "author-name"
  version: "1.0"
---

# Skill Title

Brief overview of what this skill does.

## Decision Tree: Choose Your Approach
(Route the user's task to the right workflow)

## Core Process
(Step-by-step instructions — the main workflow)

## Anti-Patterns
(❌/✅ pairs documenting common mistakes)

## Reference Files
(Links to references/, scripts/, assets/)
```

### Phase 3: Create Resources

**Progressive disclosure is key:**
- SKILL.md body: **< 500 lines** (~5000 tokens) — routing logic and core instructions
- `references/`: Detailed checklists, examples, API docs — loaded on demand
- `scripts/`: Executable code — run with `--help`, don't read source
- `assets/`: Templates, data files — used in output

### Phase 4: Register

```python
# Option A: Use manage_skills tool
manage_skills(action="create", name="my-skill", description="...", instructions="...")

# Option B: Create files directly (more control)
run_shell("mkdir -p skills/my-skill/{references,scripts,assets}")
write_file("skills/my-skill/SKILL.md", content)
write_file("skills/my-skill/references/...", content)
manage_skills(action="reload")
```

## Skill Writing Guide

### Descriptions — Be Pushy

Claude tends to **under-trigger** skills. Make descriptions assertive:

```yaml
# ❌ Too passive
description: Guide for building dashboards

# ✅ Pushy — explicit trigger conditions
description: >-
  Build data dashboards and visualizations. Use this skill whenever the user 
  mentions dashboards, data viz, charts, graphs, metrics display, or wants
  to visualize any kind of data, even if they don't explicitly say "dashboard".
  TRIGGER on: "show me a chart", "visualize this data", "dashboard", "metrics".
  DO NOT TRIGGER on: static reports, CSV exports without visualization.
```

### Instructions — Decision Trees, Not Checklists

```markdown
# ❌ Flat checklist (generic, doesn't guide decisions)
1. Check for X
2. Check for Y
3. Check for Z

# ✅ Decision tree (guides the agent to the right approach)
What kind of input?
├─ Type A → Use approach 1
├─ Type B → Use approach 2
└─ Unknown → Ask the user
```

### Anti-Patterns — Show ❌/✅ Pairs

```markdown
❌ **Don't** do X because [reason]
✅ **Do** Y instead because [reason]
```

**Why pairs?** The ❌ shows what to watch for, the ✅ shows what to do instead. Just saying "do Y" misses the cases where the agent might accidentally do X.

### Resources — Explain the Why

From Anthropic's skill-creator:
> "Try hard to explain the **why** behind everything you're asking the model to do. Today's LLMs are smart. If you find yourself writing ALWAYS or NEVER in all caps, that's a yellow flag — reframe and explain the reasoning."

## Name Rules

- Lowercase letters, numbers, and hyphens only
- 1-64 characters, no `--`, no leading/trailing hyphens
- Must match the directory name
- Examples: `data-analysis`, `api-testing`, `react-components`

## Quality Checklist

- [ ] `name` matches directory name
- [ ] `description` has TRIGGER and DO NOT TRIGGER guidance
- [ ] Description is "pushy" enough to trigger when relevant
- [ ] SKILL.md body has a decision tree (not just flat steps)
- [ ] Anti-patterns documented with ❌/✅ pairs
- [ ] Reference files for detailed content (body < 500 lines)
- [ ] Instructions explain WHY, not just WHAT

## Reference

See [📋 Agent Skills Spec](references/spec-summary.md) for the full specification.
