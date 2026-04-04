---
name: create-skill
description: >-
  Create new Agent Skills, modify existing skills, and improve skill quality
  using eval-driven development. Use this skill whenever the user wants to
  create a skill from scratch, edit or improve an existing skill, turn a
  workflow into a reusable skill, optimize a skill's description for better
  triggering, run trigger tests, or package a skill for distribution.
  TRIGGER on: "create a skill", "make a skill for", "turn this into a skill",
  "improve this skill", "skill for X", "test this skill", "package this skill",
  packaging expertise, creating reusable agent instructions.
  DO NOT TRIGGER on: using an existing skill (that skill should trigger instead),
  general coding tasks, or asking about what skills are available.
metadata:
  author: jy-agent
  version: "3.0"
---

# Skill Creator

Create and iteratively improve Agent Skills using eval-driven development.

The core loop: **Draft → Test → Review → Improve → Repeat**

Don't just write a skill and hope it works. Measure trigger accuracy,
grade output quality, and iterate based on data.

## Decision Tree: What's the User Doing?

```
User request →
├─ "Create a skill for X" → Full Creation Flow
│
├─ "Turn this conversation into a skill" → Extract from Context
│   1. Identify the workflow from conversation history
│   2. Extract: tools used, step sequence, corrections made
│   3. Skip interview — infer from context, confirm gaps
│   4. Jump to Phase 2 (Write)
│
├─ "Improve/fix this skill" → Improvement Flow
│   1. manage_skills(action="info", name="skill-name")
│   2. Read current SKILL.md and resources
│   3. Run trigger tests: python scripts/test_trigger.py <skill-dir> -v
│   4. Run validation: python scripts/validate_skill.py <skill-dir>
│   5. Fix issues based on test + validation results
│
├─ "Test/optimize this skill" → Eval Flow
│   1. Run tests: python scripts/test_trigger.py <skill-dir> -v
│   2. If imperfect: python scripts/improve_description.py <skill-dir>
│   3. Re-validate: python scripts/validate_skill.py <skill-dir>
│
├─ "Package this skill" → Package Flow
│   1. Validate: python scripts/validate_skill.py <skill-dir>
│   2. Package: python scripts/package_skill.py <skill-dir> [output-dir]
│
└─ "What skills do I have?" → Inventory
    1. manage_skills(action="list")
    2. Summarize each skill's purpose and quality
```

## Full Creation Flow

### Phase 1: Capture Intent

Understand what the user wants. Infer from context first — don't ask what
you already know. Fill gaps by asking:

1. **What** should this skill enable the agent to do?
2. **When** should it trigger? (specific phrases, contexts)
3. **Output format** — what does good output look like?
4. **Edge cases** — what could go wrong?

**Communication awareness**: Skill users range from expert developers to
non-technical users. Watch for context cues and adjust your language.
Terms like "eval" and "assertion" need brief explanations for general
audiences: "...run the tests (checks that verify the skill works)..."

### Phase 2: Write the SKILL.md

#### Anatomy of a Skill
```
skill-name/
├── SKILL.md              # Required — YAML frontmatter + instructions
├── references/           # Detailed docs, loaded on demand
├── scripts/              # Executable helpers (run, don't read)
├── agents/               # Subagent instructions (grader, etc.)
├── assets/               # Templates, icons, data files
└── evals/                # Test data (excluded from packaging)
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

Brief overview.

## Decision Tree: Choose Your Approach
(Route the user's task to the right workflow)

## Core Process
(Step-by-step instructions — the main workflow)

## Anti-Patterns
(❌/✅ pairs documenting common mistakes and correct approaches)

## Reference Files
(Links to references/ and scripts/ — loaded on demand)
```

#### Key Writing Principles

**Decision trees over checklists** — Route to the right approach first.
See [📋 Writing Guide](references/writing-guide.md) for detailed patterns.

**Be "pushy" in descriptions** — Skills under-trigger. Explicitly list
trigger phrases and contexts.

**Explain the why** — Smart models work better with reasoning than rigid
rules. If you're writing ALWAYS/NEVER in caps, reframe with explanation.

**Anti-patterns with ❌/✅ pairs** — The ❌ catches what to watch for,
the ✅ shows what to do instead. Just saying "do Y" misses the cases
where the agent might accidentally do X.

**Progressive disclosure** — SKILL.md is the router (< 500 lines).
Details go in references/. Scripts run as black boxes.

### Phase 3: Validate

Run the validator to catch structural issues:
```bash
python scripts/validate_skill.py <skill-dir>
```

It checks: frontmatter format, name rules, description quality,
body length, decision trees, broken references, and ALL-CAPS rules.

### Phase 4: Test Triggers

Run trigger tests to measure if the description activates correctly:
```bash
python scripts/test_trigger.py <skill-dir> --verbose
```

This uses the LLM skill router with all installed skills present
(realistic routing scenario). It auto-generates test queries from
the description, or you can provide a custom eval set:

```bash
python scripts/test_trigger.py <skill-dir> --eval-set evals/trigger_evals.json -v
```

Create custom eval sets for thorough testing. Format:
```json
[
    {"query": "create a skill for X", "should_trigger": true},
    {"query": "deploy to kubernetes", "should_trigger": false}
]
```

### Phase 5: Optimize Description (if needed)

If trigger tests aren't perfect, run automated optimization:
```bash
python scripts/improve_description.py <skill-dir> --iterations 3 --verbose
```

This loop:
1. Tests current description → identifies failures
2. Calls LLM to generate improved description based on failures
3. Re-tests → compares scores
4. Keeps the best-scoring version
5. Writes it back to SKILL.md

### Phase 6: Register

```python
# Option A: Use manage_skills tool
manage_skills(action="create", name="my-skill", description="...", instructions="...")

# Option B: Create files directly (more control for complex skills)
run_shell("mkdir -p skills/my-skill/{references,scripts,assets,evals}")
write_file("skills/my-skill/SKILL.md", content)
write_file("skills/my-skill/references/...", content)
manage_skills(action="reload")
```

### Phase 7: Package (optional)

```bash
python scripts/package_skill.py <skill-dir> [output-dir]
```

Creates a `.skill` file (ZIP archive) after validation. Excludes
`evals/`, `__pycache__/`, `.DS_Store`.

## Anti-Patterns

❌ **Don't** write a skill and assume it works
✅ **Do** run `test_trigger.py` to verify the description triggers correctly

❌ **Don't** write passive descriptions ("This skill handles dashboards")
✅ **Do** write pushy descriptions ("Use this skill whenever the user mentions
   dashboards, charts, or data visualization")

❌ **Don't** use flat checklists (1. Check X  2. Check Y  3. Check Z)
✅ **Do** use decision trees that route to the right approach

❌ **Don't** write ALWAYS/NEVER rules without explaining why
✅ **Do** explain the reasoning — models work better with understanding

❌ **Don't** put all details in SKILL.md (it bloats context for every query)
✅ **Do** use progressive disclosure — SKILL.md routes, references/ details

❌ **Don't** keep instructions that aren't being followed
✅ **Do** remove noise — if the model ignores a rule, it's not helping

## Reference Files

- [📋 Spec Summary](references/spec-summary.md) — Agent Skills specification
- [📝 Writing Guide](references/writing-guide.md) — Detailed writing patterns, quality checklist
- [📊 Schemas](references/schemas.md) — JSON formats for evals, results, grading
- [🎯 Grader Agent](agents/grader.md) — Instructions for evaluating skill output quality

## Scripts

Run with `--help` for usage. Don't read source — treat as black boxes.

- `scripts/validate_skill.py` — Validate SKILL.md format and quality
- `scripts/test_trigger.py` — Test if description triggers correctly
- `scripts/improve_description.py` — Automated description optimization loop
- `scripts/package_skill.py` — Package skill into distributable .skill file

## Name Rules

- Lowercase letters, numbers, hyphens only
- 1-64 characters, no `--`, no leading/trailing hyphens
- Must match directory name
