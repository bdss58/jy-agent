# Skill Writing Guide

Detailed guide for writing high-quality skills. Distilled from Anthropic's production
patterns and iterative testing.

## The Core Insight

> "We're trying to create skills that can be used a million times across many different
> prompts. Rather than put in fiddly overfitty changes, or oppressively constrictive
> MUSTs, if there's some stubborn issue, try branching out and using different metaphors."
>
> — Anthropic skill-creator

## Description Writing

### Be Pushy (Skills Under-Trigger)

The #1 problem: skills don't activate when they should. Be assertive:

```yaml
# ❌ Too passive — LLM won't know when to use it
description: Guide for building dashboards

# ✅ Pushy — explicit about when to activate
description: >-
  Build data dashboards and visualizations. Use this skill whenever the user
  mentions dashboards, data viz, charts, graphs, metrics display, or wants
  to visualize any kind of data, even if they don't explicitly say "dashboard".
  TRIGGER on: "show me a chart", "visualize this data", "dashboard", "metrics".
  DO NOT TRIGGER on: static reports, CSV exports without visualization.
```

### Description Structure

1. **What** it does (1 sentence)
2. **When** to use it — specific trigger phrases and contexts
3. **When NOT** to use it — prevent false triggers
4. Keep under 200 words / 1024 characters

### Optimization

Run `scripts/test_trigger.py` to measure trigger accuracy. If imperfect,
run `scripts/improve_description.py` for automated optimization.

## Instruction Writing

### Decision Trees Over Checklists

```markdown
# ❌ Flat (doesn't guide decisions)
1. Check the input
2. Process the data
3. Generate output

# ✅ Decision tree (routes to right approach)
What's the input type?
├─ Structured data (CSV, JSON) → Use pandas workflow
├─ Unstructured text → Use NLP extraction
├─ Mixed → Split and process separately
└─ Unknown format → Ask the user
```

### Anti-Patterns with Pairs

Always show what NOT to do alongside what TO do:

```markdown
❌ **Don't** hardcode API endpoints — they change between environments
✅ **Do** use environment variables or config files for endpoints
```

Why pairs? The ❌ catches cases the agent might accidentally fall into.
Just saying "use env vars" misses the temptation to hardcode.

### Explain the Why

> "Try hard to explain the **why** behind everything. If you find yourself
> writing ALWAYS or NEVER in all caps, that's a yellow flag — reframe and
> explain the reasoning."

```markdown
# ❌ Rigid rule (model doesn't understand why)
ALWAYS validate inputs. NEVER skip validation.

# ✅ Explained reasoning (model can generalize)
Validate inputs before processing because malformed data causes silent
failures downstream — the agent will produce wrong results without
knowing it, and the user won't catch it until much later.
```

### Progressive Disclosure

| Level | What | Budget | When loaded |
|-------|------|--------|-------------|
| Metadata | name + description | ~100 tokens | Always (system prompt) |
| Instructions | SKILL.md body | < 5000 tokens | When skill activates |
| Resources | references/, scripts/ | Unlimited | On demand via read_file |

**Key insight**: SKILL.md is the **router**. It should help the agent decide
what to do and point to the right reference file. Details live in references/.

### Keep It Lean

After each iteration, review the skill and remove instructions that:
- Aren't being followed (model ignores them → they're noise)
- Are obvious to the model (smart models don't need "be thorough")
- Duplicate each other
- Only apply to one edge case (move to references/)

## Scripts as Black Boxes

```markdown
# ❌ Don't explain script internals in SKILL.md
The validate_skill.py script parses YAML frontmatter using regex,
extracts the name and description fields...

# ✅ Treat scripts as tools — describe inputs/outputs
Run `python scripts/validate_skill.py <skill-dir>` to check your skill.
It validates frontmatter, name format, description quality, and body structure.
Returns ✅ on success or lists errors/warnings.
```

Scripts should:
- Be self-contained (minimal dependencies)
- Support `--help`
- Have clear error messages
- Be deterministic when possible

## Communication Awareness

Skill users range from experienced developers to "plumbers opening their terminals."
Pay attention to context cues:

- **Safe terms**: "test", "check", "improve", "create"
- **Borderline**: "evaluation", "benchmark" — OK but consider explaining
- **Needs context**: "JSON", "assertion", "frontmatter" — explain briefly if unsure

When in doubt, add a short definition: "...run the evals (test cases that check
if the skill works correctly)..."

## Quality Checklist

### Must Have
- [ ] name matches directory name
- [ ] description has TRIGGER and DO NOT TRIGGER guidance
- [ ] Description is "pushy" enough
- [ ] Body has routing logic (decision tree or clear branching)
- [ ] Instructions explain WHY, not just WHAT

### Should Have
- [ ] Anti-patterns documented (❌/✅ pairs)
- [ ] Reference files for detailed content
- [ ] Body under 500 lines
- [ ] Trigger test passing (`scripts/test_trigger.py`)

### Nice to Have
- [ ] Scripts for deterministic tasks
- [ ] Example inputs and outputs
- [ ] Environment-specific sections (if applicable)
- [ ] Communication awareness for non-technical users
