---
name: create-skill
description: >-
  Create new Agent Skills following the open standard (agentskills.io). Use when 
  asked to create, write, or package a new skill, or when the user wants to add 
  domain expertise to the agent. Guides proper SKILL.md format and structure.
metadata:
  author: agent-builtin
  version: "1.0"
---

## Instructions

When creating a new Agent Skill:

### 1. Gather Requirements
Ask the user:
- What should the skill do? (→ becomes `description`)
- What's a good short name? (→ becomes `name`, lowercase-hyphenated)
- What tools does it need? (→ becomes `allowed-tools`)
- Any reference material to include? (→ goes in `references/`)

### 2. Name Rules
- Lowercase letters, numbers, and hyphens only
- 1-64 characters
- No consecutive hyphens (`--`)
- No starting/ending with hyphens
- Must match the directory name
- Examples: `data-analysis`, `api-testing`, `react-components`

### 3. SKILL.md Template

```markdown
---
name: my-skill-name
description: >-
  Clear description of what this skill does and when to use it.
  Include specific keywords that help the agent identify relevant tasks.
  Max 1024 characters.
metadata:
  author: "your-name"
  version: "1.0"
allowed-tools: tool1 tool2
---

## Instructions

Step-by-step instructions for the agent to follow.

### Step 1: ...
### Step 2: ...

## Examples

Show input/output examples.

## Edge Cases

Document known edge cases and how to handle them.
```

### 4. Progressive Disclosure
- Keep SKILL.md body **under 500 lines** (~5000 tokens)
- Move detailed reference material to `references/` directory
- Move templates to `assets/` directory
- Move executable code to `scripts/` directory

### 5. Quality Checklist
- [ ] `name` matches directory name
- [ ] `description` clearly states what AND when to use
- [ ] Instructions are step-by-step and actionable
- [ ] Tools the skill needs are listed in `allowed-tools`
- [ ] Examples are provided for common use cases
- [ ] Edge cases are documented

### 6. Creation Method
Use the `manage_skills` tool with action `create`:
```
manage_skills(action="create", name="my-skill", description="...", instructions="...")
```

Or create files directly:
```
1. mkdir -p skills/my-skill/{scripts,references,assets}
2. write_file skills/my-skill/SKILL.md with proper content
3. manage_skills(action="reload") to pick up the new skill
```

### 7. Reference: Full Spec
See `references/spec-summary.md` for the complete Agent Skills specification summary.
