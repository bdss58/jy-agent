# Prompt Templates for Dual Planning

Ready-to-use templates. Copy, fill in the brackets, send to both agents.

## Architecture Decision

```
I need to decide on the approach for: <description>

Current state: <what exists>
Requirements: <what needs to change>
Constraints: <budget, timeline, compatibility>

Give me a detailed plan with tradeoffs. Don't implement yet.
```

## Migration Strategy

```
We need to migrate from <old> to <new>.

Codebase: <size, language, framework>
Critical paths: <what must not break>
Timeline: <deadline>

Produce a phased migration plan. Identify the riskiest step.
Don't implement yet.
```

## Feature Implementation

```
New feature: <description>

Existing patterns to follow: @src/handlers/example.py
Tests required: unit + integration
Must not break: <existing functionality>

Plan the implementation. List every file to create or modify.
Don't implement yet.
```

## Performance Optimization

```
Performance problem: <description, metrics, SLA targets>

Profiling data: <what you've measured>
Architecture: <relevant infra — DB, cache, queue, etc.>
Constraint: must not change <API contracts, data format, etc.>

Where should we optimize? What's the expected impact of each change?
Don't implement yet.
```

## Bug Investigation

```
Bug: <description, repro steps, error message>

Suspected area: <files, modules>
Already tried: <what you've ruled out>

Analyze the root cause. Propose fixes with risk assessment.
Don't implement yet.
```

## Refactor

```
Refactor goal: <what's wrong with current code, what "better" looks like>

Files in scope: <list or glob>
Must preserve: <behavior, API contracts, test coverage>
Style guide: <relevant conventions>

Plan the refactor. Group changes into safe, atomic steps.
Don't implement yet.
```

## The Golden Rule for All Templates

Always end with "Don't implement yet." — this prevents agents from
making changes when you only want a plan. Both Claude Code and Codex
will respect this instruction.
