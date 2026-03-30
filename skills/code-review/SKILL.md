---
name: code-review
description: >-
  Review code for quality, bugs, security issues, and best practices. Use when 
  asked to review a file, PR, codebase, refactor suggestions, or find bugs. 
  Covers Python, JavaScript, TypeScript, and general software engineering principles.
metadata:
  author: agent-builtin
  version: "1.0"
---

## Instructions

When reviewing code:

### 1. Understand Context First
- Read the file(s) being reviewed with `read_file`
- Check surrounding files for imports and dependencies: `list_directory`
- Understand the project structure before making judgments

### 2. Review Checklist

#### Correctness
- [ ] Does the code do what it's supposed to do?
- [ ] Are edge cases handled (empty input, None, boundary values)?
- [ ] Are error conditions handled properly?
- [ ] Are there off-by-one errors?
- [ ] Are return values correct in all code paths?

#### Security
- [ ] SQL injection risks (string formatting in queries)?
- [ ] Path traversal (user input in file paths)?
- [ ] Command injection (user input in shell commands)?
- [ ] Sensitive data exposure (logging passwords, API keys)?
- [ ] Input validation and sanitization?

#### Performance
- [ ] O(n²) or worse algorithms where O(n) is possible?
- [ ] Unnecessary database queries in loops?
- [ ] Memory leaks (unclosed resources, growing collections)?
- [ ] Expensive operations that could be cached?

#### Style & Maintainability
- [ ] Clear variable and function names?
- [ ] Functions doing one thing (Single Responsibility)?
- [ ] DRY — duplicated code that should be extracted?
- [ ] Consistent error handling patterns?
- [ ] Appropriate comments (why, not what)?

#### Python-Specific
- [ ] Using `with` for resource management?
- [ ] Proper exception types (not bare `except:`)?
- [ ] Type hints on function signatures?
- [ ] f-strings vs format() vs % formatting consistency?

#### JavaScript/TypeScript-Specific
- [ ] Proper async/await error handling?
- [ ] Memory leaks from event listeners?
- [ ] Proper null/undefined checks?
- [ ] TypeScript types actually constraining behavior?

### 3. Output Format
Structure your review as:
```
## Summary
One-paragraph overview of the code quality.

## Issues Found
### 🔴 Critical (must fix)
- Issue description with file:line reference

### 🟡 Warning (should fix)
- Issue description with file:line reference

### 🔵 Suggestion (nice to have)
- Suggestion with rationale

## Positive Notes
- What the code does well
```

### 4. Principles
- **Be specific**: Reference exact lines and provide fix examples
- **Be constructive**: Explain why something is an issue, not just that it is
- **Prioritize**: Critical bugs > security > performance > style
- **Only claim what you've read**: Don't invent issues in files you haven't examined
