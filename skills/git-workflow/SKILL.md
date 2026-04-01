---
name: git-workflow
description: >-
  Systematic git operations: branching, committing, conflict resolution, recovery, and
  workflow selection. TRIGGER when: user asks about git commands, branching strategy,
  merge conflicts, commit messages, git history rewriting, recovering lost work, PR
  workflows, conventional commits, rebase vs merge, reflog, cherry-pick, stash, bisect,
  gitflow, trunk-based development, or any git trouble. Also trigger when helping user
  commit code, create branches, or prepare PRs.
  DO NOT TRIGGER when: user is just editing code (not git-related), asking about
  GitHub/GitLab UI features unrelated to git itself, or general DevOps (use devops skill).
---

# Git Workflow

## Decision Tree: Choose Your Approach

```
What does the user need?
│
├─ Branching / workflow strategy
│   → See "Workflow Selection" below
│   → Read references/branching-strategies.md for details
│
├─ Write a commit message
│   → See "Conventional Commits" below
│
├─ Resolve a conflict
│   ├─ Merge conflict → Read references/conflict-resolution.md
│   ├─ Rebase conflict → Read references/conflict-resolution.md
│   └─ Diverged branches → Determine: rebase or merge? (see decision criteria)
│
├─ Recover from a mistake ("oh shit" moment)
│   → Read references/recovery-recipes.md
│   → reflog is your time machine
│
├─ Rewrite history (squash, amend, rebase -i)
│   → See "History Rewriting" below
│   → ⚠️ NEVER on pushed/shared branches
│
├─ PR / code review workflow
│   → See "PR Workflow" below
│
└─ Dangerous operation (force push, reset --hard, clean -fd)
    → Read references/recovery-recipes.md "Dangerous Commands" section
    → ALWAYS warn the user, suggest safer alternatives
```

## Workflow Selection

Choose based on team context:

| Signal | → Use |
|--------|-------|
| Solo project or small team (1-5), deploying continuously | **Trunk-based** (short-lived branches, merge to main daily) |
| Team > 5, multiple features in parallel, weekly+ releases | **GitHub Flow** (feature branches + PRs to main) |
| Versioned releases, mobile apps, libraries, LTS support | **GitFlow** (develop/release/hotfix branches) |
| Open source project with external contributors | **Fork & PR** model |

For most modern teams: **trunk-based with short-lived branches** is the default recommendation. See `references/branching-strategies.md` for detailed comparison.

## Conventional Commits

Every commit message follows this format:

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

### Types (memorize these)

| Type | When |
|------|------|
| `feat` | New feature (→ MINOR version bump) |
| `fix` | Bug fix (→ PATCH version bump) |
| `docs` | Documentation only |
| `style` | Formatting, no code change |
| `refactor` | Neither fix nor feature |
| `perf` | Performance improvement |
| `test` | Adding/fixing tests |
| `build` | Build system, dependencies |
| `ci` | CI configuration |
| `chore` | Maintenance tasks |

### Breaking changes

```
feat(api)!: change authentication to OAuth2

BREAKING CHANGE: API key authentication is removed.
All clients must migrate to OAuth2 tokens.
```

### Good vs bad examples

```
❌ git commit -m "fix"
❌ git commit -m "update code"
❌ git commit -m "WIP"
❌ git commit -m "fixed the thing that was broken in the login page when user clicks submit"

✅ git commit -m "fix(auth): prevent session timeout during OAuth refresh"
✅ git commit -m "feat(search): add fuzzy matching for product names"
✅ git commit -m "refactor: extract database connection pool to shared module"
```

## History Rewriting

### Safe operations (local only, not yet pushed)

```bash
# Amend last commit (message or content)
git add .
git commit --amend --no-edit    # keep message
git commit --amend              # edit message

# Squash last N commits interactively
git rebase -i HEAD~N
# In editor: change 'pick' to 'squash' (or 's') for commits to merge

# Reorder commits
git rebase -i HEAD~N
# In editor: reorder the lines
```

### ⚠️ DANGER ZONE — History rewriting rules

```
Has this been pushed to a shared branch?
├─ YES → ❌ DO NOT rebase/amend/force-push
│        → Use `git revert` instead (creates new commit undoing the change)
│
└─ NO  → ✅ Safe to rebase/amend/squash
         → After rebase, use `git push --force-with-lease` (NOT --force)
```

**Why `--force-with-lease`?** It refuses to push if someone else pushed to the remote since your last fetch. `--force` blindly overwrites.

## PR Workflow

### Before creating a PR

```bash
# Update your branch with latest main
git fetch origin
git rebase origin/main           # preferred: linear history
# OR
git merge origin/main            # if rebase causes too many conflicts

# Run tests locally
make test  # or whatever the project uses

# Check what you're about to push
git log origin/main..HEAD --oneline
git diff origin/main --stat
```

### PR best practices

1. **Keep PRs small** — under 400 lines changed. Split large changes.
2. **One concern per PR** — don't mix refactoring with features.
3. **Descriptive title** — use conventional commit format: `feat(auth): add SSO support`
4. **Link issues** — reference related issues in the description.
5. **Self-review first** — review your own diff before requesting reviews.

### Review turnaround

- Small PR (< 200 lines): review within 4 hours
- Medium PR (200-400 lines): review within 1 day
- Large PR (400+ lines): consider splitting; review within 2 days

## Rebase vs Merge Decision

```
Should I rebase or merge?
│
├─ Updating feature branch with latest main?
│   → REBASE (keeps history clean): git rebase origin/main
│
├─ Merging feature branch INTO main?
│   → MERGE (preserves branch history): git merge --no-ff feature/x
│   → Or SQUASH MERGE for single logical commit: git merge --squash feature/x
│
├─ Branch has been pushed and others are working on it?
│   → MERGE (don't rewrite shared history)
│
└─ Personal branch, want clean history?
    → REBASE freely
```

## Quick Reference: Common Operations

```bash
# Undo last commit but keep changes
git reset --soft HEAD~1

# Unstage a file
git restore --staged <file>

# Discard local changes to a file
git restore <file>

# See what changed (staged vs unstaged)
git diff              # unstaged changes
git diff --staged     # staged changes

# Stash work temporarily
git stash push -m "WIP: implementing search"
git stash list
git stash pop         # apply and remove latest stash

# Find which commit introduced a bug
git bisect start
git bisect bad        # current commit is bad
git bisect good <hash>  # this commit was good
# git will binary search; mark each as good/bad
git bisect reset      # when done

# Show file at specific commit
git show <hash>:<path/to/file>

# Find who changed a line
git blame <file>
git blame -L 10,20 <file>  # lines 10-20 only
```

## Anti-Patterns

❌ Committing directly to main without review (except solo projects)
✅ Use short-lived feature branches with PR review

❌ Giant commits with "update everything"
✅ Small, focused commits with conventional messages

❌ `git push --force` on shared branches
✅ `git push --force-with-lease` on personal branches only

❌ Resolving conflicts by accepting "mine" or "theirs" blindly
✅ Read and understand both sides; test after resolution

❌ Long-lived feature branches (weeks/months)
✅ Merge/rebase from main daily; keep branches < 2 days

❌ Ignoring `.gitignore` — committing secrets, build artifacts, node_modules
✅ Set up `.gitignore` first; use `git-secrets` or `trufflehog` to scan
