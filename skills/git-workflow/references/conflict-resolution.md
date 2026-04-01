# Conflict Resolution Guide

## Understanding Conflict Markers

When Git can't auto-merge, it inserts markers:

```
<<<<<<< HEAD (or "ours")
your version of the code
=======
their version of the code
>>>>>>> feature/their-branch (or "theirs")
```

**Rules:**
1. **NEVER** blindly accept "ours" or "theirs" without reading both
2. **ALWAYS** test after resolving conflicts
3. **ALWAYS** understand the intent of both changes

## Merge Conflicts

### Step-by-step resolution

```bash
# 1. Start the merge
git merge feature/other-branch
# CONFLICT! Git tells you which files conflict

# 2. See which files have conflicts
git status
# "both modified" files need manual resolution

# 3. Open each conflicted file, resolve markers
# Use your editor's merge tool, or manually edit

# 4. After resolving all markers in a file
git add <resolved-file>

# 5. After all files are resolved
git commit  # Git auto-generates merge commit message

# ⚡ If you want to abort and start over
git merge --abort
```

### Resolution strategies (per file)

```
What kind of conflict?
│
├─ Both sides changed different parts of same function
│   → Combine both changes manually
│   → Test: does the function still work with both changes?
│
├─ Both sides changed the same lines
│   → Understand WHY each change was made
│   → Write a version that satisfies both intents
│   → If unsure, ask the other author
│
├─ One side deleted a file, other modified it
│   → Was the deletion intentional?
│   ├─ Yes → Accept deletion, verify modifications aren't needed
│   └─ No  → Keep the file with modifications
│
├─ Both sides added different content at same location
│   → Usually keep both additions (in appropriate order)
│
└─ Dependency/import conflicts (package.json, go.mod, etc.)
    → Usually need both additions
    → Run the package manager after: npm install, go mod tidy, etc.
```

## Rebase Conflicts

Rebase replays your commits one at a time onto the target. Each commit can conflict separately.

```bash
# 1. Start rebase
git rebase origin/main
# CONFLICT on commit X!

# 2. Resolve the conflict (same as merge)
# Edit files, remove markers

# 3. Stage resolved files
git add <resolved-file>

# 4. Continue to next commit
git rebase --continue
# May conflict again on the next commit

# ⚡ Abort and go back to pre-rebase state
git rebase --abort

# ⚡ Skip this particular commit (CAREFUL — drops the commit)
git rebase --skip
```

### Rebase conflict tips

- **Conflicts compound**: if commit 1 conflicted, commit 2-N may also conflict
- **If too many conflicts**: abort and use merge instead
- **Pre-rebase safety**: `git branch backup-branch` before starting

## Using Merge Tools

### VS Code (built-in)
```bash
git config --global merge.tool vscode
git config --global mergetool.vscode.cmd 'code --wait --merge $REMOTE $LOCAL $BASE $MERGED'
# Then:
git mergetool
```

### IntelliJ / other IDEs
Most IDEs detect conflicts and offer 3-way merge UI automatically.

### Command-line: vimdiff
```bash
git config --global merge.tool vimdiff
git mergetool
# Shows 3 panes: LOCAL | BASE | REMOTE
# Edit the bottom pane (MERGED)
```

## Advanced: Rerere (Reuse Recorded Resolution)

If you keep hitting the same conflict (e.g., rebasing repeatedly):

```bash
# Enable rerere globally
git config --global rerere.enabled true

# Git remembers how you resolved a conflict
# Next time the same conflict appears, it auto-resolves!

# See recorded resolutions
git rerere status
git rerere diff
```

## Preventing Conflicts

1. **Keep branches short-lived** — merge within 1-2 days
2. **Pull/rebase from main daily** — don't let your branch drift
3. **Communicate** — if two people touch the same file, coordinate
4. **Small commits** — easier to resolve conflicts at the commit level
5. **Use `.editorconfig`** — prevent whitespace/formatting conflicts
6. **Lock files** — for binary files that can't be merged (Git LFS)

## Common Conflict Scenarios & Solutions

### Scenario: "I rebased and now everything is messed up"
```bash
# Undo the rebase entirely
git reflog
# Find the entry BEFORE the rebase
git reset --hard HEAD@{N}
```

### Scenario: "I merged the wrong branch"
```bash
# If not yet committed
git merge --abort

# If already committed but not pushed
git reset --hard HEAD~1

# If already pushed
git revert -m 1 HEAD  # creates a new commit undoing the merge
```

### Scenario: "Merge conflicts in package-lock.json / yarn.lock"
```bash
# Don't try to resolve lock file conflicts manually!
# Accept either version, then regenerate:
git checkout --theirs package-lock.json  # or --ours
npm install
git add package-lock.json
```

### Scenario: "Conflicts in auto-generated files"
```bash
# Accept one side, regenerate
git checkout --theirs <generated-file>
# Run the generator
make generate  # or whatever regenerates the file
git add <generated-file>
```
