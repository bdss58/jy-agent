# Git Recovery Recipes (a.k.a "Oh Shit, Git!")

Inspired by [ohshitgit.com](https://ohshitgit.com/) — real-world recovery recipes for when things go wrong.

## 🔮 The Magic Time Machine: reflog

`git reflog` is your nuclear option. It records **every** HEAD movement, across all branches.

```bash
git reflog
# Shows a list of everything you've done:
# abc1234 HEAD@{0}: commit: feat: add search
# def5678 HEAD@{1}: rebase: checkout origin/main
# ghi9012 HEAD@{2}: commit: the state before you broke everything

# Go back to any point:
git reset --hard HEAD@{2}
# You're back to before it broke!
```

**⚠️ reflog entries expire** (default: 90 days for reachable, 30 days for unreachable). Don't wait too long.

---

## Recipe: "I committed to the wrong branch!"

```bash
# Option A: Move last commit to correct branch
git reset --soft HEAD~1          # undo commit, keep changes staged
git stash                        # stash the changes
git checkout correct-branch
git stash pop                    # apply changes
git add . && git commit -m "..."

# Option B: Cherry-pick to correct branch
git checkout correct-branch
git cherry-pick main             # grab the commit
git checkout main
git reset --hard HEAD~1          # remove from wrong branch
```

## Recipe: "I committed to main instead of a new branch!"

```bash
# Create branch from current state (preserves your commit)
git branch feature/new-thing

# Reset main back to before your commit
git reset --hard HEAD~1

# Switch to your new branch
git checkout feature/new-thing
# Your commit is safely on the new branch!
```

## Recipe: "I need to undo the last commit"

```bash
# Keep the changes (unstaged)
git reset HEAD~1

# Keep the changes (staged)
git reset --soft HEAD~1

# Nuke the changes completely (⚠️ DESTRUCTIVE)
git reset --hard HEAD~1
```

## Recipe: "I need to undo a commit from 5 commits ago"

```bash
# Find the commit hash
git log --oneline

# Create a NEW commit that undoes that specific commit
git revert <hash>
# This is safe — it doesn't rewrite history
```

## Recipe: "I amended a commit but I shouldn't have!"

```bash
# The original commit is in reflog
git reflog
# Find the entry BEFORE the amend
git reset --hard HEAD@{1}
```

## Recipe: "I force-pushed and lost commits on remote!"

```bash
# If you still have the commits locally:
git reflog
# Find the commit before force-push
git push origin <hash>:refs/heads/branch-name

# If someone else has the commits:
# Ask them to push their copy

# If the commits are truly gone from all local copies:
# 🪦 They're gone. This is why --force-with-lease exists.
```

## Recipe: "I deleted a branch I need!"

```bash
# Find the branch tip in reflog
git reflog | grep "branch-name"
# OR
git reflog --all | grep "branch-name"

# Recreate the branch at that commit
git branch branch-name <hash>
```

## Recipe: "I need to completely undo a merge"

```bash
# If not yet pushed:
git reset --hard HEAD~1   # or use reflog

# If already pushed (safe way):
git revert -m 1 <merge-commit-hash>
# -m 1 means "keep the mainline parent"
```

## Recipe: "I have changes I don't want to commit but don't want to lose"

```bash
# Stash with a description
git stash push -m "WIP: half-finished search feature"

# List stashes
git stash list

# Apply most recent stash (keep it in stash list)
git stash apply

# Apply and remove most recent stash
git stash pop

# Apply a specific stash
git stash apply stash@{2}

# See what's in a stash
git stash show -p stash@{0}
```

## Recipe: "I need to split a commit into two"

```bash
# Interactive rebase to the commit BEFORE the one to split
git rebase -i HEAD~3
# Change 'pick' to 'edit' for the commit to split

# When rebase stops at that commit:
git reset HEAD~1         # undo the commit, keep changes
git add file1.py         # stage first part
git commit -m "feat: first logical change"
git add file2.py         # stage second part
git commit -m "refactor: second logical change"
git rebase --continue
```

## Recipe: "I want to find which commit broke something"

```bash
git bisect start
git bisect bad                 # current commit is broken
git bisect good v1.0           # this tag/commit was working

# Git checks out middle commit. Test it, then:
git bisect good   # if this commit works
git bisect bad    # if this commit is broken

# Git binary-searches until it finds THE commit
# Automated version:
git bisect run make test       # auto-run tests at each step

git bisect reset               # when done, return to original HEAD
```

## Recipe: "Everything is FUBAR, I want to match remote exactly"

```bash
# ⚠️ DESTRUCTIVE — loses all local changes
git fetch origin
git checkout main
git reset --hard origin/main
git clean -fd                  # remove untracked files and directories

# For ALL branches:
# Repeat checkout/reset/clean for each branch
```

---

## 🚨 Dangerous Commands Reference

Commands that can cause permanent data loss. **Always warn the user.**

| Command | Danger Level | What it does | Safer alternative |
|---------|-------------|--------------|-------------------|
| `git reset --hard` | 🔴 HIGH | Destroys uncommitted changes | `git stash` first, or `git reset --soft` |
| `git push --force` | 🔴 HIGH | Overwrites remote history | `git push --force-with-lease` |
| `git clean -fd` | 🔴 HIGH | Deletes untracked files permanently | `git clean -fdn` (dry run first!) |
| `git checkout -- <file>` | 🟡 MEDIUM | Discards uncommitted changes to file | `git stash` the file first |
| `git rebase` (on shared branch) | 🟡 MEDIUM | Rewrites shared history | `git merge` instead |
| `git branch -D` | 🟡 MEDIUM | Force-deletes branch | Verify with `git log branch-name` first |
| `git filter-branch` | 🔴 HIGH | Rewrites entire history | `git-filter-repo` (faster, safer) |

### Before any dangerous operation

```bash
# 1. Create a backup branch
git branch backup-$(date +%Y%m%d-%H%M%S)

# 2. For remote operations, fetch first
git fetch origin

# 3. For clean operations, dry run first
git clean -fdn   # -n = dry run, shows what WOULD be deleted

# 4. For reset, know where you are
git log --oneline -5
git stash         # save any uncommitted work
```

### The safety net hierarchy

```
Lost uncommitted changes?
├─ Were they staged (git add)?
│   → Maybe recoverable: git fsck --lost-found
│   → Check .git/lost-found/
│
├─ Were they stashed?
│   → git stash list
│
├─ Were they committed (even if branch was deleted)?
│   → git reflog (recoverable for ~30 days)
│
└─ Never staged, never committed?
    → 🪦 Gone forever. This is why you commit often.
```
