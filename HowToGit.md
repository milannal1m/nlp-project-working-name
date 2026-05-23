# Git How-To

## Clone the repository

```bash
git clone git@github.com:milannal1m/nlp-project-working-name.git
cd nlp-project-working-name
```


## Create a branch from main

```bash
git checkout main
git pull origin main
git checkout -b your-branch-name
```

The `-b` flag creates and switches to the new branch in one step. Branch names should be short and descriptive, e.g. `feature/evaluation-metrics` or `fix/dataset-loading`.

---

## Push to your branch

Stage and commit your changes, then push:

```bash
git add <file>          # stage a specific file
git add .               # stage all changes
git commit -m "your message"
git push -u origin your-branch-name
```

The `-u` flag sets the upstream so subsequent pushes only need `git push`.

---

## Quick reference

| Command | Description |
|---|---|
| `git status` | Show changed and staged files |
| `git log --oneline` | Compact commit history |
| `git diff` | Show unstaged changes |
| `git checkout main` | Switch back to main |
| `git branch` | List local branches |
