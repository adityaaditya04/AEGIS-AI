History rewrite: models/ removed

What changed
- The repository history was rewritten to remove the `models/` directory and large model artifacts.
- A forced push was performed to update `main` with the cleaned history.

Action required (collaborators)
- Recommended: reclone the repository to avoid history conflicts:

  git clone https://github.com/adityaaditya04/AEGIS-AI.git

- Alternative (advanced): reset an existing clone (this will discard local changes):

  git fetch origin
  git checkout main
  git reset --hard origin/main
  git clean -fdx

Why this was done
- Large model files were accidentally committed; removing them reduces repository size and prevents future leaks.

LFS objects and storage
- This rewrite removes tracked files from history, but Git LFS objects may still exist in GitHub storage. To fully purge LFS objects you can either:
  - Use `git lfs migrate` locally and force-push a new history (destructive), then contact GitHub Support to remove orphaned LFS objects; OR
  - Ask me to coordinate the LFS purge and I will prepare the mirror+filter steps and request removal.

If you need me to proceed with LFS object purge, reply with "purge-lfs" and I will start the process and list required permissions.
