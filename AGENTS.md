## Agent skills

### Issue tracker

Issues and specs live as markdown files under `.scratch/<feature>/`. See `docs/agents/issue-tracker.md`.

### Triage labels

Canonical triage roles use the default labels in this repo's issue tracker. See `docs/agents/triage-labels.md`.

### Domain docs

Domain documentation uses the single-context layout. See `docs/agents/domain.md`.

## Remote Development and Testing

The Hailo Apps framework is available on the `home` remote host under `~/hailo-apps/`, but is not assumed to be installed locally. All project code execution must happen on `home`; do not run the project, tests, linters, builds, or Python commands locally.

Before running code, ensure `~/hailo-apps/` on `home` is initialized. If it is missing or not a usable checkout, pull the Hailo Apps project from its configured upstream and set it up following its instructions for the Hailo Apps virtual environment. Use that environment for project execution and tests.

Run only committed changes from the active feature branch. Commit and push the branch changes so they are available remotely, then on `home` fetch and check out/update that same branch in the adapter's remote checkout before testing. Do not test uncommitted local-only changes or copy source files around the remote checkout as a substitute for syncing the branch.

When reporting results, include the feature branch, tested commit, commands run on `home`, and their outcomes. If the remote host, checkout, or environment is unavailable, report that tests were not run rather than running them locally.