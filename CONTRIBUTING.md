# Contributing to cneuromod.train

This repository follows a disciplined engineering workflow even during early development, including for self-merged pull requests.

## Development workflow

Every meaningful unit of work follows:

```text
issue → branch → implementation → tests → PR → CI → merge
```

1. Define or reference a GitHub issue before substantive work.
2. Create a focused branch from `main` (e.g. `feat/…`, `fix/…`, `docs/…`).
3. Implement the smallest coherent change.
4. Add or update tests for behavior changes.
5. Run local validation (see below).
6. Open a pull request using the PR template; document the problem, design decisions, tests, and limitations.
7. Merge only when CI is green.

Large changes should begin with an issue so design and ownership are clear before implementation.

## Git rules

- `main` is a stable integration branch, not a development branch.
- No substantive direct commits to `main`.
- Self-merged PRs are acceptable during early development, but they must remain reviewable engineering records: focused purpose, complete PR description, green CI.
- Prefer small, coherent commits with descriptive messages.
- Do not mix unrelated refactoring with feature work.
- Never force-push `main` or rewrite shared history.
- Never merge with failing tests.
- Inspect `git diff` before committing and the full diff against `main` before opening a PR.

## Never commit

- credentials, tokens, or secrets;
- private datasets or research data;
- model checkpoints or generated artifacts;
- notebook outputs;
- cluster logs or local environment files.

## Local development

Requires Python 3.11+.

```bash
pip install -e ".[dev]"
```

Run the same checks CI enforces:

```bash
ruff check .
ruff format --check .
mypy
pytest
```

## Code standards

- Code is linted and formatted with [ruff](https://docs.astral.sh/ruff/); configuration lives in `pyproject.toml`.
- Type checking is [mypy](https://mypy.readthedocs.io/) in strict mode; new code must be fully typed.
- Core behavior must be tested with pytest.
- Configuration should define experiments; avoid ad-hoc scripts for reproducible work.
- Prefer explicit interfaces over hidden coupling; isolate unstable upstream dependencies behind adapters.
- Avoid premature abstraction: do not generalize a component until real experiments demonstrate the need.

## AI-assisted development

AI coding tools may be used, but contributors remain responsible for architecture, scientific correctness, understanding generated code, reviewing diffs, tests, dependency decisions, security, and reproducibility. Generated code must remain understandable and maintainable by humans.
