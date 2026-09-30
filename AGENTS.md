# Contributor guide

## Project structure

- `src/agentic_rag/` contains application and domain logic.
- `tests/` mirrors the source modules and contains offline tests.
- `scripts/` contains evaluation and maintenance entrypoints.
- `app.py` is the Streamlit UI entrypoint.
- `data/` is local runtime storage and must not contain committed user data.
- `.agents/skills/` contains repeatable repository workflows.

## Conventions

- Support Python 3.10 and newer.
- Add type hints to public and internal functions.
- Keep model, embedding, and vector-store boundaries injectable so tests remain offline.
- Keep modules focused; separate parsing, chunking, storage, retrieval, generation, and UI code.
- Require citations for answers that use retrieved context.
- Never commit API keys, uploaded documents, local vector databases, or `.env` files.
- Add or update tests for every core-logic change.

## Required checks

Run these commands before every PR:

```bash
python -m ruff format --check .
python -m ruff check .
python -m mypy
python -m pytest
```

Follow `.agents/skills/test-app/SKILL.md` for installation, application startup, and UI testing.
PRs that touch the UI require a recorded browser test.
