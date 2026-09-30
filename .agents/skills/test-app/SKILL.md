---
name: test-app
description: Install, statically check, test, run, and validate the Agentic RAG demo before a PR.
---

# Test the Agentic RAG application

Run every step from the repository root. Stop and fix failures before continuing.

## 1. Install

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
pre-commit install
```

Reuse an existing `.venv` when it is already healthy, but rerun the two install commands after
dependency changes.

## 2. Static checks and automated tests

```bash
source .venv/bin/activate
python -m ruff format --check .
python -m ruff check .
python -m mypy
python -m pytest
```

All checks must pass. Tests must not make live OpenAI calls.

## 3. Run the application

```bash
source .venv/bin/activate
test -n "$OPENAI_API_KEY"
streamlit run app.py --server.headless true
```

Use the URL printed by Streamlit. Do not put the API key in a file.

## 4. Validate the UI

For a UI-changing PR, record the complete validation:

1. Open the Streamlit application.
2. Upload one supported document.
3. Confirm ingestion succeeds and reports a positive chunk count.
4. Ask a question whose answer is present in the document.
5. Confirm the answer includes source citations.
6. Confirm the interface displays request latency.
7. Ask an unrelated question and confirm the app handles missing evidence clearly.
8. Save the recording and attach it to the PR.

For a non-UI PR, application startup is optional when `app.py` is unaffected.
