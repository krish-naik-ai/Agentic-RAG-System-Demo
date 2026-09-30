# Agentic RAG System Demo

An end-to-end retrieval-augmented generation application with document ingestion, local Chroma
storage, an agent that plans retrieval, cited answers, comparative evaluation, and a Streamlit UI.

## Requirements

- Python 3.10+
- `OPENAI_API_KEY` in the process environment for live embeddings and answers

## Install

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
pre-commit install
```

## Verify

```bash
python -m ruff format --check .
python -m ruff check .
python -m mypy
python -m pytest
```

## Run the completed application

```bash
streamlit run app.py
```

The Streamlit entrypoint is added in the UI slice. The application stores local runtime data under
`data/`; that directory is ignored by Git.

## Project layout

```text
.
├── app.py
├── scripts/
├── src/agentic_rag/
├── tests/
└── data/
```

See `AGENTS.md` for repository conventions and `.agents/skills/test-app/SKILL.md` for the complete
pre-PR validation workflow.
