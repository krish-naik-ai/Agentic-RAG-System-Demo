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

## Ingest documents programmatically

```python
from pathlib import Path

from agentic_rag.embeddings import OpenAIEmbeddingProvider
from agentic_rag.ingestion import DocumentIngestor
from agentic_rag.vector_store import ChromaVectorStore

store = ChromaVectorStore(Path("data/chroma"))
ingestor = DocumentIngestor(
    embeddings=OpenAIEmbeddingProvider(),
    vector_store=store,
)
result = ingestor.ingest(Path("example.pdf"))
print(f"Stored {result.chunk_count} chunks")
```

## Ask cited questions programmatically

```python
from agentic_rag.agent import RetrievalAgent
from agentic_rag.llm import OpenAILanguageModel

agent = RetrievalAgent(
    language_model=OpenAILanguageModel(),
    embeddings=OpenAIEmbeddingProvider(),
    vector_store=store,
)
response = agent.answer("What are the document's main recommendations?")
print(response.answer)
```

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
