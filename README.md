# Agentic RAG System Demo

An end-to-end retrieval-augmented generation application with document ingestion, local Chroma
storage, an agent that plans retrieval, open-source cross-encoder reranking, cited answers,
comparative evaluation, and a Streamlit UI.

## Requirements

- Python 3.10+
- `OPENAI_API_KEY` in the process environment for live embeddings (always) and OpenAI answers
- Optional: `LLM_PROVIDER=gemini` and `GEMINI_API_KEY` to generate answers with Gemini while
  keeping OpenAI embeddings. `LLM_MODEL` selects `gemini-2.5-flash` (default) or
  `gemini-3.5-flash`.

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

## Run the application

```bash
streamlit run app.py
```

Upload a PDF, DOCX, Markdown, or text document from the sidebar, ingest it, and ask questions in the
chat. Each answer reports latency and whether retrieval was used; source-backed responses expose
their citation details. Uploads can be up to 10 MB per file. The application stores uploads and
local Chroma data under ignored `data/`.

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
from agentic_rag.reranking import FlashRankReranker

agent = RetrievalAgent(
    language_model=OpenAILanguageModel(),
    embeddings=OpenAIEmbeddingProvider(),
    vector_store=store,
    reranker=FlashRankReranker(Path("data/models")),
)
response = agent.answer("What are the document's main recommendations?")
print(response.answer)
```

The agent retrieves up to 20 vector-search candidates, removes weak matches, and reranks the
remaining chunks before sending the best five to the answer model. FlashRank runs the open-source
`ms-marco-TinyBERT-L-2-v2` cross-encoder locally on CPU and caches it under ignored `data/`.

## Compare agentic and naive RAG

The fixed evaluation ingests the included reference document, runs both systems over the same
question set, and prints aggregate answer-quality, citation, and latency metrics.

```bash
python -m scripts.evaluate
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
