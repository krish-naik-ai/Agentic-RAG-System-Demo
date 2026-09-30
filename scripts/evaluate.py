"""Run the fixed agentic-versus-naive RAG benchmark."""

from pathlib import Path

from pydantic import TypeAdapter

from agentic_rag.agent import RetrievalAgent
from agentic_rag.baselines import NaiveRAG
from agentic_rag.embeddings import OpenAIEmbeddingProvider
from agentic_rag.evaluation import EvaluationCase, compare_systems, render_comparison_table
from agentic_rag.ingestion import DocumentIngestor
from agentic_rag.llm import OpenAILanguageModel
from agentic_rag.vector_store import ChromaVectorStore


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    cases = TypeAdapter(list[EvaluationCase]).validate_json(
        (root / "eval" / "questions.json").read_bytes()
    )
    vector_store = ChromaVectorStore(
        root / "data" / "eval_chroma",
        collection_name="agentic-rag-evaluation",
    )
    embeddings = OpenAIEmbeddingProvider()
    language_model = OpenAILanguageModel()
    ingestor = DocumentIngestor(embeddings=embeddings, vector_store=vector_store)
    ingestor.ingest(root / "eval" / "fixtures" / "agentic_rag_overview.txt")

    report = compare_systems(
        [
            (
                "Agentic RAG",
                RetrievalAgent(
                    language_model=language_model,
                    embeddings=embeddings,
                    vector_store=vector_store,
                ),
            ),
            (
                "Naive RAG",
                NaiveRAG(
                    language_model=language_model,
                    embeddings=embeddings,
                    vector_store=vector_store,
                ),
            ),
        ],
        cases,
    )
    print(render_comparison_table(report))


if __name__ == "__main__":
    main()
