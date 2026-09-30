from agentic_rag.evaluation import EvaluationCase, compare_systems, render_comparison_table
from agentic_rag.models import AgentResponse, Citation


class StaticSystem:
    def __init__(self, response: AgentResponse) -> None:
        self._response = response

    def answer(self, question: str) -> AgentResponse:
        return self._response


def test_compares_keyword_citation_and_latency_metrics() -> None:
    case = EvaluationCase(
        question="How is evidence stored?",
        expected_terms=["chroma", "locally"],
    )
    cited_response = AgentResponse(
        answer="Evidence is stored locally in Chroma [S1].",
        retrieval_used=True,
        citations=[
            Citation(
                label="S1",
                source="guide.txt",
                chunk_id="one",
                score=0.9,
            )
        ],
    )
    uncited_response = AgentResponse(
        answer="Evidence is stored in Chroma.",
        retrieval_used=True,
    )

    report = compare_systems(
        [
            ("Agentic RAG", StaticSystem(cited_response)),
            ("Naive RAG", StaticSystem(uncited_response)),
        ],
        [case],
    )

    assert report.summaries[0].average_score == 1.0
    assert report.summaries[0].citation_rate == 1.0
    assert report.summaries[1].average_score == 0.4
    assert report.summaries[1].average_latency_ms >= 0.0


def test_renders_comparison_table() -> None:
    report = compare_systems(
        [
            (
                "Agentic RAG",
                StaticSystem(
                    AgentResponse(
                        answer="Chroma [S1]",
                        retrieval_used=True,
                        citations=[
                            Citation(
                                label="S1",
                                source="guide.txt",
                                chunk_id="one",
                                score=0.9,
                            )
                        ],
                    )
                ),
            )
        ],
        [EvaluationCase(question="Store?", expected_terms=["chroma"])],
    )

    table = render_comparison_table(report)

    assert "System" in table
    assert "Agentic RAG" in table
    assert "Avg latency" in table
