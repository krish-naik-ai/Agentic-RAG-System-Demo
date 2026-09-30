"""Comparative evaluation utilities for RAG systems."""

from collections.abc import Sequence
from statistics import fmean
from time import perf_counter
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from agentic_rag.models import AgentResponse


class AnsweringSystem(Protocol):
    """Question-answering interface shared by evaluated systems."""

    def answer(self, question: str) -> AgentResponse:
        """Return an answer and its retrieval trace."""


class EvaluationCase(BaseModel):
    """One fixed benchmark question and expected answer terms."""

    model_config = ConfigDict(frozen=True)

    question: str = Field(min_length=1)
    expected_terms: list[str] = Field(min_length=1)


class EvaluationRecord(BaseModel):
    """Per-question score for one system."""

    model_config = ConfigDict(frozen=True)

    system: str
    question: str
    answer: str
    keyword_recall: float = Field(ge=0.0, le=1.0)
    citation_score: float = Field(ge=0.0, le=1.0)
    total_score: float = Field(ge=0.0, le=1.0)
    latency_ms: float = Field(ge=0.0)


class EvaluationSummary(BaseModel):
    """Aggregate metrics for one evaluated system."""

    model_config = ConfigDict(frozen=True)

    system: str
    average_score: float
    keyword_recall: float
    citation_rate: float
    average_latency_ms: float


class EvaluationReport(BaseModel):
    """Complete comparison result."""

    model_config = ConfigDict(frozen=True)

    records: list[EvaluationRecord]
    summaries: list[EvaluationSummary]


def compare_systems(
    systems: Sequence[tuple[str, AnsweringSystem]],
    cases: Sequence[EvaluationCase],
) -> EvaluationReport:
    """Run every fixed question against each system and aggregate metrics."""

    records: list[EvaluationRecord] = []
    summaries: list[EvaluationSummary] = []
    for name, system in systems:
        system_records = [_evaluate_case(name, system, case) for case in cases]
        records.extend(system_records)
        summaries.append(
            EvaluationSummary(
                system=name,
                average_score=fmean(record.total_score for record in system_records),
                keyword_recall=fmean(record.keyword_recall for record in system_records),
                citation_rate=fmean(record.citation_score for record in system_records),
                average_latency_ms=fmean(record.latency_ms for record in system_records),
            )
        )
    return EvaluationReport(records=records, summaries=summaries)


def render_comparison_table(report: EvaluationReport) -> str:
    """Render aggregate results as an aligned console table."""

    headers = ("System", "Avg score", "Keyword recall", "Citation rate", "Avg latency")
    rows = [
        (
            summary.system,
            f"{summary.average_score:.3f}",
            f"{summary.keyword_recall:.3f}",
            f"{summary.citation_rate:.3f}",
            f"{summary.average_latency_ms:.1f} ms",
        )
        for summary in report.summaries
    ]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]

    def format_row(row: tuple[str, ...]) -> str:
        return " | ".join(value.ljust(widths[index]) for index, value in enumerate(row))

    separator = "-+-".join("-" * width for width in widths)
    return "\n".join([format_row(headers), separator, *(format_row(row) for row in rows)])


def _evaluate_case(
    system_name: str,
    system: AnsweringSystem,
    case: EvaluationCase,
) -> EvaluationRecord:
    started = perf_counter()
    response = system.answer(case.question)
    latency_ms = (perf_counter() - started) * 1_000
    normalized_answer = response.answer.casefold()
    matches = sum(term.casefold() in normalized_answer for term in case.expected_terms)
    keyword_recall = matches / len(case.expected_terms)
    citation_score = float(
        bool(response.citations)
        and all(f"[{citation.label}]" in response.answer for citation in response.citations)
    )
    total_score = (0.8 * keyword_recall) + (0.2 * citation_score)
    return EvaluationRecord(
        system=system_name,
        question=case.question,
        answer=response.answer,
        keyword_recall=keyword_recall,
        citation_score=citation_score,
        total_score=total_score,
        latency_ms=latency_ms,
    )
