"""Streamlit entrypoint for the Agentic RAG demo."""

import os
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Literal, TypedDict, cast

import streamlit as st
from openai import APIError

from agentic_rag.agent import RetrievalAgent
from agentic_rag.embeddings import OpenAIEmbeddingProvider
from agentic_rag.ingestion import DocumentIngestor
from agentic_rag.llm import create_language_model, required_api_keys
from agentic_rag.models import AgentResponse
from agentic_rag.reranking import FlashRankReranker
from agentic_rag.ui import format_citation, save_uploaded_document
from agentic_rag.vector_store import ChromaVectorStore

DATA_DIRECTORY = Path("data")
UPLOAD_DIRECTORY = DATA_DIRECTORY / "uploads"
VECTOR_DIRECTORY = DATA_DIRECTORY / "chroma"
MODEL_DIRECTORY = DATA_DIRECTORY / "models"


@dataclass(frozen=True)
class AppServices:
    """Long-lived application dependencies."""

    ingestor: DocumentIngestor
    agent: RetrievalAgent
    vector_store: ChromaVectorStore


class ChatMessage(TypedDict):
    """Serializable message rendered in the chat timeline."""

    role: Literal["user", "assistant"]
    content: str
    latency_ms: float | None
    retrieval_used: bool | None
    citations: list[str]


@st.cache_resource
def build_services() -> AppServices:
    """Create shared embedding, language model, and local Chroma adapters."""

    embeddings = OpenAIEmbeddingProvider()
    language_model = create_language_model()
    vector_store = ChromaVectorStore(VECTOR_DIRECTORY)
    reranker = FlashRankReranker(MODEL_DIRECTORY)
    return AppServices(
        ingestor=DocumentIngestor(
            embeddings=embeddings,
            vector_store=vector_store,
        ),
        agent=RetrievalAgent(
            language_model=language_model,
            embeddings=embeddings,
            vector_store=vector_store,
            reranker=reranker,
        ),
        vector_store=vector_store,
    )


def render_answer(response: AgentResponse, latency_ms: float) -> None:
    """Render a generated answer and its retrieval trace."""

    st.markdown(response.answer)
    retrieval_status = "retrieval used" if response.retrieval_used else "no retrieval"
    st.caption(f"{latency_ms:.0f} ms · {retrieval_status}")
    if response.citations:
        with st.expander("Source details"):
            for citation in response.citations:
                st.markdown(f"- {format_citation(citation)}")


def render_history(messages: list[ChatMessage]) -> None:
    """Render prior chat messages."""

    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                retrieval_status = "retrieval used" if message["retrieval_used"] else "no retrieval"
                latency_ms = message["latency_ms"] or 0.0
                st.caption(f"{latency_ms:.0f} ms · {retrieval_status}")
                citations = message["citations"]
                if citations:
                    with st.expander("Source details"):
                        for citation in citations:
                            st.markdown(f"- {citation}")


st.set_page_config(
    page_title="Agentic RAG",
    page_icon="◆",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at 12% 2%, rgba(43, 116, 111, .13), transparent 34rem),
            #f7f8f5;
    }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stSidebar"] {
        background: #132b29;
        color: #f4f5ef;
    }
    [data-testid="stSidebar"] * { color: #f4f5ef; }
    [data-testid="stSidebar"] .stButton > button {
        background: #d6ff70;
        border: 0;
        color: #132b29;
        font-weight: 700;
    }
    .hero-label {
        color: #27746f;
        font-size: .78rem;
        font-weight: 800;
        letter-spacing: .14em;
        margin-bottom: .3rem;
        text-transform: uppercase;
    }
    .hero-title {
        color: #132b29;
        font-family: Georgia, serif;
        font-size: clamp(2.4rem, 5vw, 4.7rem);
        font-weight: 500;
        letter-spacing: -.055em;
        line-height: .96;
        margin: 0;
        max-width: 850px;
    }
    .hero-copy {
        color: #546562;
        font-size: 1.05rem;
        margin: 1rem 0 2.2rem;
        max-width: 680px;
    }
    [data-testid="stChatMessage"] {
        background: rgba(255, 255, 255, .72);
        border: 1px solid rgba(19, 43, 41, .09);
        border-radius: 18px;
        margin-bottom: .75rem;
        padding: .35rem;
    }
    [data-testid="stChatInput"] {
        background: white;
        border-radius: 18px;
        box-shadow: 0 12px 32px rgba(19, 43, 41, .08);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

missing_keys = [key for key in required_api_keys() if not os.environ.get(key)]
if missing_keys:
    st.error(f"Set {', '.join(missing_keys)} in the process environment before starting the app.")
    st.stop()

services = build_services()
messages = cast(list[ChatMessage], st.session_state.setdefault("messages", []))

with st.sidebar:
    st.markdown("## Knowledge base")
    st.caption("Upload a PDF, DOCX, Markdown, or text document.")
    uploaded_file = st.file_uploader(
        "Document",
        type=["pdf", "docx", "md", "txt"],
        max_upload_size=250,
        label_visibility="collapsed",
    )
    if uploaded_file is not None and st.button("Ingest document", use_container_width=True):
        try:
            path = save_uploaded_document(
                filename=uploaded_file.name,
                content=uploaded_file.getvalue(),
                upload_directory=UPLOAD_DIRECTORY,
            )
            with st.spinner("Reading, chunking, and embedding…"):
                result = services.ingestor.ingest(path)
        except (APIError, OSError, ValueError) as error:
            st.error(str(error))
        else:
            st.session_state["active_document"] = result.source
            st.success(f"Stored {result.chunk_count} chunks from {result.source}")

    st.divider()
    active_document = st.session_state.get("active_document")
    if active_document:
        st.markdown("**Active document**")
        st.write(active_document)
        st.caption(f"{services.vector_store.count()} searchable chunks")
    else:
        st.caption("No document ingested in this session.")

st.markdown('<p class="hero-label">Plan · Retrieve · Cite</p>', unsafe_allow_html=True)
st.markdown(
    '<h1 class="hero-title">Ask your documents.<br>See the evidence.</h1>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="hero-copy">The agent decides when to search, rewrites the query, '
    "and returns source-backed answers with latency and retrieval details.</p>",
    unsafe_allow_html=True,
)

render_history(messages)

has_documents = services.vector_store.count() > 0
prompt = st.chat_input(
    "Ask a question about the uploaded document",
    disabled=not has_documents,
)
if not has_documents:
    st.info("Ingest a document from the sidebar to start asking questions.")

if prompt:
    user_message: ChatMessage = {
        "role": "user",
        "content": prompt,
        "latency_ms": None,
        "retrieval_used": None,
        "citations": [],
    }
    messages.append(user_message)
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"), st.spinner("Planning retrieval…"):
        started_at = perf_counter()
        try:
            response = services.agent.answer(prompt)
        except (APIError, RuntimeError, ValueError) as error:
            st.error(f"Unable to answer: {error}")
        else:
            latency_ms = (perf_counter() - started_at) * 1_000
            render_answer(response, latency_ms)
            messages.append(
                {
                    "role": "assistant",
                    "content": response.answer,
                    "latency_ms": latency_ms,
                    "retrieval_used": response.retrieval_used,
                    "citations": [format_citation(citation) for citation in response.citations],
                }
            )
