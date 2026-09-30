"""Persistent local Chroma vector storage."""

from pathlib import Path
from typing import cast

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.api.types import Metadata, PyEmbeddings

from agentic_rag.models import DocumentChunk, RetrievedChunk


class ChromaVectorStore:
    """Stores explicitly generated embeddings in a persistent Chroma collection."""

    def __init__(
        self,
        persist_path: Path,
        *,
        collection_name: str = "agentic-rag",
    ) -> None:
        persist_path.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(persist_path))
        self._collection: Collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def replace(self, chunks: list[DocumentChunk], embeddings: list[list[float]]) -> None:
        """Replace stored chunks for the affected sources."""

        if len(chunks) != len(embeddings):
            raise ValueError("Each chunk must have exactly one embedding")
        if not chunks:
            return

        for source in {chunk.source for chunk in chunks}:
            self._collection.delete(where={"source": source})

        metadatas: list[Metadata] = []
        for chunk in chunks:
            metadata: dict[str, str | int] = {
                "source": chunk.source,
                "chunk_index": chunk.chunk_index,
            }
            if chunk.page is not None:
                metadata["page"] = chunk.page
            metadatas.append(metadata)

        embedding_values = cast(PyEmbeddings, embeddings)
        self._collection.upsert(
            ids=[chunk.chunk_id for chunk in chunks],
            documents=[chunk.text for chunk in chunks],
            embeddings=embedding_values,
            metadatas=metadatas,
        )

    def query(self, embedding: list[float], *, limit: int = 5) -> list[RetrievedChunk]:
        """Return the closest stored chunks for a query embedding."""

        if limit <= 0:
            raise ValueError("limit must be positive")

        result = self._collection.query(
            query_embeddings=cast(PyEmbeddings, [embedding]),
            n_results=limit,
            include=["documents", "metadatas", "distances"],
        )
        ids = result["ids"][0]
        documents = result["documents"][0] if result["documents"] else []
        metadatas = result["metadatas"][0] if result["metadatas"] else []
        distances = result["distances"][0] if result["distances"] else []

        retrieved: list[RetrievedChunk] = []
        for chunk_id, text, metadata, distance in zip(
            ids,
            documents,
            metadatas,
            distances,
            strict=True,
        ):
            source = metadata.get("source")
            chunk_index = metadata.get("chunk_index")
            page = metadata.get("page")
            if not isinstance(source, str) or not isinstance(chunk_index, int):
                raise ValueError("Stored chunk metadata is invalid")
            if page is not None and not isinstance(page, int):
                raise ValueError("Stored page metadata is invalid")

            chunk = DocumentChunk(
                chunk_id=chunk_id,
                text=text,
                source=source,
                chunk_index=chunk_index,
                page=page,
            )
            retrieved.append(RetrievedChunk(chunk=chunk, score=1.0 - float(distance)))
        return retrieved

    def count(self) -> int:
        return self._collection.count()
