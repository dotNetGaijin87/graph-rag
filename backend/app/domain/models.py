"""Framework-agnostic domain models."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Chunk:
    id: str
    document_id: str
    index: int
    text: str
    embedding: list[float] | None = None


@dataclass(frozen=True)
class Entity:
    name: str
    type: str = "Concept"
    description: str = ""


@dataclass(frozen=True)
class Relationship:
    source: str
    target: str
    type: str = "RELATED_TO"
    description: str = ""


@dataclass(frozen=True)
class ExtractionResult:
    entities: list[Entity] = field(default_factory=list)
    relationships: list[Relationship] = field(default_factory=list)


@dataclass(frozen=True)
class Document:
    id: str
    title: str
    text: str
    chunks: list[Chunk] = field(default_factory=list)


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    document_id: str
    text: str
    score: float
    entities: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class GraphFact:
    source: str
    type: str
    target: str
    description: str = ""

    def as_sentence(self) -> str:
        rel = self.type.replace("_", " ").lower()
        base = f"{self.source} {rel} {self.target}"
        return f"{base} ({self.description})" if self.description else base


@dataclass(frozen=True)
class RetrievalContext:
    chunks: list[RetrievedChunk] = field(default_factory=list)
    facts: list[GraphFact] = field(default_factory=list)


@dataclass(frozen=True)
class Answer:
    question: str
    answer: str
    context: RetrievalContext


@dataclass(frozen=True)
class IngestionReport:
    document_id: str
    title: str
    chunk_count: int
    entity_count: int
    relationship_count: int
