"""Abstract interface for knowledge sources in RadClaw."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class KBSearchResult:
    source_id: str
    source_system: str
    preferred_term: str
    content: str
    score: float = 0.0
    metadata: dict = field(default_factory=dict)


class KnowledgeSource(ABC):
    @abstractmethod
    async def search(self, query: str, max_results: int = 10) -> list[KBSearchResult]:
        ...

    @abstractmethod
    async def lookup(self, code: str) -> KBSearchResult | None:
        ...

    @abstractmethod
    def is_available(self) -> bool:
        ...