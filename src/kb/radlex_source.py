"""RadLex knowledge source backed by FAISS vector index."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from src.kb.base import KBSearchResult, KnowledgeSource
from src.kb.vector_store import VectorStore
from src.kb.bm25_index import BM25Index

logger = logging.getLogger(__name__)


class RadLexSource(KnowledgeSource):
    def __init__(
        self,
        indexes_dir: str = "./data/indexes",
        embedder=None,
    ) -> None:
        self._indexes_dir = Path(indexes_dir)
        self._embedder = embedder
        self._store: VectorStore | None = None
        self._bm25: BM25Index | None = None
        self._loaded = False

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        index_path = self._indexes_dir / "radlex.index"
        meta_path = self._indexes_dir / "radlex_meta.json"
        if index_path.exists() and meta_path.exists():
            self._store = VectorStore(str(index_path), str(meta_path))
        else:
            logger.warning("RadLex index not found at %s", self._indexes_dir)
        bm25_path = self._indexes_dir / "radlex.bm25.pkl"
        bm25_meta = self._indexes_dir / "radlex_bm25_meta.json"
        if bm25_path.exists() and bm25_meta.exists():
            self._bm25 = BM25Index(str(bm25_path), str(bm25_meta))

    def is_available(self) -> bool:
        self._ensure_loaded()
        return self._store is not None and self._store.is_loaded and self._embedder is not None

    async def search(self, query: str, max_results: int = 10) -> list[KBSearchResult]:
        if not self.is_available():
            return []

        query_emb = self._embedder.embed_query(query)
        if self._bm25 and self._bm25.is_loaded:
            hits = self._store.search_hybrid(query_emb, query, self._bm25, k=max_results)
        else:
            hits = self._store.search(query_emb, k=max_results)

        results = []
        for idx, score in hits:
            meta = self._store.get_metadata(idx)
            if not meta:
                continue
            results.append(
                KBSearchResult(
                    source_id=meta.get("code", ""),
                    source_system="radlex",
                    preferred_term=meta.get("preferred_name", ""),
                    content=meta.get("definition", ""),
                    score=score,
                    metadata={
                        "synonyms": meta.get("synonyms", []),
                        "category": meta.get("category", ""),
                        "parent_code": meta.get("parent_code", ""),
                    },
                )
            )
        return results

    async def lookup(self, code: str) -> KBSearchResult | None:
        self._ensure_loaded()
        if self._store is None:
            return None

        for i in range(self._store.size):
            meta = self._store.get_metadata(i)
            if meta.get("code") == code:
                return KBSearchResult(
                    source_id=meta.get("code", ""),
                    source_system="radlex",
                    preferred_term=meta.get("preferred_name", ""),
                    content=meta.get("definition", ""),
                    score=1.0,
                    metadata={
                        "synonyms": meta.get("synonyms", []),
                        "category": meta.get("category", ""),
                        "parent_code": meta.get("parent_code", ""),
                    },
                )
        return None