"""Structured clinical guidelines store with FAISS semantic search."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from src.kb.base import KBSearchResult, KnowledgeSource
from src.kb.vector_store import VectorStore
from src.kb.bm25_index import BM25Index

logger = logging.getLogger(__name__)


def _meta_matches(entry_values: list[str], filter_value: str) -> bool:
    if not entry_values:
        return True
    return filter_value in entry_values


class GuidelinesStore(KnowledgeSource):
    """Search structured clinical guidelines via FAISS vector index."""

    def __init__(
        self,
        indexes_dir: str = "./data/indexes",
        knowledge_dir: str = "./data/knowledge",
        embedder=None,
    ) -> None:
        self._indexes_dir = Path(indexes_dir)
        self._knowledge_dir = Path(knowledge_dir)
        self._embedder = embedder
        self._store: VectorStore | None = None
        self._bm25: BM25Index | None = None
        self._loaded = False

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        index_path = self._indexes_dir / "guidelines.index"
        meta_path = self._indexes_dir / "guidelines_meta.json"
        if index_path.exists() and meta_path.exists():
            self._store = VectorStore(str(index_path), str(meta_path))
        else:
            logger.info("Guidelines index not found at %s", self._indexes_dir)
        bm25_path = self._indexes_dir / "guidelines.bm25.pkl"
        bm25_meta = self._indexes_dir / "guidelines_bm25_meta.json"
        if bm25_path.exists() and bm25_meta.exists():
            self._bm25 = BM25Index(str(bm25_path), str(bm25_meta))

    def is_available(self) -> bool:
        self._ensure_loaded()
        return self._store is not None and self._store.is_loaded and self._embedder is not None

    async def search(
        self,
        query: str,
        max_results: int = 10,
        modality: str | None = None,
        disease_type: str | None = None,
    ) -> list[KBSearchResult]:
        if not self.is_available():
            return []

        fetch_k = max_results * 3 if (modality or disease_type) else max_results
        query_emb = self._embedder.embed_query(query)
        if self._bm25 and self._bm25.is_loaded:
            hits = self._store.search_hybrid(query_emb, query, self._bm25, k=fetch_k)
        else:
            hits = self._store.search(query_emb, k=fetch_k)

        results = []
        for idx, score in hits:
            meta = self._store.get_metadata(idx)
            if not meta:
                continue

            if modality and not _meta_matches(meta.get("modality", []), modality):
                continue
            if disease_type and not _meta_matches(meta.get("disease_type", []), disease_type):
                continue

            source_id = meta.get("guideline_id", meta.get("id", meta.get("code", f"GL_{idx}")))
            source_name = meta.get("source", meta.get("system", "guideline"))

            title = ""
            for _tk in ("condition", "title", "name", "topic", "region", "finding",
                         "severity", "term", "section", "code", "nodule_type", "organ"):
                title = meta.get(_tk, "")
                if title:
                    break

            content_parts = []
            for _ck in ("imaging_criteria", "criteria", "definition", "recommendation", "text", "description"):
                if meta.get(_ck):
                    content_parts.append(meta[_ck])
                    break
            if meta.get("management"):
                content_parts.append(f"Management: {meta['management']}")
            if not content_parts and meta.get("content"):
                content_parts.append(meta["content"])
            content = " ".join(content_parts) if content_parts else title

            results.append(
                KBSearchResult(
                    source_id=source_id,
                    source_system="guideline",
                    preferred_term=title,
                    content=content,
                    score=score,
                    metadata={
                        "source": source_name,
                        "references": meta.get("references", []),
                        "key_findings": meta.get("key_findings", []),
                        "category": meta.get("category", ""),
                        "modality": meta.get("modality", []),
                        "disease_type": meta.get("disease_type", []),
                    },
                )
            )
            if len(results) >= max_results:
                break
        return results

    async def lookup(self, code: str) -> KBSearchResult | None:
        self._ensure_loaded()
        if self._store is None:
            return None

        for i in range(self._store.size):
            meta = self._store.get_metadata(i)
            gid = meta.get("guideline_id", meta.get("id", ""))
            if gid == code:
                title = meta.get("condition", meta.get("title", ""))
                content = meta.get("imaging_criteria", meta.get("content", ""))
                return KBSearchResult(
                    source_id=gid,
                    source_system="guideline",
                    preferred_term=title,
                    content=content,
                    score=1.0,
                    metadata=meta,
                )
        return None