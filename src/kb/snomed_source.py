"""SNOMED CT knowledge source via Snowstorm FHIR API."""

from __future__ import annotations

import asyncio
import json
import logging
import urllib.request
import urllib.parse
from pathlib import Path
from typing import Any

from src.kb.base import KBSearchResult, KnowledgeSource
from src.kb.vector_store import VectorStore
from src.kb.bm25_index import BM25Index

logger = logging.getLogger(__name__)

_DEFAULT_FHIR_TX_URL = "https://tx.fhir.org/r4"
_DEFAULT_EDITION = "900000000000207008"


class SnomedSource(KnowledgeSource):
    """SNOMED CT source with dual mode: local FAISS index + online API fallback."""

    def __init__(
        self,
        indexes_dir: str = "./data/indexes",
        embedder=None,
        snowstorm_url: str = _DEFAULT_FHIR_TX_URL,
        edition: str = _DEFAULT_EDITION,
        mode: str = "api",
    ) -> None:
        self._indexes_dir = Path(indexes_dir)
        self._embedder = embedder
        self._snowstorm_url = snowstorm_url.rstrip("/")
        self._edition = edition
        self._mode = mode
        self._store: VectorStore | None = None
        self._bm25: BM25Index | None = None
        self._loaded = False
        self._cache: dict[str, Any] = {}

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        index_path = self._indexes_dir / "snomed.index"
        meta_path = self._indexes_dir / "snomed_meta.json"
        if index_path.exists() and meta_path.exists():
            self._store = VectorStore(str(index_path), str(meta_path))
        bm25_path = self._indexes_dir / "snomed.bm25.pkl"
        bm25_meta = self._indexes_dir / "snomed_bm25_meta.json"
        if bm25_path.exists() and bm25_meta.exists():
            self._bm25 = BM25Index(str(bm25_path), str(bm25_meta))

    def is_available(self) -> bool:
        self._ensure_loaded()
        has_local = self._store is not None and self._store.is_loaded and self._embedder is not None
        has_api = self._mode in ("api", "hybrid")
        return has_local or has_api

    def _api_search(self, query: str, max_results: int = 10) -> list[KBSearchResult]:
        cache_key = f"search:{query}:{max_results}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        params = urllib.parse.urlencode({
            "filter": query,
            "count": str(max_results),
            "url": "http://snomed.info/sct?fhir_vs",
            "includeDesignations": "true",
        })
        url = f"{self._snowstorm_url}/ValueSet/$expand?{params}"

        try:
            req = urllib.request.Request(url, headers={"Accept": "application/fhir+json"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            logger.warning("Snowstorm API search failed: %s", e)
            return []

        results = []
        expansion = data.get("expansion", {})
        for item in expansion.get("contains", []):
            code = item.get("code", "")
            display = item.get("display", "")
            if not code:
                continue

            synonyms = []
            for desig in item.get("designation", []):
                val = desig.get("value", "").strip()
                if val and val.lower() != display.lower():
                    synonyms.append(val)

            content_parts = [display]
            if synonyms:
                content_parts.append("synonyms: " + ", ".join(synonyms[:5]))
            content = ". ".join(content_parts)

            results.append(
                KBSearchResult(
                    source_id=code,
                    source_system="snomed",
                    preferred_term=display,
                    content=content,
                    score=0.9,
                    metadata={
                        "system": item.get("system", ""),
                        "synonyms": synonyms,
                    },
                )
            )

        self._cache[cache_key] = results
        return results

    def _api_lookup(self, code: str) -> KBSearchResult | None:
        cache_key = f"lookup:{code}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        url = (
            f"{self._snowstorm_url}/CodeSystem/$lookup"
            f"?system=http://snomed.info/sct&code={code}"
        )

        try:
            req = urllib.request.Request(url, headers={"Accept": "application/fhir+json"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            logger.warning("SNOMED FHIR API lookup failed for %s: %s", code, e)
            return None

        display = ""
        designations = []
        for param in data.get("parameter", []):
            if param.get("name") == "display":
                display = param.get("valueString", "")
            if param.get("name") == "designation":
                for part in param.get("part", []):
                    if part.get("name") == "value":
                        designations.append(part.get("valueString", ""))

        if not display:
            return None

        result = KBSearchResult(
            source_id=code,
            source_system="snomed",
            preferred_term=display,
            content=display,
            score=1.0,
            metadata={"synonyms": designations},
        )
        self._cache[cache_key] = result
        return result

    async def search(self, query: str, max_results: int = 10) -> list[KBSearchResult]:
        self._ensure_loaded()

        if self._store and self._store.is_loaded and self._embedder:
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
                        source_id=meta.get("sctid", ""),
                        source_system="snomed",
                        preferred_term=meta.get("term", ""),
                        content=meta.get("definition", ""),
                        score=score,
                        metadata={
                            "synonyms": meta.get("synonyms", []),
                            "semantic_type": meta.get("semantic_type", ""),
                        },
                    )
                )
            if results:
                return results

        if self._mode in ("api", "hybrid"):
            return await asyncio.to_thread(self._api_search, query, max_results)

        return []

    async def lookup(self, code: str) -> KBSearchResult | None:
        self._ensure_loaded()

        if self._store and self._store.is_loaded:
            for i in range(self._store.size):
                meta = self._store.get_metadata(i)
                if meta.get("sctid") == code:
                    return KBSearchResult(
                        source_id=code,
                        source_system="snomed",
                        preferred_term=meta.get("term", ""),
                        content=meta.get("definition", ""),
                        score=1.0,
                        metadata={
                            "synonyms": meta.get("synonyms", []),
                            "semantic_type": meta.get("semantic_type", ""),
                        },
                    )

        if self._mode in ("api", "hybrid"):
            return await asyncio.to_thread(self._api_lookup, code)

        return None
