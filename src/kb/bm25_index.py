"""BM25 sparse index for hybrid retrieval alongside FAISS dense search."""

from __future__ import annotations

import json
import logging
import pickle
import re
from pathlib import Path

logger = logging.getLogger(__name__)

_STOP_WORDS = frozenset(
    "a an the is are was were be been being of in to for on with and or "
    "at by from as it its that this these those do does did has have had "
    "will would shall should may might can could".split()
)


def _tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in tokens if t not in _STOP_WORDS and len(t) > 1]


class BM25Index:
    def __init__(self, index_path: str, meta_path: str) -> None:
        self._index_path = Path(index_path)
        self._meta_path = Path(meta_path)
        self._bm25 = None
        self._metadata: list[dict] = []

        if self._index_path.exists() and self._meta_path.exists():
            with open(self._index_path, "rb") as f:
                self._bm25 = pickle.load(f)
            with open(self._meta_path, "r", encoding="utf-8") as f:
                self._metadata = json.load(f)
            logger.info(
                "Loaded BM25 index %s (%d documents)",
                self._index_path.name,
                len(self._metadata),
            )

    @property
    def is_loaded(self) -> bool:
        return self._bm25 is not None and len(self._metadata) > 0

    @property
    def size(self) -> int:
        return len(self._metadata)

    def search(self, query: str, k: int = 10) -> list[tuple[int, float]]:
        if not self.is_loaded:
            return []

        tokens = _tokenize(query)
        if not tokens:
            return []

        scores = self._bm25.get_scores(tokens)
        top_k = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:k]
        return [(idx, float(score)) for idx, score in top_k if score > 0]

    def get_metadata(self, idx: int) -> dict:
        if 0 <= idx < len(self._metadata):
            return self._metadata[idx]
        return {}

    @classmethod
    def build(
        cls,
        texts: list[str],
        metadata: list[dict],
        output_dir: str,
        name: str,
    ) -> BM25Index:
        from rank_bm25 import BM25Okapi

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        index_path = out / f"{name}.bm25.pkl"
        meta_path = out / f"{name}_bm25_meta.json"

        corpus = [_tokenize(t) or ["_empty_"] for t in texts]

        bm25 = BM25Okapi(corpus)

        with open(index_path, "wb") as f:
            pickle.dump(bm25, f)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False)

        logger.info("Built BM25 index %s with %d documents", name, len(texts))
        return cls(str(index_path), str(meta_path))