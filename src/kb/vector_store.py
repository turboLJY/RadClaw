"""FAISS vector store with JSON metadata sidecar."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class VectorStore:
    def __init__(self, index_path: str, metadata_path: str) -> None:
        import faiss

        self._index_path = Path(index_path)
        self._metadata_path = Path(metadata_path)
        self._index = None
        self._metadata: list[dict] = []

        if self._index_path.exists() and self._metadata_path.exists():
            self._index = faiss.read_index(str(self._index_path))
            with open(self._metadata_path, "r", encoding="utf-8") as f:
                self._metadata = json.load(f)
            logger.info(
                "Loaded FAISS index %s (%d vectors)",
                self._index_path.name,
                self._index.ntotal,
            )

    @property
    def is_loaded(self) -> bool:
        return self._index is not None and len(self._metadata) > 0

    @property
    def size(self) -> int:
        return self._index.ntotal if self._index else 0

    def search(
        self, query_embedding: np.ndarray, k: int = 10
    ) -> list[tuple[int, float]]:
        if not self.is_loaded:
            return []
        query = np.asarray(query_embedding, dtype=np.float32)
        if query.ndim == 1:
            query = query.reshape(1, -1)
        k = min(k, self._index.ntotal)
        scores, indices = self._index.search(query, k)
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx >= 0:
                results.append((int(idx), float(score)))
        return results

    def search_hybrid(
        self,
        query_embedding: np.ndarray,
        query_text: str,
        bm25_index,
        k: int = 10,
        rrf_k: int = 60,
    ) -> list[tuple[int, float]]:
        dense_results = self.search(query_embedding, k=k * 2)
        sparse_results = bm25_index.search(query_text, k=k * 2) if bm25_index and bm25_index.is_loaded else []

        if not sparse_results:
            return dense_results[:k]
        if not dense_results:
            return sparse_results[:k]

        dense_rank = {idx: rank for rank, (idx, _) in enumerate(dense_results)}
        sparse_rank = {idx: rank for rank, (idx, _) in enumerate(sparse_results)}
        max_rank = k * 2 + 1

        all_ids = set(dense_rank.keys()) | set(sparse_rank.keys())
        rrf_scores = []
        for idx in all_ids:
            dr = dense_rank.get(idx, max_rank)
            sr = sparse_rank.get(idx, max_rank)
            score = 1.0 / (rrf_k + dr) + 1.0 / (rrf_k + sr)
            rrf_scores.append((idx, score))

        rrf_scores.sort(key=lambda x: x[1], reverse=True)
        return rrf_scores[:k]

    def get_metadata(self, idx: int) -> dict:
        if 0 <= idx < len(self._metadata):
            return self._metadata[idx]
        return {}

    @classmethod
    def build(
        cls,
        embeddings: np.ndarray,
        metadata: list[dict],
        output_dir: str,
        name: str,
    ) -> VectorStore:
        import faiss

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        index_path = out / f"{name}.index"
        meta_path = out / f"{name}_meta.json"

        emb = np.asarray(embeddings, dtype=np.float32)
        faiss.normalize_L2(emb)
        index = faiss.IndexFlatIP(emb.shape[1])
        index.add(emb)

        faiss.write_index(index, str(index_path))
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False)

        logger.info("Built FAISS index %s with %d vectors", name, index.ntotal)
        return cls(str(index_path), str(meta_path))