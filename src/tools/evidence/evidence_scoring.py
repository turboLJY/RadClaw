"""
EvidenceScoringTool — re-ranks retrieved evidence by relevance to visual findings.
Uses cosine similarity via MedicalEmbedder when available, falls back to keyword overlap.
"""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING

import numpy as np

from src.core.tool import Tool, ToolUseContext
from src.core.types import ToolResult, ValidationResult

if TYPE_CHECKING:
    from src.kb.embedder import MedicalEmbedder

logger = logging.getLogger(__name__)


class EvidenceScoringTool(Tool):
    name = "evidence_score_and_rank"
    description = (
        "Re-score and rank all retrieved evidence by relevance to the current "
        "visual findings.  Uses semantic similarity when available, "
        "otherwise keyword overlap."
    )
    input_schema: dict = {
        "type": "object",
        "properties": {
            "evidence_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Optional list of evidence IDs to score.  If omitted or "
                    "empty, all retrieved evidence will be scored."
                ),
            }
        },
        "additionalProperties": False,
    }
    is_read_only = False
    is_concurrency_safe = False
    max_result_size_chars = 30000

    def __init__(
        self,
        knowledge_dir: str = "./data/knowledge",
        embedder: MedicalEmbedder | None = None,
    ) -> None:
        self._knowledge_dir = knowledge_dir
        self._embedder = embedder

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    _STOP_WORDS = frozenset(
        "a an the is are was were be been being of in to for on with and or "
        "at by from as it its that this these those no not".split()
    )

    @classmethod
    def _tokenize(cls, text: str) -> set[str]:
        tokens = set(re.findall(r"[a-z0-9]+", text.lower()))
        return tokens - cls._STOP_WORDS

    @classmethod
    def _overlap_score(cls, reference_tokens: set[str], candidate_tokens: set[str]) -> float:
        """Recall-weighted overlap: fraction of reference tokens found in candidate."""
        if not reference_tokens:
            return 0.0
        overlap = reference_tokens & candidate_tokens
        return round(len(overlap) / len(reference_tokens), 4)

    # ------------------------------------------------------------------
    # Tool interface
    # ------------------------------------------------------------------

    def prompt(self) -> str:
        return (
            "Use evidence_score_and_rank to re-rank all retrieved evidence by "
            "relevance to the visual findings.  Call this after accumulating "
            "evidence from literature_search and case_retrieval to prioritize "
            "the most relevant evidence for report generation."
        )

    def validate_input(self, args: dict) -> ValidationResult:
        ids = args.get("evidence_ids")
        if ids is not None:
            if not isinstance(ids, list):
                return ValidationResult(valid=False, message="evidence_ids must be a list of strings")
            if not all(isinstance(i, str) for i in ids):
                return ValidationResult(valid=False, message="all evidence_ids must be strings")
        return ValidationResult(valid=True)

    def _cosine_score(self, ref_text: str, candidate_text: str) -> float | None:
        """Compute cosine similarity using the embedder. Returns None if unavailable."""
        if self._embedder is None:
            return None
        try:
            ref_emb = self._embedder.embed_query(ref_text)
            cand_emb = self._embedder.embed_query(candidate_text)
            return float(np.dot(ref_emb, cand_emb))
        except Exception:
            return None

    async def call(self, args: dict, context: ToolUseContext) -> ToolResult:
        pipeline_state = context.extra.get("pipeline_state")
        if pipeline_state is None:
            return ToolResult(
                data=json.dumps({"error": "No pipeline_state available in context"}),
                is_error=True,
            )

        evidence_list = pipeline_state.retrieved_evidence
        if not evidence_list:
            return ToolResult(
                data=json.dumps({"message": "No retrieved evidence to score", "ranked": []}),
            )

        requested_ids = args.get("evidence_ids") or []
        if requested_ids:
            id_set = set(requested_ids)
            evidence_list = [e for e in evidence_list if e.evidence_id in id_set]

        visual_text = pipeline_state.visual_findings_text or ""
        ref_tokens = self._tokenize(visual_text)

        use_semantic = self._embedder is not None and visual_text.strip()
        ref_emb = None
        if use_semantic:
            try:
                ref_emb = self._embedder.embed_query(visual_text)
            except Exception:
                use_semantic = False

        # Batch embed all evidence at once for efficiency
        cand_embs: list | None = None
        if use_semantic and ref_emb is not None:
            try:
                texts = [ev.content[:512] for ev in evidence_list]
                cand_embs = self._embedder.embed_texts(texts)
            except Exception:
                use_semantic = False

        scored: list[tuple[float, object]] = []
        for idx, ev in enumerate(evidence_list):
            keyword_score = self._overlap_score(ref_tokens, self._tokenize(ev.content))

            if use_semantic and cand_embs is not None:
                cosine_score = float(np.dot(ref_emb, cand_embs[idx]))
                score = 0.6 * cosine_score + 0.4 * keyword_score
            else:
                score = keyword_score

            score = round(score, 4)
            ev.relevance_score = score
            scored.append((score, ev))

        # Sort descending
        scored.sort(key=lambda x: x[0], reverse=True)

        ranked_output: list[dict] = []
        for rank, (score, ev) in enumerate(scored[:20], 1):
            ranked_output.append({
                "rank": rank,
                "evidence_id": ev.evidence_id,
                "source_type": ev.source_type,
                "relevance_score": score,
                "content_preview": ev.content[:100],
            })

        output = {
            "total_scored": len(scored),
            "showing_top": len(ranked_output),
            "ranked": ranked_output,
        }
        return ToolResult(data=json.dumps(output, indent=2, ensure_ascii=False))
