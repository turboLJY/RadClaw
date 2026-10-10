"""
LiteratureSearchTool — dual-mode literature search:
1) Remote HTTP service for clinical guidelines (POST /search/guidelines)
2) PubMed E-utilities for online literature
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING

import httpx

from src.core.tool import Tool, ToolUseContext
from src.core.types import RetrievedEvidence, ToolResult, ValidationResult
from src.kb.abbreviation_expander import AbbreviationExpander

if TYPE_CHECKING:
    from src.kb.pubmed_client import PubMedClient

logger = logging.getLogger(__name__)
_expander = AbbreviationExpander()

_MODALITY_TO_CATEGORY: dict[str, str] = {
    "chest_xray": "chest_xray",
    "cxr": "chest_xray",
    "chest_x-ray": "chest_xray",
    "chest_ct": "chest_ct",
    "brain_ct": "brain_ct",
    "head_ct": "brain_ct",
    "brain_mri": "brain_mri",
    "abdominal_ct": "abdominal_ct",
    "abdomen_ct": "abdominal_ct",
}
_DEFAULT_CATEGORY = "chest_xray"


class LiteratureSearchTool(Tool):
    name = "literature_search"
    description = (
        "Search clinical guidelines and PubMed for evidence related to a query. "
        "Returns matched guidelines and articles ranked by relevance and stores "
        "them as RetrievedEvidence records."
    )
    input_schema: dict = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural-language query to search (e.g. 'management of solitary pulmonary nodule').",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results to return.",
                "default": 5,
                "minimum": 1,
                "maximum": 30,
            },
        },
        "required": ["query"],
        "additionalProperties": False,
    }
    is_read_only = False
    is_concurrency_safe = False
    max_result_size_chars = 30000

    def __init__(
        self,
        remote_base_url: str | None = None,
        timeout: float = 10.0,
        pubmed_client: PubMedClient | None = None,
    ) -> None:
        self._base_url = remote_base_url.rstrip("/") if remote_base_url else None
        self._timeout = timeout
        self._pubmed_client = pubmed_client
        self._evidence_counter = 0

    def _next_evidence_id(self) -> str:
        self._evidence_counter += 1
        return f"LIT_{self._evidence_counter:03d}"

    def prompt(self) -> str:
        return (
            "Use literature_search to find clinical guidelines and PubMed articles "
            "relevant to a specific radiologic finding or clinical question."
        )

    def validate_input(self, args: dict) -> ValidationResult:
        query = args.get("query")
        if not query or not isinstance(query, str) or not query.strip():
            return ValidationResult(valid=False, message="query must be a non-empty string")
        max_r = args.get("max_results", 5)
        if not isinstance(max_r, int) or max_r < 1:
            return ValidationResult(valid=False, message="max_results must be a positive integer")
        return ValidationResult(valid=True)

    async def call(self, args: dict, context: ToolUseContext) -> ToolResult:
        query = _expander.expand_query(args["query"].strip())
        max_results = args.get("max_results", 5)
        pipeline_state = context.extra.get("pipeline_state")

        modality = (pipeline_state.modality if pipeline_state else "") or ""
        category = _MODALITY_TO_CATEGORY.get(modality.lower(), _DEFAULT_CATEGORY)

        async def _search_guidelines() -> list[dict]:
            if not self._base_url:
                logger.info("literature_search: no remote_base_url configured")
                return []
            try:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    resp = await client.post(
                        f"{self._base_url}/search/guidelines",
                        json={"query": query, "category": category, "max_results": max_results},
                    )
                    resp.raise_for_status()
                    data = resp.json()
            except Exception as exc:
                logger.warning("Guidelines remote search failed: %s", exc)
                return []

            gl_results = []
            for hit in data.get("results", []):
                eid = self._next_evidence_id()
                gl_results.append({
                    "evidence_id": eid,
                    "chunk_id": hit.get("chunk_id", ""),
                    "doc_id": hit.get("doc_id", ""),
                    "title": hit.get("title", ""),
                    "section": hit.get("section", ""),
                    "content": hit.get("text", ""),
                    "relevance_score": hit.get("score", 0.0),
                    "retrieval_mode": "remote_guidelines",
                })
                if pipeline_state is not None:
                    pipeline_state.retrieved_evidence.append(RetrievedEvidence(
                        evidence_id=eid,
                        source_type="guideline",
                        content=hit.get("text", ""),
                        relevance_score=hit.get("score", 0.0),
                        metadata={
                            "chunk_id": hit.get("chunk_id", ""),
                            "doc_id": hit.get("doc_id", ""),
                            "title": hit.get("title", ""),
                            "section": hit.get("section", ""),
                        },
                    ))
            return gl_results

        async def _search_pubmed() -> list[dict]:
            if not self._pubmed_client:
                return []
            try:
                pm_hits = await self._pubmed_client.search(query, max_results)
                pm_results = []
                for hit in pm_hits:
                    eid = self._next_evidence_id()
                    pm_results.append({
                        "evidence_id": eid,
                        "pmid": hit.metadata.get("pmid", ""),
                        "title": hit.preferred_term,
                        "citation": hit.metadata.get("citation", ""),
                        "abstract_preview": hit.content[:300],
                        "relevance_score": hit.score,
                        "retrieval_mode": "pubmed",
                    })
                    if pipeline_state is not None:
                        pipeline_state.retrieved_evidence.append(RetrievedEvidence(
                            evidence_id=eid,
                            source_type="pubmed",
                            content=hit.content,
                            relevance_score=hit.score,
                            metadata=hit.metadata,
                        ))
                return pm_results
            except Exception as exc:
                logger.warning("PubMed search failed: %s", exc)
                return []

        gl_results, pm_results = await asyncio.gather(
            _search_guidelines(), _search_pubmed()
        )
        results = gl_results + pm_results

        output = {
            "query": query,
            "category": category,
            "total_matches": len(results),
            "results": results,
        }
        return ToolResult(data=json.dumps(output, indent=2, ensure_ascii=False))
