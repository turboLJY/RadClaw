"""
CaseRetrievalTool — retrieves similar cases via remote HTTP service.
Sends joined observations as a dense query to POST /search/cases.
"""

from __future__ import annotations

import json
import logging

import httpx

from src.core.tool import Tool, ToolUseContext
from src.core.types import RetrievedEvidence, ToolResult, ValidationResult

logger = logging.getLogger(__name__)

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


class CaseRetrievalTool(Tool):
    name = "case_retrieval"
    description = (
        "Find similar cases whose findings best match a given set of observations. "
        "Returns case reports with findings, impressions, and structured findings lists."
    )
    input_schema: dict = {
        "type": "object",
        "properties": {
            "observations": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of observation terms to match against case reports.",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of cases to return.",
                "default": 5,
                "minimum": 1,
                "maximum": 50,
            },
        },
        "required": ["observations"],
        "additionalProperties": False,
    }
    is_read_only = False
    is_concurrency_safe = False

    def __init__(
        self,
        remote_base_url: str | None = None,
        timeout: float = 10.0,
    ) -> None:
        self._base_url = remote_base_url.rstrip("/") if remote_base_url else None
        self._timeout = timeout
        self._evidence_counter = 0

    def _next_evidence_id(self) -> str:
        self._evidence_counter += 1
        return f"CASE_{self._evidence_counter:03d}"

    def prompt(self) -> str:
        return (
            "Use case_retrieval to find similar cases whose findings overlap "
            "with the current observations. Helps correlate the current study "
            "with known diagnostic patterns."
        )

    def validate_input(self, args: dict) -> ValidationResult:
        obs = args.get("observations")
        if not obs or not isinstance(obs, list) or len(obs) == 0:
            return ValidationResult(valid=False, message="observations must be a non-empty list of strings")
        if not all(isinstance(o, str) for o in obs):
            return ValidationResult(valid=False, message="all observation items must be strings")
        max_r = args.get("max_results", 5)
        if not isinstance(max_r, int) or max_r < 1:
            return ValidationResult(valid=False, message="max_results must be a positive integer")
        return ValidationResult(valid=True)

    async def call(self, args: dict, context: ToolUseContext) -> ToolResult:
        observations: list[str] = args["observations"]
        max_results: int = args.get("max_results", 5)
        pipeline_state = context.extra.get("pipeline_state")

        modality = (pipeline_state.modality if pipeline_state else "") or ""
        category = _MODALITY_TO_CATEGORY.get(modality.lower(), _DEFAULT_CATEGORY)

        query = " ".join(observations)
        empty = json.dumps(
            {"observations_query": observations, "category": category, "total_matches": 0, "results": []},
            indent=2,
        )

        if not self._base_url:
            logger.info("case_retrieval: no remote_base_url configured, returning empty")
            return ToolResult(data=empty)

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(
                    f"{self._base_url}/search/cases",
                    json={"query": query, "category": category, "max_results": max_results},
                )
                resp.raise_for_status()
                data = resp.json()
        except Exception as exc:
            logger.warning(
                "Cases remote search failed: type=%s repr=%r category=%s timeout=%s base_url=%s",
                type(exc).__name__,
                exc,
                category,
                self._timeout,
                self._base_url,
            )
            return ToolResult(data=empty)

        results: list[dict] = []
        for hit in data.get("results", []):
            eid = self._next_evidence_id()
            sections = hit.get("sections", {})
            findings_text = sections.get("findings", "")
            impression_text = sections.get("impression", "")
            results.append({
                "evidence_id": eid,
                "case_id": hit.get("case_id", ""),
                "findings_text": findings_text,
                "impression_text": impression_text,
                "positive_findings": hit.get("positive_findings", []),
                "negative_findings": hit.get("negative_findings", []),
                "relevance_score": hit.get("score", 0.0),
            })
            if pipeline_state is not None:
                content = f"Findings: {findings_text} Impression: {impression_text}"
                pipeline_state.retrieved_evidence.append(RetrievedEvidence(
                    evidence_id=eid,
                    source_type="case",
                    content=content,
                    relevance_score=hit.get("score", 0.0),
                    metadata={
                        "case_id": hit.get("case_id", ""),
                        "positive_findings": hit.get("positive_findings", []),
                        "negative_findings": hit.get("negative_findings", []),
                    },
                ))

        output = {
            "observations_query": observations,
            "category": category,
            "total_matches": len(results),
            "results": results,
        }
        return ToolResult(data=json.dumps(output, indent=2, ensure_ascii=False))