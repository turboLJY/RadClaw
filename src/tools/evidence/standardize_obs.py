"""
StandardizeObservationTool — maps free-text observations to structured
RadLex / SNOMED terms.  Prefers FAISS vector search when available,
falls back to fuzzy matching against local JSON knowledge bases.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.core.tool import Tool, ToolUseContext
from src.core.types import ToolResult, ValidationResult
from src.kb.abbreviation_expander import AbbreviationExpander

if TYPE_CHECKING:
    from src.kb.base import KnowledgeSource

logger = logging.getLogger(__name__)
_expander = AbbreviationExpander()


class StandardizeObservationTool(Tool):
    name = "standardize_observation"
    description = (
        "Map a free-text radiologic observation to standardized RadLex and "
        "SNOMED CT terms.  Returns the best-matching standardized term, codes, "
        "synonyms, and definition.  Stores the result in the observation graph."
    )
    input_schema: dict = {
        "type": "object",
        "properties": {
            "observation_text": {
                "type": "string",
                "description": "Free-text observation to standardize (e.g. 'fluid around the lungs').",
            }
        },
        "required": ["observation_text"],
        "additionalProperties": False,
    }
    is_read_only = False
    is_concurrency_safe = False

    def __init__(
        self,
        knowledge_dir: str = "./data/knowledge",
        radlex_source: KnowledgeSource | None = None,
        snomed_source: KnowledgeSource | None = None,
    ) -> None:
        self._knowledge_dir = Path(knowledge_dir)
        self._radlex_source = radlex_source
        self._snomed_source = snomed_source

    # ------------------------------------------------------------------
    # Tool interface
    # ------------------------------------------------------------------

    def prompt(self) -> str:
        return (
            "Use standardize_observation to map a free-text radiologic observation "
            "(e.g. 'bilateral haziness at the bases') to its standardized RadLex and "
            "SNOMED CT codes.  This is typically the first step when processing each "
            "observation from the visual analysis stage."
        )

    def validate_input(self, args: dict) -> ValidationResult:
        text = args.get("observation_text")
        if not text or not isinstance(text, str) or not text.strip():
            return ValidationResult(valid=False, message="observation_text must be a non-empty string")
        return ValidationResult(valid=True)

    async def _vector_radlex_match(self, text: str) -> tuple[dict | None, float]:
        try:
            if self._radlex_source and self._radlex_source.is_available():
                hits = await self._radlex_source.search(text, max_results=1)
                if hits:
                    h = hits[0]
                    return {
                        "preferred_name": h.preferred_term,
                        "code": h.source_id,
                        "synonyms": h.metadata.get("synonyms", []),
                        "definition": h.content,
                        "category": h.metadata.get("category", ""),
                    }, h.score
        except Exception as e:
            logger.warning("RadLex vector search failed: %s", e)
        return None, 0.0

    async def _vector_snomed_match(self, text: str) -> tuple[dict | None, float]:
        try:
            if self._snomed_source and self._snomed_source.is_available():
                hits = await self._snomed_source.search(text, max_results=1)
                if hits:
                    h = hits[0]
                    return {
                        "sctid": h.source_id,
                        "term": h.preferred_term,
                        "synonyms": h.metadata.get("synonyms", []),
                        "definition": h.content,
                    }, h.score
        except Exception as e:
            logger.warning("SNOMED vector search failed: %s", e)
        return None, 0.0

    async def call(self, args: dict, context: ToolUseContext) -> ToolResult:
        observation_text: str = _expander.expand_query(args["observation_text"].strip())

        radlex_match, radlex_score = await self._vector_radlex_match(observation_text)

        snomed_match, snomed_score = await self._vector_snomed_match(observation_text)

        result: dict[str, Any] = {
            "input_text": observation_text,
            "standardized_term": None,
            "radlex_code": None,
            "snomed_code": None,
            "synonyms": [],
            "definition": None,
            "match_scores": {"radlex": radlex_score, "snomed": snomed_score},
        }

        if radlex_match:
            result["standardized_term"] = radlex_match["preferred_name"]
            result["radlex_code"] = radlex_match["code"]
            result["synonyms"] = radlex_match.get("synonyms", [])
            result["definition"] = radlex_match.get("definition")

        if snomed_match:
            result["snomed_code"] = snomed_match["sctid"]
            if result["standardized_term"] is None:
                result["standardized_term"] = snomed_match["term"]
                result["synonyms"] = snomed_match.get("synonyms", [])
                result["definition"] = snomed_match.get("definition")

        # Store in pipeline state
        pipeline_state = context.extra.get("pipeline_state")
        if pipeline_state is not None:
            obs_node = {
                "original_text": observation_text,
                "standardized_term": result["standardized_term"],
                "radlex_code": result["radlex_code"],
                "snomed_code": result["snomed_code"],
                "synonyms": result["synonyms"],
                "definition": result["definition"],
            }
            pipeline_state.observation_graph.append(obs_node)

        return ToolResult(data=json.dumps(result, indent=2, ensure_ascii=False))
