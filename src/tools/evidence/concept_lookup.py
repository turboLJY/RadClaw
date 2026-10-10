"""
ConceptLookupTool — search RadLex and/or SNOMED CT for matching concepts.
Prefers FAISS vector search when available, falls back to fuzzy matching
against local JSON knowledge bases.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from src.core.tool import Tool, ToolUseContext
from src.core.types import ToolResult, ValidationResult
from src.kb.abbreviation_expander import AbbreviationExpander

if TYPE_CHECKING:
    from src.kb.base import KnowledgeSource

logger = logging.getLogger(__name__)
_expander = AbbreviationExpander()


class ConceptLookupTool(Tool):
    name = "concept_lookup"
    description = (
        "Search local RadLex and/or SNOMED CT terminologies for concepts "
        "matching a query term.  Returns matching concepts with codes, "
        "definitions, synonyms, and relationships."
    )
    input_schema: dict = {
        "type": "object",
        "properties": {
            "term": {
                "type": "string",
                "description": "The term to look up (e.g. 'pneumothorax').",
            },
            "terminology": {
                "type": "string",
                "enum": ["radlex", "snomed", "all"],
                "description": "Which terminology to search: 'radlex', 'snomed', or 'all'.",
                "default": "all",
            },
        },
        "required": ["term"],
        "additionalProperties": False,
    }
    is_read_only = True
    is_concurrency_safe = True

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
            "Use concept_lookup to search RadLex and SNOMED CT for medical "
            "concepts by name.  You can restrict the search to one terminology "
            "or search both.  Useful for getting definitions, codes, and "
            "relationships for a specific finding or anatomy term."
        )

    def validate_input(self, args: dict) -> ValidationResult:
        term = args.get("term")
        if not term or not isinstance(term, str) or not term.strip():
            return ValidationResult(valid=False, message="term must be a non-empty string")
        terminology = args.get("terminology", "all")
        if terminology not in ("radlex", "snomed", "all"):
            return ValidationResult(
                valid=False,
                message="terminology must be 'radlex', 'snomed', or 'all'",
            )
        return ValidationResult(valid=True)

    async def _vector_search_radlex(self, query: str, max_results: int = 8) -> list[dict]:
        try:
            if not (self._radlex_source and self._radlex_source.is_available()):
                return []
            hits = await self._radlex_source.search(query, max_results=max_results)
            return [
                {
                    "source": "radlex",
                    "code": h.source_id,
                    "preferred_name": h.preferred_term,
                    "synonyms": h.metadata.get("synonyms", [])[:3],
                    "definition": h.content[:200],
                    "category": h.metadata.get("category", ""),
                    "match_score": round(h.score, 3),
                }
                for h in hits
            ]
        except Exception as e:
            logger.warning("RadLex vector search failed: %s", e)
            return []

    async def _vector_search_snomed(self, query: str, max_results: int = 8) -> list[dict]:
        try:
            if not (self._snomed_source and self._snomed_source.is_available()):
                return []
            hits = await self._snomed_source.search(query, max_results=max_results)
            return [
                {
                    "source": "snomed",
                    "sctid": h.source_id,
                    "term": h.preferred_term,
                    "synonyms": h.metadata.get("synonyms", [])[:3],
                    "definition": h.content[:200],
                    "semantic_type": h.metadata.get("semantic_type", ""),
                    "match_score": round(h.score, 3),
                }
                for h in hits
            ]
        except Exception as e:
            logger.warning("SNOMED vector search failed: %s", e)
            return []

    async def call(self, args: dict, context: ToolUseContext) -> ToolResult:
        term = _expander.expand_query(args["term"].strip())
        terminology = args.get("terminology", "all")

        results: list[dict] = []
        if terminology in ("radlex", "all"):
            results.extend(await self._vector_search_radlex(term))
        if terminology in ("snomed", "all"):
            results.extend(await self._vector_search_snomed(term))

        results.sort(key=lambda x: x.get("match_score", 0), reverse=True)

        output = {
            "query": term,
            "terminology": terminology,
            "total_matches": len(results),
            "results": results,
        }
        return ToolResult(data=json.dumps(output, indent=2, ensure_ascii=False))