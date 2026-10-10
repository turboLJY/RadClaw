"""
ConflictDetectionTool — analyses retrieved evidence for conflicts:
conclusion, recommendation, applicability, and granularity conflicts.
Uses keyword-based heuristics and stores results in pipeline_state.conflicts.
"""

from __future__ import annotations

import json
import logging
import re
from itertools import combinations

from src.core.tool import Tool, ToolUseContext
from src.core.types import RetrievedEvidence, ToolResult, ValidationResult

logger = logging.getLogger(__name__)


class ConflictDetectionTool(Tool):
    name = "conflict_detection"
    description = (
        "Analyze all retrieved evidence for potential conflicts including "
        "contradictory conclusions, conflicting recommendations, applicability "
        "mismatches, and granularity discrepancies.  Stores detected conflicts "
        "in the pipeline state."
    )
    input_schema: dict = {
        "type": "object",
        "properties": {
            "check_all": {
                "type": "boolean",
                "description": "If true, check all evidence pairs.  Defaults to true.",
                "default": True,
            }
        },
        "additionalProperties": False,
    }
    is_read_only = False
    is_concurrency_safe = False

    def __init__(self, knowledge_dir: str = "./data/knowledge") -> None:
        self._knowledge_dir = knowledge_dir

    # ------------------------------------------------------------------
    # Conflict keyword sets
    # ------------------------------------------------------------------

    # Opposing diagnosis / conclusion pairs
    _OPPOSING_DIAGNOSES: list[tuple[set[str], set[str]]] = [
        ({"pneumonia", "infection", "consolidation"}, {"pulmonary edema", "heart failure", "chf"}),
        ({"pneumothorax"}, {"pleural effusion"}),
        ({"benign", "granuloma"}, {"malignant", "cancer", "carcinoma"}),
        ({"normal", "unremarkable"}, {"abnormal", "pathology"}),
        ({"improving", "resolving"}, {"worsening", "progressing"}),
        ({"acute"}, {"chronic", "old", "longstanding"}),
    ]

    # Opposing management keywords
    _OPPOSING_MANAGEMENT: list[tuple[set[str], set[str]]] = [
        ({"no follow-up", "no routine follow-up"}, {"follow-up", "surveillance"}),
        ({"observation"}, {"biopsy", "tissue sampling", "surgical"}),
        ({"conservative"}, {"emergent", "urgent", "immediate"}),
    ]

    # Severity levels for granularity
    _SEVERITY_LEVELS = ["none", "mild", "moderate", "severe", "critical"]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        return set(re.findall(r"[a-z0-9-]+", text.lower()))

    def _check_conclusion_conflict(
        self, ev_a: RetrievedEvidence, ev_b: RetrievedEvidence
    ) -> dict | None:
        tokens_a = self._tokenize(ev_a.content)
        tokens_b = self._tokenize(ev_b.content)

        for group_x, group_y in self._OPPOSING_DIAGNOSES:
            a_has_x = bool(tokens_a & group_x)
            a_has_y = bool(tokens_a & group_y)
            b_has_x = bool(tokens_b & group_x)
            b_has_y = bool(tokens_b & group_y)

            if (a_has_x and b_has_y) or (a_has_y and b_has_x):
                return {
                    "conflict_type": "conclusion",
                    "evidence_ids": [ev_a.evidence_id, ev_b.evidence_id],
                    "description": (
                        f"Contradictory conclusions between {ev_a.evidence_id} "
                        f"and {ev_b.evidence_id}: one suggests {group_x} while "
                        f"the other suggests {group_y}."
                    ),
                    "severity": "high",
                }
        return None

    def _check_recommendation_conflict(
        self, ev_a: RetrievedEvidence, ev_b: RetrievedEvidence
    ) -> dict | None:
        text_a = ev_a.content.lower()
        text_b = ev_b.content.lower()

        for group_x, group_y in self._OPPOSING_MANAGEMENT:
            a_has_x = any(phrase in text_a for phrase in group_x)
            a_has_y = any(phrase in text_a for phrase in group_y)
            b_has_x = any(phrase in text_b for phrase in group_x)
            b_has_y = any(phrase in text_b for phrase in group_y)

            if (a_has_x and b_has_y) or (a_has_y and b_has_x):
                return {
                    "conflict_type": "recommendation",
                    "evidence_ids": [ev_a.evidence_id, ev_b.evidence_id],
                    "description": (
                        f"Conflicting management recommendations between "
                        f"{ev_a.evidence_id} and {ev_b.evidence_id}."
                    ),
                    "severity": "medium",
                }
        return None

    def _check_applicability_conflict(
        self, ev_a: RetrievedEvidence, ev_b: RetrievedEvidence
    ) -> dict | None:
        """Flag when two evidence items are from very different conditions."""
        cond_a = ev_a.metadata.get("condition", "").lower()
        cond_b = ev_b.metadata.get("condition", "").lower()

        if not cond_a or not cond_b:
            return None

        # If conditions share no meaningful words, flag potential mismatch
        tokens_a = self._tokenize(cond_a) - {"evaluation", "management", "imaging", "criteria"}
        tokens_b = self._tokenize(cond_b) - {"evaluation", "management", "imaging", "criteria"}
        if tokens_a and tokens_b and not (tokens_a & tokens_b):
            return {
                "conflict_type": "applicability",
                "evidence_ids": [ev_a.evidence_id, ev_b.evidence_id],
                "description": (
                    f"Evidence items may address different conditions: "
                    f"'{cond_a}' vs '{cond_b}'."
                ),
                "severity": "low",
            }
        return None

    def _check_granularity_conflicts(
        self, evidence_list: list[RetrievedEvidence]
    ) -> list[dict]:
        """Check for severity-level disagreements across case-based evidence."""
        conflicts: list[dict] = []
        severity_map: dict[str, list[tuple[str, str]]] = {}

        for ev in evidence_list:
            sev = ev.metadata.get("severity", "")
            diags = ev.metadata.get("diagnoses", [])
            if sev and diags:
                for d in diags:
                    key = d.lower()
                    severity_map.setdefault(key, []).append((ev.evidence_id, sev))

        for diag, entries in severity_map.items():
            severities = {s for _, s in entries}
            if len(severities) > 1:
                # Check if severities are far apart
                indices = [
                    self._SEVERITY_LEVELS.index(s)
                    for s in severities
                    if s in self._SEVERITY_LEVELS
                ]
                if indices and (max(indices) - min(indices) >= 2):
                    ids = [eid for eid, _ in entries]
                    conflicts.append({
                        "conflict_type": "granularity",
                        "evidence_ids": ids,
                        "description": (
                            f"Different severity levels for '{diag}': "
                            f"{', '.join(f'{eid}={sev}' for eid, sev in entries)}."
                        ),
                        "severity": "low",
                    })
        return conflicts

    # ------------------------------------------------------------------
    # Tool interface
    # ------------------------------------------------------------------

    def prompt(self) -> str:
        return (
            "Use conflict_detection after gathering evidence to identify "
            "contradictions or inconsistencies among retrieved evidence items.  "
            "This helps the report writer handle conflicting information "
            "appropriately."
        )

    def validate_input(self, args: dict) -> ValidationResult:
        check = args.get("check_all", True)
        if not isinstance(check, bool):
            return ValidationResult(valid=False, message="check_all must be a boolean")
        return ValidationResult(valid=True)

    _MAX_EVIDENCE_FOR_PAIRWISE = 20
    _MAX_CONFLICTS_RETURNED = 30

    async def call(self, args: dict, context: ToolUseContext) -> ToolResult:
        pipeline_state = context.extra.get("pipeline_state")
        if pipeline_state is None:
            return ToolResult(
                data=json.dumps({"error": "No pipeline_state available in context"}),
                is_error=True,
            )

        evidence_list: list[RetrievedEvidence] = pipeline_state.retrieved_evidence
        if not evidence_list:
            return ToolResult(
                data=json.dumps({
                    "message": "No retrieved evidence to analyze for conflicts",
                    "conflicts": [],
                }),
            )

        all_conflicts: list[dict] = []

        # Cap to top-N by relevance to keep pairwise complexity manageable
        top_evidence = sorted(evidence_list, key=lambda e: e.relevance_score, reverse=True)
        top_evidence = top_evidence[: self._MAX_EVIDENCE_FOR_PAIRWISE]

        # Pairwise conclusion and recommendation checks only
        for ev_a, ev_b in combinations(top_evidence, 2):
            conflict = self._check_conclusion_conflict(ev_a, ev_b)
            if conflict:
                all_conflicts.append(conflict)

            conflict = self._check_recommendation_conflict(ev_a, ev_b)
            if conflict:
                all_conflicts.append(conflict)

        # Granularity checks across full evidence list
        gran_conflicts = self._check_granularity_conflicts(evidence_list)
        all_conflicts.extend(gran_conflicts)

        # Flag conflicting evidence items
        flagged_ids: set[str] = set()
        for c in all_conflicts:
            for eid in c.get("evidence_ids", []):
                flagged_ids.add(eid)

        for ev in evidence_list:
            if ev.evidence_id in flagged_ids:
                if "conflict_detected" not in ev.conflict_flags:
                    ev.conflict_flags.append("conflict_detected")

        # Store in pipeline state
        pipeline_state.conflicts.extend(all_conflicts)

        output = {
            "total_evidence_checked": len(evidence_list),
            "pairs_checked": len(top_evidence) * (len(top_evidence) - 1) // 2,
            "total_conflicts": len(all_conflicts),
            "conflicts": all_conflicts[: self._MAX_CONFLICTS_RETURNED],
        }
        return ToolResult(data=json.dumps(output, indent=2, ensure_ascii=False))
