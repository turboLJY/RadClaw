"""
EvidenceSummaryTool — compiles all retrieved evidence into a structured
summary text including concept definitions, guideline references,
case comparisons, and detected conflicts.
"""

from __future__ import annotations

import json
import logging

from src.core.tool import Tool, ToolUseContext
from src.core.types import RetrievedEvidence, ToolResult, ValidationResult

logger = logging.getLogger(__name__)


class EvidenceSummaryTool(Tool):
    name = "evidence_summary"
    description = (
        "Compile all retrieved evidence into a structured summary.  Includes "
        "concept definitions, guideline references, case comparisons, and "
        "(optionally) detected conflicts.  Stores the summary in "
        "pipeline_state.evidence_summary."
    )
    input_schema: dict = {
        "type": "object",
        "properties": {
            "include_conflicts": {
                "type": "boolean",
                "description": "Whether to include conflict information in the summary.  Defaults to true.",
                "default": True,
            }
        },
        "additionalProperties": False,
    }
    is_read_only = False
    is_concurrency_safe = False
    max_result_size_chars = 30000

    def __init__(self, knowledge_dir: str = "./data/knowledge") -> None:
        self._knowledge_dir = knowledge_dir

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_concept_section(concepts: list[RetrievedEvidence]) -> str:
        if not concepts:
            return ""
        lines = ["## Concept Definitions", ""]
        for ev in concepts:
            lines.append(f"- **{ev.evidence_id}** (score {ev.relevance_score:.2f}): {ev.content}")
        lines.append("")
        return "\n".join(lines)

    @staticmethod
    def _format_literature_section(literature: list[RetrievedEvidence]) -> str:
        if not literature:
            return ""
        lines = ["## Clinical Guidelines & Literature", ""]
        for ev in literature:
            source = ev.metadata.get("source", "Unknown")
            condition = ev.metadata.get("condition", "")
            refs = ev.metadata.get("references", [])
            lines.append(f"- **{ev.evidence_id}** [{source}] {condition} (score {ev.relevance_score:.2f})")
            content_preview = ev.content[:300]
            if len(ev.content) > 300:
                content_preview += "..."
            lines.append(f"  {content_preview}")
            if refs:
                lines.append(f"  References: {'; '.join(refs)}")
        lines.append("")
        return "\n".join(lines)

    @staticmethod
    def _format_pubmed_section(pubmed: list[RetrievedEvidence]) -> str:
        if not pubmed:
            return ""
        lines = ["## PubMed Literature", ""]
        for ev in pubmed:
            citation = ev.metadata.get("citation", "")
            pmid = ev.metadata.get("pmid", "")
            title = ev.metadata.get("title", ev.preferred_term if hasattr(ev, "preferred_term") else "")
            lines.append(f"- **{ev.evidence_id}** PMID:{pmid} (score {ev.relevance_score:.2f})")
            if citation:
                lines.append(f"  {citation}")
            content_preview = ev.content[:300]
            if len(ev.content) > 300:
                content_preview += "..."
            lines.append(f"  {content_preview}")
        lines.append("")
        return "\n".join(lines)

    @staticmethod
    def _format_guideline_section(guidelines: list[RetrievedEvidence]) -> str:
        if not guidelines:
            return ""
        lines = ["## Structured Guidelines", ""]
        for ev in guidelines:
            source = ev.metadata.get("source", "guideline")
            condition = ev.metadata.get("condition", "")
            lines.append(f"- **{ev.evidence_id}** [{source}] {condition} (score {ev.relevance_score:.2f})")
            content_preview = ev.content[:300]
            if len(ev.content) > 300:
                content_preview += "..."
            lines.append(f"  {content_preview}")
            refs = ev.metadata.get("references", [])
            if refs:
                lines.append(f"  References: {'; '.join(refs)}")
        lines.append("")
        return "\n".join(lines)

    @staticmethod
    def _format_case_section(cases: list[RetrievedEvidence]) -> str:
        if not cases:
            return ""
        lines = ["## Case Comparisons", ""]
        for ev in cases:
            scenario = ev.metadata.get("clinical_scenario", "")
            diagnoses = ev.metadata.get("diagnoses", [])
            severity = ev.metadata.get("severity", "")
            lines.append(
                f"- **{ev.evidence_id}** (score {ev.relevance_score:.2f}) "
                f"— {scenario}"
            )
            if diagnoses:
                lines.append(f"  Diagnoses: {', '.join(diagnoses)}")
            if severity:
                lines.append(f"  Severity: {severity}")
            content_preview = ev.content[:250]
            if len(ev.content) > 250:
                content_preview += "..."
            lines.append(f"  {content_preview}")
        lines.append("")
        return "\n".join(lines)

    @staticmethod
    def _format_conflict_section(conflicts: list[dict]) -> str:
        if not conflicts:
            return ""
        lines = ["## Detected Conflicts", ""]
        for idx, c in enumerate(conflicts, 1):
            ctype = c.get("conflict_type", "unknown")
            desc = c.get("description", "")
            sev = c.get("severity", "")
            ids = c.get("evidence_ids", [])
            lines.append(f"{idx}. **{ctype.upper()}** ({sev}): {desc}")
            lines.append(f"   Involved evidence: {', '.join(ids)}")
        lines.append("")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Tool interface
    # ------------------------------------------------------------------

    def prompt(self) -> str:
        return (
            "Use evidence_summary after all evidence has been gathered and "
            "scored to produce a structured text summary.  This summary is "
            "consumed by the report-writing agent to generate the final report."
        )

    def validate_input(self, args: dict) -> ValidationResult:
        incl = args.get("include_conflicts", True)
        if not isinstance(incl, bool):
            return ValidationResult(valid=False, message="include_conflicts must be a boolean")
        return ValidationResult(valid=True)

    async def call(self, args: dict, context: ToolUseContext) -> ToolResult:
        pipeline_state = context.extra.get("pipeline_state")
        if pipeline_state is None:
            return ToolResult(
                data=json.dumps({"error": "No pipeline_state available in context"}),
                is_error=True,
            )

        include_conflicts = args.get("include_conflicts", True)
        evidence_list: list[RetrievedEvidence] = pipeline_state.retrieved_evidence

        if not evidence_list:
            summary_text = "No evidence has been retrieved for this study."
            pipeline_state.evidence_summary = summary_text
            return ToolResult(
                data=json.dumps({
                    "summary": summary_text,
                    "evidence_count": 0,
                }),
            )

        # Partition by source type
        concepts = [e for e in evidence_list if e.source_type == "concept"]
        literature = [e for e in evidence_list if e.source_type == "literature"]
        cases = [e for e in evidence_list if e.source_type == "case"]
        pubmed = [e for e in evidence_list if e.source_type == "pubmed"]
        guidelines = [e for e in evidence_list if e.source_type == "guideline"]

        # Sort each group by relevance score descending, cap to top-N
        concepts.sort(key=lambda e: e.relevance_score, reverse=True)
        literature.sort(key=lambda e: e.relevance_score, reverse=True)
        cases.sort(key=lambda e: e.relevance_score, reverse=True)
        pubmed.sort(key=lambda e: e.relevance_score, reverse=True)
        guidelines.sort(key=lambda e: e.relevance_score, reverse=True)

        concepts = concepts[:8]
        literature = literature[:8]
        guidelines = guidelines[:10]
        pubmed = pubmed[:8]
        cases = cases[:5]

        # Build summary
        sections: list[str] = ["# Evidence Summary", ""]

        # Observation graph info
        obs_graph = pipeline_state.observation_graph
        if obs_graph:
            obs_lines = ["## Standardized Observations", ""]
            for obs in obs_graph:
                std_term = obs.get("standardized_term", obs.get("original_text", "?"))
                radlex = obs.get("radlex_code", "")
                snomed = obs.get("snomed_code", "")
                codes = []
                if radlex:
                    codes.append(f"RadLex: {radlex}")
                if snomed:
                    codes.append(f"SNOMED: {snomed}")
                code_str = f" ({', '.join(codes)})" if codes else ""
                obs_lines.append(f"- {std_term}{code_str}")
            obs_lines.append("")
            sections.append("\n".join(obs_lines))

        concept_text = self._format_concept_section(concepts)
        if concept_text:
            sections.append(concept_text)

        lit_text = self._format_literature_section(literature)
        if lit_text:
            sections.append(lit_text)

        guideline_text = self._format_guideline_section(guidelines)
        if guideline_text:
            sections.append(guideline_text)

        pubmed_text = self._format_pubmed_section(pubmed)
        if pubmed_text:
            sections.append(pubmed_text)

        case_text = self._format_case_section(cases)
        if case_text:
            sections.append(case_text)

        if include_conflicts:
            conflicts = pipeline_state.conflicts
            conflict_text = self._format_conflict_section(conflicts)
            if conflict_text:
                sections.append(conflict_text)

        # Stats
        sections.append("## Summary Statistics")
        sections.append(f"- Total evidence items: {len(evidence_list)}")
        sections.append(f"- Concepts: {len(concepts)}")
        sections.append(f"- Literature/Guidelines (legacy): {len(literature)}")
        sections.append(f"- Structured Guidelines: {len(guidelines)}")
        sections.append(f"- PubMed articles: {len(pubmed)}")
        sections.append(f"- Case comparisons: {len(cases)}")
        if include_conflicts:
            sections.append(f"- Conflicts detected: {len(pipeline_state.conflicts)}")

        summary_text = "\n".join(sections)

        # Store in pipeline state
        pipeline_state.evidence_summary = summary_text

        output = {
            "summary": summary_text,
            "evidence_count": len(evidence_list),
            "concept_count": len(concepts),
            "literature_count": len(literature),
            "guideline_count": len(guidelines),
            "pubmed_count": len(pubmed),
            "case_count": len(cases),
            "conflict_count": len(pipeline_state.conflicts) if include_conflicts else 0,
        }
        return ToolResult(data=json.dumps(output, indent=2, ensure_ascii=False))
