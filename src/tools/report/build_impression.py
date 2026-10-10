"""Build a structured Impression object and add it to the report."""

import re

from src.core.tool import Tool, ToolUseContext
from src.core.types import Impression, SupportEvidence, ToolResult, ValidationResult


def _strip_leading_number(text: str) -> str:
    """Remove leading number prefix like '1. ' or '2) ' from impression text."""
    return re.sub(r"^\d+[\.\)]\s*", "", text)


class BuildImpressionTool(Tool):
    name = "build_structured_impression"
    description = "Create a structured impression (diagnostic conclusion) linked to findings and evidence"
    input_schema = {
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "The impression text"},
            "type": {
                "type": "string",
                "enum": ["diagnosis", "recommendation", "differential"],
                "description": "Impression type",
            },
            "linked_finding_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "IDs of findings this impression is based on",
            },
            "confidence": {"type": "number", "description": "Confidence score 0.0-1.0"},
            "image_evidence_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "IDs of supporting image evidence",
            },
            "retrieved_evidence_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "IDs of supporting retrieved evidence",
            },
        },
        "required": ["text", "linked_finding_ids"],
    }

    def prompt(self) -> str:
        return (
            "Use build_structured_impression to create each impression. "
            "Every impression MUST link to at least one finding via linked_finding_ids. "
            "List impressions in order of clinical significance."
        )

    def validate_input(self, args: dict) -> ValidationResult:
        if not args.get("text", "").strip():
            return ValidationResult(valid=False, message="Impression text cannot be empty")
        if not args.get("linked_finding_ids"):
            return ValidationResult(valid=False, message="Must link to at least one finding")
        return ValidationResult(valid=True)

    async def call(self, args: dict, context: ToolUseContext) -> ToolResult:
        state = context.extra.get("pipeline_state")
        if state is None:
            return ToolResult(data="Error: no pipeline state", is_error=True)

        if state.structured_report is None:
            from src.core.types import StructuredReport
            state.structured_report = StructuredReport()

        imp_num = len(state.structured_report.impressions) + 1
        impression_id = f"I{imp_num:03d}"

        linked_ids = args.get("linked_finding_ids", [])
        img_ids = args.get("image_evidence_ids", [])
        ev_ids = args.get("retrieved_evidence_ids", [])

        img_evidence = [e for e in state.image_evidence if e.image_id in img_ids]
        ret_evidence = [e for e in state.retrieved_evidence if e.evidence_id in ev_ids]

        impression = Impression(
            impression_id=impression_id,
            text=_strip_leading_number(args["text"]),
            type=args.get("type", "diagnosis"),
            linked_findings=linked_ids,
            confidence=args.get("confidence", 0.0),
            support=SupportEvidence(
                image_evidence=img_evidence,
                retrieved_evidence=ret_evidence,
            ),
        )

        state.structured_report.impressions.append(impression)

        return ToolResult(
            data=(
                f"Impression {impression_id} created:\n"
                f"  Text: {impression.text}\n"
                f"  Type: {impression.type}, Confidence: {impression.confidence}\n"
                f"  Linked findings: {linked_ids}\n"
                f"  Image evidence: {img_ids}\n"
                f"  Retrieved evidence: {ev_ids}"
            )
        )