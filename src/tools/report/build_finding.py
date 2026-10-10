"""Build a structured Finding object and add it to the report."""

from src.core.tool import Tool, ToolUseContext
from src.core.types import Finding, SupportEvidence, ImageEvidence, RetrievedEvidence, ToolResult, ValidationResult


class BuildFindingTool(Tool):
    name = "build_structured_finding"
    description = "Create a structured finding with text, anatomy, observation type, confidence, and linked evidence IDs"
    input_schema = {
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "The finding text in standard radiology language"},
            "type": {"type": "string", "enum": ["normal", "abnormal", "uncertain"], "description": "Finding type"},
            "anatomy": {"type": "string", "description": "Anatomical location (RadLex term preferred)"},
            "observation": {"type": "string", "description": "Standardized observation term"},
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
        "required": ["text", "anatomy"],
    }

    def prompt(self) -> str:
        return (
            "Use build_structured_finding to create each finding in the report. "
            "For ABNORMAL findings, include at least one image_evidence_id (IMG_*) "
            "and one retrieved_evidence_id (LIT_* or CASE_*). "
            "For NORMAL/negative findings (e.g., 'no pneumothorax', 'clear lung fields'), "
            "image_evidence_ids and retrieved_evidence_ids are optional. "
            "Use standardized RadLex anatomy terms."
        )

    def validate_input(self, args: dict) -> ValidationResult:
        if not args.get("text", "").strip():
            return ValidationResult(valid=False, message="Finding text cannot be empty")
        return ValidationResult(valid=True)

    async def call(self, args: dict, context: ToolUseContext) -> ToolResult:
        state = context.extra.get("pipeline_state")
        if state is None:
            return ToolResult(data="Error: no pipeline state", is_error=True)

        if state.structured_report is None:
            from src.core.types import StructuredReport
            state.structured_report = StructuredReport()

        finding_num = len(state.structured_report.findings) + 1
        finding_id = f"F{finding_num:03d}"

        img_ids = args.get("image_evidence_ids", [])
        ev_ids = args.get("retrieved_evidence_ids", [])

        img_evidence = [e for e in state.image_evidence if e.image_id in img_ids]
        ret_evidence = [e for e in state.retrieved_evidence if e.evidence_id in ev_ids]

        finding = Finding(
            finding_id=finding_id,
            text=args["text"],
            type=args.get("type", "abnormal"),
            anatomy=args.get("anatomy", ""),
            observation=args.get("observation", ""),
            confidence=args.get("confidence", 0.0),
            support=SupportEvidence(
                image_evidence=img_evidence,
                retrieved_evidence=ret_evidence,
            ),
        )

        state.structured_report.findings.append(finding)

        return ToolResult(
            data=(
                f"Finding {finding_id} created:\n"
                f"  Text: {finding.text}\n"
                f"  Anatomy: {finding.anatomy}\n"
                f"  Type: {finding.type}, Confidence: {finding.confidence}\n"
                f"  Image evidence: {img_ids}\n"
                f"  Retrieved evidence: {ev_ids}"
            )
        )