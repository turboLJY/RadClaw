"""Assemble all findings and impressions into a complete StructuredReport."""

from src.core.tool import Tool, ToolUseContext
from src.core.types import ToolResult, ValidationResult


class AssembleReportTool(Tool):
    name = "assemble_report_object"
    description = "Assemble all created findings and impressions into a complete structured report object"
    input_schema = {
        "type": "object",
        "properties": {
            "clinical_indication": {
                "type": "string",
                "description": "Clinical indication for the study (if known)",
            },
        },
        "required": [],
    }

    def prompt(self) -> str:
        return (
            "Use assemble_report_object after creating all findings and impressions. "
            "This finalizes the structured report with exam info and metadata."
        )

    async def call(self, args: dict, context: ToolUseContext) -> ToolResult:
        state = context.extra.get("pipeline_state")
        if state is None:
            return ToolResult(data="Error: no pipeline state", is_error=True)

        if state.structured_report is None:
            return ToolResult(data="Error: no findings or impressions created yet", is_error=True)

        report = state.structured_report

        report.exam_info.update(state.exam_info)
        if args.get("clinical_indication"):
            report.exam_info["clinical_indication"] = args["clinical_indication"]

        report.metadata = {
            "pipeline_iterations": state.iteration_count,
            "total_image_evidence": len(state.image_evidence),
            "total_retrieved_evidence": len(state.retrieved_evidence),
            "total_findings": len(report.findings),
            "total_impressions": len(report.impressions),
            "conflicts_detected": len(state.conflicts),
        }

        issues = []
        for f in report.findings:
            if not f.support.image_evidence and not f.support.retrieved_evidence:
                issues.append(f"Finding {f.finding_id} has no supporting evidence")
        for imp in report.impressions:
            if not imp.linked_findings:
                issues.append(f"Impression {imp.impression_id} not linked to any findings")

        summary = (
            f"Report assembled:\n"
            f"  Exam: {report.exam_info.get('study_type', 'unknown')}\n"
            f"  Findings: {len(report.findings)}\n"
            f"  Impressions: {len(report.impressions)}\n"
            f"  Image evidence: {report.metadata['total_image_evidence']}\n"
            f"  Retrieved evidence: {report.metadata['total_retrieved_evidence']}"
        )

        if issues:
            summary += "\n\nWarnings:\n" + "\n".join(f"  - {i}" for i in issues)

        # Compact findings listing (IDs + short description + evidence counts)
        if report.findings:
            summary += "\n\nFindings:"
            for f in report.findings:
                n_img = len(f.support.image_evidence) if f.support else 0
                n_ev = len(f.support.retrieved_evidence) if f.support else 0
                text = (f.text or "")[:120]
                summary += f"\n  {f.finding_id} [{f.type}] {f.anatomy}: {text} (img:{n_img} ev:{n_ev})"

        if report.impressions:
            summary += "\n\nImpressions:"
            for imp in report.impressions:
                text = (imp.text or "")[:120]
                summary += f"\n  {imp.impression_id}: {text}"

        return ToolResult(data=summary)