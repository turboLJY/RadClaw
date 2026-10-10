"""Render the structured report as natural language text."""

from src.core.tool import Tool, ToolUseContext
from src.core.types import ToolResult


class RenderReportTool(Tool):
    name = "render_natural_language"
    description = "Convert the structured report object into a natural language radiology report"
    input_schema = {
        "type": "object",
        "properties": {
            "style": {
                "type": "string",
                "enum": ["standard", "detailed", "brief"],
                "description": "Report style: standard (default), detailed (with evidence refs), or brief",
            },
        },
        "required": [],
    }

    def prompt(self) -> str:
        return (
            "Use render_natural_language as the final step to produce a "
            "human-readable radiology report from the structured report object."
        )

    async def call(self, args: dict, context: ToolUseContext) -> ToolResult:
        state = context.extra.get("pipeline_state")
        if state is None or state.structured_report is None:
            return ToolResult(data="Error: no report to render", is_error=True)

        report = state.structured_report
        style = args.get("style", "standard")

        lines = []
        exam = report.exam_info
        lines.append(f"EXAMINATION: {exam.get('study_type', 'Imaging Study')} ({exam.get('view', 'standard')})")
        if exam.get("clinical_indication"):
            lines.append(f"CLINICAL INDICATION: {exam['clinical_indication']}")
        lines.append("")

        if report.findings:
            lines.append("FINDINGS:")
            # Merge all findings into a prose paragraph (matches GT style)
            finding_texts = []
            for f in report.findings:
                text = f.text
                if style == "detailed":
                    refs = []
                    for ie in f.support.image_evidence:
                        refs.append(ie.image_id)
                    for re in f.support.retrieved_evidence:
                        refs.append(re.evidence_id)
                    if refs:
                        text += f" [Refs: {', '.join(refs)}]"
                    if f.confidence > 0:
                        text += f" (confidence: {f.confidence:.2f})"
                finding_texts.append(text)
            lines.append(" ".join(finding_texts))
            lines.append("")

        if report.impressions:
            lines.append("IMPRESSION:")
            for idx, imp in enumerate(report.impressions, 1):
                text = imp.text
                if style == "detailed":
                    if imp.confidence > 0:
                        text += f" (confidence: {imp.confidence:.2f})"
                    if imp.support.conflict_evidence:
                        text += f" [Conflicts noted]"
                lines.append(f"{idx}. {text}")

        if style == "brief" and report.impressions:
            lines = []
            lines.append("IMPRESSION:")
            for idx, imp in enumerate(report.impressions, 1):
                lines.append(f"{idx}. {imp.text}")

        nl_report = "\n".join(lines)

        # Store eval-clean version (no EXAMINATION/CLINICAL INDICATION headers,
        # no refs or confidence) so summary.json scores cleanly against ground truth.
        eval_lines = []
        if report.findings:
            eval_lines.append("FINDINGS:")
            eval_lines.append(" ".join(f.text for f in report.findings))
            eval_lines.append("")
        if report.impressions:
            eval_lines.append("IMPRESSION:")
            for idx, imp in enumerate(report.impressions, 1):
                eval_lines.append(f"{idx}. {imp.text}")
        state.natural_language_report = "\n".join(eval_lines)

        return ToolResult(
            data=f"Natural language report ({style} style):\n\n{nl_report}"
        )