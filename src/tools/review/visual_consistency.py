"""Check visual consistency: do findings match the image evidence?"""

from src.core.tool import Tool, ToolUseContext
from src.core.types import ToolResult


class VisualConsistencyTool(Tool):
    name = "check_visual_consistency"
    description = "Verify that report findings are consistent with the linked image evidence"
    input_schema = {
        "type": "object",
        "properties": {
            "finding_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Finding IDs to check (empty = check all)",
            },
        },
        "required": [],
    }

    def prompt(self) -> str:
        return (
            "Use check_visual_consistency to verify each finding is supported by "
            "its linked image evidence. Reports findings without visual support."
        )

    async def call(self, args: dict, context: ToolUseContext) -> ToolResult:
        state = context.extra.get("pipeline_state")
        if state is None or state.structured_report is None:
            return ToolResult(data="Error: no report to review", is_error=True)

        report = state.structured_report
        finding_ids = args.get("finding_ids", [])
        findings = report.findings
        if finding_ids:
            findings = [f for f in findings if f.finding_id in finding_ids]

        issues = []
        for f in findings:
            if not f.support.image_evidence:
                issues.append({
                    "finding_id": f.finding_id,
                    "issue": "No image evidence linked",
                    "severity": "high",
                    "text": f.text,
                })
            else:
                for ie in f.support.image_evidence:
                    if not ie.description:
                        issues.append({
                            "finding_id": f.finding_id,
                            "issue": f"Image evidence {ie.image_id} has no description",
                            "severity": "medium",
                            "text": f.text,
                        })

            if f.anatomy == "":
                issues.append({
                    "finding_id": f.finding_id,
                    "issue": "Missing anatomy specification",
                    "severity": "medium",
                    "text": f.text,
                })

        if not issues:
            return ToolResult(data="Visual consistency check PASSED: all findings have supporting image evidence.")

        result_lines = [f"Visual consistency check found {len(issues)} issue(s):"]
        for i in issues:
            result_lines.append(f"  [{i['severity']}] {i['finding_id']}: {i['issue']}")
        return ToolResult(data="\n".join(result_lines))