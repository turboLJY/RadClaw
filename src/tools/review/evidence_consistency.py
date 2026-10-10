"""Check evidence consistency: do findings match retrieved evidence?"""

from src.core.tool import Tool, ToolUseContext
from src.core.types import ToolResult


class EvidenceConsistencyTool(Tool):
    name = "check_evidence_consistency"
    description = "Verify that report findings are consistent with retrieved medical evidence"
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
            "Use check_evidence_consistency to verify findings match their "
            "linked literature and case evidence."
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
            if not f.support.retrieved_evidence:
                issues.append({
                    "finding_id": f.finding_id,
                    "issue": "No retrieved evidence linked",
                    "severity": "high",
                })

            for conflict in state.conflicts:
                related_findings = conflict.get("related_findings", [])
                if f.finding_id in related_findings or f.observation in related_findings:
                    issues.append({
                        "finding_id": f.finding_id,
                        "issue": f"Evidence conflict: {conflict.get('description', 'unspecified')}",
                        "severity": "medium",
                    })

        if not issues:
            return ToolResult(data="Evidence consistency check PASSED: all findings have supporting retrieved evidence.")

        result_lines = [f"Evidence consistency check found {len(issues)} issue(s):"]
        for i in issues:
            result_lines.append(f"  [{i['severity']}] {i['finding_id']}: {i['issue']}")
        return ToolResult(data="\n".join(result_lines))