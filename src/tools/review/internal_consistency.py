"""Check internal consistency: findings vs impressions coherence."""

from src.core.tool import Tool, ToolUseContext
from src.core.types import ToolResult


class InternalConsistencyTool(Tool):
    name = "check_internal_consistency"
    description = "Verify that findings and impressions are internally consistent and logically coherent"
    input_schema = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    def prompt(self) -> str:
        return (
            "Use check_internal_consistency to verify findings and impressions "
            "support each other without contradictions."
        )

    async def call(self, args: dict, context: ToolUseContext) -> ToolResult:
        state = context.extra.get("pipeline_state")
        if state is None or state.structured_report is None:
            return ToolResult(data="Error: no report to review", is_error=True)

        report = state.structured_report
        issues = []

        # Check: every impression links to at least one finding
        finding_ids = {f.finding_id for f in report.findings}
        for imp in report.impressions:
            if not imp.linked_findings:
                issues.append({
                    "target": imp.impression_id,
                    "issue": "Impression not linked to any finding",
                    "severity": "high",
                })
            else:
                for fid in imp.linked_findings:
                    if fid not in finding_ids:
                        issues.append({
                            "target": imp.impression_id,
                            "issue": f"Links to non-existent finding {fid}",
                            "severity": "high",
                        })

        # Check: no orphan findings (every finding referenced by at least one impression)
        referenced = set()
        for imp in report.impressions:
            referenced.update(imp.linked_findings)
        for f in report.findings:
            if f.finding_id not in referenced and f.type == "abnormal":
                issues.append({
                    "target": f.finding_id,
                    "issue": "Abnormal finding not referenced by any impression",
                    "severity": "medium",
                })

        # Check: confidence alignment
        for imp in report.impressions:
            linked = [f for f in report.findings if f.finding_id in imp.linked_findings]
            if linked and imp.confidence > 0:
                avg_finding_conf = sum(f.confidence for f in linked if f.confidence > 0)
                count = sum(1 for f in linked if f.confidence > 0)
                if count > 0:
                    avg_finding_conf /= count
                    if imp.confidence > avg_finding_conf + 0.3:
                        issues.append({
                            "target": imp.impression_id,
                            "issue": f"Impression confidence ({imp.confidence:.2f}) much higher than supporting findings ({avg_finding_conf:.2f})",
                            "severity": "medium",
                        })

        if not issues:
            return ToolResult(data="Internal consistency check PASSED: findings and impressions are coherent.")

        result_lines = [f"Internal consistency check found {len(issues)} issue(s):"]
        for i in issues:
            result_lines.append(f"  [{i['severity']}] {i['target']}: {i['issue']}")
        return ToolResult(data="\n".join(result_lines))