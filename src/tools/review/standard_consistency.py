"""Check medical standard consistency: terminology and radiology norms."""

from src.core.tool import Tool, ToolUseContext
from src.core.types import ToolResult


HEDGING_TERMS = {
    "may", "might", "could", "possibly", "likely", "probable", "suggests",
    "consistent with", "cannot exclude", "concerning for", "suspicious",
}

REQUIRED_ANATOMY_ZONES = {
    "cardiac", "heart", "mediastin", "lung", "pleura",
    "diaphragm", "bone", "rib", "airway", "trachea",
}


class StandardConsistencyTool(Tool):
    name = "check_medical_standard"
    description = "Verify terminology correctness, report structure, and adherence to radiology norms"
    input_schema = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    def prompt(self) -> str:
        return (
            "Use check_medical_standard to verify the report follows radiology "
            "conventions: proper terminology, hedging language for uncertain findings, "
            "and systematic anatomical coverage."
        )

    async def call(self, args: dict, context: ToolUseContext) -> ToolResult:
        state = context.extra.get("pipeline_state")
        if state is None or state.structured_report is None:
            return ToolResult(data="Error: no report to review", is_error=True)

        report = state.structured_report
        issues = []

        # Check systematic anatomy coverage
        all_text = " ".join(f.text.lower() for f in report.findings)
        all_text += " " + " ".join(f.anatomy.lower() for f in report.findings)
        covered = set()
        for zone in REQUIRED_ANATOMY_ZONES:
            if zone in all_text:
                covered.add(zone)
        uncovered = REQUIRED_ANATOMY_ZONES - covered
        if len(uncovered) > 5:
            issues.append({
                "issue": f"Limited anatomical coverage. Zones not addressed: {', '.join(sorted(uncovered))}",
                "severity": "low",
            })

        # Check hedging on uncertain findings
        for f in report.findings:
            if f.type == "uncertain" and f.confidence < 0.5:
                text_lower = f.text.lower()
                has_hedging = any(h in text_lower for h in HEDGING_TERMS)
                if not has_hedging:
                    issues.append({
                        "issue": f"Finding {f.finding_id} is uncertain (conf={f.confidence:.2f}) but uses definitive language",
                        "severity": "medium",
                    })

        # Check: high-confidence findings should not use hedging
        for f in report.findings:
            if f.confidence >= 0.9:
                text_lower = f.text.lower()
                if any(h in text_lower for h in {"possibly", "might", "may"}):
                    issues.append({
                        "issue": f"Finding {f.finding_id} has high confidence ({f.confidence:.2f}) but uses hedging language",
                        "severity": "low",
                    })

        # Check report has both findings and impressions
        if not report.findings:
            issues.append({"issue": "Report has no findings", "severity": "critical"})
        if not report.impressions:
            issues.append({"issue": "Report has no impressions", "severity": "critical"})

        if not issues:
            return ToolResult(data="Medical standard check PASSED: report follows radiology conventions.")

        result_lines = [f"Medical standard check found {len(issues)} issue(s):"]
        for i in issues:
            result_lines.append(f"  [{i['severity']}] {i['issue']}")
        return ToolResult(data="\n".join(result_lines))