"""Attribute responsibility for an issue to the appropriate agent."""

from src.core.tool import Tool, ToolUseContext
from src.core.types import AgentRole, ReviewIssue, ToolResult, ValidationResult


ATTRIBUTION_MAP = {
    "visual": AgentRole.VISUAL_ANALYSIS,
    "retrieval": AgentRole.EVIDENCE_RETRIEVAL,
    "generation": AgentRole.REPORT_WRITING,
    "wording": AgentRole.REPORT_WRITING,
}


class AttributeIssueTool(Tool):
    name = "attribute_responsibility"
    description = "Attribute a discovered issue to the responsible agent in the pipeline"
    input_schema = {
        "type": "object",
        "properties": {
            "issue_description": {"type": "string", "description": "Description of the issue found"},
            "issue_type": {
                "type": "string",
                "enum": [
                    "visual_inconsistency", "missing_evidence", "overclaim",
                    "underclaim", "terminology_error", "internal_contradiction",
                    "confidence_miscalibration", "missing_anatomy",
                ],
                "description": "Type of issue",
            },
            "attribution": {
                "type": "string",
                "enum": ["visual", "retrieval", "generation", "wording"],
                "description": "Which pipeline stage is responsible",
            },
            "severity": {
                "type": "string",
                "enum": ["low", "medium", "high", "critical"],
                "description": "Issue severity",
            },
            "suggested_fix": {"type": "string", "description": "Suggested fix for the issue"},
            "related_finding_id": {"type": "string", "description": "Related finding or impression ID"},
        },
        "required": ["issue_description", "issue_type", "attribution", "severity"],
    }

    def prompt(self) -> str:
        return (
            "Use attribute_responsibility for EACH issue found during review. "
            "Attribution determines which agent will re-run to fix the issue:\n"
            "  - 'visual': Visual analysis MISSED an anatomical region entirely, "
            "or fundamentally misidentified pathology (e.g., called consolidation normal). "
            "Do NOT use 'visual' for missing evidence links — that is 'generation'.\n"
            "  - 'retrieval': Evidence retrieval returned irrelevant results or missed "
            "a critical knowledge source.\n"
            "  - 'generation': Report writing failed to link available image/literature "
            "evidence to ABNORMAL findings, created unsupported claims, or added duplicate "
            "findings. Note: normal/negative findings (type='normal', e.g. 'no pneumothorax', "
            "'clear lung fields') do NOT need image or retrieved evidence links — do not "
            "flag them for missing evidence.\n"
            "  - 'wording': Terminology, confidence calibration, or phrasing issue only."
        )

    def validate_input(self, args: dict) -> ValidationResult:
        if not args.get("issue_description", "").strip():
            return ValidationResult(valid=False, message="Issue description required")
        if args.get("attribution") not in ATTRIBUTION_MAP:
            return ValidationResult(valid=False, message=f"Attribution must be one of: {list(ATTRIBUTION_MAP.keys())}")
        return ValidationResult(valid=True)

    async def call(self, args: dict, context: ToolUseContext) -> ToolResult:
        state = context.extra.get("pipeline_state")
        if state is None:
            return ToolResult(data="Error: no pipeline state", is_error=True)

        issue_num = len(state.review_issues) + 1
        attribution = args["attribution"]

        issue = ReviewIssue(
            issue_id=f"RI{issue_num:03d}",
            issue_type=args["issue_type"],
            description=args["issue_description"],
            severity=args.get("severity", "medium"),
            attribution=attribution,
            target_agent=ATTRIBUTION_MAP[attribution],
            suggested_fix=args.get("suggested_fix", ""),
        )

        state.review_issues.append(issue)

        return ToolResult(
            data=(
                f"Issue {issue.issue_id} attributed:\n"
                f"  Type: {issue.issue_type}\n"
                f"  Description: {issue.description}\n"
                f"  Attribution: {issue.attribution} → {issue.target_agent.value}\n"
                f"  Severity: {issue.severity}\n"
                f"  Suggested fix: {issue.suggested_fix}"
            )
        )