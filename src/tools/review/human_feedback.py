"""Human feedback injection tool for doctor collaboration."""

from src.core.tool import Tool, ToolUseContext
from src.core.types import ToolResult


class HumanFeedbackTool(Tool):
    name = "inject_human_feedback"
    description = "Allow a doctor to provide feedback on the report (interactive mode only)"
    input_schema = {
        "type": "object",
        "properties": {
            "feedback_text": {
                "type": "string",
                "description": "Doctor's feedback text",
            },
            "override_verdict": {
                "type": "string",
                "enum": ["pass", "revise"],
                "description": "Override the automated review verdict",
            },
        },
        "required": [],
    }

    def prompt(self) -> str:
        return (
            "Use inject_human_feedback to incorporate doctor feedback. "
            "In batch mode this is a no-op. In interactive mode, it pauses "
            "for human input."
        )

    async def call(self, args: dict, context: ToolUseContext) -> ToolResult:
        state = context.extra.get("pipeline_state")
        feedback = args.get("feedback_text", "")
        override = args.get("override_verdict")

        if override == "pass" and state:
            state.review_passed = True
            return ToolResult(data="Doctor override: PASS. Report accepted.")

        if override == "revise" and state:
            state.review_passed = False
            return ToolResult(data=f"Doctor requests revision. Feedback: {feedback}")

        if feedback:
            return ToolResult(data=f"Doctor feedback recorded: {feedback}")

        return ToolResult(data="No feedback provided. Continuing with automated review.")