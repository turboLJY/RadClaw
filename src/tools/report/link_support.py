"""Link additional evidence to an existing finding or impression."""

from src.core.tool import Tool, ToolUseContext
from src.core.types import ToolResult, ValidationResult


class LinkSupportTool(Tool):
    name = "link_support_evidence"
    description = "Link additional image or retrieved evidence to an existing finding or impression"
    input_schema = {
        "type": "object",
        "properties": {
            "target_id": {
                "type": "string",
                "description": "Finding ID (F001) or Impression ID (I001) to link evidence to",
            },
            "image_evidence_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Additional image evidence IDs to link",
            },
            "retrieved_evidence_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Additional retrieved evidence IDs to link",
            },
        },
        "required": ["target_id"],
    }

    def prompt(self) -> str:
        return (
            "Use link_support_evidence to add additional evidence to a finding "
            "or impression after it has been created."
        )

    async def call(self, args: dict, context: ToolUseContext) -> ToolResult:
        state = context.extra.get("pipeline_state")
        if state is None or state.structured_report is None:
            return ToolResult(data="Error: no report in pipeline state", is_error=True)

        target_id = args["target_id"]
        new_img_ids = args.get("image_evidence_ids", [])
        new_ev_ids = args.get("retrieved_evidence_ids", [])

        target = None
        for f in state.structured_report.findings:
            if f.finding_id == target_id:
                target = f
                break
        if target is None:
            for imp in state.structured_report.impressions:
                if imp.impression_id == target_id:
                    target = imp
                    break

        if target is None:
            return ToolResult(data=f"Error: target {target_id} not found", is_error=True)

        added_img = []
        for img_id in new_img_ids:
            ev = next((e for e in state.image_evidence if e.image_id == img_id), None)
            if ev and ev not in target.support.image_evidence:
                target.support.image_evidence.append(ev)
                added_img.append(img_id)

        added_ev = []
        for ev_id in new_ev_ids:
            ev = next((e for e in state.retrieved_evidence if e.evidence_id == ev_id), None)
            if ev and ev not in target.support.retrieved_evidence:
                target.support.retrieved_evidence.append(ev)
                added_ev.append(ev_id)

        return ToolResult(
            data=(
                f"Evidence linked to {target_id}:\n"
                f"  Added image evidence: {added_img}\n"
                f"  Added retrieved evidence: {added_ev}"
            )
        )