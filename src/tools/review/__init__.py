from src.core.tool import ToolRegistry


def build_review_registry() -> ToolRegistry:
    from src.tools.review.visual_consistency import VisualConsistencyTool
    from src.tools.review.evidence_consistency import EvidenceConsistencyTool
    from src.tools.review.internal_consistency import InternalConsistencyTool
    from src.tools.review.standard_consistency import StandardConsistencyTool
    from src.tools.review.attribute_issue import AttributeIssueTool
    from src.tools.review.human_feedback import HumanFeedbackTool

    registry = ToolRegistry()
    registry.register(VisualConsistencyTool())
    registry.register(EvidenceConsistencyTool())
    registry.register(InternalConsistencyTool())
    registry.register(StandardConsistencyTool())
    registry.register(AttributeIssueTool())
    registry.register(HumanFeedbackTool())
    return registry