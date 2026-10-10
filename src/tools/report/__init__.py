from src.core.tool import ToolRegistry


def build_report_registry() -> ToolRegistry:
    from src.tools.report.build_finding import BuildFindingTool
    from src.tools.report.build_impression import BuildImpressionTool
    from src.tools.report.link_support import LinkSupportTool
    from src.tools.report.assemble_report import AssembleReportTool
    from src.tools.report.render_report import RenderReportTool

    registry = ToolRegistry()
    registry.register(BuildFindingTool())
    registry.register(BuildImpressionTool())
    registry.register(LinkSupportTool())
    registry.register(AssembleReportTool())
    registry.register(RenderReportTool())
    return registry