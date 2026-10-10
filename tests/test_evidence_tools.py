"""Test evidence retrieval tools."""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.tool import ToolUseContext
from src.core.types import PipelineState


def _make_context():
    state = PipelineState(session_id="test")
    state.visual_findings_text = "Bilateral pleural effusions. Cardiomegaly noted."
    return ToolUseContext(
        session_id="test",
        extra={
            "pipeline_state": state,
            "knowledge_dir": str(Path(__file__).resolve().parent.parent / "data" / "knowledge"),
        },
    )


@pytest.mark.asyncio
async def test_concept_lookup():
    from src.tools.evidence.concept_lookup import ConceptLookupTool

    tool = ConceptLookupTool()
    ctx = _make_context()
    result = await tool.call({"term": "pleural effusion", "terminology": "all"}, ctx)
    assert not result.is_error
    assert "pleural" in result.data.lower() or "No matching" in result.data


@pytest.mark.asyncio
async def test_literature_search():
    from src.tools.evidence.literature_search import LiteratureSearchTool

    tool = LiteratureSearchTool()
    ctx = _make_context()
    result = await tool.call({"query": "pleural effusion management", "max_results": 3}, ctx)
    assert not result.is_error


@pytest.mark.asyncio
async def test_case_retrieval():
    from src.tools.evidence.case_retrieval import CaseRetrievalTool

    tool = CaseRetrievalTool()
    ctx = _make_context()
    result = await tool.call(
        {"observations": ["cardiomegaly", "pleural effusion"], "max_results": 3},
        ctx,
    )
    assert not result.is_error


@pytest.mark.asyncio
async def test_evidence_summary():
    from src.tools.evidence.evidence_summary import EvidenceSummaryTool

    tool = EvidenceSummaryTool()
    ctx = _make_context()
    result = await tool.call({"include_conflicts": True}, ctx)
    assert not result.is_error


def test_evidence_tool_schemas():
    from src.tools.evidence import build_evidence_registry

    knowledge_dir = str(Path(__file__).resolve().parent.parent / "data" / "knowledge")
    registry = build_evidence_registry(knowledge_dir)
    schemas = registry.get_api_schemas()
    assert len(schemas) == 7


if __name__ == "__main__":
    pytest.main([__file__, "-v"])