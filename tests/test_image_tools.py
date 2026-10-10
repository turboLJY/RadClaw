"""Test visual analysis tools."""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image
from src.data.image_store import SessionImageStore
from src.core.tool import ToolUseContext
from src.core.types import PipelineState


def _make_context():
    state = PipelineState(session_id="test")
    state.original_image = Image.new("RGB", (512, 512), color=(128, 128, 128))
    store = SessionImageStore("test")
    return ToolUseContext(
        session_id="test",
        extra={"pipeline_state": state, "image_store": store},
    )


@pytest.mark.asyncio
async def test_crop_roi():
    from src.tools.visual.crop_roi import CropROITool

    tool = CropROITool()
    ctx = _make_context()
    result = await tool.call(
        {"x": 10, "y": 10, "width": 100, "height": 100, "description": "test crop"},
        ctx,
    )
    assert not result.is_error
    assert "IMG_" in result.data


@pytest.mark.asyncio
async def test_select_slice():
    from src.tools.visual.select_slice import SelectSliceTool

    tool = SelectSliceTool()
    ctx = _make_context()
    result = await tool.call({"view_type": "frontal_PA"}, ctx)
    assert not result.is_error
    assert "frontal_PA" in result.data


@pytest.mark.asyncio
async def test_link_evidence():
    from src.tools.visual.link_evidence import LinkEvidenceTool

    tool = LinkEvidenceTool()
    ctx = _make_context()
    store = ctx.extra["image_store"]
    img = Image.new("RGB", (50, 50))
    img_id = store.save(img, "test", "crop")

    result = await tool.call(
        {
            "finding_description": "Test finding",
            "image_evidence_ids": [img_id],
            "confidence": 0.8,
        },
        ctx,
    )
    assert not result.is_error

    state = ctx.extra["pipeline_state"]
    assert len(state.image_evidence) == 1
    assert state.image_evidence[0].confidence == 0.8


@pytest.mark.asyncio
async def test_multi_window():
    from src.tools.visual.multi_window import MultiWindowTool

    tool = MultiWindowTool()
    ctx = _make_context()
    result = await tool.call({"window_types": ["bone", "lung"]}, ctx)
    assert not result.is_error
    assert "bone" in result.data.lower()


def test_tool_schemas():
    from src.tools.visual import build_visual_registry

    registry = build_visual_registry()
    schemas = registry.get_api_schemas()
    assert len(schemas) == 8
    for s in schemas:
        assert s["type"] == "function"
        assert "name" in s["function"]
        assert "parameters" in s["function"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])