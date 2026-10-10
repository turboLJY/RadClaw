"""
Ablation study baselines for RadClaw.

Each function mirrors run_pipeline()'s exact interface:
    async def run_*(image, exam_info, settings, llm_client) -> AsyncGenerator[StreamEvent]

The DONE event payload matches run_pipeline() so evaluate.py and demo_server.py
work without any changes.
"""

from __future__ import annotations

import logging
import uuid
from typing import AsyncGenerator

from PIL import Image

from src.core.orchestrator import (
    MODALITY_DISPLAY,
    _build_agent_query,
    _image_to_base64,
    _resolve_modality,
    _update_pipeline_state,
)
from src.core.loop import QueryParams, query_loop
from src.core.types import AgentRole, EventType, PipelineState, StreamEvent
from src.agents.base import build_agent
from src.data.image_store import SessionImageStore
from src.llm.client import LLMClient

logger = logging.getLogger(__name__)


_DIRECT_VLM_SYSTEM = """\
You are an expert radiologist. Generate a radiology report for the provided medical image.

Your report MUST use this exact format:

FINDINGS:
  [Describe all relevant findings systematically, covering all anatomical regions visible in the image.]

IMPRESSION:
  1. [Most important clinical conclusion]
  2. [Additional conclusions or recommendations if needed]

Rules:
- Use standard radiological terminology.
- Include pertinent negatives for clinically relevant structures.
- Be concise and clinically actionable.\
"""


async def run_direct_vlm(
    image: Image.Image,
    exam_info: dict,
    settings: dict,
    llm_client: LLMClient,
) -> AsyncGenerator[StreamEvent, None]:
    """
    B1 baseline: single LLM call with image, no tools, no agents.

    Produces the same StreamEvent stream and DONE payload as run_pipeline()
    so TraceCollector, evaluate.py, and demo_server.py all work unchanged.
    """
    session_id = str(uuid.uuid4())[:8]
    modality = _resolve_modality(exam_info)
    modality_display = MODALITY_DISPLAY.get(modality, modality or "medical image")

    yield StreamEvent(
        type=EventType.STATUS,
        data={"message": f"[B1 direct_vlm] session={session_id}"},
    )
    yield StreamEvent(
        type=EventType.PIPELINE_UPDATE,
        data={"iteration": 1, "agents": ["direct_vlm"]},
    )
    yield StreamEvent(
        type=EventType.AGENT_START,
        data={"agent": "direct_vlm", "iteration": 1},
    )

    # Build user message with image + exam context
    b64 = _image_to_base64(image)
    parts: list[dict] = []

    view = exam_info.get("view", "")
    indication = exam_info.get("indication", "")
    context_lines = [f"Modality: {modality_display}"]
    if view:
        context_lines.append(f"View: {view}")
    if indication:
        context_lines.append(f"Indication: {indication}")

    parts.append({
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{b64}"},
    })
    parts.append({
        "type": "text",
        "text": (
            f"Generate a radiology report for this {modality_display}.\n"
            + "\n".join(context_lines)
        ),
    })

    messages = [{"role": "user", "content": parts}]
    text_parts: list[str] = []

    async for event in llm_client.stream(
        messages=messages,
        system_prompt=_DIRECT_VLM_SYSTEM,
        tools=None,
    ):
        if event.type == EventType.TEXT_DELTA:
            text_parts.append(event.data.get("text", ""))
        yield event

    report_text = "".join(text_parts).strip()
    logger.info("[direct_vlm] report length: %d chars", len(report_text))

    yield StreamEvent(
        type=EventType.AGENT_END,
        data={"agent": "direct_vlm", "iteration": 1},
    )

    yield StreamEvent(
        type=EventType.DONE,
        data={
            "session_id": session_id,
            "iterations": 1,
            "review_passed": True,       # not applicable; set True so summary shows PASS
            "image_evidence_count": 0,
            "retrieved_evidence_count": 0,
            "natural_language_report": report_text,
            "structured_report": None,
            "image_evidence": [],
            "retrieved_evidence": [],
            "agent_run_log": [
                {
                    "agent": "direct_vlm",
                    "iteration": 1,
                    "output_length": len(report_text),
                }
            ],
        },
    )


async def run_retrieval_only(
    image: Image.Image,
    exam_info: dict,
    settings: dict,
    llm_client: LLMClient,
) -> AsyncGenerator[StreamEvent, None]:
    """
    B3 baseline: Evidence Retrieval + Report Writing, no Visual Analysis, no QA loop.

    The ER agent receives the raw image directly (no prior visual analysis),
    so it must infer observations from the image itself before retrieving evidence.
    Agent sequence: [EVIDENCE_RETRIEVAL → REPORT_WRITING]
    """
    _AGENT_SEQUENCE = [AgentRole.EVIDENCE_RETRIEVAL, AgentRole.REPORT_WRITING]

    session_id = str(uuid.uuid4())[:8]
    image_store = SessionImageStore(session_id)
    modality = _resolve_modality(exam_info)
    modality_display = MODALITY_DISPLAY.get(modality, modality or "medical image")
    image_store.save(image, f"Original {modality_display}", "input")

    state = PipelineState(
        session_id=session_id,
        original_image=image,
        exam_info=exam_info,
        modality=modality,
        max_iterations=1,
    )
    state.iteration_count = 1

    yield StreamEvent(
        type=EventType.STATUS,
        data={"message": f"[B3 retrieval_only] session={session_id}"},
    )
    yield StreamEvent(
        type=EventType.PIPELINE_UPDATE,
        data={"iteration": 1, "agents": [a.value for a in _AGENT_SEQUENCE]},
    )

    modality_display = MODALITY_DISPLAY.get(state.modality, state.modality or "medical image")

    for agent_role in _AGENT_SEQUENCE:
        yield StreamEvent(
            type=EventType.AGENT_START,
            data={"agent": agent_role.value, "iteration": 1},
        )

        agent_config = build_agent(agent_role, settings, state)

        if agent_role == AgentRole.EVIDENCE_RETRIEVAL:
            # No prior visual analysis — ask ER agent to observe the image itself
            query = (
                f"You are given a {modality_display} image with no prior visual analysis. "
                f"First, carefully observe the image and identify the key findings. "
                f"Then use standardize_observation for each finding, followed by "
                f"concept_lookup, literature_search, and case_retrieval to retrieve "
                f"supporting evidence. Use evidence_score_and_rank to rank results, "
                f"conflict_detection to check for contradictions, and evidence_summary "
                f"to produce a consolidated evidence summary."
            )
            b64 = _image_to_base64(image)
            images = [{"url": f"data:image/png;base64,{b64}"}]
        else:
            query = _build_agent_query(agent_role, state)
            b64 = _image_to_base64(image)
            images = [{"url": f"data:image/png;base64,{b64}"}]

        params = QueryParams(
            query=query,
            system_prompt=agent_config.system_prompt,
            tool_registry=agent_config.tool_registry,
            llm_client=llm_client,
            max_turns=agent_config.max_turns,
            session_id=session_id,
            hook_engine=agent_config.hook_engine,
            temperature=agent_config.temperature,
            images=images,
        )
        params._extra_context = {
            "pipeline_state": state,
            "image_store": image_store,
            "knowledge_dir": settings.get("data", {}).get("knowledge_dir", "./data/knowledge"),
        }

        agent_text_parts: list[str] = []
        async for event in query_loop(params):
            yield event
            if event.type == EventType.TEXT_DELTA:
                agent_text_parts.append(event.data.get("text", ""))
            elif event.type == EventType.DONE:
                _update_pipeline_state(state, agent_role, "".join(agent_text_parts))

        yield StreamEvent(
            type=EventType.AGENT_END,
            data={"agent": agent_role.value, "iteration": 1},
        )

    nl_report = state.natural_language_report or ""
    if not nl_report and state.structured_report:
        nl_report = state.structured_report.to_natural_language(eval_format=True)
    logger.info(
        "[retrieval_only] retrieved_evidence=%d  nl_report=%d chars",
        len(state.retrieved_evidence), len(nl_report),
    )

    result_data: dict = {
        "session_id": session_id,
        "iterations": 1,
        "review_passed": True,
        "image_evidence_count": 0,
        "retrieved_evidence_count": len(state.retrieved_evidence),
        "natural_language_report": nl_report,
        "structured_report": state.structured_report.to_dict() if state.structured_report else None,
        "image_evidence": [],
        "retrieved_evidence": [e.to_dict() for e in state.retrieved_evidence],
        "agent_run_log": state.agent_run_log,
    }

    images_data: dict = {}
    for item in image_store.list_all():
        img_id = item["image_id"]
        b64 = image_store.to_base64(img_id)
        if b64:
            images_data[img_id] = {
                "base64": b64,
                "description": item["description"],
                "tool_chain": item["tool_chain"],
                "size": list(item["size"]),
                "metadata": item.get("metadata", {}),
            }
    result_data["images"] = images_data

    yield StreamEvent(type=EventType.DONE, data=result_data)


async def run_visual_only(
    image: Image.Image,
    exam_info: dict,
    settings: dict,
    llm_client: LLMClient,
) -> AsyncGenerator[StreamEvent, None]:
    """
    B2 baseline: Visual Analysis + Report Writing, no Evidence Retrieval, no QA loop.

    Agent sequence: [VISUAL_ANALYSIS → REPORT_WRITING]
    Report Writing receives visual findings and image evidence but no retrieved evidence.
    """
    _AGENT_SEQUENCE = [AgentRole.VISUAL_ANALYSIS, AgentRole.REPORT_WRITING]

    session_id = str(uuid.uuid4())[:8]
    image_store = SessionImageStore(session_id)
    modality = _resolve_modality(exam_info)
    modality_display = MODALITY_DISPLAY.get(modality, modality or "medical image")
    image_store.save(image, f"Original {modality_display}", "input")

    state = PipelineState(
        session_id=session_id,
        original_image=image,
        exam_info=exam_info,
        modality=modality,
        max_iterations=1,
    )
    state.iteration_count = 1

    yield StreamEvent(
        type=EventType.STATUS,
        data={"message": f"[B2 visual_only] session={session_id}"},
    )
    yield StreamEvent(
        type=EventType.PIPELINE_UPDATE,
        data={"iteration": 1, "agents": [a.value for a in _AGENT_SEQUENCE]},
    )

    for agent_role in _AGENT_SEQUENCE:
        yield StreamEvent(
            type=EventType.AGENT_START,
            data={"agent": agent_role.value, "iteration": 1},
        )

        agent_config = build_agent(agent_role, settings, state)
        query = _build_agent_query(agent_role, state)

        images = None
        if agent_role in (AgentRole.VISUAL_ANALYSIS, AgentRole.REPORT_WRITING):
            b64 = _image_to_base64(image)
            images = [{"url": f"data:image/png;base64,{b64}"}]

        params = QueryParams(
            query=query,
            system_prompt=agent_config.system_prompt,
            tool_registry=agent_config.tool_registry,
            llm_client=llm_client,
            max_turns=agent_config.max_turns,
            session_id=session_id,
            hook_engine=agent_config.hook_engine,
            temperature=agent_config.temperature,
            images=images,
        )
        params._extra_context = {
            "pipeline_state": state,
            "image_store": image_store,
            "knowledge_dir": settings.get("data", {}).get("knowledge_dir", "./data/knowledge"),
        }

        agent_text_parts: list[str] = []
        async for event in query_loop(params):
            yield event
            if event.type == EventType.TEXT_DELTA:
                agent_text_parts.append(event.data.get("text", ""))
            elif event.type == EventType.DONE:
                _update_pipeline_state(state, agent_role, "".join(agent_text_parts))

        yield StreamEvent(
            type=EventType.AGENT_END,
            data={"agent": agent_role.value, "iteration": 1},
        )

    nl_report = state.natural_language_report or ""
    if not nl_report and state.structured_report:
        nl_report = state.structured_report.to_natural_language(eval_format=True)
    logger.info(
        "[visual_only] image_evidence=%d  nl_report=%d chars",
        len(state.image_evidence), len(nl_report),
    )

    result_data: dict = {
        "session_id": session_id,
        "iterations": 1,
        "review_passed": True,
        "image_evidence_count": len(state.image_evidence),
        "retrieved_evidence_count": 0,
        "natural_language_report": nl_report,
        "structured_report": state.structured_report.to_dict() if state.structured_report else None,
        "image_evidence": [e.to_dict() for e in state.image_evidence],
        "retrieved_evidence": [],
        "agent_run_log": state.agent_run_log,
    }

    images_data: dict = {}
    for item in image_store.list_all():
        img_id = item["image_id"]
        b64 = image_store.to_base64(img_id)
        if b64:
            images_data[img_id] = {
                "base64": b64,
                "description": item["description"],
                "tool_chain": item["tool_chain"],
                "size": list(item["size"]),
                "metadata": item.get("metadata", {}),
            }
    result_data["images"] = images_data

    yield StreamEvent(type=EventType.DONE, data=result_data)
