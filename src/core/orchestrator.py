"""
Multi-agent orchestrator for RadClaw.

Sequences 4 agents (Visual Analysis → Evidence Retrieval → Report Writing → Quality Review)
with a quality review feedback loop that triggers targeted re-execution.

This is the major new component not present in SearchClaw.
"""

from __future__ import annotations

import base64
import io
import logging
import re
import uuid
from typing import AsyncGenerator

from PIL import Image

from src.core.types import (
    AgentRole,
    ContentBlock,
    EventType,
    ImageEvidence,
    Message,
    PipelineState,
    StreamEvent,
    StructuredReport,
)
from src.core.loop import QueryParams, query_loop
from src.agents.base import build_agent
from src.data.image_store import SessionImageStore
from src.llm.client import LLMClient

logger = logging.getLogger(__name__)

AGENT_SEQUENCE = [
    AgentRole.VISUAL_ANALYSIS,
    AgentRole.EVIDENCE_RETRIEVAL,
    AgentRole.REPORT_WRITING,
    AgentRole.QUALITY_REVIEW,
]

ATTRIBUTION_TO_AGENT = {
    "visual": AgentRole.VISUAL_ANALYSIS,
    "retrieval": AgentRole.EVIDENCE_RETRIEVAL,
    "generation": AgentRole.REPORT_WRITING,
    "wording": AgentRole.REPORT_WRITING,
}


MODALITY_MAP = {
    "X-Ray": "CXR",
    "x-ray": "CXR",
    "xray": "CXR",
    "CXR": "CXR",
    "cxr": "CXR",
    "chest_ct": "chest_ct",
    "CT": "chest_ct",
    "ct": "chest_ct",
    "abdominal_ct": "abdominal_ct",
    "brain_ct": "brain_ct",
    "brain_mri": "brain_mri",
    "MRI": "brain_mri",
    "mri": "brain_mri",
}

MODALITY_DISPLAY = {
    "CXR": "chest X-ray",
    "chest_ct": "chest CT",
    "abdominal_ct": "abdominal CT",
    "brain_ct": "brain CT",
    "brain_mri": "brain MRI",
}


def _resolve_modality(exam_info: dict) -> str:
    raw = exam_info.get("modality", "")
    return MODALITY_MAP.get(raw, raw)


def _image_to_base64(image: Image.Image) -> str:
    max_dim = 1024
    if max(image.size) > max_dim:
        ratio = max_dim / max(image.size)
        new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
        image = image.resize(new_size, Image.LANCZOS)
        logger.info("Resized image from original to %s for LLM input", new_size)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def _determine_agents_to_run(state: PipelineState) -> list[AgentRole]:
    if state.iteration_count <= 1:
        return list(AGENT_SEQUENCE)

    agents_needed = set()
    for issue in state.review_issues:
        target = ATTRIBUTION_TO_AGENT.get(issue.attribution)
        if target:
            agents_needed.add(target)

    if not agents_needed:
        return []

    # Always run all downstream agents from the earliest needed one,
    # so that visual changes propagate through evidence → report → review.
    pipeline_without_qr = [r for r in AGENT_SEQUENCE if r != AgentRole.QUALITY_REVIEW]
    earliest_idx = min(pipeline_without_qr.index(r) for r in agents_needed)
    result = pipeline_without_qr[earliest_idx:]
    result.append(AgentRole.QUALITY_REVIEW)
    return result


def _build_agent_query(agent_role: AgentRole, state: PipelineState) -> str:
    modality_display = MODALITY_DISPLAY.get(state.modality, state.modality or "medical image")

    if agent_role == AgentRole.VISUAL_ANALYSIS:
        return (
            f"Analyze this {modality_display} image. Use your visual analysis tools to "
            "systematically examine all anatomical regions following the ABCDEF approach. "
            "For each finding, use the appropriate tools (crop_roi, zoom_and_enhance, "
            "multi_window_render, overlay_annotation) and bind evidence using link_visual_evidence. "
            "Provide a comprehensive summary of all findings."
        )

    elif agent_role == AgentRole.EVIDENCE_RETRIEVAL:
        findings = state.visual_findings_text or "No visual findings available."
        evidence_list = "\n".join(
            f"  - {e.description} (image: {e.image_id})"
            for e in state.image_evidence[:10]
        )
        return (
            f"Given these visual findings from {modality_display} analysis:\n\n"
            f"{findings}\n\n"
            f"Image evidence:\n{evidence_list}\n\n"
            f"Retrieve supporting medical evidence. Use standardize_observation for each "
            f"finding, then use concept_lookup, literature_search, and case_retrieval "
            f"to search knowledge bases. Use evidence_score_and_rank to rank results, "
            f"conflict_detection to check for contradictions, and evidence_summary to "
            f"generate a consolidated summary."
        )

    elif agent_role == AgentRole.REPORT_WRITING:
        findings = state.visual_findings_text or ""
        evidence = state.evidence_summary or ""
        img_refs = ", ".join(e.image_id for e in state.image_evidence[:10])
        ev_refs = ", ".join(e.evidence_id for e in state.retrieved_evidence[:10])
        conflicts = ""
        if state.conflicts:
            conflicts = "\nConflicts detected:\n" + "\n".join(
                f"  - {c.get('description', str(c))}" for c in state.conflicts[:5]
            )
        return (
            f"Generate a structured radiology report from the following analysis:\n\n"
            f"VISUAL FINDINGS:\n{findings}\n\n"
            f"EVIDENCE SUMMARY:\n{evidence}\n"
            f"{conflicts}\n\n"
            f"Available image evidence IDs: {img_refs}\n"
            f"Available retrieved evidence IDs: {ev_refs}\n\n"
        f"Use build_structured_finding for each finding, build_structured_impression "
            f"for each impression, link_support_evidence to attach evidence, then "
            f"assemble_report_object and render_natural_language."
        )

    elif agent_role == AgentRole.QUALITY_REVIEW:
        report_json = ""
        if state.structured_report:
            import json
            report_json = json.dumps(state.structured_report.to_dict(), indent=2, ensure_ascii=False)
        return (
            f"Review this radiology report for quality and consistency.\n\n"
            f"STRUCTURED REPORT:\n{report_json}\n\n"
        f"Check all four dimensions:\n"
            f"1. Visual consistency (check_visual_consistency)\n"
            f"2. Evidence consistency (check_evidence_consistency)\n"
            f"3. Internal consistency (check_internal_consistency)\n"
            f"4. Medical standard compliance (check_medical_standard)\n\n"
            f"For EACH issue found, use attribute_responsibility to assign it.\n"
            f"If no issues, the report passes review."
        )

    return ""


def _build_rerun_query(agent_role: AgentRole, state: PipelineState) -> str:
    relevant_issues = [
        i for i in state.review_issues
        if ATTRIBUTION_TO_AGENT.get(i.attribution) == agent_role
    ]

    if not relevant_issues:
        return _build_agent_query(agent_role, state)

    feedback_text = "\n".join(
        f"  - [{i.severity}] {i.description} (suggested fix: {i.suggested_fix})"
        for i in relevant_issues
    )

    base_query = _build_agent_query(agent_role, state)

    return (
        f"REVISION REQUIRED\n\n"
        f"The quality review found these issues with your previous output:\n"
        f"{feedback_text}\n\n"
        f"Please address these specific issues.\n\n"
        f"---\n\n{base_query}"
    )


async def run_pipeline(
    image: Image.Image,
    exam_info: dict,
    settings: dict,
    llm_client: LLMClient,
) -> AsyncGenerator[StreamEvent, str | None]:
    """
    Run the full 4-agent pipeline with quality review feedback loop.

    Yields StreamEvents for real-time streaming to UI.
    """
    session_id = str(uuid.uuid4())[:8]
    image_store = SessionImageStore(session_id)

    modality = _resolve_modality(exam_info)
    modality_display = MODALITY_DISPLAY.get(modality, modality or "medical image")

    # Store original image
    image_store.save(image, f"Original {modality_display}", "input")

    state = PipelineState(
        session_id=session_id,
        original_image=image,
        exam_info=exam_info,
        modality=modality,
        max_iterations=settings.get("pipeline", {}).get("max_iterations", 3),
    )

    yield StreamEvent(
        type=EventType.STATUS,
        data={"message": f"Pipeline started (session: {session_id})"},
    )

    while state.iteration_count < state.max_iterations:
        state.iteration_count += 1

        agents_to_run = _determine_agents_to_run(state)
        # Clear issues after determining which agents to re-run
        state.review_issues = []
        state.review_passed = False

        if not agents_to_run:
            break

        yield StreamEvent(
            type=EventType.PIPELINE_UPDATE,
            data={
                "iteration": state.iteration_count,
                "agents": [a.value for a in agents_to_run],
            },
        )

        for agent_role in agents_to_run:
            # Clear stale report before report_writing re-runs so findings don't stack
            if agent_role == AgentRole.REPORT_WRITING and state.iteration_count > 1:
                state.structured_report = None
                state.natural_language_report = ""
            # Clear stale evidence before evidence_retrieval re-runs
            if agent_role == AgentRole.EVIDENCE_RETRIEVAL and state.iteration_count > 1:
                state.retrieved_evidence.clear()
                state.conflicts.clear()

            yield StreamEvent(
                type=EventType.AGENT_START,
                data={
                    "agent": agent_role.value,
                    "iteration": state.iteration_count,
                },
            )

            agent_config = build_agent(agent_role, settings, state)

            if state.iteration_count == 1:
                query = _build_agent_query(agent_role, state)
            else:
                query = _build_rerun_query(agent_role, state)

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

            # Pass shared state through tool context
            params._extra_context = {
                "pipeline_state": state,
                "image_store": image_store,
                "knowledge_dir": settings.get("data", {}).get("knowledge_dir", "./data/knowledge"),
            }

            agent_text_parts = []
            async for event in query_loop(params):
                yield event

                if event.type == EventType.TEXT_DELTA:
                    agent_text_parts.append(event.data.get("text", ""))

                elif event.type == EventType.DONE:
                    agent_text = "".join(agent_text_parts)
                    _update_pipeline_state(state, agent_role, agent_text)

            yield StreamEvent(
                type=EventType.AGENT_END,
                data={
                    "agent": agent_role.value,
                    "iteration": state.iteration_count,
                },
            )

        # After quality review
        if state.review_passed or not state.review_issues:
            state.review_passed = True
            break

        yield StreamEvent(
            type=EventType.STATUS,
            data={
                "message": (
                    f"Quality review found {len(state.review_issues)} issue(s). "
                    f"Re-running targeted agents (iteration {state.iteration_count + 1})..."
                ),
            },
        )

    # Finalize — if max iterations exhausted, accept with warnings
    if not state.review_passed and state.structured_report:
        state.review_passed = True
        logger.warning(
            "Max iterations reached with %d remaining issue(s); accepting report with warnings",
            len(state.review_issues),
        )

    result_data = {
        "session_id": session_id,
        "iterations": state.iteration_count,
        "review_passed": state.review_passed,
        "image_evidence_count": len(state.image_evidence),
        "retrieved_evidence_count": len(state.retrieved_evidence),
    }

    if state.structured_report:
        result_data["structured_report"] = state.structured_report.to_dict()
        result_data["natural_language_report"] = state.natural_language_report or state.structured_report.to_natural_language(eval_format=True)

    if state.review_issues:
        result_data["remaining_issues"] = [i.to_dict() for i in state.review_issues]

    result_data["image_evidence"] = [e.to_dict() for e in state.image_evidence]
    result_data["retrieved_evidence"] = [e.to_dict() for e in state.retrieved_evidence]
    result_data["agent_run_log"] = state.agent_run_log

    # Include all derived images as base64 for visualization / demo export
    images_data = {}
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


def _fallback_link_image_evidence(state: PipelineState, agent_text: str) -> None:
    """Auto-link image evidence when the visual agent referenced images in text
    but never called link_visual_evidence."""
    img_pattern = re.compile(r"IMG_[a-z0-9]+_\d{3}")
    if not img_pattern.search(agent_text):
        return

    finding_pattern = re.compile(
        r"\*\*\d+\.\s*(?:Finding:?\s*)?(.+?)\*\*"
        r"(.*?)(?=\*\*\d+\.\s*(?:Finding)?|\Z)",
        re.DOTALL,
    )

    for match in finding_pattern.finditer(agent_text):
        title = match.group(1).strip()
        body = match.group(2)
        found_ids = img_pattern.findall(body)
        if not found_ids:
            continue

        obs_match = re.search(r"\*\*Observation\*\*:\s*(.+)", body)
        description = obs_match.group(1).strip() if obs_match else title

        for img_id in found_ids:
            evidence = ImageEvidence(
                image_id=img_id,
                description=f"[auto-linked] {description}",
                tool_chain=["fallback_auto_link"],
                finding_description=description,
                confidence=0.7,
            )
            state.image_evidence.append(evidence)

    if state.image_evidence:
        logger.info(
            "Fallback auto-linked %d image evidence records from visual analysis text",
            len(state.image_evidence),
        )


def _update_pipeline_state(state: PipelineState, agent_role: AgentRole, agent_text: str) -> None:
    state.agent_run_log.append({
        "agent": agent_role.value,
        "iteration": state.iteration_count,
        "output_length": len(agent_text),
    })

    if agent_role == AgentRole.VISUAL_ANALYSIS:
        state.visual_findings_text = agent_text
        if not state.image_evidence:
            _fallback_link_image_evidence(state, agent_text)
        if state.iteration_count > 1:
            state.structured_report = None
            state.natural_language_report = ""

    elif agent_role == AgentRole.EVIDENCE_RETRIEVAL:
        state.evidence_summary = agent_text if agent_text.strip() else state.evidence_summary

    elif agent_role == AgentRole.REPORT_WRITING:
        # render_natural_language tool may have already written the formatted report
        # into state.natural_language_report — only fall back to agent text if not set.
        if agent_text.strip() and not state.natural_language_report:
            state.natural_language_report = agent_text

    elif agent_role == AgentRole.QUALITY_REVIEW:
        text_lower = agent_text.lower()
        if "pass" in text_lower and "no issue" in text_lower:
            state.review_passed = True
        elif not state.review_issues:
            state.review_passed = True
        else:
            high_or_critical = [
                i for i in state.review_issues
                if i.severity in ("high", "critical")
            ]
            if not high_or_critical:
                state.review_passed = True
