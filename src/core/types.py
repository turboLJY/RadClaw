"""
Core types for RadClaw multi-agent medical imaging system.

Extends SearchClaw's type system with medical imaging-specific types:
PipelineState, ImageEvidence, RetrievedEvidence, ReviewIssue,
StructuredReport components, and AgentRole enum.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Literal


# ---------------------------------------------------------------------------
# Agent Roles
# ---------------------------------------------------------------------------

class AgentRole(str, Enum):
    VISUAL_ANALYSIS = "visual_analysis"
    EVIDENCE_RETRIEVAL = "evidence_retrieval"
    REPORT_WRITING = "report_writing"
    QUALITY_REVIEW = "quality_review"


# ---------------------------------------------------------------------------
# Evidence Types
# ---------------------------------------------------------------------------

@dataclass
class ImageEvidence:
    image_id: str
    description: str
    tool_chain: list[str] = field(default_factory=list)
    finding_description: str = ""
    confidence: float = 0.0
    crop_box: dict | None = None
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        d = {
            "image_id": self.image_id,
            "description": self.description,
            "tool_chain": self.tool_chain,
            "finding_description": self.finding_description,
            "confidence": self.confidence,
        }
        if self.crop_box:
            d["crop_box"] = self.crop_box
        return d


@dataclass
class RetrievedEvidence:
    evidence_id: str
    source_type: Literal["concept", "literature", "case", "pubmed", "guideline", "umls"]
    content: str
    relevance_score: float = 0.0
    conflict_flags: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "evidence_id": self.evidence_id,
            "source_type": self.source_type,
            "content": self.content,
            "relevance_score": self.relevance_score,
            "conflict_flags": self.conflict_flags,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Report Components
# ---------------------------------------------------------------------------

@dataclass
class SupportEvidence:
    image_evidence: list[ImageEvidence] = field(default_factory=list)
    retrieved_evidence: list[RetrievedEvidence] = field(default_factory=list)
    conflict_evidence: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "image_evidence": [e.to_dict() for e in self.image_evidence],
            "retrieved_evidence": [e.to_dict() for e in self.retrieved_evidence],
            "conflict_evidence": self.conflict_evidence,
        }


@dataclass
class Finding:
    finding_id: str
    text: str
    type: str = "abnormal"
    anatomy: str = ""
    observation: str = ""
    confidence: float = 0.0
    support: SupportEvidence = field(default_factory=SupportEvidence)

    def to_dict(self) -> dict:
        return {
            "finding_id": self.finding_id,
            "text": self.text,
            "type": self.type,
            "anatomy": self.anatomy,
            "observation": self.observation,
            "confidence": self.confidence,
            "support": self.support.to_dict(),
        }


@dataclass
class Impression:
    impression_id: str
    text: str
    type: str = "diagnosis"
    linked_findings: list[str] = field(default_factory=list)
    confidence: float = 0.0
    support: SupportEvidence = field(default_factory=SupportEvidence)

    def to_dict(self) -> dict:
        return {
            "impression_id": self.impression_id,
            "text": self.text,
            "type": self.type,
            "linked_findings": self.linked_findings,
            "confidence": self.confidence,
            "support": self.support.to_dict(),
        }


@dataclass
class StructuredReport:
    exam_info: dict = field(default_factory=dict)
    findings: list[Finding] = field(default_factory=list)
    impressions: list[Impression] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "exam_info": self.exam_info,
            "findings": [f.to_dict() for f in self.findings],
            "impressions": [i.to_dict() for i in self.impressions],
            "metadata": self.metadata,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

    def to_natural_language(self) -> str:
        lines = []
        if self.exam_info:
            study = self.exam_info.get("study_type", "Imaging Study")
            view = self.exam_info.get("view", "")
            lines.append(f"EXAMINATION: {study}" + (f" ({view})" if view else ""))
            lines.append("")

        if self.findings:
            lines.append("FINDINGS:")
            for f in self.findings:
                lines.append(f"  {f.text}")
            lines.append("")

        if self.impressions:
            lines.append("IMPRESSION:")
            for idx, imp in enumerate(self.impressions, 1):
                lines.append(f"  {idx}. {imp.text}")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Review Issue (Quality Review output)
# ---------------------------------------------------------------------------

@dataclass
class ReviewIssue:
    issue_id: str
    issue_type: str
    description: str
    severity: Literal["low", "medium", "high", "critical"] = "medium"
    attribution: Literal["visual", "retrieval", "generation", "wording"] = "generation"
    target_agent: AgentRole = AgentRole.REPORT_WRITING
    suggested_fix: str = ""

    def to_dict(self) -> dict:
        return {
            "issue_id": self.issue_id,
            "issue_type": self.issue_type,
            "description": self.description,
            "severity": self.severity,
            "attribution": self.attribution,
            "target_agent": self.target_agent.value,
            "suggested_fix": self.suggested_fix,
        }


# ---------------------------------------------------------------------------
# Pipeline State (cross-agent shared state)
# ---------------------------------------------------------------------------

@dataclass
class PipelineState:
    session_id: str = ""
    original_image: Any = None  # PIL.Image.Image
    exam_info: dict = field(default_factory=dict)
    modality: str = ""

    # Visual Analysis outputs
    image_evidence: list[ImageEvidence] = field(default_factory=list)
    visual_findings_text: str = ""

    # Evidence Retrieval outputs
    observation_graph: list[dict] = field(default_factory=list)
    retrieved_evidence: list[RetrievedEvidence] = field(default_factory=list)
    evidence_summary: str = ""
    conflicts: list[dict] = field(default_factory=list)

    # Report Writing outputs
    structured_report: StructuredReport | None = None
    natural_language_report: str = ""

    # Quality Review outputs
    review_issues: list[ReviewIssue] = field(default_factory=list)
    review_passed: bool = False

    # Orchestration tracking
    iteration_count: int = 0
    max_iterations: int = 3
    agent_run_log: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "exam_info": self.exam_info,
            "image_evidence_count": len(self.image_evidence),
            "retrieved_evidence_count": len(self.retrieved_evidence),
            "structured_report": self.structured_report.to_dict() if self.structured_report else None,
            "review_passed": self.review_passed,
            "review_issues": [i.to_dict() for i in self.review_issues],
            "iteration_count": self.iteration_count,
            "agent_run_log": self.agent_run_log,
        }


# ---------------------------------------------------------------------------
# Tool Results
# ---------------------------------------------------------------------------

@dataclass
class ToolResult:
    data: str
    citations: list[dict] = field(default_factory=list)
    truncated: bool = False
    cached_path: str | None = None
    is_error: bool = False
    metadata: dict = field(default_factory=dict)


@dataclass
class ValidationResult:
    valid: bool
    message: str = ""


# ---------------------------------------------------------------------------
# Stream Events
# ---------------------------------------------------------------------------

class EventType(str, Enum):
    TEXT_DELTA = "text_delta"
    TOOL_USE = "tool_use"
    TOOL_RESULT = "tool_result"
    STATUS = "status"
    ERROR = "error"
    DONE = "done"
    AGENT_START = "agent_start"
    AGENT_END = "agent_end"
    PIPELINE_UPDATE = "pipeline_update"
    USER_QUESTION = "user_question"


@dataclass
class StreamEvent:
    type: EventType
    data: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"type": self.type.value, "data": self.data}


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------

@dataclass
class ContentBlock:
    type: Literal["text", "tool_use", "tool_result", "image_url"]
    text: str | None = None
    tool_name: str | None = None
    tool_use_id: str | None = None
    tool_input: dict | None = None
    content: str | None = None
    is_error: bool = False
    image_url: dict | None = None


@dataclass
class Message:
    role: Literal["user", "assistant", "system", "tool"]
    content: str | list[ContentBlock]
    metadata: dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def text_content(self) -> str:
        if isinstance(self.content, str):
            return self.content
        return " ".join(
            block.text or block.content or ""
            for block in self.content
            if block.type in ("text", "tool_result")
        )

    def to_api_dict(self) -> dict:
        if self.role == "tool":
            return {
                "role": "tool",
                "content": self.content if isinstance(self.content, str) else self.text_content,
                "tool_call_id": self.metadata.get("tool_call_id", ""),
            }

        if isinstance(self.content, str):
            return {"role": self.role, "content": self.content}

        if self.role == "assistant":
            msg: dict = {"role": "assistant"}
            text_parts = []
            tool_calls = []

            for block in self.content:
                if block.type == "text" and block.text:
                    text_parts.append(block.text)
                elif block.type == "tool_use":
                    tool_calls.append({
                        "id": block.tool_use_id,
                        "type": "function",
                        "function": {
                            "name": block.tool_name,
                            "arguments": json.dumps(block.tool_input or {}),
                        },
                    })

            msg["content"] = " ".join(text_parts) if text_parts else None
            if tool_calls:
                msg["tool_calls"] = tool_calls
            return msg

        # User messages with possible image content
        api_content = []
        for block in self.content:
            if block.type == "text" and block.text:
                api_content.append({"type": "text", "text": block.text})
            elif block.type == "image_url" and block.image_url:
                api_content.append({"type": "image_url", "image_url": block.image_url})
        return {"role": self.role, "content": api_content if api_content else self.text_content}


# ---------------------------------------------------------------------------
# Loop State
# ---------------------------------------------------------------------------

@dataclass
class LoopState:
    messages: list[Message]
    turn_count: int = 0
    compaction_count: int = 0

    @property
    def last_assistant_message(self) -> str | None:
        for msg in reversed(self.messages):
            if msg.role == "assistant":
                return msg.text_content
        return None
