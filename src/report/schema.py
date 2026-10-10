"""
Pydantic models for RadClaw structured radiology reports.
Provides validation, JSON serialization, and natural language conversion.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class SupportEvidenceModel(BaseModel):
    image_evidence_ids: list[str] = Field(default_factory=list)
    retrieved_evidence_ids: list[str] = Field(default_factory=list)
    conflict_notes: list[str] = Field(default_factory=list)


class FindingModel(BaseModel):
    finding_id: str
    text: str
    type: str = "abnormal"
    anatomy: str = ""
    observation: str = ""
    confidence: float = 0.0
    support: SupportEvidenceModel = Field(default_factory=SupportEvidenceModel)


class ImpressionModel(BaseModel):
    impression_id: str
    text: str
    type: str = "diagnosis"
    linked_findings: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    support: SupportEvidenceModel = Field(default_factory=SupportEvidenceModel)


class ExamInfoModel(BaseModel):
    study_type: str = "chest_xray"
    view: str = "frontal"
    modality: str = "X-Ray"
    body_part: str = "thorax"
    clinical_indication: str = ""


class StructuredReportModel(BaseModel):
    exam_info: ExamInfoModel = Field(default_factory=ExamInfoModel)
    findings: list[FindingModel] = Field(default_factory=list)
    impressions: list[ImpressionModel] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)

    def to_natural_language(self, eval_format: bool = False) -> str:
        lines = []
        if not eval_format:
            lines.append(f"EXAMINATION: {self.exam_info.study_type} ({self.exam_info.view})")
            if self.exam_info.clinical_indication:
                lines.append(f"CLINICAL INDICATION: {self.exam_info.clinical_indication}")
            lines.append("")

        if self.findings:
            lines.append("FINDINGS:")
            finding_texts = []
            for f in self.findings:
                conf = f" [confidence: {f.confidence:.2f}]" if (f.confidence > 0 and not eval_format) else ""
                finding_texts.append(f"{f.text}{conf}")
            lines.append("  " + " ".join(finding_texts))
            lines.append("")

        if self.impressions:
            lines.append("IMPRESSION:")
            for idx, imp in enumerate(self.impressions, 1):
                conf = f" [confidence: {imp.confidence:.2f}]" if (imp.confidence > 0 and not eval_format) else ""
                lines.append(f"  {idx}. {imp.text}{conf}")

        return "\n".join(lines)

    def validate_evidence_links(self, available_image_ids: set[str], available_evidence_ids: set[str]) -> list[str]:
        issues = []
        for f in self.findings:
            for img_id in f.support.image_evidence_ids:
                if img_id not in available_image_ids:
                    issues.append(f"Finding {f.finding_id}: unknown image evidence {img_id}")
            for ev_id in f.support.retrieved_evidence_ids:
                if ev_id not in available_evidence_ids:
                    issues.append(f"Finding {f.finding_id}: unknown retrieved evidence {ev_id}")

        for imp in self.impressions:
            for fid in imp.linked_findings:
                if not any(f.finding_id == fid for f in self.findings):
                    issues.append(f"Impression {imp.impression_id}: links to unknown finding {fid}")

        return issues