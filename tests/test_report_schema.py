"""Test report schema and structured report."""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.types import (
    Finding, Impression, StructuredReport, SupportEvidence,
    ImageEvidence, RetrievedEvidence,
)


def test_finding_creation():
    f = Finding(
        finding_id="F001",
        text="Right pleural effusion",
        type="abnormal",
        anatomy="right costophrenic angle",
        observation="pleural_effusion",
        confidence=0.9,
    )
    d = f.to_dict()
    assert d["finding_id"] == "F001"
    assert d["confidence"] == 0.9


def test_impression_creation():
    imp = Impression(
        impression_id="I001",
        text="Findings consistent with CHF",
        type="diagnosis",
        linked_findings=["F001", "F002"],
        confidence=0.85,
    )
    d = imp.to_dict()
    assert len(d["linked_findings"]) == 2


def test_structured_report():
    report = StructuredReport(
        exam_info={"study_type": "chest_xray", "view": "frontal_PA"},
        findings=[
            Finding(finding_id="F001", text="Cardiomegaly", confidence=0.9),
            Finding(finding_id="F002", text="Bilateral pleural effusions", confidence=0.85),
        ],
        impressions=[
            Impression(
                impression_id="I001",
                text="CHF with bilateral effusions",
                linked_findings=["F001", "F002"],
                confidence=0.8,
            ),
        ],
    )

    d = report.to_dict()
    assert len(d["findings"]) == 2
    assert len(d["impressions"]) == 1

    j = report.to_json()
    assert "Cardiomegaly" in j

    nl = report.to_natural_language()
    assert "FINDINGS:" in nl
    assert "IMPRESSION:" in nl
    assert "Cardiomegaly" in nl


def test_support_evidence():
    ie = ImageEvidence(
        image_id="IMG_001",
        description="Cropped CPA",
        tool_chain=["crop_roi", "zoom_and_enhance"],
    )
    re = RetrievedEvidence(
        evidence_id="LIT_001",
        source_type="literature",
        content="ACR criteria for pleural effusion",
        relevance_score=0.9,
    )
    support = SupportEvidence(
        image_evidence=[ie],
        retrieved_evidence=[re],
    )
    d = support.to_dict()
    assert len(d["image_evidence"]) == 1
    assert d["retrieved_evidence"][0]["relevance_score"] == 0.9


def test_pydantic_report_model():
    from src.report.schema import StructuredReportModel, FindingModel, ImpressionModel

    report = StructuredReportModel(
        findings=[
            FindingModel(finding_id="F001", text="Test finding", confidence=0.8),
        ],
        impressions=[
            ImpressionModel(
                impression_id="I001",
                text="Test impression",
                linked_findings=["F001"],
            ),
        ],
    )
    nl = report.to_natural_language()
    assert "Test finding" in nl

    issues = report.validate_evidence_links(set(), set())
    assert len(issues) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])