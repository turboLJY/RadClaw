"""Tests for relevant evidence segment annotation helpers."""

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.add_relevant_evidence_segments import (  # noqa: E402
    collect_evidence_refs,
    response_to_segment,
    select_relevant_segment,
)


def test_collect_evidence_refs_targets_guideline_and_pubmed_only():
    result_data = {
        "structured_report": {
            "findings": [
                {
                    "text": "右侧少量胸腔积液。",
                    "support": {
                        "retrieved_evidence": [
                            {
                                "source_type": "guideline",
                                "content": "胸腔积液可表现为肋膈角变钝。",
                            },
                            {
                                "source_type": "case",
                                "content": "既往病例内容。",
                            },
                        ]
                    },
                }
            ],
            "impressions": [
                {
                    "text": "心影增大。",
                    "support": {
                        "retrieved_evidence": [
                            {
                                "source_type": "pubmed",
                                "content": "胸片显示心脏增大可提示心脏病。",
                                "relevant_segment": [0, 10],
                            }
                        ]
                    },
                }
            ],
        }
    }

    refs = collect_evidence_refs(
        result_data,
        "sample_1",
        {"guideline", "pubmed"},
        overwrite=False,
    )

    assert len(refs) == 1
    assert refs[0].source_type == "guideline"
    assert refs[0].path == (
        "structured_report.findings[0].support.retrieved_evidence[0]"
    )


def test_collect_evidence_refs_overwrite_includes_existing_segments():
    result_data = {
        "structured_report": {
            "findings": [
                {
                    "text": "心影增大。",
                    "support": {
                        "retrieved_evidence": [
                            {
                                "source_type": "pubmed",
                                "content": "胸片显示心脏增大可提示心脏病。",
                                "relevant_segment": [0, 10],
                            }
                        ]
                    },
                }
            ],
            "impressions": [],
        }
    }

    refs = collect_evidence_refs(
        result_data,
        "sample_1",
        {"guideline", "pubmed"},
        overwrite=True,
    )

    assert len(refs) == 1
    assert refs[0].source_type == "pubmed"


def test_response_to_segment_prefers_exact_segment_text_location():
    content = "无关背景。胸腔积液可表现为肋膈角变钝。其他内容。"
    response = '{"segment_text": "胸腔积液可表现为肋膈角变钝。", "start_idx": 999, "end_idx": 1000}'

    assert response_to_segment(content, response) == (5, 19)
    assert content[5:19] == "胸腔积液可表现为肋膈角变钝。"


class RecordingCompletions:
    def __init__(self):
        self.request = None

    def create(self, **request):
        self.request = request
        message = SimpleNamespace(
            content='{"segment_text": "胸片显示心脏增大可提示心脏病。", "start_idx": 0, "end_idx": 15}'
        )
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def test_select_relevant_segment_enables_deepseek_thinking_mode():
    completions = RecordingCompletions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    args = SimpleNamespace(
        model="deepseek-v4-flash",
        temperature=0.0,
        max_tokens=2048,
        no_json_response_format=False,
        retries=1,
        retry_sleep=0.0,
        reasoning_effort="high",
    )
    ref = SimpleNamespace(
        report_text="心影增大。",
        source_type="pubmed",
        content="胸片显示心脏增大可提示心脏病。",
    )

    segment = select_relevant_segment(client, args, ref)

    assert segment == (0, 15)
    assert completions.request["model"] == "deepseek-v4-flash"
    assert completions.request["reasoning_effort"] == "high"
    assert completions.request["extra_body"] == {"thinking": {"type": "enabled"}}
    assert "temperature" not in completions.request
