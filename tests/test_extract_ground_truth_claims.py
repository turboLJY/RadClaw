"""Tests for ground-truth claim extraction helpers."""

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.extract_ground_truth_claims import (  # noqa: E402
    extract_claims,
    normalize_claims_response,
    split_ground_truth_sections,
)


def test_split_ground_truth_sections_extracts_chinese_headings():
    text = "影像表现：\n双肺清晰。未见胸腔积液。\n\n诊断意见：\n双肺清晰。"

    sections = split_ground_truth_sections(text)

    assert sections == {
        "FINDINGS": "双肺清晰。未见胸腔积液。",
        "IMPRESSION": "双肺清晰。",
    }


def test_normalize_claims_response_validates_schema():
    content = """
    {
      "claims": [
        {"section": "FINDINGS", "claim": "双肺清晰。"},
        {"section": "IMPRESSION", "claim": "未见急性心肺异常。"}
      ]
    }
    """

    claims = normalize_claims_response(content)

    assert claims == [
        {"section": "FINDINGS", "claim": "双肺清晰。"},
        {"section": "IMPRESSION", "claim": "未见急性心肺异常。"},
    ]


class RecordingCompletions:
    def __init__(self):
        self.request = None

    def create(self, **request):
        self.request = request
        message = SimpleNamespace(
            content='{"claims": [{"section": "FINDINGS", "claim": "双肺清晰。"}]}'
        )
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def test_extract_claims_enables_deepseek_thinking_mode():
    completions = RecordingCompletions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    args = SimpleNamespace(
        model="deepseek-v4-flash",
        max_tokens=4096,
        no_json_response_format=False,
        retries=1,
        retry_sleep=0.0,
        reasoning_effort="high",
    )

    claims = extract_claims(client, args, {"FINDINGS": "双肺清晰。", "IMPRESSION": ""})

    assert claims == [{"section": "FINDINGS", "claim": "双肺清晰。"}]
    assert completions.request["model"] == "deepseek-v4-flash"
    assert completions.request["reasoning_effort"] == "high"
    assert completions.request["extra_body"] == {"thinking": {"type": "enabled"}}
    assert "temperature" not in completions.request
