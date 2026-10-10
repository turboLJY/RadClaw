"""Tests for annotation sample translation helpers."""

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.translate_annotation_samples_to_chinese import (  # noqa: E402
    DEFAULT_GUIDELINE_TEXT,
    apply_translations,
    collect_result_refs,
    collect_summary_refs,
    load_guideline_text,
    translate_batch,
)


def test_default_guideline_text_uses_chest_xray_writing_guideline():
    assert DEFAULT_GUIDELINE_TEXT == Path("chest_xray_writing_guideline.txt")


def test_load_guideline_text_reads_text_file(tmp_path):
    guideline_path = tmp_path / "guideline.txt"
    guideline_path.write_text("指南标题\x0c\n\n\n\n正文  \n", encoding="utf-8")
    args = SimpleNamespace(guideline_text=guideline_path, guideline_max_chars=0)

    assert load_guideline_text(args) == "指南标题\n\n\n正文"


def test_summary_ground_truth_translation_preserves_original():
    original_ground_truth = "FINDINGS:\nLungs are clear.\n\nIMPRESSION:\nClear lungs."
    translated_ground_truth = "影像表现：\n双肺清晰。\n\n诊断意见：\n双肺清晰。"
    summary_data = {
        "sample_index": 64,
        "ground_truth": original_ground_truth,
        "natural_language_report": "FINDINGS:\nLungs are clear.",
    }

    refs = collect_summary_refs(summary_data)
    assert len(refs) == 1
    assert refs[0].key == "ground_truth_chinese"

    apply_translations(refs, {original_ground_truth: translated_ground_truth})

    assert summary_data["ground_truth"] == original_ground_truth
    assert summary_data["ground_truth_chinese"] == translated_ground_truth


def test_result_refs_include_retrieved_evidence_content():
    result_data = {
        "structured_report": {
            "findings": [
                {
                    "text": "Right pleural effusion.",
                    "support": {
                        "image_evidence": [{"description": "Blunted right costophrenic angle."}],
                        "retrieved_evidence": [{"content": "Pleural effusions appear as blunting."}],
                    },
                }
            ],
            "impressions": [
                {
                    "text": "Small right pleural effusion.",
                    "support": {
                        "retrieved_evidence": [{"content": "Small effusions can be subtle."}],
                    },
                }
            ],
        }
    }

    refs = collect_result_refs(result_data)
    paths = {ref.path for ref in refs}

    assert "structured_report.findings[0].support.retrieved_evidence[0].content" in paths
    assert "structured_report.impressions[0].support.retrieved_evidence[0].content" in paths

    apply_translations(
        refs,
        {
            "Pleural effusions appear as blunting.": "胸腔积液可表现为肋膈角变钝。",
            "Small effusions can be subtle.": "少量积液表现可能较隐匿。",
        },
    )

    finding_support = result_data["structured_report"]["findings"][0]["support"]
    impression_support = result_data["structured_report"]["impressions"][0]["support"]
    assert finding_support["retrieved_evidence"][0]["content"] == "胸腔积液可表现为肋膈角变钝。"
    assert impression_support["retrieved_evidence"][0]["content"] == "少量积液表现可能较隐匿。"


class RecordingCompletions:
    def __init__(self):
        self.request = None

    def create(self, **request):
        self.request = request
        message = SimpleNamespace(content='{"T0001": "译文"}')
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def test_deepseek_v4_requests_enable_thinking_mode():
    for model in ("deepseek-v4-flash", "deepseek-v4-pro"):
        completions = RecordingCompletions()
        client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        args = SimpleNamespace(
            model=model,
            temperature=0.0,
            max_tokens=4096,
            no_json_response_format=False,
            retries=1,
            retry_sleep=0.0,
            reasoning_effort="high",
        )

        translations = translate_batch(client, args, "system prompt", [("T0001", "source")])

        assert translations == {"T0001": "译文"}
        assert completions.request["reasoning_effort"] == "high"
        assert completions.request["extra_body"] == {"thinking": {"type": "enabled"}}
        assert "temperature" not in completions.request

