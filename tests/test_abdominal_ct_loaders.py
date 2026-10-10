"""Tests for abdominal CT JSONL dataset loaders."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_merlin_loader_uses_image_path_and_report_text(tmp_path: Path):
    jsonl_path = tmp_path / "official_test.jsonl"
    volume_path = tmp_path / "AC423ccbe.nii.gz"
    _write_jsonl(
        jsonl_path,
        [
            {
                "case_id": "AC423ccbe",
                "study_id": "AC423ccbe",
                "split": "test",
                "image_path": str(volume_path),
                "reference": {
                    "findings": "FINDINGS: Liver and biliary tree: Normal.",
                    "report_text": "FINDINGS: Liver and biliary tree: Normal.",
                },
                "metadata": {
                    "Age": "70.927",
                    "Gender": "Female",
                    "contrast": "True",
                    "phase": "portal_venous",
                },
            }
        ],
    )

    from src.data.merlin_loader import MerlinLoader

    loader = MerlinLoader(data_dir=tmp_path, load_image=False)

    assert len(loader) == 1
    sample = loader[0]
    assert sample["image"] is None
    assert sample["volume_path"] == str(volume_path)
    assert sample["text"] == "FINDINGS: Liver and biliary tree: Normal."
    assert sample["sample_id"] == "AC423ccbe"

    exam_info = loader.get_exam_info(0)
    assert exam_info["study_type"] == "abdominal_ct"
    assert exam_info["modality"] == "abdominal_ct"
    assert exam_info["retrieval_category"] == "abdominal_ct"
    assert exam_info["body_part"] == "abdomen"
    assert exam_info["dataset"] == "Merlin"
    assert exam_info["case_id"] == "AC423ccbe"
    assert exam_info["metadata"]["patient_sex"] == "Female"
    assert exam_info["metadata"]["phase"] == "portal_venous"


def test_abdomenatlas_loader_prefixes_relative_image_path(tmp_path: Path):
    data_dir = tmp_path / "hf"
    data_dir.mkdir()
    jsonl_path = data_dir / "iid_test.jsonl"
    _write_jsonl(
        jsonl_path,
        [
            {
                "case_id": "BDMAP_00000005",
                "study_id": "BDMAP_00000005",
                "split": "iid_test",
                "image_path": str(tmp_path / "clinical_project/abdominal_ct/abdomenatlas3_mini_hf/images/BDMAP_00000005/ct.nii.gz"),
                "reference": {
                    "report_text": "The liver demonstrates a mean HU value.\n\nIMPRESSION:\nNo tumor.",
                    "structured_report": "FINDINGS:\nLiver normal.",
                },
                "metadata": {"spacing": "[0.78125 0.78125 0.8 ]", "shape": "(512, 512, 509)"},
            }
        ],
    )

    from src.data.abdomenatlas_loader import AbdomenAtlasLoader, AbdomenAtlasMiniLoader

    loader = AbdomenAtlasLoader(data_dir=data_dir, load_image=False)
    expected_path = tmp_path / "clinical_project/abdominal_ct/abdomenatlas3_mini_hf/images/BDMAP_00000005/ct.nii.gz"

    assert AbdomenAtlasMiniLoader is AbdomenAtlasLoader
    assert len(loader) == 1
    sample = loader[0]
    assert sample["volume_path"] == str(expected_path)
    assert sample["text"].startswith("The liver demonstrates")

    exam_info = loader.get_exam_info(0)
    assert exam_info["modality"] == "abdominal_ct"
    assert exam_info["dataset"] == "AbdomenAtlas3.0Mini"
    assert exam_info["volume_path"] == str(expected_path)
    assert exam_info["metadata"]["spacing"] == "[0.78125 0.78125 0.8 ]"
    assert exam_info["metadata"]["shape"] == "(512, 512, 509)"
