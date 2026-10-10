"""Tests for JSONL chest CT dataset loaders."""

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


def test_ct_rate_loader_metadata_without_image_loading(tmp_path: Path):
    jsonl_path = tmp_path / "ct_rate_validation.jsonl"
    _write_jsonl(
        jsonl_path,
        [
            {
                "sample_id": "valid_1_a_1",
                "split": "validation",
                "volume_name": "valid_1_a_1.nii.gz",
                "image_paths": ["volumes/valid_1_a_1.nii.gz"],
                "clinical_context": {"indication": "cough", "technique": "HRCT"},
                "reference": {
                    "findings": "Airways are patent.",
                    "impression": "No acute abnormality.",
                    "report_text": "FINDINGS: Airways are patent.\n\nIMPRESSION: No acute abnormality.",
                    "labels": {"VolumeName": "valid_1_a_1.nii.gz", "Lung nodule": 1},
                },
                "metadata": {"PatientSex": "M", "NumberofSlices": 251},
            }
        ],
    )

    from src.data.ct_rate_loader import CTRateLoader

    loader = CTRateLoader(data_dir=tmp_path, load_image=False)

    assert len(loader) == 1
    sample = loader[0]
    assert sample["image"] is None
    assert sample["findings"] == "Airways are patent."
    assert sample["impression"] == "No acute abnormality."
    assert sample["volume_path"] == str(tmp_path / "volumes/valid_1_a_1.nii.gz")

    exam_info = loader.get_exam_info(0)
    assert exam_info["study_type"] == "chest_ct"
    assert exam_info["modality"] == "chest_ct"
    assert exam_info["retrieval_category"] == "chest_ct"
    assert exam_info["dataset"] == "CT-RATE"
    assert exam_info["positive_labels"] == ["Lung nodule"]
    assert exam_info["metadata"]["patient_sex"] == "M"


def test_radgenome_loader_disorders_without_image_loading(tmp_path: Path):
    jsonl_path = tmp_path / "radgenome_chestct_validation.jsonl"
    _write_jsonl(
        jsonl_path,
        [
            {
                "sample_id": "valid_861_a_1",
                "split": "validation",
                "volume_name": "valid_861_a_1.nii.gz",
                "image_paths": ["/abs/valid_861_a_1.nii.gz"],
                "reference": {
                    "findings": "Ground-glass opacities are present.",
                    "impression": "",
                    "report_text": "FINDINGS: Ground-glass opacities are present.\n\nIMPRESSION:",
                    "disorders": "covid-19 pneumonia, pericardial effusion",
                },
            }
        ],
    )

    from src.data.radgenome_loader import RadGenomeChestCTLoader, RadGenomeLoader

    loader = RadGenomeChestCTLoader(data_dir=tmp_path, load_image=False)

    assert RadGenomeLoader is RadGenomeChestCTLoader
    assert len(loader) == 1
    sample = loader[0]
    assert sample["image_paths"] == ["/abs/valid_861_a_1.nii.gz"]
    assert sample["disorders"] == ["covid-19 pneumonia", "pericardial effusion"]

    exam_info = loader.get_exam_info(0)
    assert exam_info["modality"] == "chest_ct"
    assert exam_info["dataset"] == "RadGenome-ChestCT"
    assert exam_info["disorders"] == ["covid-19 pneumonia", "pericardial effusion"]
