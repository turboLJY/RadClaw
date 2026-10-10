"""Tests for brain MRI JSONL dataset loaders."""

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


def test_mr_rate_loader_brain_mri_exam_info(tmp_path: Path):
    jsonl_path = tmp_path / "mr_rate_test.jsonl"
    volume_path = tmp_path / "SZ5AHVADEH_t1w-raw-sag.nii.gz"
    _write_jsonl(
        jsonl_path,
        [
            {
                "study_id": "SZ5AHVADEH",
                "split": "subset1000",
                "image_path": str(volume_path),
                "image_paths": [str(volume_path)],
                "modal": "T1WI",
                "reference": {
                    "findings": "White matter signal changes are present.",
                    "impression": "Nonspecific white matter lesions.",
                    "report_text": "BRAIN MRI:\nWhite matter signal changes are present.",
                },
            }
        ],
    )

    from src.data.mr_rate_loader import MRRateLoader, MRRATELoader

    loader = MRRateLoader(data_dir=tmp_path, load_image=False)

    assert MRRATELoader is MRRateLoader
    assert len(loader) == 1
    sample = loader[0]
    assert sample["image"] is None
    assert sample["volume_path"] == str(volume_path)
    assert sample["text"].startswith("BRAIN MRI")
    assert sample["findings"] == "White matter signal changes are present."
    assert sample["modal"] == "T1WI"

    exam_info = loader.get_exam_info(0)
    assert exam_info["study_type"] == "brain_mri"
    assert exam_info["modality"] == "brain_mri"
    assert exam_info["retrieval_category"] == "brain_mri"
    assert exam_info["body_part"] == "brain"
    assert exam_info["dataset"] == "MR-RATE"
    assert exam_info["study_id"] == "SZ5AHVADEH"
    assert exam_info["modal"] == "T1WI"


def test_radgenome_brain_mri_loader_brain_mri_exam_info(tmp_path: Path):
    jsonl_path = tmp_path / "radgenome_brain_mri_held_out.jsonl"
    volume_path = tmp_path / "sub-strokecase0001_ses-0001_adc.nii.gz"
    _write_jsonl(
        jsonl_path,
        [
            {
                "study_id": "sub-strokecase0001_ses-0001_adc",
                "split": "held_out",
                "source_dataset": "ISLES22",
                "sequence": "adc",
                "image_path": str(volume_path),
                "reference": {
                    "report_text": "Restricted diffusion involves the left hemisphere.",
                },
            }
        ],
    )

    from src.data.radgenome_brain_mri_loader import RadGenomeBrainLoader, RadGenomeBrainMRILoader

    loader = RadGenomeBrainMRILoader(data_dir=tmp_path, load_image=False)

    assert RadGenomeBrainLoader is RadGenomeBrainMRILoader
    assert len(loader) == 1
    sample = loader[0]
    assert sample["volume_path"] == str(volume_path)
    assert sample["text"] == "Restricted diffusion involves the left hemisphere."
    assert sample["sequence"] == "adc"
    assert sample["source_dataset"] == "ISLES22"

    exam_info = loader.get_exam_info(0)
    assert exam_info["modality"] == "brain_mri"
    assert exam_info["retrieval_category"] == "brain_mri"
    assert exam_info["dataset"] == "RadGenome-Brain_MRI"
    assert exam_info["study_id"] == "sub-strokecase0001_ses-0001_adc"
    assert exam_info["sequence"] == "adc"
    assert exam_info["source_dataset"] == "ISLES22"
