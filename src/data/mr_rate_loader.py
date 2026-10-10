"""
MR-RATE brain MRI dataset loader.

Loads ``mr_rate_test.jsonl`` from the local MR-RATE snapshot.  Each row uses
``image_path`` as the NIfTI input volume and ``reference.report_text`` as the
target report.
"""

from __future__ import annotations

from pathlib import Path

from src.data.chest_ct_jsonl_loader import JsonlChestCTLoader


class MRRateLoader(JsonlChestCTLoader):
    dataset_name = "MR-RATE"
    default_data_dir = "/mnt/yulan_pretrain_131/clinical_project/hf/Forithmus/MR-RATE"
    default_jsonl_name = "mr_rate_test.jsonl"
    study_type = "brain_mri"
    modality = "brain_mri"
    retrieval_category = "brain_mri"
    body_part = "brain"
    prompt = "Generate a radiology report for this brain MRI study."

    def __init__(
        self,
        data_dir: str | Path | None = None,
        jsonl_file: str | Path | None = None,
        split: str = "test",
        load_image: bool = True,
        image_root: str | Path | None = None,
        plane: str = "axial",
        slice_index: int | None = None,
        window: str = "mri",
    ):
        super().__init__(
            data_dir=data_dir,
            jsonl_file=jsonl_file,
            split=split,
            load_image=load_image,
            image_root=image_root,
            plane=plane,
            slice_index=slice_index,
            window=window,
        )


MRRATELoader = MRRateLoader
