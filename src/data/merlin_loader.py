"""
Merlin abdominal CT dataset loader.

Loads ``official_test.jsonl`` from the local Merlin snapshot.  Each row uses
``image_path`` as the NIfTI input volume and ``reference.report_text`` as the
target radiology report.
"""

from __future__ import annotations

from pathlib import Path

from src.data.chest_ct_jsonl_loader import JsonlChestCTLoader


class MerlinLoader(JsonlChestCTLoader):
    dataset_name = "Merlin"
    default_data_dir = "/mnt/yulan_pretrain_131/clinical_project/hf/merlin"
    default_jsonl_name = "official_test.jsonl"
    study_type = "abdominal_ct"
    modality = "abdominal_ct"
    retrieval_category = "abdominal_ct"
    body_part = "abdomen"
    prompt = "Generate a radiology report for this abdominal CT study."

    def __init__(
        self,
        data_dir: str | Path | None = None,
        jsonl_file: str | Path | None = None,
        split: str = "test",
        load_image: bool = True,
        image_root: str | Path | None = None,
        plane: str = "axial",
        slice_index: int | None = None,
        window: str = "soft_tissue",
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
