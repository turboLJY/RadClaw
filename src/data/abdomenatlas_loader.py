"""
AbdomenAtlas 3.0 Mini abdominal CT dataset loader.

Loads ``iid_test.jsonl`` from the local AbdomenAtlas3.0Mini snapshot.  Each row
uses ``image_path`` as an absolute NIfTI path and ``reference.report_text`` as
the target report.
"""

from __future__ import annotations

from pathlib import Path

from src.data.chest_ct_jsonl_loader import JsonlChestCTLoader


class AbdomenAtlasLoader(JsonlChestCTLoader):
    dataset_name = "AbdomenAtlas3.0Mini"
    default_data_dir = "/mnt/yulan_pretrain_131/clinical_project/hf/AbdomenAtlas/AbdomenAtlas3.0Mini"
    default_jsonl_name = "iid_test.jsonl"
    study_type = "abdominal_ct"
    modality = "abdominal_ct"
    retrieval_category = "abdominal_ct"
    body_part = "abdomen"
    prompt = "Generate a radiology report for this abdominal CT study."

    def __init__(
        self,
        data_dir: str | Path | None = None,
        jsonl_file: str | Path | None = None,
        split: str = "iid_test",
        load_image: bool = True,
        plane: str = "axial",
        slice_index: int | None = None,
        window: str = "soft_tissue",
    ):
        super().__init__(
            data_dir=data_dir,
            jsonl_file=jsonl_file,
            split=split,
            load_image=load_image,
            plane=plane,
            slice_index=slice_index,
            window=window,
        )


AbdomenAtlasMiniLoader = AbdomenAtlasLoader
