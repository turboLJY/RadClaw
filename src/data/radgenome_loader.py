"""
RadGenome-ChestCT dataset loader.

Loads the validation JSONL exported under the local Hugging Face snapshot and
returns a chest CT sample with report text plus the source NIfTI volume path.
"""

from __future__ import annotations

from pathlib import Path

from src.data.chest_ct_jsonl_loader import JsonlChestCTLoader


class RadGenomeChestCTLoader(JsonlChestCTLoader):
    dataset_name = "RadGenome-ChestCT"
    default_data_dir = "/mnt/yulan_pretrain_131/clinical_project/hf/RadGenome/RadGenome-ChestCT"
    default_jsonl_name = "radgenome_chestct_validation.jsonl"
    prompt = "Generate a radiology report for this chest CT study."

    def __init__(
        self,
        data_dir: str | Path | None = None,
        jsonl_file: str | Path | None = None,
        split: str = "validation",
        load_image: bool = True,
        plane: str = "axial",
        slice_index: int | None = None,
        window: str = "lung",
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


RadGenomeLoader = RadGenomeChestCTLoader
