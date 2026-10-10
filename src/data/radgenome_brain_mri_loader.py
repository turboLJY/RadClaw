"""
RadGenome-Brain MRI dataset loader.

Loads ``radgenome_brain_mri_held_out.jsonl`` from the local RadGenome-Brain MRI
snapshot.  Each row uses ``image_path`` as the NIfTI input volume and
``reference.report_text`` as the target report.
"""

from __future__ import annotations

from pathlib import Path

from src.data.chest_ct_jsonl_loader import JsonlChestCTLoader


class RadGenomeBrainMRILoader(JsonlChestCTLoader):
    dataset_name = "RadGenome-Brain_MRI"
    default_data_dir = "/mnt/yulan_pretrain_131/clinical_project/hf/JiayuLei/RadGenome-Brain_MRI"
    default_jsonl_name = "radgenome_brain_mri_held_out.jsonl"
    study_type = "brain_mri"
    modality = "brain_mri"
    retrieval_category = "brain_mri"
    body_part = "brain"
    prompt = "Generate a radiology report for this brain MRI study."

    def __init__(
        self,
        data_dir: str | Path | None = None,
        jsonl_file: str | Path | None = None,
        split: str = "held_out",
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


RadGenomeBrainLoader = RadGenomeBrainMRILoader
