"""
MIMIC-CXR-RRG dataset loader.
Richer metadata than CheXpert (indication, view, impression in same row).
"""

from __future__ import annotations

import logging

from PIL import Image

logger = logging.getLogger(__name__)


class MIMICCXRLoader:
    def __init__(
        self,
        data_dir: str = "/mnt/yulan_pretrain_131/clinical_project/hf/Yamini-1628/MIMIC-CXR-RRG",
        split_name: str = "findings_section",
        split: str = "test",
    ):
        from datasets import load_dataset

        self.split_name = split_name
        self.dataset = load_dataset(data_dir, name=split_name, split=split)
        logger.info("Loaded MIMIC-CXR-RRG %s/%s: %d samples", split_name, split, len(self.dataset))

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        sample = self.dataset[idx]
        image = sample["main_image"]
        if hasattr(image, "convert"):
            pil_image = image.convert("RGB")
        else:
            pil_image = image

        findings = sample.get("findings_section", "")
        impression = sample.get("impression_section", "")

        if impression:
            text = f"FINDINGS:\n{findings}\n\nIMPRESSION:\n{impression}"
        else:
            text = f"FINDINGS:\n{findings}"

        return {
            "image": pil_image,
            "text": text,
            "findings": findings,
            "impression": impression,
            "prompt": sample.get("default_prompt", ""),
            "index": idx,
        }

    def get_exam_info(self, idx: int) -> dict:
        sample = self.dataset[idx]
        view = sample.get("view", sample.get("ViewPosition", "frontal"))
        indication = sample.get("indication_section", "") or ""

        return {
            "study_type": "chest_xray",
            "view": view or "frontal",
            "modality": "X-Ray",
            "body_part": "thorax",
            "dataset": "MIMIC-CXR-RRG",
            "split": self.split_name,
            "sample_index": idx,
            "indication": indication,
        }