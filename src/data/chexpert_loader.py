"""
CheXpert-plus-RRG dataset loader.
Loads parquet files with embedded chest X-ray images and report text.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Iterator

from PIL import Image

logger = logging.getLogger(__name__)


def _image_hash(img) -> str:
    """Hash an image regardless of whether it's a PIL Image or a bytes dict."""
    if isinstance(img, dict) and "bytes" in img:
        return hashlib.md5(img["bytes"]).hexdigest()
    import numpy as np
    if not isinstance(img, Image.Image):
        img = Image.fromarray(img)
    return hashlib.md5(np.array(img).tobytes()).hexdigest()


class CheXpertLoader:
    def __init__(
        self,
        data_dir: str = "/cpfs01/user/zhaizhai.cxx/hf/data/Doorad/CheXpert-plus-RRG",
        split_name: str = "findings_section",
    ):
        from datasets import load_dataset
        self.split_name = split_name
        self.dataset = load_dataset(data_dir, name=split_name, split="valid")
        logger.info(f"Loaded CheXpert {split_name}: {len(self.dataset)} samples")

        # Build image-hash → impression mapping so we can join complete GT
        self._impression_map: dict[str, str] = {}
        try:
            imp_ds = load_dataset(data_dir, name="impression_section", split="valid")
            for row in imp_ds:
                h = _image_hash(row["main_image"])
                self._impression_map[h] = row["findings_section"]
            logger.info("Loaded impression index: %d entries", len(self._impression_map))
        except Exception as e:
            logger.warning("Could not load impression_section: %s", e)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        sample = self.dataset[idx]
        image = sample["main_image"]
        if hasattr(image, "convert"):
            pil_image = image.convert("RGB")
        else:
            pil_image = image

        findings = sample.get(self.split_name, "")
        prompt = sample.get("default_prompt", "")

        # Join impression if available for this image
        img_hash = _image_hash(sample["main_image"])
        impression = self._impression_map.get(img_hash, "")
        if impression:
            text = f"FINDINGS:\n{findings}\n\nIMPRESSION:\n{impression}"
        else:
            text = f"FINDINGS:\n{findings}"

        return {
            "image": pil_image,
            "text": text,
            "findings": findings,
            "impression": impression,
            "prompt": prompt,
            "index": idx,
        }

    def iter_samples(self) -> Iterator[dict]:
        for i in range(len(self)):
            yield self[i]

    def get_exam_info(self, idx: int) -> dict:
        return {
            "study_type": "chest_xray",
            "view": "frontal",
            "modality": "X-Ray",
            "body_part": "thorax",
            "dataset": "CheXpert-plus-RRG",
            "split": self.split_name,
            "sample_index": idx,
        }