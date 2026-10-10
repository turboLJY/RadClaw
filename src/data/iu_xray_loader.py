"""
IU-Xray-RRG dataset loader.
Same parquet format as CheXpert-plus-RRG but uses 'test' split instead of 'valid'.
"""

from __future__ import annotations

import hashlib
import logging

from PIL import Image

logger = logging.getLogger(__name__)


def _image_hash(img) -> str:
    if isinstance(img, dict) and "bytes" in img:
        return hashlib.md5(img["bytes"]).hexdigest()
    import numpy as np
    if not isinstance(img, Image.Image):
        img = Image.fromarray(img)
    return hashlib.md5(np.array(img).tobytes()).hexdigest()


class IUXrayLoader:
    def __init__(
        self,
        data_dir: str = "/mnt/yulan_pretrain_131/clinical_project/hf/X-iZhang/IU-Xray-RRG",
        split_name: str = "findings_section",
    ):
        from datasets import load_dataset

        self.split_name = split_name
        self.dataset = load_dataset(data_dir, name=split_name, split="test")
        logger.info("Loaded IU-Xray-RRG %s/test: %d samples", split_name, len(self.dataset))

        # Join impression from impression_section split
        self._impression_map: dict[str, str] = {}
        try:
            imp_ds = load_dataset(data_dir, name="impression_section", split="test")
            for row in imp_ds:
                h = _image_hash(row["main_image"])
                self._impression_map[h] = row["findings_section"]
            logger.info("Loaded IU-Xray impression index: %d entries", len(self._impression_map))
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
            "prompt": sample.get("default_prompt", ""),
            "index": idx,
        }

    def get_exam_info(self, idx: int) -> dict:
        return {
            "study_type": "chest_xray",
            "view": "frontal",
            "modality": "X-Ray",
            "body_part": "thorax",
            "dataset": "IU-Xray-RRG",
            "split": self.split_name,
            "sample_index": idx,
        }