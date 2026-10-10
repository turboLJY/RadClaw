"""
Open-I dataset loader.
Test split (770 samples) is fixed in data/openi_test_split.json.
Only loads samples with frontal images and non-empty findings.
"""

from __future__ import annotations

import io
import json
import logging
from pathlib import Path

from PIL import Image

logger = logging.getLogger(__name__)

SPLIT_FILE = Path(__file__).resolve().parent.parent.parent / "data" / "openi_test_split.json"


class OpenILoader:
    def __init__(
        self,
        data_dir: str = "/cpfs01/user/zhaizhai.cxx/hf/data/ykumards/open-i",
        split: str = "test",
    ):
        from datasets import load_dataset

        full_ds = load_dataset(data_dir, split="train")

        if SPLIT_FILE.exists():
            with open(SPLIT_FILE) as f:
                split_data = json.load(f)
            test_indices = split_data["test_indices"]
            train_indices = split_data["train_indices"]
            logger.info("Loaded fixed Open-I split from %s", SPLIT_FILE)
        else:
            import numpy as np
            valid_indices = [
                i for i in range(len(full_ds))
                if full_ds[i].get("img_frontal") is not None
                and full_ds[i].get("findings")
                and len(full_ds[i]["findings"].strip()) > 10
            ]
            rng = np.random.default_rng(42)
            rng.shuffle(valid_indices)
            test_indices = sorted(valid_indices[:770])
            train_indices = sorted(valid_indices[770:])

            SPLIT_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(SPLIT_FILE, "w") as f:
                json.dump({
                    "test_indices": test_indices,
                    "train_indices": train_indices,
                    "total_valid": len(valid_indices),
                    "seed": 42,
                    "n_test": 770,
                }, f, indent=2)
            logger.info("Generated and saved Open-I split to %s", SPLIT_FILE)

        if split == "test":
            self._indices = test_indices
        else:
            self._indices = train_indices

        self._full_ds = full_ds
        self._split = split
        logger.info("Open-I %s split: %d samples", split, len(self._indices))

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> dict:
        real_idx = self._indices[idx]
        sample = self._full_ds[real_idx]

        img_bytes = sample["img_frontal"]
        pil_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        findings = sample.get("findings", "").strip()
        impression = sample.get("impression", "").strip()

        if impression:
            text = f"FINDINGS:\n{findings}\n\nIMPRESSION:\n{impression}"
        else:
            text = f"FINDINGS:\n{findings}"

        return {
            "image": pil_image,
            "text": text,
            "findings": findings,
            "impression": impression,
            "prompt": "Provide a detailed description of the findings in the radiology image.",
            "index": idx,
        }

    def get_exam_info(self, idx: int) -> dict:
        real_idx = self._indices[idx]
        sample = self._full_ds[real_idx]
        indication = sample.get("indication", "") or ""

        return {
            "study_type": "chest_xray",
            "view": "frontal",
            "modality": "X-Ray",
            "body_part": "thorax",
            "dataset": "Open-I",
            "split": self._split,
            "sample_index": idx,
            "indication": indication,
        }