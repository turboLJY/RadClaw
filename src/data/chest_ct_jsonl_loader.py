"""
Shared JSONL loader utilities for volumetric report generation datasets.

The volumetric validation files use one JSON object per sample.  Each sample points to
one or more NIfTI volumes through ``image_paths`` or ``image_path`` and stores
the target report text either at top level or under ``reference``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterator

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

_CT_WINDOWS: dict[str, tuple[float, float]] = {
    "lung": (-600.0, 1500.0),
    "mediastinal": (40.0, 400.0),
    "soft_tissue": (50.0, 350.0),
    "bone": (500.0, 2000.0),
}


def _text(value) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} at line {line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object in {path} at line {line_no}")
            rows.append(row)
    return rows


def _resolve_jsonl_path(data_dir: Path, jsonl_file: str | Path | None, default_name: str) -> Path:
    path = Path(jsonl_file or default_name).expanduser()
    if not path.is_absolute():
        path = data_dir / path
    return path


def _coerce_image_paths(
    raw_paths,
    data_dir: Path,
    image_root: Path | None = None,
) -> list[str]:
    if raw_paths is None:
        values: list[str] = []
    elif isinstance(raw_paths, str):
        values = [raw_paths]
    elif isinstance(raw_paths, (list, tuple)):
        values = [str(p) for p in raw_paths if p]
    else:
        values = [str(raw_paths)]

    root = image_root or data_dir
    resolved: list[str] = []
    for value in values:
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = root / path
        resolved.append(str(path))
    return resolved


def _report_text(record: dict) -> tuple[str, str, str]:
    reference = record.get("reference") if isinstance(record.get("reference"), dict) else {}
    findings = _text(record.get("findings") or reference.get("findings"))
    impression = _text(record.get("impression") or reference.get("impression"))
    report_text = _text(record.get("report_text") or reference.get("report_text"))

    if not report_text:
        if impression:
            report_text = f"FINDINGS:\n{findings}\n\nIMPRESSION:\n{impression}"
        else:
            report_text = f"FINDINGS:\n{findings}"

    return report_text, findings, impression


def _positive_labels(labels: dict) -> list[str]:
    positives: list[str] = []
    for key, value in labels.items():
        if str(key).lower() in {"volumename", "volume_name"}:
            continue
        is_positive = False
        if isinstance(value, bool):
            is_positive = value
        elif isinstance(value, (int, float)):
            is_positive = value > 0
        elif isinstance(value, str):
            is_positive = value.strip().lower() in {"1", "true", "yes", "positive"}
        if is_positive:
            positives.append(str(key))
    return positives


def _disorders(record: dict) -> list[str]:
    reference = record.get("reference") if isinstance(record.get("reference"), dict) else {}
    raw = record.get("disorders") or reference.get("disorders")
    if raw is None:
        return []
    if isinstance(raw, str):
        return [item.strip() for item in raw.split(",") if item.strip()]
    if isinstance(raw, (list, tuple)):
        return [_text(item) for item in raw if _text(item)]
    return [_text(raw)] if _text(raw) else []


def _record_labels(record: dict) -> dict:
    reference = record.get("reference") if isinstance(record.get("reference"), dict) else {}
    labels = record.get("labels") or reference.get("labels") or {}
    return labels if isinstance(labels, dict) else {}


def _percentile_slice(slice_data: np.ndarray) -> Image.Image:
    finite = np.isfinite(slice_data)
    if not finite.any():
        arr = np.zeros(slice_data.shape, dtype=np.uint8)
        return Image.fromarray(arr).convert("RGB")

    valid = slice_data[finite].astype(np.float32)
    lower, upper = np.percentile(valid, [1, 99])
    if float(upper) == float(lower):
        arr = np.zeros(slice_data.shape, dtype=np.uint8)
        return Image.fromarray(arr).convert("RGB")

    clipped = np.clip(slice_data.astype(np.float32), lower, upper)
    clipped = np.nan_to_num(clipped, nan=lower, posinf=upper, neginf=lower)
    arr = ((clipped - lower) / (upper - lower) * 255.0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr).convert("RGB")


def _window_slice(slice_data: np.ndarray, window: str) -> Image.Image:
    if window.strip().lower() in {"mri", "auto", "percentile"}:
        return _percentile_slice(slice_data)

    center, width = _CT_WINDOWS.get(window, _CT_WINDOWS["lung"])
    finite = np.isfinite(slice_data)
    if not finite.any():
        arr = np.zeros(slice_data.shape, dtype=np.uint8)
        return Image.fromarray(arr).convert("RGB")

    lower = center - width / 2.0
    upper = center + width / 2.0
    clipped = np.clip(slice_data.astype(np.float32), lower, upper)
    clipped = np.nan_to_num(clipped, nan=lower, posinf=upper, neginf=lower)

    if float(clipped.max()) == float(clipped.min()):
        return _percentile_slice(slice_data)

    arr = ((clipped - lower) / (upper - lower) * 255.0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr).convert("RGB")


def load_nifti_preview(
    volume_path: str | Path,
    *,
    plane: str = "axial",
    slice_index: int | None = None,
    window: str = "lung",
) -> tuple[Image.Image, dict]:
    try:
        import nibabel as nib
    except ImportError as exc:
        raise ImportError(
            "Reading NIfTI volumes requires nibabel. "
            "Install project dependencies or instantiate the loader with load_image=False "
            "when only metadata/report text is needed."
        ) from exc

    path = Path(volume_path)
    img = nib.load(str(path))
    volume = np.asanyarray(img.dataobj)
    if volume.ndim > 3:
        volume = volume[..., 0]
    volume = np.squeeze(volume)
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D imaging volume in {path}, got shape {volume.shape}")

    axis_by_plane = {"sagittal": 0, "coronal": 1, "axial": 2}
    if plane not in axis_by_plane:
        raise ValueError(f"Unknown plane '{plane}'. Expected one of: {sorted(axis_by_plane)}")
    axis = axis_by_plane[plane]

    if slice_index is None:
        selected = volume.shape[axis] // 2
    else:
        selected = int(slice_index)
        if selected < 0 or selected >= volume.shape[axis]:
            raise IndexError(
                f"slice_index {selected} out of range for {plane} axis with "
                f"{volume.shape[axis]} slices"
            )

    slice_data = np.take(volume, selected, axis=axis)
    slice_data = np.rot90(slice_data)
    pil_image = _window_slice(slice_data, window)
    return pil_image, {
        "volume_shape": list(volume.shape),
        "slice_plane": plane,
        "slice_index": selected,
        "window": window,
    }


class JsonlChestCTLoader:
    dataset_name: str = "ChestCT"
    default_data_dir: str = ""
    default_jsonl_name: str = ""
    default_image_root: str | None = None
    study_type: str = "chest_ct"
    modality: str = "chest_ct"
    retrieval_category: str = "chest_ct"
    body_part: str = "thorax"
    prompt: str = "Generate a radiology report for this chest CT study."

    def __init__(
        self,
        data_dir: str | Path | None = None,
        jsonl_file: str | Path | None = None,
        split: str = "validation",
        load_image: bool = True,
        image_root: str | Path | None = None,
        plane: str = "axial",
        slice_index: int | None = None,
        window: str = "lung",
    ):
        self.data_dir = Path(data_dir or self.default_data_dir).expanduser()
        default_root = self.default_image_root or self.data_dir
        self.image_root = Path(image_root or default_root).expanduser()
        self.jsonl_path = _resolve_jsonl_path(self.data_dir, jsonl_file, self.default_jsonl_name)
        self.split = split
        self.load_image = load_image
        self.plane = plane
        self.slice_index = slice_index
        self.window = window

        if not self.jsonl_path.exists():
            raise FileNotFoundError(f"{self.dataset_name} JSONL file not found: {self.jsonl_path}")

        self.dataset = _read_jsonl(self.jsonl_path)
        logger.info(
            "Loaded %s %s: %d samples",
            self.dataset_name,
            self.jsonl_path.name,
            len(self.dataset),
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def _image_paths_for_record(self, record: dict) -> list[str]:
        raw_paths = record.get("image_paths")
        if raw_paths is None:
            raw_paths = record.get("image_path")
        return _coerce_image_paths(raw_paths, self.data_dir, self.image_root)

    def __getitem__(self, idx: int) -> dict:
        record = self.dataset[idx]
        image_paths = self._image_paths_for_record(record)
        report_text, findings, impression = _report_text(record)
        sample_id = record.get("sample_id") or record.get("case_id") or record.get("study_id") or ""

        image: Image.Image | None = None
        image_metadata: dict = {}
        if self.load_image:
            if not image_paths:
                raise ValueError(f"{self.dataset_name} sample {idx} has no image_paths")
            image, image_metadata = load_nifti_preview(
                image_paths[0],
                plane=self.plane,
                slice_index=self.slice_index,
                window=self.window,
            )

        return {
            "image": image,
            "image_paths": image_paths,
            "volume_path": image_paths[0] if image_paths else "",
            "text": report_text,
            "findings": findings,
            "impression": impression,
            "prompt": self.prompt,
            "index": idx,
            "sample_id": sample_id,
            "case_id": record.get("case_id", ""),
            "study_id": record.get("study_id", ""),
            "patient_id": record.get("patient_id", ""),
            "volume_name": record.get("volume_name", Path(image_paths[0]).name if image_paths else ""),
            "clinical_context": record.get("clinical_context", {}) or {},
            "labels": _record_labels(record),
            "disorders": _disorders(record),
            "metadata": record.get("metadata", {}) or {},
            "modal": record.get("modal", ""),
            "sequence": record.get("sequence", ""),
            "source_dataset": record.get("source_dataset", ""),
            "image_metadata": image_metadata,
        }

    def iter_samples(self) -> Iterator[dict]:
        for i in range(len(self)):
            yield self[i]

    def get_exam_info(self, idx: int) -> dict:
        record = self.dataset[idx]
        clinical_context = record.get("clinical_context", {}) or {}
        metadata = record.get("metadata", {}) or {}
        image_paths = self._image_paths_for_record(record)
        labels = _record_labels(record)
        positive_labels = _positive_labels(labels)
        disorders = _disorders(record)

        sample_id = record.get("sample_id") or record.get("case_id") or record.get("study_id") or ""
        volume_name = record.get("volume_name") or metadata.get("VolumeName")
        if not volume_name and image_paths:
            volume_name = Path(image_paths[0]).name

        exam_info = {
            "study_type": self.study_type,
            "modality": self.modality,
            "retrieval_category": self.retrieval_category,
            "body_part": self.body_part,
            "view": self.plane,
            "dataset": self.dataset_name,
            "split": record.get("split", self.split),
            "sample_index": idx,
            "sample_id": sample_id,
            "case_id": record.get("case_id", ""),
            "study_id": record.get("study_id", ""),
            "patient_id": record.get("patient_id", ""),
            "volume_name": volume_name or "",
            "volume_path": image_paths[0] if image_paths else "",
            "image_paths": image_paths,
            "indication": _text(clinical_context.get("indication")),
            "technique": _text(clinical_context.get("technique")),
            "comparison": _text(clinical_context.get("comparison")),
        }
        if record.get("modal"):
            exam_info["modal"] = record.get("modal")
        if record.get("sequence"):
            exam_info["sequence"] = record.get("sequence")
        if record.get("source_dataset"):
            exam_info["source_dataset"] = record.get("source_dataset")

        selected_metadata = {
            "patient_age": metadata.get("PatientAge") or metadata.get("Age") or metadata.get("age"),
            "patient_sex": metadata.get("PatientSex") or metadata.get("Gender") or metadata.get("sex"),
            "series_description": metadata.get("SeriesDescription") or metadata.get("series_description"),
            "patient_position": metadata.get("PatientPosition") or metadata.get("patient_position"),
            "number_of_slices": metadata.get("NumberofSlices") or metadata.get("number_of_slices"),
            "xy_spacing": metadata.get("XYSpacing") or metadata.get("xy_spacing"),
            "z_spacing": metadata.get("ZSpacing") or metadata.get("z_spacing"),
            "spacing": metadata.get("spacing"),
            "shape": metadata.get("shape"),
            "contrast": metadata.get("contrast"),
            "phase": metadata.get("phase"),
            "scanner": metadata.get("scanner") or metadata.get("manufacturermodelname") or metadata.get("ManufacturerModelName"),
            "manufacturer": metadata.get("manufacturer") or metadata.get("Manufacturer"),
            "slice_thickness": metadata.get("slicethickness") or metadata.get("SliceThickness"),
        }
        selected_metadata = {k: v for k, v in selected_metadata.items() if v not in (None, "")}
        if selected_metadata:
            exam_info["metadata"] = selected_metadata
        if positive_labels:
            exam_info["positive_labels"] = positive_labels
        if disorders:
            exam_info["disorders"] = disorders

        return exam_info
