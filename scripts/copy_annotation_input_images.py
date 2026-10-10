#!/usr/bin/env python3
"""Export original input chest X-ray images for annotation samples."""

from __future__ import annotations

import argparse
import io
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from PIL import Image


DEFAULT_DATASET_PATHS = {
    "chexpertplus": Path("/mnt/yulan_pretrain_131/clinical_project/hf/Doorad/CheXpert-plus-RRG"),
    "iu_xray": Path("/mnt/yulan_pretrain_131/clinical_project/hf/X-iZhang/IU-Xray-RRG"),
    "mimic": Path("/mnt/yulan_pretrain_131/clinical_project/hf/Yamini-1628/MIMIC-CXR-RRG"),
    "openi": Path("/mnt/yulan_pretrain_131/clinical_project/hf/ykumards/open-i"),
}

SAMPLE_RE = re.compile(r"^(?P<dataset>chexpertplus|iu_xray|mimic|openi)_sample_(?P<id>\d+)$")


@dataclass(frozen=True)
class ExportRecord:
    sample: str
    dataset: str
    sample_id: int
    hf_row_index: int
    source_parquet: str
    image_column: str
    output_path: str
    status: str


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser(
        description="Copy original HF input images to annotation_samples/*/images/input.png."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=repo_root,
        help="Repository root used to resolve relative annotation paths.",
    )
    parser.add_argument(
        "--annotation-dir",
        type=Path,
        default=Path("annotation_samples"),
        help="Directory containing annotation sample folders.",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="input.png",
        help="Output filename inside each sample's images directory.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing images/input.png.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report exports without writing image files.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on unsupported sample folders instead of skipping them.",
    )
    return parser.parse_args()


def resolve_path(repo_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def parse_sample_dir(sample_dir: Path) -> tuple[str, int] | None:
    match = SAMPLE_RE.match(sample_dir.name)
    if not match:
        return None
    return match.group("dataset"), int(match.group("id"))


def sorted_parquet_files(path: Path) -> list[Path]:
    files = sorted(path.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {path}")
    return files


class ParquetImageReader:
    def __init__(self, files: list[Path], image_column: str):
        self.files = files
        self.image_column = image_column
        self._table_cache: dict[Path, Any] = {}
        self._ranges: list[tuple[int, int, Path]] = []

        start = 0
        for path in files:
            rows = pq.ParquetFile(path).metadata.num_rows
            self._ranges.append((start, start + rows, path))
            start += rows
        self.num_rows = start

    def locate(self, row_index: int) -> tuple[Path, int]:
        if row_index < 0 or row_index >= self.num_rows:
            raise IndexError(f"Row index {row_index} out of range 0..{self.num_rows - 1}")
        for start, end, path in self._ranges:
            if start <= row_index < end:
                return path, row_index - start
        raise IndexError(f"Row index {row_index} could not be located")

    def read_image_value(self, row_index: int) -> tuple[Any, Path]:
        path, local_index = self.locate(row_index)
        if path not in self._table_cache:
            self._table_cache[path] = pq.read_table(path, columns=[self.image_column])
        values = self._table_cache[path][self.image_column].slice(local_index, 1).to_pylist()
        if not values:
            raise RuntimeError(f"No image value at row {row_index} in {path}")
        return values[0], path


def image_bytes_from_value(value: Any, dataset_root: Path) -> bytes:
    if isinstance(value, dict):
        raw = value.get("bytes")
        if raw is not None:
            return raw
        image_path = value.get("path")
        if image_path:
            path = Path(image_path)
            if not path.is_absolute():
                path = dataset_root / path
            return path.read_bytes()
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    raise TypeError(f"Unsupported image value type: {type(value)!r}")


def save_png(image_bytes: bytes, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(io.BytesIO(image_bytes)) as image:
        image.convert("RGB").save(output_path, format="PNG")


def load_openi_test_indices(repo_root: Path) -> list[int]:
    split_path = repo_root / "data" / "openi_test_split.json"
    if not split_path.exists():
        raise FileNotFoundError(
            f"Open-I split file not found: {split_path}. "
            "This file is needed to map openi_sample_<id> to the original parquet row."
        )
    with split_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    indices = payload.get("test_indices")
    if not isinstance(indices, list):
        raise ValueError(f"{split_path} must contain a list field named test_indices")
    return [int(index) for index in indices]


def build_readers(repo_root: Path) -> tuple[dict[str, ParquetImageReader], list[int]]:
    readers = {
        "chexpertplus": ParquetImageReader(
            sorted_parquet_files(DEFAULT_DATASET_PATHS["chexpertplus"] / "findings_section"),
            image_column="main_image",
        ),
        "iu_xray": ParquetImageReader(
            sorted_parquet_files(DEFAULT_DATASET_PATHS["iu_xray"] / "findings_section"),
            image_column="main_image",
        ),
        "mimic": ParquetImageReader(
            sorted_parquet_files(DEFAULT_DATASET_PATHS["mimic"] / "findings_section"),
            image_column="main_image",
        ),
        "openi": ParquetImageReader(
            sorted_parquet_files(DEFAULT_DATASET_PATHS["openi"] / "data"),
            image_column="img_frontal",
        ),
    }
    return readers, load_openi_test_indices(repo_root)


def hf_row_index(dataset: str, sample_id: int, openi_test_indices: list[int]) -> int:
    if dataset == "openi":
        if sample_id < 0 or sample_id >= len(openi_test_indices):
            raise IndexError(
                f"Open-I sample index {sample_id} out of test split range 0..{len(openi_test_indices) - 1}"
            )
        return openi_test_indices[sample_id]
    return sample_id


def export_sample(
    sample_dir: Path,
    dataset: str,
    sample_id: int,
    reader: ParquetImageReader,
    openi_test_indices: list[int],
    output_name: str,
    overwrite: bool,
    dry_run: bool,
) -> ExportRecord:
    row_index = hf_row_index(dataset, sample_id, openi_test_indices)
    output_path = sample_dir / "images" / output_name
    value, source_parquet = reader.read_image_value(row_index)

    if output_path.exists() and not overwrite:
        status = "exists"
    else:
        status = "would_write" if dry_run else "written"
        if not dry_run:
            image_bytes = image_bytes_from_value(value, DEFAULT_DATASET_PATHS[dataset])
            save_png(image_bytes, output_path)

    return ExportRecord(
        sample=sample_dir.name,
        dataset=dataset,
        sample_id=sample_id,
        hf_row_index=row_index,
        source_parquet=str(source_parquet),
        image_column=reader.image_column,
        output_path=str(output_path),
        status=status,
    )


def write_manifest(annotation_dir: Path, records: list[ExportRecord], dry_run: bool) -> None:
    status_counts: dict[str, int] = {}
    for record in records:
        status_counts[record.status] = status_counts.get(record.status, 0) + 1

    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "annotation_dir": str(annotation_dir),
        "dry_run": dry_run,
        "dataset_paths": {name: str(path) for name, path in DEFAULT_DATASET_PATHS.items()},
        "status_counts": dict(sorted(status_counts.items())),
        "records": [asdict(record) for record in records],
    }

    manifest_path = annotation_dir / "input_image_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"Wrote manifest: {manifest_path}")


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    annotation_dir = resolve_path(repo_root, args.annotation_dir)

    if not annotation_dir.exists():
        raise FileNotFoundError(f"Annotation directory does not exist: {annotation_dir}")

    readers, openi_test_indices = build_readers(repo_root)
    records: list[ExportRecord] = []
    skipped: list[str] = []

    for sample_dir in sorted(annotation_dir.iterdir(), key=lambda path: path.name):
        if not sample_dir.is_dir():
            continue
        parsed = parse_sample_dir(sample_dir)
        if parsed is None:
            skipped.append(sample_dir.name)
            continue
        dataset, sample_id = parsed
        record = export_sample(
            sample_dir=sample_dir,
            dataset=dataset,
            sample_id=sample_id,
            reader=readers[dataset],
            openi_test_indices=openi_test_indices,
            output_name=args.output_name,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )
        records.append(record)

    if skipped:
        message = f"Skipped unsupported sample folders: {', '.join(skipped)}"
        if args.strict:
            raise RuntimeError(message)
        print(f"[WARN] {message}")

    status_counts: dict[str, int] = {}
    for record in records:
        status_counts[record.status] = status_counts.get(record.status, 0) + 1
    print(f"Processed {len(records)} samples: {dict(sorted(status_counts.items()))}")

    if not args.dry_run:
        write_manifest(annotation_dir, records, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
