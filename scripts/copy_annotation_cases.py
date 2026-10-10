#!/usr/bin/env python3
"""
Copy retrieved similar-case JSON files into annotation sample folders.

For each sample under annotation_samples/, this script reads result.json,
collects retrieved_evidence entries where source_type == "case", resolves
metadata.case_id to <case_id>.json under the MIMIC-CXR case base, and copies
the matched JSON files into <sample>/cases/.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_ANNOTATION_DIR = Path("annotation_samples")
DEFAULT_CASEBASE_DIR = Path(
    "/mnt/yulan_pretrain_131/clinical_project/CaseBase/MIMIC-CXR/train"
)


@dataclass
class CopyRecord:
    sample: str
    case_id: str
    source_path: str | None
    destination_path: str | None
    status: str


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser(
        description="Copy original similar-case JSON files for annotation samples."
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
        default=DEFAULT_ANNOTATION_DIR,
        help="Directory containing selected annotation sample folders.",
    )
    parser.add_argument(
        "--casebase-dir",
        type=Path,
        default=DEFAULT_CASEBASE_DIR,
        help="Root directory to search for <case_id>.json files.",
    )
    parser.add_argument(
        "--cases-subdir",
        type=str,
        default="cases",
        help="Subdirectory created inside each sample folder for copied case JSONs.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite already copied case JSON files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be copied without writing files.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any required case_id cannot be found.",
    )
    return parser.parse_args()


def resolve_path(repo_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def iter_report_items(structured_report: dict[str, Any]):
    for section in ("findings", "impressions"):
        entries = structured_report.get(section) or []
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if isinstance(entry, dict):
                yield section, entry


def collect_sample_case_ids(annotation_dir: Path) -> dict[Path, set[str]]:
    sample_case_ids: dict[Path, set[str]] = {}

    for result_path in sorted(annotation_dir.glob("*/result.json")):
        try:
            data = load_json(result_path)
        except json.JSONDecodeError as exc:
            print(f"[WARN] Invalid result.json, skipping {result_path}: {exc}")
            continue

        structured_report = data.get("structured_report") or {}
        if not isinstance(structured_report, dict):
            continue

        case_ids: set[str] = set()
        for _, entry in iter_report_items(structured_report):
            support = entry.get("support") or {}
            if not isinstance(support, dict):
                continue
            retrieved = support.get("retrieved_evidence") or []
            if not isinstance(retrieved, list):
                continue
            for evidence in retrieved:
                if not isinstance(evidence, dict):
                    continue
                if evidence.get("source_type") != "case":
                    continue
                metadata = evidence.get("metadata") or {}
                if not isinstance(metadata, dict):
                    continue
                case_id = metadata.get("case_id")
                if case_id:
                    case_ids.add(str(case_id))

        if case_ids:
            sample_case_ids[result_path.parent] = case_ids

    return sample_case_ids


def build_case_index(casebase_dir: Path, needed_case_ids: set[str]) -> dict[str, Path]:
    index: dict[str, Path] = {}
    remaining = set(needed_case_ids)

    for path in casebase_dir.rglob("*.json"):
        case_id = path.stem
        if case_id not in remaining:
            continue
        index[case_id] = path
        remaining.remove(case_id)
        if not remaining:
            break

    return index


def copy_cases(
    sample_case_ids: dict[Path, set[str]],
    case_index: dict[str, Path],
    cases_subdir: str,
    overwrite: bool,
    dry_run: bool,
) -> list[CopyRecord]:
    records: list[CopyRecord] = []

    for sample_dir, case_ids in sorted(sample_case_ids.items(), key=lambda item: item[0].name):
        destination_dir = sample_dir / cases_subdir
        if case_ids and not dry_run:
            destination_dir.mkdir(parents=True, exist_ok=True)

        for case_id in sorted(case_ids):
            source_path = case_index.get(case_id)
            if source_path is None:
                records.append(
                    CopyRecord(
                        sample=sample_dir.name,
                        case_id=case_id,
                        source_path=None,
                        destination_path=None,
                        status="missing",
                    )
                )
                continue

            destination_path = destination_dir / source_path.name
            if destination_path.exists() and not overwrite:
                status = "exists"
            else:
                status = "would_copy" if dry_run else "copied"
                if not dry_run:
                    shutil.copy2(source_path, destination_path)

            records.append(
                CopyRecord(
                    sample=sample_dir.name,
                    case_id=case_id,
                    source_path=str(source_path),
                    destination_path=str(destination_path),
                    status=status,
                )
            )

    return records


def write_manifest(
    annotation_dir: Path,
    casebase_dir: Path,
    records: list[CopyRecord],
    dry_run: bool,
) -> None:
    by_status: dict[str, int] = defaultdict(int)
    by_sample: dict[str, int] = defaultdict(int)
    for record in records:
        by_status[record.status] += 1
        if record.status in {"copied", "exists", "would_copy"}:
            by_sample[record.sample] += 1

    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "annotation_dir": str(annotation_dir),
        "casebase_dir": str(casebase_dir),
        "dry_run": dry_run,
        "num_records": len(records),
        "status_counts": dict(sorted(by_status.items())),
        "sample_case_counts": dict(sorted(by_sample.items())),
        "records": [asdict(record) for record in records],
    }

    manifest_path = annotation_dir / "case_copy_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"Wrote manifest: {manifest_path}")


def print_summary(records: list[CopyRecord]) -> None:
    by_status: dict[str, int] = defaultdict(int)
    by_sample: dict[str, int] = defaultdict(int)
    for record in records:
        by_status[record.status] += 1
        if record.status in {"copied", "exists", "would_copy"}:
            by_sample[record.sample] += 1

    print("Status counts:")
    for status, count in sorted(by_status.items()):
        print(f"  {status}: {count}")

    print("Samples with copied/found cases:")
    for sample, count in sorted(by_sample.items()):
        print(f"  {sample}: {count}")


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    annotation_dir = resolve_path(repo_root, args.annotation_dir)
    casebase_dir = args.casebase_dir

    if not annotation_dir.exists():
        raise FileNotFoundError(f"Annotation directory does not exist: {annotation_dir}")
    if not casebase_dir.exists():
        raise FileNotFoundError(f"CaseBase directory does not exist: {casebase_dir}")

    sample_case_ids = collect_sample_case_ids(annotation_dir)
    needed_case_ids = set().union(*sample_case_ids.values()) if sample_case_ids else set()
    print(
        f"Found {len(needed_case_ids)} unique case IDs across "
        f"{len(sample_case_ids)} annotation samples."
    )

    if not needed_case_ids:
        return

    case_index = build_case_index(casebase_dir, needed_case_ids)
    missing = sorted(needed_case_ids - set(case_index))
    if missing:
        message = f"Missing {len(missing)} case JSON files: {', '.join(missing)}"
        if args.strict:
            raise FileNotFoundError(message)
        print(f"[WARN] {message}")

    records = copy_cases(
        sample_case_ids=sample_case_ids,
        case_index=case_index,
        cases_subdir=args.cases_subdir,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )
    print_summary(records)
    if not args.dry_run:
        write_manifest(annotation_dir, casebase_dir, records, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
