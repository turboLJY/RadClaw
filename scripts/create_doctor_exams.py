#!/usr/bin/env python3
"""Create a balanced incomplete crossover assignment for two doctors."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path


CONDITIONS = ("manual", "ai_draft_only", "radclaw")
DOCTORS = ("doctor_1", "doctor_2")
PAIRINGS = (
    ("manual_vs_ai_draft_only", "manual", "ai_draft_only"),
    ("manual_vs_radclaw", "manual", "radclaw"),
    ("ai_draft_only_vs_radclaw", "ai_draft_only", "radclaw"),
)
SAMPLE_RE = re.compile(r"^(?P<dataset>.+)_sample_(?P<id>.+)$")


@dataclass(frozen=True)
class Assignment:
    sample: str
    dataset: str
    sample_id: str
    pairing: str
    orientation: str
    doctor_1_condition: str
    doctor_2_condition: str


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Split 30 annotation samples across two doctors and three conditions."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=repo_root,
        help="Repository root used to resolve relative paths.",
    )
    parser.add_argument(
        "--annotation-dir",
        type=Path,
        default=Path("annotation_samples"),
        help="Directory containing the 30 selected sample folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("Doctor_exams"),
        help="Output directory for doctor exam folders.",
    )
    parser.add_argument(
        "--expected-samples",
        type=int,
        default=30,
        help="Expected number of sample folders to assign.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output directory.",
    )
    return parser.parse_args()


def resolve_path(repo_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def sample_sort_key(path: Path) -> tuple[str, int, int | str]:
    match = SAMPLE_RE.match(path.name)
    if not match:
        return (path.name, 1, path.name)
    sample_id = match.group("id")
    parsed_id: int | str = int(sample_id) if sample_id.isdigit() else sample_id
    return (match.group("dataset"), 0 if sample_id.isdigit() else 1, parsed_id)


def parse_sample_name(sample_name: str) -> tuple[str, str]:
    match = SAMPLE_RE.match(sample_name)
    if not match:
        raise ValueError(f"Sample directory name does not match *_sample_*: {sample_name}")
    return match.group("dataset"), match.group("id")


def load_samples(annotation_dir: Path, expected_samples: int) -> list[Path]:
    samples = sorted(
        [path for path in annotation_dir.iterdir() if path.is_dir() and SAMPLE_RE.match(path.name)],
        key=sample_sort_key,
    )
    if len(samples) != expected_samples:
        raise RuntimeError(
            f"Expected {expected_samples} sample folders in {annotation_dir}, found {len(samples)}"
        )
    return samples


def assign_pairing_groups(samples: list[Path]) -> dict[str, list[Path]]:
    groups = {pairing[0]: [] for pairing in PAIRINGS}
    pair_names = [pairing[0] for pairing in PAIRINGS]

    for idx, sample in enumerate(samples):
        pair_name = pair_names[idx % len(pair_names)]
        groups[pair_name].append(sample)

    for pair_name, group in groups.items():
        if len(group) != 10:
            raise RuntimeError(f"{pair_name} expected 10 samples, found {len(group)}")
    return groups


def build_assignments(groups: dict[str, list[Path]]) -> list[Assignment]:
    assignments: list[Assignment] = []
    pairing_lookup = {name: (condition_a, condition_b) for name, condition_a, condition_b in PAIRINGS}

    for pairing_name, group in groups.items():
        condition_a, condition_b = pairing_lookup[pairing_name]
        ordered_group = sorted(group, key=sample_sort_key)
        for idx, sample in enumerate(ordered_group):
            dataset, sample_id = parse_sample_name(sample.name)
            if idx % 2 == 0:
                doctor_1_condition = condition_a
                doctor_2_condition = condition_b
                orientation = f"doctor_1_{condition_a}__doctor_2_{condition_b}"
            else:
                doctor_1_condition = condition_b
                doctor_2_condition = condition_a
                orientation = f"doctor_1_{condition_b}__doctor_2_{condition_a}"

            assignments.append(
                Assignment(
                    sample=sample.name,
                    dataset=dataset,
                    sample_id=sample_id,
                    pairing=pairing_name,
                    orientation=orientation,
                    doctor_1_condition=doctor_1_condition,
                    doctor_2_condition=doctor_2_condition,
                )
            )

    return sorted(assignments, key=lambda item: (item.pairing, item.sample))


def validate_assignments(assignments: list[Assignment]) -> dict[str, dict[str, int]]:
    if len(assignments) != 30:
        raise RuntimeError(f"Expected 30 assignments, found {len(assignments)}")

    doctor_condition_counts: dict[str, Counter[str]] = {
        doctor: Counter() for doctor in DOCTORS
    }
    pair_counts: Counter[str] = Counter()
    orientation_counts: dict[str, Counter[str]] = defaultdict(Counter)
    sample_counts: Counter[str] = Counter()

    for item in assignments:
        if item.doctor_1_condition == item.doctor_2_condition:
            raise RuntimeError(f"Sample has duplicate condition across doctors: {item.sample}")
        doctor_condition_counts["doctor_1"][item.doctor_1_condition] += 1
        doctor_condition_counts["doctor_2"][item.doctor_2_condition] += 1
        pair_counts[item.pairing] += 1
        orientation_counts[item.pairing][item.orientation] += 1
        sample_counts[item.sample] += 2

    for doctor, counts in doctor_condition_counts.items():
        for condition in CONDITIONS:
            if counts[condition] != 10:
                raise RuntimeError(f"{doctor}/{condition} expected 10 samples, found {counts[condition]}")

    for pairing_name, _, _ in PAIRINGS:
        if pair_counts[pairing_name] != 10:
            raise RuntimeError(f"{pairing_name} expected 10 samples, found {pair_counts[pairing_name]}")
        if sorted(orientation_counts[pairing_name].values()) != [5, 5]:
            raise RuntimeError(
                f"{pairing_name} expected two 5-case orientations, got {dict(orientation_counts[pairing_name])}"
            )

    if any(count != 2 for count in sample_counts.values()):
        raise RuntimeError("Each sample must appear exactly twice, once per doctor")

    return {
        "doctor_1": dict(sorted(doctor_condition_counts["doctor_1"].items())),
        "doctor_2": dict(sorted(doctor_condition_counts["doctor_2"].items())),
        "pairings": dict(sorted(pair_counts.items())),
    }


def prepare_output(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory already exists: {output_dir}")
        shutil.rmtree(output_dir)

    for doctor in DOCTORS:
        for condition in CONDITIONS:
            (output_dir / doctor / condition).mkdir(parents=True, exist_ok=True)


def copy_assignments(
    annotation_dir: Path,
    output_dir: Path,
    assignments: list[Assignment],
) -> None:
    for item in assignments:
        source = annotation_dir / item.sample
        destinations = (
            output_dir / "doctor_1" / item.doctor_1_condition / item.sample,
            output_dir / "doctor_2" / item.doctor_2_condition / item.sample,
        )
        for destination in destinations:
            shutil.copytree(source, destination, symlinks=True)


def write_manifest(
    output_dir: Path,
    annotation_dir: Path,
    assignments: list[Assignment],
    summary: dict[str, dict[str, int]],
) -> None:
    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "annotation_dir": str(annotation_dir),
        "design": {
            "num_cases": 30,
            "num_doctors": 2,
            "conditions": list(CONDITIONS),
            "pairings": [
                {"name": name, "conditions": [condition_a, condition_b]}
                for name, condition_a, condition_b in PAIRINGS
            ],
            "tasks_per_doctor": 30,
            "tasks_per_condition_per_doctor": 10,
            "tasks_total": 60,
        },
        "summary": summary,
        "assignments": [asdict(item) for item in assignments],
    }

    with (output_dir / "assignment_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    with (output_dir / "assignment_manifest.tsv").open("w", encoding="utf-8") as f:
        f.write(
            "sample\tdataset\tsample_id\tpairing\torientation\t"
            "doctor_1_condition\tdoctor_2_condition\n"
        )
        for item in assignments:
            f.write(
                f"{item.sample}\t{item.dataset}\t{item.sample_id}\t{item.pairing}\t"
                f"{item.orientation}\t{item.doctor_1_condition}\t{item.doctor_2_condition}\n"
            )


def print_summary(summary: dict[str, dict[str, int]], assignments: list[Assignment]) -> None:
    print("Doctor condition counts:")
    for doctor in DOCTORS:
        print(f"  {doctor}: {summary[doctor]}")
    print("Pairing counts:")
    print(f"  {summary['pairings']}")
    print(f"Total reading tasks: {len(assignments) * 2}")


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    annotation_dir = resolve_path(repo_root, args.annotation_dir)
    output_dir = resolve_path(repo_root, args.output_dir)

    if not annotation_dir.exists():
        raise FileNotFoundError(f"Annotation directory does not exist: {annotation_dir}")

    samples = load_samples(annotation_dir, expected_samples=args.expected_samples)
    groups = assign_pairing_groups(samples)
    assignments = build_assignments(groups)
    summary = validate_assignments(assignments)

    prepare_output(output_dir, overwrite=args.overwrite)
    copy_assignments(annotation_dir, output_dir, assignments)
    write_manifest(output_dir, annotation_dir, assignments, summary)
    print_summary(summary, assignments)
    print(f"Wrote doctor exam folders to {output_dir}")
    print(f"Wrote manifest: {output_dir / 'assignment_manifest.json'}")


if __name__ == "__main__":
    main()
