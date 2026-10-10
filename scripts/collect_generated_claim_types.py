#!/usr/bin/env python3
"""Collect generated claim_type values from hallucination detection outputs."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scan sample_* folders under one or more output directories and report "
            "unique generated_claims[*].claim_type values from hallucination_detection.json."
        )
    )
    parser.add_argument(
        "target_dirs",
        nargs="+",
        type=Path,
        help="Output directories containing sample_* folders, e.g. output/mimic/full_radclaw.",
    )
    parser.add_argument(
        "--sample-prefix",
        default="sample_",
        help="Only scan immediate child directories with this prefix.",
    )
    parser.add_argument(
        "--filename",
        default="hallucination_detection.json",
        help="Per-sample hallucination detection JSON filename.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Write machine-readable JSON instead of text.",
    )
    return parser.parse_args()


def sample_sort_key(path: Path) -> tuple[int, int | str]:
    match = re.search(r"(\d+)$", path.name)
    if match:
        return (0, int(match.group(1)))
    return (1, path.name)


def warn(message: str) -> None:
    print(f"warning: {message}", file=sys.stderr)


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        warn(f"{path}: invalid JSON: {exc.msg}")
        return None
    except OSError as exc:
        warn(f"{path}: cannot read file: {exc}")
        return None

    if not isinstance(data, dict):
        warn(f"{path}: root JSON value is not an object")
        return None
    return data


def iter_sample_dirs(target_dir: Path, sample_prefix: str) -> list[Path]:
    if not target_dir.exists():
        warn(f"{target_dir}: directory does not exist")
        return []
    if not target_dir.is_dir():
        warn(f"{target_dir}: not a directory")
        return []
    return sorted(
        (
            path
            for path in target_dir.iterdir()
            if path.is_dir() and path.name.startswith(sample_prefix)
        ),
        key=sample_sort_key,
    )


def collect_claim_types(
    target_dirs: list[Path],
    *,
    sample_prefix: str,
    filename: str,
) -> dict[str, Any]:
    claim_type_counts: Counter[str] = Counter()
    per_target_counts: dict[str, Counter[str]] = {}
    scanned_samples = 0
    samples_with_file = 0
    generated_claims = 0
    claims_missing_type = 0

    for target_dir in target_dirs:
        target_counter: Counter[str] = Counter()
        for sample_dir in iter_sample_dirs(target_dir, sample_prefix):
            scanned_samples += 1
            json_path = sample_dir / filename
            if not json_path.exists():
                continue

            samples_with_file += 1
            data = load_json(json_path)
            if data is None:
                continue

            claims = data.get("generated_claims")
            if not isinstance(claims, list):
                warn(f"{json_path}: generated_claims is not a list")
                continue

            for index, claim in enumerate(claims):
                generated_claims += 1
                if not isinstance(claim, dict):
                    warn(f"{json_path}: generated_claims[{index}] is not an object")
                    claims_missing_type += 1
                    continue

                claim_type = claim.get("claim_type")
                if not isinstance(claim_type, str) or not claim_type.strip():
                    claims_missing_type += 1
                    continue

                clean_type = claim_type.strip()
                claim_type_counts[clean_type] += 1
                target_counter[clean_type] += 1

        per_target_counts[str(target_dir)] = target_counter

    return {
        "target_dirs": [str(path) for path in target_dirs],
        "scanned_samples": scanned_samples,
        "samples_with_file": samples_with_file,
        "generated_claims": generated_claims,
        "claims_missing_type": claims_missing_type,
        "unique_claim_type_count": len(claim_type_counts),
        "claim_type_counts": dict(sorted(claim_type_counts.items())),
        "per_target_claim_type_counts": {
            target: dict(sorted(counter.items()))
            for target, counter in per_target_counts.items()
        },
    }


def print_text(summary: dict[str, Any]) -> None:
    print("Scanned target dirs:")
    for target_dir in summary["target_dirs"]:
        print(f"  {target_dir}")

    print()
    print(f"sample folders scanned: {summary['scanned_samples']}")
    print(f"samples with hallucination JSON: {summary['samples_with_file']}")
    print(f"generated claims scanned: {summary['generated_claims']}")
    print(f"claims missing claim_type: {summary['claims_missing_type']}")
    print(f"unique claim_type count: {summary['unique_claim_type_count']}")

    print()
    print("unique generated claim_type values:")
    for claim_type in summary["claim_type_counts"]:
        print(f"  {claim_type}")

    print()
    print("counts by claim_type:")
    for claim_type, count in summary["claim_type_counts"].items():
        print(f"  {claim_type}: {count}")


def main() -> int:
    args = parse_args()
    summary = collect_claim_types(
        args.target_dirs,
        sample_prefix=args.sample_prefix,
        filename=args.filename,
    )
    if args.json:
        json.dump(summary, sys.stdout, ensure_ascii=False, indent=2)
        print()
    else:
        print_text(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
