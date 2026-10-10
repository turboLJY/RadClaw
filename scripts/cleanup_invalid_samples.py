#!/usr/bin/env python3
"""Remove sample output directories with incomplete agent or report outputs."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_REQUIRED_AGENTS = (
    "visual_analysis",
    "evidence_retrieval",
    "report_writing",
    "quality_review",
)

DEFAULT_REQUIRED_REPORT_FIELDS = (
    "natural_language_report",
    "structured_report",
)


@dataclass
class InvalidSample:
    sample_id: str
    sample_dir: Path
    reasons: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scan sample_* directories under an output directory, report samples whose "
            "result.json has missing/zero-length required agent output or missing final "
            "report fields, then delete them."
        )
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Output directory containing sample folders, e.g. output/chexpert/ablation_radclaw",
    )
    parser.add_argument(
        "--agents",
        nargs="+",
        default=list(DEFAULT_REQUIRED_AGENTS),
        help="Required agent names to check in agent_run_log.",
    )
    parser.add_argument(
        "--sample-prefix",
        default="sample_",
        help="Only inspect immediate child directories with this prefix.",
    )
    parser.add_argument(
        "--report-fields",
        nargs="+",
        default=list(DEFAULT_REQUIRED_REPORT_FIELDS),
        help="Required non-empty final report fields to check in result.json.",
    )
    parser.add_argument(
        "--list-file",
        type=Path,
        default=None,
        help=(
            "File to write invalid sample ids to, one per line. "
            "Default: <output_dir>/invalid_samples.txt"
        ),
    )
    parser.add_argument(
        "--no-list-file",
        action="store_true",
        help="Do not write the invalid sample id list to disk.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print invalid samples without deleting their directories.",
    )
    parser.add_argument(
        "--ignore-missing",
        action="store_true",
        help=(
            "Only mark samples invalid when a required agent has output_length == 0. "
            "By default, missing result.json, missing agent_run_log, and missing agents "
            "are also treated as invalid. Missing final report fields are still checked "
            "when result.json can be read."
        ),
    )
    return parser.parse_args()


def sample_sort_key(path: Path) -> tuple[int, int | str]:
    match = re.search(r"(\d+)$", path.name)
    if match:
        return (0, int(match.group(1)))
    return (1, path.name)


def load_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return None, "missing result.json"
    except json.JSONDecodeError as exc:
        return None, f"invalid result.json: {exc.msg}"

    if not isinstance(data, dict):
        return None, "result.json root is not an object"
    return data, None


def get_output_length(entry: dict[str, Any]) -> int | float | None:
    value = entry.get("output_length")
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            try:
                return float(value)
            except ValueError:
                return None
    return None


def has_non_empty_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (dict, list, tuple, set)):
        return bool(value)
    return True


def check_sample(
    sample_dir: Path,
    required_agents: set[str],
    required_report_fields: set[str],
    treat_missing_as_invalid: bool,
) -> InvalidSample | None:
    result_json = sample_dir / "result.json"
    data, load_error = load_json(result_json)
    if load_error:
        if treat_missing_as_invalid:
            return InvalidSample(sample_dir.name, sample_dir, [load_error])
        return None

    assert data is not None
    reasons: list[str] = []

    for field in sorted(required_report_fields):
        if not has_non_empty_value(data.get(field)):
            reasons.append(f"{field}: missing or empty")

    agent_run_log = data.get("agent_run_log")
    if not isinstance(agent_run_log, list):
        if treat_missing_as_invalid:
            reasons.append("missing or invalid agent_run_log")
        return InvalidSample(sample_dir.name, sample_dir, reasons) if reasons else None

    seen_agents: set[str] = set()

    for i, entry in enumerate(agent_run_log):
        if not isinstance(entry, dict):
            if treat_missing_as_invalid:
                reasons.append(f"agent_run_log[{i}] is not an object")
            continue

        agent = entry.get("agent")
        if agent not in required_agents:
            continue

        seen_agents.add(agent)
        output_length = get_output_length(entry)
        if output_length is None:
            if treat_missing_as_invalid:
                reasons.append(f"{agent}: missing or invalid output_length")
        elif output_length == 0:
            iteration = entry.get("iteration", "?")
            reasons.append(f"{agent} iteration {iteration}: output_length=0")

    if treat_missing_as_invalid:
        missing_agents = sorted(required_agents - seen_agents)
        reasons.extend(f"{agent}: missing from agent_run_log" for agent in missing_agents)

    if reasons:
        return InvalidSample(sample_dir.name, sample_dir, reasons)
    return None


def find_invalid_samples(
    output_dir: Path,
    sample_prefix: str,
    required_agents: set[str],
    required_report_fields: set[str],
    treat_missing_as_invalid: bool,
) -> list[InvalidSample]:
    invalid_samples: list[InvalidSample] = []
    sample_dirs = sorted(
        (
            path
            for path in output_dir.iterdir()
            if path.is_dir() and path.name.startswith(sample_prefix)
        ),
        key=sample_sort_key,
    )

    for sample_dir in sample_dirs:
        invalid_sample = check_sample(
            sample_dir=sample_dir,
            required_agents=required_agents,
            required_report_fields=required_report_fields,
            treat_missing_as_invalid=treat_missing_as_invalid,
        )
        if invalid_sample is not None:
            invalid_samples.append(invalid_sample)

    return invalid_samples


def write_invalid_list(path: Path, invalid_samples: list[InvalidSample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for sample in invalid_samples:
            f.write(f"{sample.sample_id}\n")


def remove_invalid_dirs(invalid_samples: list[InvalidSample]) -> None:
    for sample in invalid_samples:
        shutil.rmtree(sample.sample_dir)


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()

    if not output_dir.exists():
        print(f"ERROR: output_dir does not exist: {output_dir}")
        return 2
    if not output_dir.is_dir():
        print(f"ERROR: output_dir is not a directory: {output_dir}")
        return 2

    required_agents = set(args.agents)
    required_report_fields = set(args.report_fields)
    invalid_samples = find_invalid_samples(
        output_dir=output_dir,
        sample_prefix=args.sample_prefix,
        required_agents=required_agents,
        required_report_fields=required_report_fields,
        treat_missing_as_invalid=not args.ignore_missing,
    )

    print(f"Scanned: {output_dir}")
    print(f"Invalid samples: {len(invalid_samples)}")

    for sample in invalid_samples:
        reasons = "; ".join(sample.reasons)
        print(f"{sample.sample_id}\t{reasons}")

    if not args.no_list_file:
        list_file = args.list_file or output_dir / "invalid_samples.txt"
        write_invalid_list(list_file, invalid_samples)
        print(f"Wrote invalid sample list: {list_file}")

    if args.dry_run:
        print("Dry run: no directories deleted.")
        return 0

    remove_invalid_dirs(invalid_samples)
    print(f"Deleted sample directories: {len(invalid_samples)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
