#!/usr/bin/env python3
"""
Select high-quality, difficult Chest CT samples for physician annotation.

scripts/evaluate_chest_ct.py reports corpus-level Macro-F1 and Micro-F1, but
does not write per-sample clinical F1 values. This selector computes per-sample
CheXbert labels once, caches them in each dataset directory, then ranks samples
by medical quality first and difficulty second:

  1. clinical_score = weighted average of Macro-F1 and Micro-F1
  2. Macro-F1
  3. Micro-F1
  4. iterations from sample_<id>/summary.json

Selected sample folders are copied into annotation_samples/ and renamed as
<dataset>_sample_<id>. Source output folders are not modified.
"""

from __future__ import annotations

import argparse
import ast
import gc
import json
import math
import re
import shutil
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_DATASET_DIRS = {
    "ct_rate": Path("output/ct_rate/full_radclaw"),
    "radgenome": Path("output/radgenome/full_radclaw"),
}
DATASET_ORDER = tuple(DEFAULT_DATASET_DIRS.keys())
SAMPLE_DIR_RE = re.compile(r"^sample_(?P<id>.+)$")


@dataclass(frozen=True)
class Candidate:
    rank: int | None
    dataset: str
    sample_id: str
    sample_index: int | str
    iterations: int
    macro_f1: float
    micro_f1: float
    clinical_score: float
    source_dir: str
    destination_dir: str | None = None


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser(
        description="Select high-quality and difficult Chest CT samples."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=repo_root,
        help="Repository root used to resolve relative input/output paths.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("annotation_samples"),
        help="Directory for renamed selected sample copies.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=15,
        help="Total number of samples to select when --dataset-counts is not set.",
    )
    parser.add_argument(
        "--dataset-counts",
        type=str,
        default=None,
        help=(
            "Optional per-dataset sample counts in order ct_rate,radgenome, "
            "e.g. '[8,7]' or '8,7'. Overrides --limit."
        ),
    )
    parser.add_argument(
        "--macro-weight",
        type=float,
        default=0.5,
        help="Weight for Macro-F1 in the clinical score.",
    )
    parser.add_argument(
        "--micro-weight",
        type=float,
        default=0.5,
        help="Weight for Micro-F1 in the clinical score.",
    )
    parser.add_argument(
        "--min-macro-f1",
        type=float,
        default=None,
        help="Optional minimum Macro-F1 threshold.",
    )
    parser.add_argument(
        "--min-micro-f1",
        type=float,
        default=None,
        help="Optional minimum Micro-F1 threshold.",
    )
    parser.add_argument(
        "--metric-cache-name",
        type=str,
        default="clinical_f1_per_sample.json",
        help="Per-dataset cache filename for computed per-sample clinical F1.",
    )
    parser.add_argument(
        "--force-recompute-metrics",
        action="store_true",
        help="Ignore existing per-sample clinical F1 cache and recompute it.",
    )
    parser.add_argument(
        "--no-compute-missing-metrics",
        action="store_true",
        help="Do not compute missing per-sample clinical F1 caches.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing destination sample folders.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print selected samples without copying folders.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if a default dataset directory is missing.",
    )
    parser.add_argument(
        "--allow-fewer",
        action="store_true",
        help="Allow copying fewer than the requested sample count.",
    )
    return parser.parse_args()


def parse_dataset_counts(raw_counts: str | None) -> dict[str, int] | None:
    if raw_counts is None:
        return None

    text = raw_counts.strip()
    if not text:
        raise ValueError("--dataset-counts cannot be empty")

    try:
        if text.startswith("["):
            values = ast.literal_eval(text)
        else:
            values = [part.strip() for part in text.split(",")]
    except (SyntaxError, ValueError) as exc:
        raise ValueError("--dataset-counts must look like '[8,7]' or '8,7'") from exc

    if not isinstance(values, (list, tuple)):
        raise ValueError("--dataset-counts must be a list of non-negative integers")
    if len(values) != len(DATASET_ORDER):
        raise ValueError(
            f"--dataset-counts must contain {len(DATASET_ORDER)} values in order: "
            f"{','.join(DATASET_ORDER)}"
        )

    counts: dict[str, int] = {}
    for dataset, value in zip(DATASET_ORDER, values):
        if isinstance(value, bool):
            raise ValueError("--dataset-counts values must be integers")
        try:
            count = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("--dataset-counts values must be integers") from exc
        if count < 0:
            raise ValueError("--dataset-counts values must be non-negative")
        counts[dataset] = count

    if sum(counts.values()) <= 0:
        raise ValueError("--dataset-counts must request at least one sample")
    return counts


def resolve_path(repo_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def sample_sort_key(path: Path) -> tuple[int, int | str]:
    match = SAMPLE_DIR_RE.match(path.name)
    if not match:
        return (2, path.name)
    sample_id = match.group("id")
    if sample_id.isdigit():
        return (0, int(sample_id))
    return (1, sample_id)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def to_finite_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
    elif isinstance(value, str):
        try:
            number = float(value)
        except ValueError:
            return None
    else:
        return None
    return number if math.isfinite(number) else None


def to_int(value: Any, default: int = 0) -> int:
    if value is None or isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def metric_from_record(record: dict[str, Any], names: tuple[str, ...]) -> float | None:
    for name in names:
        if name in record:
            value = to_finite_float(record[name])
            if value is not None:
                return value
    return None


def discover_sample_metadata(dataset_dir: Path) -> dict[str, dict[str, Any]]:
    metadata: dict[str, dict[str, Any]] = {}

    for sample_dir in sorted(dataset_dir.glob("sample_*"), key=sample_sort_key):
        if not sample_dir.is_dir():
            continue
        match = SAMPLE_DIR_RE.match(sample_dir.name)
        if not match:
            continue

        sample_id = match.group("id")
        summary_path = sample_dir / "summary.json"
        if not summary_path.exists():
            print(f"[WARN] Missing summary.json, skipping: {sample_dir}")
            continue

        try:
            summary = load_json(summary_path)
        except json.JSONDecodeError as exc:
            print(f"[WARN] Invalid summary.json, skipping {summary_path}: {exc}")
            continue

        sample_index = summary.get("sample_index", sample_id)
        metadata[str(sample_index)] = {
            "sample_id": sample_id,
            "sample_index": sample_index,
            "iterations": to_int(summary.get("iterations"), default=0),
            "source_dir": sample_dir,
            "macro_f1": metric_from_record(summary, ("macro_f1", "Macro-F1")),
            "micro_f1": metric_from_record(summary, ("micro_f1", "Micro-F1")),
        }

    return metadata


def load_predictions_vs_gt_metrics(dataset_dir: Path) -> dict[str, dict[str, float]]:
    path = dataset_dir / "predictions_vs_gt.json"
    if not path.exists():
        return {}

    records = load_json(path)
    if not isinstance(records, list):
        raise ValueError(f"{path} must contain a list of per-sample records")

    metrics: dict[str, dict[str, float]] = {}
    for fallback_index, record in enumerate(records):
        if not isinstance(record, dict):
            continue
        sample_index = str(record.get("sample_index", record.get("index", fallback_index)))
        macro_f1 = metric_from_record(record, ("macro_f1", "Macro-F1"))
        micro_f1 = metric_from_record(record, ("micro_f1", "Micro-F1"))
        if macro_f1 is None and micro_f1 is None:
            continue
        metrics[sample_index] = {}
        if macro_f1 is not None:
            metrics[sample_index]["macro_f1"] = macro_f1
        if micro_f1 is not None:
            metrics[sample_index]["micro_f1"] = micro_f1

    return metrics


def load_cached_clinical_metrics(dataset_dir: Path, cache_name: str) -> dict[str, dict[str, float]]:
    path = dataset_dir / cache_name
    if not path.exists():
        return {}

    payload = load_json(path)
    records = payload.get("records", payload) if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        raise ValueError(f"{path} must contain a list or an object with a records list")

    metrics: dict[str, dict[str, float]] = {}
    for fallback_index, record in enumerate(records):
        if not isinstance(record, dict):
            continue
        sample_index = str(record.get("sample_index", record.get("index", fallback_index)))
        macro_f1 = metric_from_record(record, ("macro_f1", "Macro-F1"))
        micro_f1 = metric_from_record(record, ("micro_f1", "Micro-F1"))
        if macro_f1 is None or micro_f1 is None:
            continue
        metrics[sample_index] = {
            "macro_f1": macro_f1,
            "micro_f1": micro_f1,
        }
    return metrics


def merge_metrics(
    metadata: dict[str, dict[str, Any]],
    metric_sources: list[dict[str, dict[str, float]]],
) -> None:
    for sample_index, item in metadata.items():
        for source in metric_sources:
            source_item = source.get(sample_index)
            if not source_item:
                continue
            if item.get("macro_f1") is None and source_item.get("macro_f1") is not None:
                item["macro_f1"] = source_item["macro_f1"]
            if item.get("micro_f1") is None and source_item.get("micro_f1") is not None:
                item["micro_f1"] = source_item["micro_f1"]


def empty_torch_cache() -> None:
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def compute_and_cache_clinical_metrics(
    dataset: str,
    dataset_dir: Path,
    cache_name: str,
) -> dict[str, dict[str, float]]:
    scripts_dir = Path(__file__).resolve().parent
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))

    from evaluate_chest_ct import truncate_for_metric
    from evaluate_chest_xray import load_predictions

    try:
        import torch
        from f1chexbert import F1CheXbert
        from sklearn.metrics import f1_score
    except ImportError as exc:
        raise RuntimeError(f"Cannot compute clinical F1 because a dependency is missing: {exc}") from exc

    results, predictions, references = load_predictions(dataset_dir)
    if not predictions:
        raise RuntimeError(f"No valid predictions found in {dataset_dir}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Loading F1CheXbert for {dataset} on device={device}")
    metric = F1CheXbert(device=device)

    try:
        hyps = truncate_for_metric(metric, predictions)
        refs = truncate_for_metric(metric, references)

        def label_texts(texts: list[str], label: str) -> list[list[int]]:
            labels = []
            for idx, text in enumerate(texts, start=1):
                labels.append(metric.get_label(text.strip()))
                if idx == 1 or idx % 25 == 0 or idx == len(texts):
                    print(f"[INFO] {dataset}: labeled {idx}/{len(texts)} {label}")
            return labels

        ref_labels = label_texts(refs, "references")
        hyp_labels = label_texts(hyps, "predictions")

        records = []
        indexed_metrics: dict[str, dict[str, float]] = {}
        for idx, (ref, hyp, result) in enumerate(zip(ref_labels, hyp_labels, results)):
            macro_f1 = float(f1_score([ref], [hyp], average="macro", zero_division=0))
            micro_f1 = float(f1_score([ref], [hyp], average="micro", zero_division=0))
            sample_index = result.get("sample_index", idx)
            record = {
                "index": idx,
                "sample_index": sample_index,
                "macro_f1": macro_f1,
                "micro_f1": micro_f1,
                "reference_labels": ref,
                "prediction_labels": hyp,
            }
            records.append(record)
            indexed_metrics[str(sample_index)] = {
                "macro_f1": macro_f1,
                "micro_f1": micro_f1,
            }

        payload = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "metric_reference": "scripts/evaluate_chest_ct.py compute_clinical_f1",
            "label_order": list(getattr(metric, "target_names", [])),
            "overall_macro_f1": float(f1_score(ref_labels, hyp_labels, average="macro", zero_division=0)),
            "overall_micro_f1": float(f1_score(ref_labels, hyp_labels, average="micro", zero_division=0)),
            "num_samples": len(records),
            "records": records,
        }
        cache_path = dataset_dir / cache_name
        with cache_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"[INFO] Wrote per-sample clinical F1 cache: {cache_path}")
        return indexed_metrics
    finally:
        del metric
        empty_torch_cache()


def load_dataset_candidates(
    dataset: str,
    dataset_dir: Path,
    args: argparse.Namespace,
) -> list[Candidate]:
    metadata = discover_sample_metadata(dataset_dir)
    if not metadata:
        print(f"[WARN] No sample metadata found in {dataset_dir}")
        return []

    cached_metrics = {}
    if not args.force_recompute_metrics:
        cached_metrics = load_cached_clinical_metrics(dataset_dir, args.metric_cache_name)

    metric_sources = [
        load_predictions_vs_gt_metrics(dataset_dir),
        cached_metrics,
    ]
    merge_metrics(metadata, metric_sources)

    missing_metrics = [
        item for item in metadata.values()
        if item.get("macro_f1") is None or item.get("micro_f1") is None
    ]
    should_compute = (
        bool(missing_metrics)
        and not args.no_compute_missing_metrics
        and (args.force_recompute_metrics or not cached_metrics)
    )
    if should_compute:
        computed = compute_and_cache_clinical_metrics(
            dataset,
            dataset_dir,
            cache_name=args.metric_cache_name,
        )
        merge_metrics(metadata, [computed])
        missing_metrics = [
            item for item in metadata.values()
            if item.get("macro_f1") is None or item.get("micro_f1") is None
        ]

    if missing_metrics:
        message = (
            f"{dataset}: {len(missing_metrics)}/{len(metadata)} samples are missing "
            "per-sample Macro-F1 or Micro-F1. Allow metric computation or provide "
            f"{dataset_dir / args.metric_cache_name}."
        )
        if args.strict:
            raise RuntimeError(message)
        print(f"[WARN] {message} These samples will be skipped.")

    weight_sum = args.macro_weight + args.micro_weight
    if weight_sum <= 0:
        raise ValueError("--macro-weight + --micro-weight must be positive")

    candidates: list[Candidate] = []
    for item in metadata.values():
        macro_f1 = to_finite_float(item.get("macro_f1"))
        micro_f1 = to_finite_float(item.get("micro_f1"))
        if macro_f1 is None or micro_f1 is None:
            continue
        if args.min_macro_f1 is not None and macro_f1 < args.min_macro_f1:
            continue
        if args.min_micro_f1 is not None and micro_f1 < args.min_micro_f1:
            continue

        clinical_score = (
            args.macro_weight * macro_f1 + args.micro_weight * micro_f1
        ) / weight_sum
        candidates.append(
            Candidate(
                rank=None,
                dataset=dataset,
                sample_id=str(item["sample_id"]),
                sample_index=item["sample_index"],
                iterations=to_int(item.get("iterations"), default=0),
                macro_f1=macro_f1,
                micro_f1=micro_f1,
                clinical_score=clinical_score,
                source_dir=str(item["source_dir"]),
            )
        )

    return candidates


def sort_candidates(candidates: list[Candidate]) -> list[Candidate]:
    return sorted(
        candidates,
        key=lambda c: (
            c.clinical_score,
            c.macro_f1,
            c.micro_f1,
            c.iterations,
            c.dataset,
            c.sample_id,
        ),
        reverse=True,
    )


def assign_ranks(candidates: list[Candidate]) -> list[Candidate]:
    return [
        Candidate(
            **{
                **asdict(candidate),
                "rank": rank,
            }
        )
        for rank, candidate in enumerate(candidates, start=1)
    ]


def rank_candidates(candidates: list[Candidate], limit: int) -> list[Candidate]:
    return assign_ranks(sort_candidates(candidates)[:limit])


def select_by_dataset_counts(
    candidates: list[Candidate],
    dataset_counts: dict[str, int],
    allow_fewer: bool,
) -> list[Candidate]:
    selected: list[Candidate] = []

    for dataset in DATASET_ORDER:
        requested = dataset_counts[dataset]
        if requested == 0:
            continue

        dataset_candidates = [item for item in candidates if item.dataset == dataset]
        if len(dataset_candidates) < requested and not allow_fewer:
            raise RuntimeError(
                f"{dataset}: only {len(dataset_candidates)} candidates have both "
                f"Macro-F1 and Micro-F1, but --dataset-counts requests {requested}. "
                "Pass --allow-fewer to continue."
            )

        selected.extend(sort_candidates(dataset_candidates)[:requested])

    return assign_ranks(selected)


def with_destinations(selected: list[Candidate], output_dir: Path) -> list[Candidate]:
    updated = []
    for candidate in selected:
        destination = output_dir / f"{candidate.dataset}_sample_{candidate.sample_id}"
        updated.append(
            Candidate(
                **{
                    **asdict(candidate),
                    "destination_dir": str(destination),
                }
            )
        )
    return updated


def print_selection(selected: list[Candidate]) -> None:
    if not selected:
        print("No samples selected.")
        return

    header = (
        f"{'rank':>4}  {'dataset':<10}  {'sample':<12}  {'Macro-F1':>8}  "
        f"{'Micro-F1':>8}  {'clinical':>9}  {'iter':>4}"
    )
    print(header)
    print("-" * len(header))
    for item in selected:
        print(
            f"{item.rank:>4}  {item.dataset:<10}  sample_{item.sample_id:<5}  "
            f"{item.macro_f1:>8.4f}  {item.micro_f1:>8.4f}  "
            f"{item.clinical_score:>9.4f}  {item.iterations:>4}"
        )


def copy_selected(selected: list[Candidate], output_dir: Path, overwrite: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    conflicts = [
        Path(item.destination_dir)
        for item in selected
        if item.destination_dir and Path(item.destination_dir).exists()
    ]
    if conflicts and not overwrite:
        conflict_text = "\n".join(f"  - {path}" for path in conflicts[:20])
        raise FileExistsError(
            "Destination sample folders already exist. Use --overwrite to replace them:\n"
            f"{conflict_text}"
        )

    for item in selected:
        if item.destination_dir is None:
            raise ValueError(f"Missing destination for {item.source_dir}")
        source = Path(item.source_dir)
        destination = Path(item.destination_dir)
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(source, destination, symlinks=True)


def write_manifest(
    selected: list[Candidate],
    output_dir: Path,
    dataset_dirs: dict[str, Path],
    args: argparse.Namespace,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "selection_rule": (
            "Sort by clinical_score desc, Macro-F1 desc, Micro-F1 desc, "
            "iterations desc. clinical_score is weighted average of Macro-F1 "
            "and Micro-F1."
        ),
        "limit": args.limit,
        "dataset_counts": parse_dataset_counts(args.dataset_counts),
        "macro_weight": args.macro_weight,
        "micro_weight": args.micro_weight,
        "min_macro_f1": args.min_macro_f1,
        "min_micro_f1": args.min_micro_f1,
        "metric_cache_name": args.metric_cache_name,
        "source_dirs": {name: str(path) for name, path in dataset_dirs.items()},
        "selected": [asdict(item) for item in selected],
    }

    json_path = output_dir / "chest_ct_selection_manifest.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    tsv_path = output_dir / "chest_ct_selection_manifest.tsv"
    with tsv_path.open("w", encoding="utf-8") as f:
        f.write(
            "rank\tdataset\tsample_id\tsample_index\tmacro_f1\tmicro_f1\t"
            "clinical_score\titerations\tsource_dir\tdestination_dir\n"
        )
        for item in selected:
            f.write(
                f"{item.rank}\t{item.dataset}\t{item.sample_id}\t"
                f"{item.sample_index}\t{item.macro_f1:.6f}\t{item.micro_f1:.6f}\t"
                f"{item.clinical_score:.6f}\t{item.iterations}\t"
                f"{item.source_dir}\t{item.destination_dir}\n"
            )


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    output_dir = resolve_path(repo_root, args.output_dir)
    dataset_counts = parse_dataset_counts(args.dataset_counts)
    dataset_dirs = {
        name: resolve_path(repo_root, rel_path)
        for name, rel_path in DEFAULT_DATASET_DIRS.items()
    }

    all_candidates: list[Candidate] = []
    for dataset, dataset_dir in dataset_dirs.items():
        if not dataset_dir.exists():
            message = f"{dataset}: dataset directory does not exist: {dataset_dir}"
            if args.strict:
                raise FileNotFoundError(message)
            print(f"[WARN] {message}")
            continue
        all_candidates.extend(load_dataset_candidates(dataset, dataset_dir, args))

    if not all_candidates:
        raise RuntimeError("No candidates with both Macro-F1 and Micro-F1 were found.")

    if dataset_counts is None and len(all_candidates) < args.limit and not args.allow_fewer:
        raise RuntimeError(
            f"Only {len(all_candidates)} candidates have both Macro-F1 and Micro-F1, "
            f"but --limit is {args.limit}. Pass --allow-fewer to continue."
        )

    if dataset_counts is None:
        selected = rank_candidates(all_candidates, min(args.limit, len(all_candidates)))
    else:
        selected = select_by_dataset_counts(
            all_candidates,
            dataset_counts=dataset_counts,
            allow_fewer=args.allow_fewer,
        )
    selected = with_destinations(selected, output_dir)
    print_selection(selected)

    if args.dry_run:
        print(f"\n[DRY-RUN] No files copied. Output directory would be: {output_dir}")
        return

    copy_selected(selected, output_dir, overwrite=args.overwrite)
    write_manifest(selected, output_dir, dataset_dirs, args)
    print(f"\nCopied {len(selected)} samples to {output_dir}")
    print(f"Wrote manifest: {output_dir / 'chest_ct_selection_manifest.json'}")


if __name__ == "__main__":
    main()
