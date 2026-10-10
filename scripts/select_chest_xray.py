#!/usr/bin/env python3
"""
Select high-quality, difficult chest X-ray samples for physician annotation.

By default this script reads per-sample GREEN and RaTEScore values produced by
scripts/evaluate_chest_xray.py:
  - <dataset_dir>/predictions_vs_gt.json with "green" and "ratescore" fields
  - or <dataset_dir>/evaluation_metrics.json with "_green_per_sample" and
    "_ratescore_per_sample" arrays

Samples are first partitioned as abnormal vs normal from summary.json
ground_truth, then ranked by medical quality first and difficulty second:
  1. clinical_score = weighted average of GREEN and RaTEScore
  2. GREEN
  3. RaTEScore
  4. iterations from sample_<id>/summary.json

The selected sample folders are copied into annotation_samples/ and renamed as
<dataset>_sample_<id>. Source output folders are not modified.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_DATASET_DIRS = {
    "mimic": Path("output/mimic/full_radclaw"),
    "chexpertplus": Path("output/chexpertplus/full_radclaw"),
    "iu_xray": Path("output/iuxray/full_radclaw"),
    "openi": Path("output/openi/full_radclaw"),
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
    green: float
    ratescore: float
    clinical_score: float
    case_status: str
    case_status_reason: str
    source_dir: str
    destination_dir: str | None = None


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser(
        description="Select 15 high-quality and difficult chest X-ray samples."
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
        help="Total number of samples to select.",
    )
    parser.add_argument(
        "--dataset-counts",
        type=str,
        default=None,
        help=(
            "Optional per-dataset sample counts in order "
            "mimic,chexpertplus,iu_xray,openi, e.g. '[4,4,4,3]' "
            "or '4,4,4,3'. Overrides --limit."
        ),
    )
    parser.add_argument(
        "--abnormal-normal-ratio",
        type=str,
        default="4,1",
        help=(
            "Target abnormal:normal case ratio inferred from ground_truth, "
            "e.g. '4,1', '4:1', or '[4,1]'."
        ),
    )
    parser.add_argument(
        "--green-weight",
        type=float,
        default=0.5,
        help="Weight for GREEN in the clinical score.",
    )
    parser.add_argument(
        "--ratescore-weight",
        type=float,
        default=0.5,
        help="Weight for RaTEScore in the clinical score.",
    )
    parser.add_argument(
        "--min-green",
        type=float,
        default=None,
        help="Optional minimum GREEN threshold.",
    )
    parser.add_argument(
        "--min-ratescore",
        type=float,
        default=None,
        help="Optional minimum RaTEScore threshold.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing destination sample folders.",
    )
    parser.add_argument(
        "--clean-output",
        action="store_true",
        help="Remove existing selected chest X-ray sample folders in --output-dir before copying.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print selected samples without copying folders.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any default dataset directory or required metric file is missing.",
    )
    parser.add_argument(
        "--compute-missing-metrics",
        action="store_true",
        help=(
            "Compute missing GREEN/RaTEScore by importing scripts/evaluate_chest_xray.py. "
            "This can be slow and may load large metric models."
        ),
    )
    parser.add_argument(
        "--allow-fewer",
        action="store_true",
        help="Allow copying fewer than --limit samples when insufficient candidates exist.",
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
        raise ValueError(
            "--dataset-counts must look like '[4,4,4,3]' or '4,4,4,3'"
        ) from exc

    if not isinstance(values, (list, tuple)):
        raise ValueError("--dataset-counts must be a list of four non-negative integers")
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


def parse_abnormal_normal_ratio(raw_ratio: str) -> tuple[int, int]:
    text = raw_ratio.strip()
    if not text:
        raise ValueError("--abnormal-normal-ratio cannot be empty")

    try:
        if text.startswith("["):
            values = ast.literal_eval(text)
        elif ":" in text:
            values = [part.strip() for part in text.split(":")]
        else:
            values = [part.strip() for part in text.split(",")]
    except (SyntaxError, ValueError) as exc:
        raise ValueError(
            "--abnormal-normal-ratio must look like '4,1', '4:1', or '[4,1]'"
        ) from exc

    if not isinstance(values, (list, tuple)) or len(values) != 2:
        raise ValueError("--abnormal-normal-ratio must contain exactly two positive integers")

    parsed = []
    for value in values:
        if isinstance(value, bool):
            raise ValueError("--abnormal-normal-ratio values must be positive integers")
        try:
            number = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("--abnormal-normal-ratio values must be positive integers") from exc
        if number <= 0:
            raise ValueError("--abnormal-normal-ratio values must be positive integers")
        parsed.append(number)

    return parsed[0], parsed[1]


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
    if value is None:
        return None
    if isinstance(value, bool):
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
    if isinstance(value, bool) or value is None:
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


ABNORMAL_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("cardiomegaly", re.compile(r"\b(?:cardiomegaly|enlarged cardiomediastinal silhouette|cardiac silhouette (?:is )?(?:mildly |moderately |severely )?(?:enlarged|increased)|heart size (?:is )?(?:borderline |mildly |moderately |severely )?(?:enlarged|increased))\b", re.I)),
    ("edema_congestion", re.compile(r"\b(?:edema|pulmonary vascular congestion|vascular congestion|pulmonary venous congestion|interstitial edema)\b", re.I)),
    ("opacity_infiltrate", re.compile(r"\b(?:opacity|opacities|airspace disease|air-space disease|infiltrate|infiltrates|infiltration)\b", re.I)),
    ("consolidation_pneumonia", re.compile(r"\b(?:consolidation|pneumonia|pneumonic)\b", re.I)),
    ("atelectasis", re.compile(r"\b(?:atelectasis|atelectatic|collapse|volume loss)\b", re.I)),
    ("pleural_effusion", re.compile(r"\b(?:pleural effusion|effusion|pleural fluid)\b", re.I)),
    ("pneumothorax", re.compile(r"\b(?:pneumothorax)\b", re.I)),
    ("nodule_mass_lesion", re.compile(r"\b(?:nodule|nodules|mass|masses|lesion|lesions)\b", re.I)),
    ("fracture", re.compile(r"\b(?:fracture|fractures|deformity|deformities)\b", re.I)),
    ("emphysema_copd", re.compile(r"\b(?:emphysema|emphysematous|copd|hyperinflation|hyperinflated)\b", re.I)),
    ("fibrosis_scarring", re.compile(r"\b(?:fibrosis|fibrotic|scarring|scar|pleural thickening)\b", re.I)),
    ("hernia", re.compile(r"\b(?:hernia|hiatal hernia)\b", re.I)),
    ("mediastinal_abnormality", re.compile(r"\b(?:mediastinal widening|widened mediastinum)\b", re.I)),
    ("calcification_atherosclerosis", re.compile(r"\b(?:calcification|calcified|atherosclerosis|atherosclerotic)\b", re.I)),
)

NEGATION_RE = re.compile(
    r"\b(?:no|without|negative for|free of|absence of|absent|not seen|"
    r"not identified|not present|no evidence of|no radiographic evidence of)\b",
    re.I,
)
NEGATION_SCOPE_CHARS = 120

NORMAL_IMPRESSION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("normal_chest", re.compile(r"\bnormal\s+chest\b", re.I)),
    ("clear_lungs", re.compile(r"\bclear\s+lungs\b", re.I)),
    ("no_acute_cardiopulmonary", re.compile(r"\bno\s+acute\s+(?:radiographic\s+)?(?:cardio[- ]?pulmonary|cardiac|pulmonary|intrathoracic)\s+(?:abnormalit(?:y|ies)|disease|process|finding|findings)\b", re.I)),
    ("no_active_disease", re.compile(r"\bno\s+(?:acute\s+or\s+)?active\s+(?:cardiac,?\s*)?(?:pulmonary\s+or\s+pleural\s+)?disease\b", re.I)),
    ("no_acute_disease", re.compile(r"\bno\s+acute\s+disease\b", re.I)),
    ("negative_acute_abnormality", re.compile(r"\bnegative\s+for\s+acute\s+abnormalit(?:y|ies)\b", re.I)),
    ("no_pneumonia", re.compile(r"\bno\s+pneumonia\b", re.I)),
)

RESOLVED_OR_STABLE_RE = re.compile(r"\b(?:resolved|resolution of|unchanged|stable|no significant change)\b", re.I)
NON_DIAGNOSTIC_ABNORMAL_RE = re.compile(
    r"\b(?:fractures?\s+may\s+not\s+be\s+demonstrated|"
    r"continued\s+concern\s+for\s+occult\s+fracture|"
    r"concern\s+for\s+occult\s+fracture|"
    r"consider\s+(?:dedicated\s+)?rib\s+series)\b",
    re.I,
)


def extract_impression(ground_truth: str) -> tuple[str, str]:
    text = (ground_truth or "").strip()
    if not text:
        return "", "empty_ground_truth"

    matches = list(re.finditer(r"(?is)\bIMPRESSION\s*:\s*", text))
    if matches:
        start = matches[-1].end()
        impression = text[start:].strip()
        impression = re.split(r"(?is)\n\s*(?:FINDINGS|INDICATION|COMPARISON|TECHNIQUE)\s*:", impression)[0].strip()
        return impression, "impression"

    return text, "fallback_full_report"


def is_negated_context(sentence: str, start: int) -> bool:
    prefix = sentence[max(0, start - NEGATION_SCOPE_CHARS):start].lower()
    recent_prefix = prefix[-80:]

    # "No change in cardiomegaly" is a positive abnormal finding, not a negation.
    if re.search(r"\bno\s+(?:significant\s+)?(?:interval\s+)?change(?:s)?\s+(?:in|of)\b", recent_prefix):
        return False
    return NEGATION_RE.search(prefix) is not None


def normalize_impression_text(text: str) -> str:
    text = re.sub(r"(?is)<[^>]+>", " ", text)
    text = re.sub(r"[_X]{2,}", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def sentence_has_positive_abnormality(sentence: str) -> str | None:
    if NON_DIAGNOSTIC_ABNORMAL_RE.search(sentence):
        return None
    if RESOLVED_OR_STABLE_RE.search(sentence) and not re.search(
        r"\b(?:increased|worsened|new|persistent|large|moderate|severe)\b", sentence, re.I
    ):
        # Pure stability/resolution statements are not enough to call a case abnormal.
        # Positive disease terms elsewhere in the impression still make it abnormal.
        return None

    for label, pattern in ABNORMAL_PATTERNS:
        for match in pattern.finditer(sentence):
            if not is_negated_context(sentence, match.start()):
                return label
    return None


def classify_ground_truth(ground_truth: str) -> tuple[str, str]:
    impression, source = extract_impression(ground_truth)
    text = normalize_impression_text(impression)
    if not text:
        return "normal", f"{source}:empty"

    sentences = [s.strip() for s in re.split(r"(?<=[.;])\s+|\n+", text) if s.strip()]
    positive_labels = []
    normal_reasons = []
    for sentence in sentences:
        for normal_label, normal_pattern in NORMAL_IMPRESSION_PATTERNS:
            if normal_pattern.search(sentence):
                normal_reasons.append(normal_label)
        positive_label = sentence_has_positive_abnormality(sentence)
        if positive_label:
            positive_labels.append(positive_label)

    if positive_labels:
        return "abnormal", f"{source}:matched:{positive_labels[0]}"
    if normal_reasons:
        return "normal", f"{source}:{normal_reasons[0]}"
    return "normal", f"{source}:no_positive_impression_abnormal_terms"


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
        case_status, case_status_reason = classify_ground_truth(
            str(summary.get("ground_truth", "") or "")
        )
        metadata[str(sample_index)] = {
            "sample_id": sample_id,
            "sample_index": sample_index,
            "iterations": to_int(summary.get("iterations"), default=0),
            "source_dir": sample_dir,
            "case_status": case_status,
            "case_status_reason": case_status_reason,
            "green": metric_from_record(summary, ("green", "GREEN")),
            "ratescore": metric_from_record(
                summary, ("ratescore", "RaTEScore", "rate_score")
            ),
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
        sample_index = record.get("sample_index", record.get("index", fallback_index))
        green = metric_from_record(record, ("green", "GREEN"))
        ratescore = metric_from_record(record, ("ratescore", "RaTEScore", "rate_score"))
        if green is None and ratescore is None:
            continue
        metrics[str(sample_index)] = {}
        if green is not None:
            metrics[str(sample_index)]["green"] = green
        if ratescore is not None:
            metrics[str(sample_index)]["ratescore"] = ratescore

    return metrics


def per_sample_array(metrics: dict[str, Any], names: tuple[str, ...]) -> list[Any]:
    for name in names:
        value = metrics.get(name)
        if isinstance(value, list):
            return value
    return []


def load_evaluation_metrics(dataset_dir: Path) -> dict[str, dict[str, float]]:
    path = dataset_dir / "evaluation_metrics.json"
    if not path.exists():
        return {}

    metrics = load_json(path)
    if not isinstance(metrics, dict):
        raise ValueError(f"{path} must contain a JSON object")

    green_scores = per_sample_array(metrics, ("_green_per_sample", "green_per_sample"))
    ratescore_scores = per_sample_array(
        metrics, ("_ratescore_per_sample", "ratescore_per_sample")
    )

    summary_files = sorted(dataset_dir.rglob("summary.json"))
    indexed_metrics: dict[str, dict[str, float]] = {}
    for idx, summary_path in enumerate(summary_files):
        try:
            summary = load_json(summary_path)
        except json.JSONDecodeError:
            continue
        sample_index = str(summary.get("sample_index", summary_path.parent.name))
        indexed_metrics.setdefault(sample_index, {})

        if idx < len(green_scores):
            green = to_finite_float(green_scores[idx])
            if green is not None:
                indexed_metrics[sample_index]["green"] = green
        if idx < len(ratescore_scores):
            ratescore = to_finite_float(ratescore_scores[idx])
            if ratescore is not None:
                indexed_metrics[sample_index]["ratescore"] = ratescore

    return indexed_metrics


def compute_and_cache_missing_metrics(dataset_dir: Path) -> dict[str, dict[str, float]]:
    from evaluate_chest_xray import compute_green, compute_ratescore, load_predictions

    results, predictions, references = load_predictions(dataset_dir)
    if not predictions:
        raise RuntimeError(f"No valid predictions found in {dataset_dir}")

    green = compute_green(predictions, references, output_dir=str(dataset_dir))
    rate = compute_ratescore(predictions, references)

    green_scores = green.get("green_per_sample", [])
    ratescore_scores = rate.get("ratescore_per_sample", [])

    output_metrics = dataset_dir / "evaluation_metrics.json"
    if output_metrics.exists():
        cached = load_json(output_metrics)
        if not isinstance(cached, dict):
            cached = {}
    else:
        cached = {}

    cached["green"] = green.get("green")
    cached["_green_per_sample"] = green_scores
    cached["ratescore"] = rate.get("ratescore")
    cached["_ratescore_per_sample"] = ratescore_scores
    cached["num_samples"] = len(results)

    with output_metrics.open("w", encoding="utf-8") as f:
        json.dump(cached, f, indent=2, ensure_ascii=False)

    records = []
    indexed_metrics: dict[str, dict[str, float]] = {}
    for idx, result in enumerate(results):
        sample_index = result.get("sample_index", idx)
        record = {
            "index": idx,
            "sample_index": sample_index,
            "prediction": predictions[idx],
            "ground_truth": references[idx],
        }
        indexed_metrics[str(sample_index)] = {}
        if idx < len(green_scores):
            green_value = to_finite_float(green_scores[idx])
            if green_value is not None:
                record["green"] = green_value
                indexed_metrics[str(sample_index)]["green"] = green_value
        if idx < len(ratescore_scores):
            ratescore_value = to_finite_float(ratescore_scores[idx])
            if ratescore_value is not None:
                record["ratescore"] = ratescore_value
                indexed_metrics[str(sample_index)]["ratescore"] = ratescore_value
        records.append(record)

    with (dataset_dir / "predictions_vs_gt.json").open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    return indexed_metrics


def merge_metrics(
    metadata: dict[str, dict[str, Any]],
    metric_sources: list[dict[str, dict[str, float]]],
) -> None:
    for sample_index, item in metadata.items():
        for source in metric_sources:
            source_item = source.get(sample_index)
            if not source_item:
                continue
            if item.get("green") is None and source_item.get("green") is not None:
                item["green"] = source_item["green"]
            if item.get("ratescore") is None and source_item.get("ratescore") is not None:
                item["ratescore"] = source_item["ratescore"]


def load_dataset_candidates(
    dataset: str,
    dataset_dir: Path,
    args: argparse.Namespace,
) -> list[Candidate]:
    metadata = discover_sample_metadata(dataset_dir)
    if not metadata:
        print(f"[WARN] No sample metadata found in {dataset_dir}")
        return []

    metric_sources = [
        load_predictions_vs_gt_metrics(dataset_dir),
        load_evaluation_metrics(dataset_dir),
    ]
    merge_metrics(metadata, metric_sources)

    missing_metrics = [
        item for item in metadata.values()
        if item.get("green") is None or item.get("ratescore") is None
    ]
    if missing_metrics and args.compute_missing_metrics:
        print(f"[INFO] Computing missing GREEN/RaTEScore for {dataset} ({dataset_dir})")
        computed = compute_and_cache_missing_metrics(dataset_dir)
        merge_metrics(metadata, [computed])
        missing_metrics = [
            item for item in metadata.values()
            if item.get("green") is None or item.get("ratescore") is None
        ]

    if missing_metrics:
        message = (
            f"{dataset}: {len(missing_metrics)}/{len(metadata)} samples are missing "
            "GREEN or RaTEScore. Run scripts/evaluate_chest_xray.py first, or pass "
            "--compute-missing-metrics."
        )
        if args.strict:
            raise RuntimeError(message)
        print(f"[WARN] {message}")

    candidates: list[Candidate] = []
    weight_sum = args.green_weight + args.ratescore_weight
    if weight_sum <= 0:
        raise ValueError("--green-weight + --ratescore-weight must be positive")

    for item in metadata.values():
        green = to_finite_float(item.get("green"))
        ratescore = to_finite_float(item.get("ratescore"))
        if green is None or ratescore is None:
            continue
        if args.min_green is not None and green < args.min_green:
            continue
        if args.min_ratescore is not None and ratescore < args.min_ratescore:
            continue

        clinical_score = (
            args.green_weight * green + args.ratescore_weight * ratescore
        ) / weight_sum
        candidates.append(
            Candidate(
                rank=None,
                dataset=dataset,
                sample_id=str(item["sample_id"]),
                sample_index=item["sample_index"],
                iterations=to_int(item.get("iterations"), default=0),
                green=green,
                ratescore=ratescore,
                clinical_score=clinical_score,
                case_status=str(item.get("case_status", "normal")),
                case_status_reason=str(item.get("case_status_reason", "unknown")),
                source_dir=str(item["source_dir"]),
            )
        )

    return candidates


def sort_candidates(candidates: list[Candidate]) -> list[Candidate]:
    return sorted(
        candidates,
        key=lambda c: (
            c.clinical_score,
            c.green,
            c.ratescore,
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


def target_status_counts(total: int, ratio: tuple[int, int]) -> dict[str, int]:
    abnormal_weight, normal_weight = ratio
    abnormal_count = int(total * abnormal_weight / (abnormal_weight + normal_weight) + 0.5)
    abnormal_count = min(max(abnormal_count, 0), total)
    return {"abnormal": abnormal_count, "normal": total - abnormal_count}


def candidates_by_status(candidates: list[Candidate]) -> dict[str, list[Candidate]]:
    return {
        "abnormal": sort_candidates([item for item in candidates if item.case_status == "abnormal"]),
        "normal": sort_candidates([item for item in candidates if item.case_status == "normal"]),
    }


def select_by_status_ratio(
    candidates: list[Candidate],
    limit: int,
    status_counts: dict[str, int],
    allow_fewer: bool,
) -> list[Candidate]:
    by_status = candidates_by_status(candidates)
    selected: list[Candidate] = []

    for status in ("abnormal", "normal"):
        requested = status_counts[status]
        available = by_status[status]
        if len(available) < requested and not allow_fewer:
            raise RuntimeError(
                f"Only {len(available)} {status} candidates have both GREEN and RaTEScore, "
                f"but the requested 4:1 split needs {requested}."
            )
        selected.extend(available[:min(requested, len(available))])

    if len(selected) < limit and not allow_fewer:
        raise RuntimeError(f"Selected {len(selected)} samples, expected {limit}.")
    return assign_ranks(sort_candidates(selected))


def allocate_abnormal_counts(
    dataset_counts: dict[str, int],
    availability: dict[str, dict[str, int]],
    target_abnormal: int,
) -> dict[str, int]:
    min_abnormal: dict[str, int] = {}
    max_abnormal: dict[str, int] = {}
    for dataset, requested in dataset_counts.items():
        normal_available = availability[dataset]["normal"]
        abnormal_available = availability[dataset]["abnormal"]
        min_abnormal[dataset] = max(0, requested - normal_available)
        max_abnormal[dataset] = min(requested, abnormal_available)

    min_total = sum(min_abnormal.values())
    max_total = sum(max_abnormal.values())
    if target_abnormal < min_total or target_abnormal > max_total:
        raise RuntimeError(
            f"Cannot satisfy abnormal target {target_abnormal} with dataset quotas. "
            f"Feasible abnormal range is {min_total}..{max_total}."
        )

    total_requested = sum(dataset_counts.values())
    ideals = {
        dataset: dataset_counts[dataset] * target_abnormal / max(total_requested, 1)
        for dataset in DATASET_ORDER
    }
    allocation = {
        dataset: min(max(math.floor(ideals[dataset]), min_abnormal[dataset]), max_abnormal[dataset])
        for dataset in DATASET_ORDER
    }

    while sum(allocation.values()) < target_abnormal:
        choices = [dataset for dataset in DATASET_ORDER if allocation[dataset] < max_abnormal[dataset]]
        if not choices:
            break
        dataset = max(choices, key=lambda name: (ideals[name] - allocation[name], dataset_counts[name]))
        allocation[dataset] += 1

    while sum(allocation.values()) > target_abnormal:
        choices = [dataset for dataset in DATASET_ORDER if allocation[dataset] > min_abnormal[dataset]]
        if not choices:
            break
        dataset = max(choices, key=lambda name: (allocation[name] - ideals[name], dataset_counts[name]))
        allocation[dataset] -= 1

    return allocation


def select_by_dataset_counts(
    candidates: list[Candidate],
    dataset_counts: dict[str, int],
    status_counts: dict[str, int],
    allow_fewer: bool,
) -> list[Candidate]:
    selected: list[Candidate] = []
    availability: dict[str, dict[str, int]] = {}

    for dataset in DATASET_ORDER:
        dataset_candidates = [item for item in candidates if item.dataset == dataset]
        requested = dataset_counts[dataset]
        if len(dataset_candidates) < requested and not allow_fewer:
            raise RuntimeError(
                f"{dataset}: only {len(dataset_candidates)} candidates have both "
                f"GREEN and RaTEScore, but --dataset-counts requests {requested}. "
                "Pass --allow-fewer to continue."
            )
        availability[dataset] = {
            "abnormal": sum(1 for item in dataset_candidates if item.case_status == "abnormal"),
            "normal": sum(1 for item in dataset_candidates if item.case_status == "normal"),
        }

    abnormal_allocation = allocate_abnormal_counts(
        dataset_counts,
        availability,
        target_abnormal=status_counts["abnormal"],
    )

    for dataset in DATASET_ORDER:
        requested = dataset_counts[dataset]
        if requested == 0:
            continue
        dataset_candidates = [item for item in candidates if item.dataset == dataset]
        by_status = candidates_by_status(dataset_candidates)
        abnormal_count = abnormal_allocation[dataset]
        normal_count = requested - abnormal_count
        selected.extend(by_status["abnormal"][:abnormal_count])
        selected.extend(by_status["normal"][:normal_count])

    return assign_ranks(sort_candidates(selected))


def with_destinations(
    selected: list[Candidate],
    output_dir: Path,
) -> list[Candidate]:
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
        f"{'rank':>4}  {'dataset':<14}  {'sample':<12}  {'status':<8}  {'GREEN':>8}  "
        f"{'RaTEScore':>9}  {'clinical':>9}  {'iter':>4}"
    )
    print(header)
    print("-" * len(header))
    for item in selected:
        print(
            f"{item.rank:>4}  {item.dataset:<14}  sample_{item.sample_id:<5}  "
            f"{item.case_status:<8}  {item.green:>8.4f}  {item.ratescore:>9.4f}  "
            f"{item.clinical_score:>9.4f}  {item.iterations:>4}"
        )


OUTPUT_SAMPLE_RE = re.compile(r"^(?:mimic|chexpertplus|iu_xray|openi)_sample_.+$")


def clean_output_dir(output_dir: Path) -> None:
    if not output_dir.exists():
        return
    for child in output_dir.iterdir():
        if child.is_dir() and OUTPUT_SAMPLE_RE.match(child.name):
            shutil.rmtree(child)
    for manifest_name in ("selection_manifest.json", "selection_manifest.tsv"):
        manifest_path = output_dir / manifest_name
        if manifest_path.exists():
            manifest_path.unlink()


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
            "First enforce abnormal:normal ratio from ground_truth, then sort by "
            "clinical_score desc, GREEN desc, RaTEScore desc, iterations desc. "
            "clinical_score is weighted average of GREEN and RaTEScore."
        ),
        "limit": args.limit,
        "abnormal_normal_ratio": parse_abnormal_normal_ratio(args.abnormal_normal_ratio),
        "case_status_counts": {
            "abnormal": sum(1 for item in selected if item.case_status == "abnormal"),
            "normal": sum(1 for item in selected if item.case_status == "normal"),
        },
        "green_weight": args.green_weight,
        "ratescore_weight": args.ratescore_weight,
        "min_green": args.min_green,
        "min_ratescore": args.min_ratescore,
        "dataset_counts": parse_dataset_counts(args.dataset_counts),
        "source_dirs": {name: str(path) for name, path in dataset_dirs.items()},
        "selected": [asdict(item) for item in selected],
    }
    with (output_dir / "selection_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    tsv_path = output_dir / "selection_manifest.tsv"
    with tsv_path.open("w", encoding="utf-8") as f:
        f.write(
            "rank\tdataset\tsample_id\tsample_index\tcase_status\tcase_status_reason\t"
            "green\tratescore\tclinical_score\titerations\tsource_dir\tdestination_dir\n"
        )
        for item in selected:
            f.write(
                f"{item.rank}\t{item.dataset}\t{item.sample_id}\t"
                f"{item.sample_index}\t{item.case_status}\t{item.case_status_reason}\t"
                f"{item.green:.6f}\t{item.ratescore:.6f}\t"
                f"{item.clinical_score:.6f}\t{item.iterations}\t"
                f"{item.source_dir}\t{item.destination_dir}\n"
            )


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    output_dir = resolve_path(repo_root, args.output_dir)
    dataset_counts = parse_dataset_counts(args.dataset_counts)
    abnormal_normal_ratio = parse_abnormal_normal_ratio(args.abnormal_normal_ratio)
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
        raise RuntimeError("No candidates with both GREEN and RaTEScore were found.")

    if dataset_counts is None and len(all_candidates) < args.limit and not args.allow_fewer:
        raise RuntimeError(
            f"Only {len(all_candidates)} candidates have both GREEN and RaTEScore, "
            f"but --limit is {args.limit}. Pass --allow-fewer to continue."
        )

    if dataset_counts is None:
        requested_total = min(args.limit, len(all_candidates))
        status_counts = target_status_counts(requested_total, abnormal_normal_ratio)
        selected = select_by_status_ratio(
            all_candidates,
            limit=requested_total,
            status_counts=status_counts,
            allow_fewer=args.allow_fewer,
        )
    else:
        requested_total = sum(dataset_counts.values())
        status_counts = target_status_counts(requested_total, abnormal_normal_ratio)
        selected = select_by_dataset_counts(
            all_candidates,
            dataset_counts=dataset_counts,
            status_counts=status_counts,
            allow_fewer=args.allow_fewer,
        )
    selected = with_destinations(selected, output_dir)
    print_selection(selected)

    if args.dry_run:
        print(f"\n[DRY-RUN] No files copied. Output directory would be: {output_dir}")
        return

    if args.clean_output:
        clean_output_dir(output_dir)
    copy_selected(selected, output_dir, overwrite=args.overwrite)
    write_manifest(selected, output_dir, dataset_dirs, args)
    print(f"\nCopied {len(selected)} samples to {output_dir}")
    print(f"Wrote manifest: {output_dir / 'selection_manifest.json'}")


if __name__ == "__main__":
    main()
