#!/usr/bin/env python3
"""
Evaluate Chest CT RadClaw outputs.

All RadClaw generated results share the same directory layout under output/:
  sample_x/summary.json contains natural_language_report and ground_truth
  sample_x/result.json optionally contains structured_report for evidence metrics

Shared metrics are reused from evaluate_chest_xray.py:
  - NLG: BLEU-4, ROUGE-L, METEOR
  - Evidence coverage: image/retrieved evidence and impression-link rates

Chest CT-specific clinical metrics:
  - Macro-F1, Micro-F1, clinical-F1 from F1CheXbert labels
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
from pathlib import Path
from statistics import mean
from typing import Any

from evaluate_chest_xray import (
    compute_bleu,
    compute_evidence_metrics,
    compute_meteor,
    compute_rouge_l,
    load_predictions,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ALL_METRICS = ["bleu", "rouge_l", "meteor", "clinical_f1", "evidence"]
NLG_METRICS = ["bleu", "rouge_l", "meteor"]
CLINICAL_METRICS = ["clinical_f1"]

CHEXPERT_LABEL_ORDER = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
]


def parse_metrics(metrics: str) -> list[str]:
    """Parse the same comma-separated metric argument style used by Chest X-ray."""
    key = metrics.strip().lower()
    if key == "all":
        return list(ALL_METRICS)
    if key == "nlg":
        return list(NLG_METRICS)
    if key == "clinical":
        return list(CLINICAL_METRICS)
    return [item.strip() for item in metrics.split(",") if item.strip()]


def empty_torch_cache() -> None:
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def truncate_for_metric(metric: Any, texts: list[str], max_tokens: int = 500) -> list[str]:
    """Keep reports within BERT's 512-token limit before CheXbert inference."""
    tokenizer = getattr(metric, "tokenizer", None)
    if tokenizer is None:
        return texts

    truncated = []
    for text in texts:
        try:
            ids = tokenizer.encode(text, add_special_tokens=False)
            if len(ids) > max_tokens:
                text = tokenizer.decode(ids[:max_tokens], skip_special_tokens=True)
        except Exception:
            pass
        truncated.append(text)
    return truncated


def extract_per_label_f1(report: dict[str, Any]) -> dict[str, float]:
    """Extract per-pathology F1 scores from F1CheXbert's classification report."""
    per_label = {}
    for label in CHEXPERT_LABEL_ORDER:
        value = report.get(label)
        if isinstance(value, dict) and "f1-score" in value:
            per_label[label] = float(value["f1-score"])
    return per_label


def compute_clinical_f1(predictions: list[str], references: list[str]) -> dict[str, Any]:
    """Chest CT clinical metric: CheXbert micro/macro F1 plus mean per-label F1."""
    try:
        import torch
        from f1chexbert import F1CheXbert
    except ImportError as exc:
        logger.warning("Clinical F1 unavailable: %s", exc)
        return {}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Loading F1CheXbert on device=%s", device)
    metric = F1CheXbert(device=device)

    try:
        hyps = truncate_for_metric(metric, predictions)
        refs = truncate_for_metric(metric, references)
        _, _, report_14, _ = metric(hyps=hyps, refs=refs)
    finally:
        del metric
        empty_torch_cache()

    per_label = extract_per_label_f1(report_14)
    return {
        "macro_f1": float(report_14.get("macro avg", {}).get("f1-score", 0.0)),
        "micro_f1": float(report_14.get("micro avg", {}).get("f1-score", 0.0)),
        "clinical_f1": float(mean(per_label.values())) if per_label else 0.0,
        "clinical_f1_per_label": per_label,
    }


def add_nlg_metrics(metrics_to_run: list[str], predictions: list[str], references: list[str], out: dict[str, Any]) -> None:
    """Compute common NLG metrics and keep per-sample scores for predictions_vs_gt.json."""
    if "bleu" in metrics_to_run:
        logger.info("Computing BLEU-4...")
        result = compute_bleu(predictions, references)
        out.update({k: v for k, v in result.items() if not k.endswith("_per_sample")})
        out["_bleu4_per_sample"] = result.get("bleu4_per_sample", [])

    if "rouge_l" in metrics_to_run:
        logger.info("Computing ROUGE-L...")
        result = compute_rouge_l(predictions, references)
        out.update({k: v for k, v in result.items() if not k.endswith("_per_sample")})
        out["_rouge_l_per_sample"] = result.get("rouge_l_per_sample", [])

    if "meteor" in metrics_to_run:
        logger.info("Computing METEOR...")
        result = compute_meteor(predictions, references)
        out.update({k: v for k, v in result.items() if not k.endswith("_per_sample")})
        out["_meteor_per_sample"] = result.get("meteor_per_sample", [])


def add_pipeline_stats(results: list[dict[str, Any]], out: dict[str, Any]) -> None:
    review_passed = sum(1 for record in results if record.get("review_passed"))
    out["review_pass_rate"] = review_passed / max(len(results), 1)
    out["avg_iterations"] = sum(record.get("iterations", 1) for record in results) / max(len(results), 1)
    out["num_samples"] = len(results)


def save_outputs(
    output_path: Path,
    metrics: dict[str, Any],
    results: list[dict[str, Any]],
    predictions: list[str],
    references: list[str],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    logger.info("Metrics saved to %s", output_path)

    per_sample_records = []
    for idx, (pred, ref, record) in enumerate(zip(predictions, references, results)):
        item = {
            "index": idx,
            "sample_index": record.get("sample_index", idx),
            "prediction": pred,
            "ground_truth": ref,
            "structured_report": record.get("structured_report"),
        }
        for key in ("_bleu4_per_sample", "_rouge_l_per_sample", "_meteor_per_sample"):
            values = metrics.get(key, [])
            if idx < len(values):
                item[key.lstrip("_").replace("_per_sample", "")] = values[idx]
        per_sample_records.append(item)

    per_sample_path = output_path.parent / "predictions_vs_gt.json"
    per_sample_path.write_text(json.dumps(per_sample_records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    logger.info("Per-sample predictions saved to %s", per_sample_path)


def print_summary(metrics: dict[str, Any]) -> None:
    print("\n" + "=" * 60)
    print("  Chest CT Evaluation Results")
    print("=" * 60)
    for key, label in (
        ("bleu4_corpus", "BLEU-4 (corpus)"),
        ("bleu4_avg", "BLEU-4 (avg)"),
        ("rouge_l", "ROUGE-L"),
        ("meteor", "METEOR"),
        ("macro_f1", "Macro-F1"),
        ("micro_f1", "Micro-F1"),
        ("clinical_f1", "clinical-F1"),
        ("image_evidence_rate", "Image evidence"),
        ("retrieved_evidence_rate", "Retrieved evidence"),
        ("impression_link_rate", "Impression links"),
    ):
        if key in metrics:
            value = metrics[key]
            print(f"  {label:<25s} {value:.4f}" if isinstance(value, float) else f"  {label:<25s} {value}")
    print("=" * 60 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Chest CT RadClaw outputs.")
    parser.add_argument("--predictions", type=str, required=True, help="RadClaw output directory")
    parser.add_argument(
        "--metrics",
        type=str,
        default="all",
        help="Comma-separated metrics: bleu,rouge_l,meteor,clinical_f1,evidence,nlg,clinical,all",
    )
    parser.add_argument("--output", type=str, default=None, help="Default: <predictions>/evaluation_metrics.json")
    args = parser.parse_args()

    pred_dir = Path(args.predictions)
    if not pred_dir.exists():
        logger.error("Prediction path does not exist: %s", pred_dir)
        return

    results, predictions, references = load_predictions(pred_dir)
    if not predictions:
        logger.error("No valid predictions found in %s", pred_dir)
        return

    metrics_to_run = parse_metrics(args.metrics)
    all_metrics: dict[str, Any] = {}

    # Shared metrics across chest X-ray, chest CT, and Brain MRI.
    add_nlg_metrics(metrics_to_run, predictions, references, all_metrics)
    if "evidence" in metrics_to_run:
        logger.info("Computing evidence coverage...")
        all_metrics.update(compute_evidence_metrics(results))

    # Chest CT-specific clinical metrics.
    if "clinical_f1" in metrics_to_run:
        logger.info("Computing Chest CT clinical F1...")
        all_metrics.update(compute_clinical_f1(predictions, references))

    add_pipeline_stats(results, all_metrics)
    print_summary(all_metrics)

    output_path = Path(args.output) if args.output else pred_dir / "evaluation_metrics.json"
    save_outputs(output_path, all_metrics, results, predictions, references)


if __name__ == "__main__":
    main()
