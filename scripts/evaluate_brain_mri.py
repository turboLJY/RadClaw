#!/usr/bin/env python3
"""
Evaluate Brain MRI RadClaw outputs.

All RadClaw generated results share the same directory layout under output/:
  sample_x/summary.json contains natural_language_report and ground_truth
  sample_x/result.json optionally contains structured_report for evidence metrics

Shared metrics are reused from evaluate_chest_xray.py:
  - NLG: BLEU-4, ROUGE-L, METEOR
  - Evidence coverage: image/retrieved evidence and impression-link rates

Brain MRI-specific metrics follow the provided Brain MRI reference scripts:
  - BERTScore: bert-base-uncased, rescale_with_baseline=False
  - RaTEScore: default RaTEScore
  - RadGraph: F1RadGraph(reward_level="partial", model_type="radgraph")
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
from pathlib import Path
from statistics import mean
from typing import Any

from evaluate_chest_xray import (
    _quiet_ratescore_noise,
    compute_bleu,
    compute_evidence_metrics,
    compute_meteor,
    compute_rouge_l,
    load_predictions,
)

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ALL_METRICS = ["bleu", "rouge_l", "meteor", "bertscore", "ratescore", "radgraph", "evidence"]
NLG_METRICS = ["bleu", "rouge_l", "meteor"]
CLINICAL_METRICS = ["bertscore", "ratescore", "radgraph"]


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


def compute_bertscore(predictions: list[str], references: list[str]) -> dict[str, Any]:
    """Brain MRI reference setting: bert-base-uncased with no baseline rescaling."""
    try:
        import torch
        from bert_score import BERTScorer
    except ImportError as exc:
        logger.warning("BERTScore unavailable: %s", exc)
        return {}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    scorer = BERTScorer(
        model_type="bert-base-uncased",
        lang="en",
        rescale_with_baseline=False,
        device=device,
    )
    try:
        _, _, f1 = scorer.score(predictions, references, batch_size=32)
        scores = [float(value) for value in f1.cpu().numpy()]
    finally:
        del scorer
        empty_torch_cache()

    return {
        "bertscore_f1": float(mean(scores)) if scores else 0.0,
        "bertscore_f1_per_sample": scores,
    }


def compute_ratescore(predictions: list[str], references: list[str]) -> dict[str, Any]:
    """Brain MRI reference setting: default RaTEScore."""
    _quiet_ratescore_noise()
    try:
        from RaTEScore import RaTEScore
    except ImportError:
        logger.warning("RaTEScore not installed.")
        return {}

    scorer = RaTEScore()
    try:
        scores = scorer.compute_score(predictions, references)
    except Exception as exc:
        logger.error("RaTEScore failed: %s", exc)
        return {"ratescore": 0.0, "ratescore_per_sample": []}

    if not isinstance(scores, list):
        scores = [float(scores)]
    scores = [float(score) for score in scores]
    return {
        "ratescore": float(mean(scores)) if scores else 0.0,
        "ratescore_per_sample": scores,
    }


def compute_radgraph(predictions: list[str], references: list[str]) -> dict[str, Any]:
    """Brain MRI reference setting: original RadGraph, partial reward."""
    try:
        from radgraph import F1RadGraph
    except ImportError:
        logger.warning("radgraph not installed.")
        return {}

    hyps = [text if text.strip() else "none" for text in predictions]
    refs = [text if text.strip() else "none" for text in references]
    scorer = F1RadGraph(reward_level="partial", model_type="radgraph")
    try:
        _, reward_list, _, _ = scorer(hyps=hyps, refs=refs)
    finally:
        del scorer
        empty_torch_cache()

    scores = [float(score) for score in reward_list]
    return {
        "radgraph": float(mean(scores)) if scores else 0.0,
        "radgraph_per_sample": scores,
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


def add_brain_metrics(metrics_to_run: list[str], predictions: list[str], references: list[str], out: dict[str, Any]) -> None:
    """Compute Brain MRI-specific metrics from the external reference scripts."""
    if "bertscore" in metrics_to_run:
        logger.info("Computing BERTScore...")
        result = compute_bertscore(predictions, references)
        out.update({k: v for k, v in result.items() if not k.endswith("_per_sample")})
        out["_bertscore_f1_per_sample"] = result.get("bertscore_f1_per_sample", [])

    if "ratescore" in metrics_to_run:
        logger.info("Computing RaTEScore...")
        result = compute_ratescore(predictions, references)
        out.update({k: v for k, v in result.items() if not k.endswith("_per_sample")})
        out["_ratescore_per_sample"] = result.get("ratescore_per_sample", [])

    if "radgraph" in metrics_to_run:
        logger.info("Computing RadGraph...")
        result = compute_radgraph(predictions, references)
        out.update({k: v for k, v in result.items() if not k.endswith("_per_sample")})
        out["_radgraph_per_sample"] = result.get("radgraph_per_sample", [])


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
        for key in (
            "_bleu4_per_sample",
            "_rouge_l_per_sample",
            "_meteor_per_sample",
            "_bertscore_f1_per_sample",
            "_ratescore_per_sample",
            "_radgraph_per_sample",
        ):
            values = metrics.get(key, [])
            if idx < len(values):
                item[key.lstrip("_").replace("_per_sample", "")] = values[idx]
        per_sample_records.append(item)

    per_sample_path = output_path.parent / "predictions_vs_gt.json"
    per_sample_path.write_text(json.dumps(per_sample_records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    logger.info("Per-sample predictions saved to %s", per_sample_path)


def print_summary(metrics: dict[str, Any]) -> None:
    print("\n" + "=" * 60)
    print("  Brain MRI Evaluation Results")
    print("=" * 60)
    for key, label in (
        ("bleu4_corpus", "BLEU-4 (corpus)"),
        ("bleu4_avg", "BLEU-4 (avg)"),
        ("rouge_l", "ROUGE-L"),
        ("meteor", "METEOR"),
        ("bertscore_f1", "BERTScore F1"),
        ("ratescore", "RaTEScore"),
        ("radgraph", "RadGraph"),
        ("image_evidence_rate", "Image evidence"),
        ("retrieved_evidence_rate", "Retrieved evidence"),
        ("impression_link_rate", "Impression links"),
    ):
        if key in metrics:
            value = metrics[key]
            print(f"  {label:<25s} {value:.4f}" if isinstance(value, float) else f"  {label:<25s} {value}")
    print("=" * 60 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Brain MRI RadClaw outputs.")
    parser.add_argument("--predictions", type=str, required=True, help="RadClaw output directory")
    parser.add_argument(
        "--metrics",
        type=str,
        default="all",
        help="Comma-separated metrics: bleu,rouge_l,meteor,bertscore,ratescore,radgraph,evidence,nlg,clinical,all",
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

    # Brain MRI-specific metrics.
    add_brain_metrics(metrics_to_run, predictions, references, all_metrics)

    add_pipeline_stats(results, all_metrics)
    print_summary(all_metrics)

    output_path = Path(args.output) if args.output else pred_dir / "evaluation_metrics.json"
    save_outputs(output_path, all_metrics, results, predictions, references)


if __name__ == "__main__":
    main()
