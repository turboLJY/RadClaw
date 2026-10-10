#!/usr/bin/env python3
"""
Evaluate Abdominal CT RadClaw outputs.

All RadClaw generated results share the same directory layout under output/:
  sample_x/summary.json contains natural_language_report and ground_truth
  sample_x/result.json optionally contains structured_report for evidence metrics

Shared metrics are reused from evaluate_chest_xray.py:
  - NLG: BLEU-4, ROUGE-L, METEOR
  - Evidence coverage: image/retrieved evidence and impression-link rates

Abdominal CT-specific clinical metrics follow abdominal_ct/baseline/evaluate_reports.py:
  - BERTScore: microsoft/deberta-xlarge-mnli
  - UMLS precision / recall: scispaCy entity linker, threshold 0.7
"""

from __future__ import annotations

import argparse
import gc
import hashlib
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

ALL_METRICS = ["bleu", "rouge_l", "meteor", "bertscore", "umls", "evidence"]
NLG_METRICS = ["bleu", "rouge_l", "meteor"]
CLINICAL_METRICS = ["bertscore", "umls"]

BERTSCORE_MODEL = "bert-base-uncased"
BERTSCORE_BATCH_SIZE = 8
UMLS_PIPELINE = "en_core_sci_sm"
UMLS_LINKER_NAME = "umls"
UMLS_THRESHOLD = 0.7
UMLS_BATCH_SIZE = 16


def parse_metrics(metrics: str) -> list[str]:
    """Parse the same comma-separated metric argument style used by the other evaluators."""
    key = metrics.strip().lower()
    if key == "all":
        return list(ALL_METRICS)
    if key == "nlg":
        return list(NLG_METRICS)
    if key == "clinical":
        return list(CLINICAL_METRICS)

    aliases = {
        "bleu4": "bleu",
        "rouge": "rouge_l",
        "rougel": "rouge_l",
        "umls_precision": "umls",
        "umls_recall": "umls",
    }
    parsed = []
    for item in metrics.split(","):
        item = item.strip()
        if not item:
            continue
        key = item.lower().replace("-", "_")
        parsed.append(aliases.get(key, key))
    return parsed


def empty_torch_cache() -> None:
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def compute_bertscore(predictions: list[str], references: list[str]) -> dict[str, Any]:
    """Abdominal CT reference setting: DeBERTa-xlarge-MNLI BERTScore."""
    try:
        from bert_score import score
    except ImportError as exc:
        logger.warning("BERTScore unavailable: %s", exc)
        return {}

    precision, recall, f1 = score(
        predictions,
        references,
        lang="en",
        model_type=BERTSCORE_MODEL,
        batch_size=BERTSCORE_BATCH_SIZE,
        verbose=True,
    )
    return {
        "bertscore_precision": float(precision.mean().item()),
        "bertscore_recall": float(recall.mean().item()),
        "bertscore_f1": float(f1.mean().item()),
        "bertscore_precision_per_sample": [float(x) for x in precision.cpu().tolist()],
        "bertscore_recall_per_sample": [float(x) for x in recall.cpu().tolist()],
        "bertscore_f1_per_sample": [float(x) for x in f1.cpu().tolist()],
    }


def text_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def load_umls_cache(path: Path) -> dict[str, list[str]]:
    if not path.exists():
        return {}
    cache = {}
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            cache[record["hash"]] = record["cuis"]
    return cache


def append_umls_cache(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_umls_pipeline():
    """Build the scispaCy UMLS linker used by the abdominal CT baseline."""
    try:
        import spacy
        import scispacy  # noqa: F401
        from scispacy.linking import EntityLinker  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "UMLS metrics require spacy, scispacy, a biomedical spaCy model, and scispacy.linking."
        ) from exc

    nlp = spacy.load(UMLS_PIPELINE)
    if "scispacy_linker" not in nlp.pipe_names:
        nlp.add_pipe(
            "scispacy_linker",
            config={
                "resolve_abbreviations": True,
                "linker_name": UMLS_LINKER_NAME,
                "threshold": UMLS_THRESHOLD,
            },
        )
    return nlp


def extract_umls_terms(texts: list[str], cache_path: Path) -> list[set[str]]:
    """Extract UMLS CUIs with a text-hash cache so reruns do not relink reports."""
    cache = load_umls_cache(cache_path)
    results: list[set[str] | None] = []
    missing_texts = []
    missing_hashes = []

    for text in texts:
        key = text_hash(text)
        if key in cache:
            results.append(set(cache[key]))
        else:
            results.append(None)
            missing_texts.append(text)
            missing_hashes.append(key)

    if missing_texts:
        nlp = build_umls_pipeline()
        extracted_by_hash = {}
        new_rows = []
        for key, doc in zip(missing_hashes, nlp.pipe(missing_texts, batch_size=UMLS_BATCH_SIZE)):
            cuis = set()
            for ent in doc.ents:
                for cui, score in getattr(ent._, "kb_ents", []):
                    if score >= UMLS_THRESHOLD:
                        cuis.add(str(cui))
            extracted_by_hash[key] = sorted(cuis)
            new_rows.append({"hash": key, "cuis": sorted(cuis)})
        append_umls_cache(cache_path, new_rows)

        for index, value in enumerate(results):
            if value is None:
                results[index] = set(extracted_by_hash[text_hash(texts[index])])

    return [value if value is not None else set() for value in results]


def compute_umls_prf(predictions: list[str], references: list[str], cache_path: Path) -> dict[str, Any]:
    """Compute UMLS precision/recall exactly as set overlap over linked CUIs."""
    ref_terms = extract_umls_terms(references, cache_path)
    pred_terms = extract_umls_terms(predictions, cache_path)

    precisions = []
    recalls = []
    for ref, pred in zip(ref_terms, pred_terms):
        precisions.append(len(ref & pred) / len(pred) if pred else 0.0)
        recalls.append(len(ref & pred) / len(ref) if ref else 0.0)

    return {
        "umls_precision": float(mean(precisions)) if precisions else 0.0,
        "umls_recall": float(mean(recalls)) if recalls else 0.0,
        "umls_precision_per_sample": precisions,
        "umls_recall_per_sample": recalls,
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


def add_abdominal_metrics(
    metrics_to_run: list[str],
    predictions: list[str],
    references: list[str],
    out: dict[str, Any],
    umls_cache_path: Path,
) -> None:
    """Compute Abdominal CT-specific clinical metrics."""
    if "bertscore" in metrics_to_run:
        logger.info("Computing BERTScore...")
        result = compute_bertscore(predictions, references)
        out.update({k: v for k, v in result.items() if not k.endswith("_per_sample")})
        out["_bertscore_precision_per_sample"] = result.get("bertscore_precision_per_sample", [])
        out["_bertscore_recall_per_sample"] = result.get("bertscore_recall_per_sample", [])
        out["_bertscore_f1_per_sample"] = result.get("bertscore_f1_per_sample", [])
        empty_torch_cache()

    if "umls" in metrics_to_run:
        logger.info("Computing UMLS precision/recall...")
        result = compute_umls_prf(predictions, references, umls_cache_path)
        out.update({k: v for k, v in result.items() if not k.endswith("_per_sample")})
        out["_umls_precision_per_sample"] = result.get("umls_precision_per_sample", [])
        out["_umls_recall_per_sample"] = result.get("umls_recall_per_sample", [])
        out["umls_cache"] = str(umls_cache_path)


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
            "_bertscore_precision_per_sample",
            "_bertscore_recall_per_sample",
            "_bertscore_f1_per_sample",
            "_umls_precision_per_sample",
            "_umls_recall_per_sample",
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
    print("  Abdominal CT Evaluation Results")
    print("=" * 60)
    for key, label in (
        ("bleu4_corpus", "BLEU-4 (corpus)"),
        ("bleu4_avg", "BLEU-4 (avg)"),
        ("rouge_l", "ROUGE-L"),
        ("meteor", "METEOR"),
        ("bertscore_precision", "BERTScore P"),
        ("bertscore_recall", "BERTScore R"),
        ("bertscore_f1", "BERTScore F1"),
        ("umls_precision", "UMLS precision"),
        ("umls_recall", "UMLS recall"),
        ("image_evidence_rate", "Image evidence"),
        ("retrieved_evidence_rate", "Retrieved evidence"),
        ("impression_link_rate", "Impression links"),
    ):
        if key in metrics:
            value = metrics[key]
            print(f"  {label:<25s} {value:.4f}" if isinstance(value, float) else f"  {label:<25s} {value}")
    print("=" * 60 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Abdominal CT RadClaw outputs.")
    parser.add_argument("--predictions", type=str, required=True, help="RadClaw output directory")
    parser.add_argument(
        "--metrics",
        type=str,
        default="all",
        help="Comma-separated metrics: bleu,rouge_l,meteor,bertscore,umls,evidence,nlg,clinical,all",
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
    output_path = Path(args.output) if args.output else pred_dir / "evaluation_metrics.json"
    umls_cache_path = output_path.parent / "umls_cache.jsonl"
    all_metrics: dict[str, Any] = {}

    # Shared metrics across all modalities.
    add_nlg_metrics(metrics_to_run, predictions, references, all_metrics)
    if "evidence" in metrics_to_run:
        logger.info("Computing evidence coverage...")
        all_metrics.update(compute_evidence_metrics(results))

    # Abdominal CT-specific clinical metrics.
    add_abdominal_metrics(metrics_to_run, predictions, references, all_metrics, umls_cache_path)

    add_pipeline_stats(results, all_metrics)
    print_summary(all_metrics)
    save_outputs(output_path, all_metrics, results, predictions, references)


if __name__ == "__main__":
    main()
