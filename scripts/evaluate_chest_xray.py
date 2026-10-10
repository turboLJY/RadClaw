#!/usr/bin/env python3
"""
Evaluate RadClaw outputs against ground truth using 6 metrics:

NLG metrics:
  - BLEU-4 (sacrebleu)
  - ROUGE-L (rouge_score)
  - METEOR (nltk)

Clinical metrics:
  - CheXbert F1 (f1chexbert)
  - GREEN (StanfordAIMI/GREEN-RadLlama2-7b)
  - RaTEScore (ratescore)

Usage:
  python scripts/evaluate.py --predictions output/traces/sample_5_v6/sample_5
  python scripts/evaluate.py --predictions output/traces/ --metrics all
  python scripts/evaluate.py --predictions output/traces/ --metrics bleu,rouge_l,meteor
  python scripts/evaluate.py --predictions output/traces/ --metrics chexbert_f1,green,ratescore
"""

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _disable_ssl_verification():
    """Bypass SSL verification for HuggingFace model downloads."""
    os.environ["CURL_CA_BUNDLE"] = ""
    os.environ["REQUESTS_CA_BUNDLE"] = ""
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    import requests
    _original_send = requests.Session.send
    def _patched_send(self, request, **kwargs):
        kwargs["verify"] = False
        return _original_send(self, request, **kwargs)
    requests.Session.send = _patched_send

_disable_ssl_verification()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class _MetricNoiseFilter(logging.Filter):
    """Drop noisy third-party metric messages while keeping RadClaw logs."""

    NOISY_MESSAGES = {
        r'"\." is not a eligible syntax.',
    }

    def filter(self, record: logging.LogRecord) -> bool:
        return record.getMessage() not in self.NOISY_MESSAGES


def _quiet_ratescore_noise() -> None:
    """Suppress RaTEScore sentence-splitter debug/info noise.

    RaTEScore initializes medspaCy/PyRuSH, which logs repeated messages such as
    '"\." is not a eligible syntax.' and verbose token maps. These are not
    metric failures and do not affect score computation.
    """

    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        if not any(isinstance(f, _MetricNoiseFilter) for f in handler.filters):
            handler.addFilter(_MetricNoiseFilter())

    for logger_name in ("PyRuSH", "medspacy", "medspacy_pyrush"):
        noisy_logger = logging.getLogger(logger_name)
        noisy_logger.setLevel(logging.WARNING)
        noisy_logger.propagate = False

    try:
        from loguru import logger as loguru_logger

        loguru_logger.disable("PyRuSH")
        loguru_logger.disable("medspacy")
    except Exception:
        pass

# ---------------------------------------------------------------------------
# NLG Metrics
# ---------------------------------------------------------------------------

def compute_bleu(predictions: list[str], references: list[str]) -> dict:
    """BLEU-4 via sacrebleu."""
    try:
        from sacrebleu.metrics import BLEU
    except ImportError:
        logger.warning("sacrebleu not installed. pip install sacrebleu")
        return {}

    bleu = BLEU(effective_order=True)

    per_sample = []
    for pred, ref in zip(predictions, references):
        score = bleu.sentence_score(pred, [ref])
        per_sample.append(score.score / 100.0)

    corpus_score = bleu.corpus_score(predictions, [references])

    return {
        "bleu4_corpus": corpus_score.score / 100.0,
        "bleu4_avg": sum(per_sample) / len(per_sample) if per_sample else 0.0,
        "bleu4_per_sample": per_sample,
    }


def compute_rouge_l(predictions: list[str], references: list[str]) -> dict:
    """ROUGE-L via rouge_score."""
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        logger.warning("rouge_score not installed. pip install rouge-score")
        return {}

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    scores = {"rouge1": [], "rouge2": [], "rougeL": []}
    for pred, ref in zip(predictions, references):
        result = scorer.score(ref, pred)
        for k in scores:
            scores[k].append(result[k].fmeasure)

    return {
        "rouge1": sum(scores["rouge1"]) / len(scores["rouge1"]) if scores["rouge1"] else 0.0,
        "rouge2": sum(scores["rouge2"]) / len(scores["rouge2"]) if scores["rouge2"] else 0.0,
        "rouge_l": sum(scores["rougeL"]) / len(scores["rougeL"]) if scores["rougeL"] else 0.0,
        "rouge_l_per_sample": scores["rougeL"],
    }


def compute_meteor(predictions: list[str], references: list[str]) -> dict:
    """METEOR via nltk."""
    try:
        import nltk
        from nltk.translate.meteor_score import meteor_score
    except ImportError:
        logger.warning("nltk not installed. pip install nltk")
        return {}

    nltk.data.path.insert(0, "/mnt/yulan_pretrain_131/clinical_project/nltk_data")
    nltk.download("wordnet", quiet=True)
    nltk.download("punkt_tab", quiet=True)

    per_sample = []
    for pred, ref in zip(predictions, references):
        ref_tokens = nltk.word_tokenize(ref)
        pred_tokens = nltk.word_tokenize(pred)
        score = meteor_score([ref_tokens], pred_tokens)
        per_sample.append(score)

    return {
        "meteor": sum(per_sample) / len(per_sample) if per_sample else 0.0,
        "meteor_per_sample": per_sample,
    }


# ---------------------------------------------------------------------------
# Clinical Metrics
# ---------------------------------------------------------------------------

def compute_chexbert_f1(predictions: list[str], references: list[str]) -> dict:
    """CheXbert F1 — extracts 14 CheXpert labels, computes F1 on 5 key conditions.

    Returns accuracy (exact match on 5 conditions), per-sample accuracy,
    and classification reports for all 14 and top 5 conditions.

    Fixes applied:
      - tokenizer() instead of deprecated encode_plus/batch_encode_plus
      - tokenizer(...)[0] to unwrap nested list from new transformers
      - _check_targets unpacking uses *_ to handle sklearn>=1.8 extra return value
      - input truncated to 512 tokens before feeding to BERT
    """
    try:
        from f1chexbert import F1CheXbert
    except ImportError:
        logger.warning("f1chexbert not installed. pip install f1chexbert")
        return {}

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Loading CheXbert model (device=%s, will download on first run)...", device)
    f1_metric = F1CheXbert(device=device)

    # Truncate inputs to BERT's 512-token limit to avoid indexing errors
    try:
        tokenizer = f1_metric.tokenizer
        MAX_TOKENS = 500  # leave room for special tokens

        def _truncate(text: str) -> str:
            ids = tokenizer.encode(text, add_special_tokens=False)
            if len(ids) > MAX_TOKENS:
                ids = ids[:MAX_TOKENS]
                text = tokenizer.decode(ids, skip_special_tokens=True)
            return text

        predictions_trunc = [_truncate(p) for p in predictions]
        references_trunc  = [_truncate(r) for r in references]
    except Exception:
        # If truncation fails for any reason, fall back to originals
        predictions_trunc = predictions
        references_trunc  = references

    # Log if any truncation happened
    n_trunc = sum(1 for a, b in zip(predictions_trunc, predictions) if a != b)
    n_trunc += sum(1 for a, b in zip(references_trunc, references) if a != b)
    if n_trunc:
        logger.warning("Truncated %d texts to fit BERT 512-token limit", n_trunc)

    # Returns: (accuracy_5, per_element_accuracy, classification_report_14, classification_report_5)
    accuracy_5, pe_accuracy, cr_14, cr_5 = f1_metric(hyps=predictions_trunc, refs=references_trunc)

    f1_micro_5  = cr_5.get("micro avg", {}).get("f1-score", accuracy_5)
    f1_macro_5  = cr_5.get("macro avg", {}).get("f1-score", 0.0)
    f1_micro_14 = cr_14.get("micro avg", {}).get("f1-score", 0.0)

    return {
        "chexbert_f1_micro_5":  f1_micro_5,
        "chexbert_f1_macro_5":  f1_macro_5,
        "chexbert_f1_micro_14": f1_micro_14,
        "chexbert_accuracy_5":  accuracy_5,
        "chexbert_5_per_sample": pe_accuracy.tolist() if hasattr(pe_accuracy, "tolist") else list(pe_accuracy),
    }


def compute_green(predictions: list[str], references: list[str], model_name: str = None, output_dir: str = ".") -> dict:
    """GREEN — LLM-based radiology report evaluation (StanfordAIMI)."""
    if model_name is None:
        model_name = "StanfordAIMI/GREEN-RadLlama2-7b"

    green_module_path = str(Path(__file__).resolve().parent.parent / "src" / "metrics")
    if green_module_path not in sys.path:
        sys.path.insert(0, green_module_path)

    # GREEN's wrapper places the model on torch.cuda.current_device(), while its
    # tokenization path moves batches to LOCAL_RANK. Keep both pointed at the GPU
    # with the most currently free memory. Setting CUDA_VISIBLE_DEVICES here is
    # too late if another metric has already initialized CUDA.
    run_on_cpu = False
    try:
        import torch

        if torch.cuda.is_available():
            device_stats = []
            for device_idx in range(torch.cuda.device_count()):
                with torch.cuda.device(device_idx):
                    free_bytes, total_bytes = torch.cuda.mem_get_info()
                device_stats.append((device_idx, free_bytes, total_bytes))

            best_device, free_bytes, total_bytes = max(device_stats, key=lambda x: x[1])
            torch.cuda.set_device(best_device)
            os.environ["LOCAL_RANK"] = str(best_device)
            logger.info(
                "GREEN using CUDA device %d (%.2f GiB free / %.2f GiB total)",
                best_device,
                free_bytes / (1024 ** 3),
                total_bytes / (1024 ** 3),
            )
        else:
            run_on_cpu = True
            logger.warning("CUDA is not available; GREEN will run on CPU")
    except Exception as e:
        logger.warning("Could not select CUDA device for GREEN: %s", e)

    try:
        from green_score import GREEN
    except ImportError:
        logger.warning("GREEN module not available. Check src/metrics/green_score/")
        return {}

    logger.info("Loading GREEN model: %s ...", model_name)
    scorer = GREEN(
        model_name=model_name,
        output_dir=output_dir,
        cpu=run_on_cpu,
        compute_summary_stats=False,
    )

    mean, std, green_scores, summary, results_df = scorer(refs=references, hyps=predictions)

    valid_scores = [s for s in green_scores if s is not None]
    avg_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0

    result = {
        "green": avg_score,
        "green_per_sample": green_scores,
    }

    # Save detailed GREEN output (LLM explanations) if results_df is available
    if results_df is not None:
        try:
            green_detail_path = os.path.join(output_dir, "green_details.csv")
            results_df.to_csv(green_detail_path, index=False)
            logger.info("GREEN detailed output saved to %s", green_detail_path)
        except Exception as e:
            logger.warning("Could not save GREEN details: %s", e)

    return result


def compute_ratescore(predictions: list[str], references: list[str]) -> dict:
    """RaTEScore — entity-aware radiology report metric."""
    _quiet_ratescore_noise()
    try:
        from RaTEScore import RaTEScore
    except ImportError:
        logger.warning("ratescore not installed. pip install ratescore")
        return {}

    logger.info("Loading RaTEScore model (will download on first run)...")
    scorer = RaTEScore()

    try:
        scores = scorer.compute_score(predictions, references)
    except Exception as e:
        logger.error("RaTEScore failed: %s", e)
        return {"ratescore": 0.0, "ratescore_per_sample": []}

    if not isinstance(scores, list):
        scores = [float(scores)]

    return {
        "ratescore": sum(scores) / len(scores) if scores else 0.0,
        "ratescore_per_sample": [float(s) for s in scores],
    }


# ---------------------------------------------------------------------------
# Evidence Coverage (RadClaw-specific)
# ---------------------------------------------------------------------------

def compute_evidence_metrics(results: list[dict]) -> dict:
    total_findings = 0
    findings_with_img = 0
    findings_with_ev = 0
    total_impressions = 0
    impressions_with_links = 0

    for r in results:
        report = r.get("structured_report") or {}  # guard against None
        for f in report.get("findings", []):
            total_findings += 1
            support = f.get("support", {})
            if support.get("image_evidence"):
                findings_with_img += 1
            if support.get("retrieved_evidence"):
                findings_with_ev += 1
        for imp in report.get("impressions", []):
            total_impressions += 1
            if imp.get("linked_findings"):
                impressions_with_links += 1

    return {
        "total_findings": total_findings,
        "findings_with_image_evidence": findings_with_img,
        "image_evidence_rate": findings_with_img / max(total_findings, 1),
        "findings_with_retrieved_evidence": findings_with_ev,
        "retrieved_evidence_rate": findings_with_ev / max(total_findings, 1),
        "total_impressions": total_impressions,
        "impressions_with_finding_links": impressions_with_links,
        "impression_link_rate": impressions_with_links / max(total_impressions, 1),
    }


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def extract_report(text: str) -> str:
    """Extract the actual radiology report from model output that may contain
    thinking traces, preambles, or other non-report content.

    Strategy:
      1. Strip <think>...</think> blocks (Qwen3 reasoning traces)
      2. If FINDINGS: or IMPRESSION: header exists, extract from the first
         occurrence to the end (or next non-report section)
      3. Strip markdown reasoning content (**Step ...**, **Finding ...**, etc.)
      4. Strip common preambles ("Here is the report", "Based on...", etc.)
    """
    if not text:
        return text

    # 1. Remove <think>...</think> blocks (greedy, handles multiline)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

    # 2. Remove everything before the first FINDINGS: or IMPRESSION: header
    match = re.search(r"(?i)(FINDINGS|IMPRESSION)\s*:", text)
    if match:
        text = text[match.start():]
    else:
        # No standard header found — try to strip common preambles
        preamble_patterns = [
            r"^.*?(?:here is|here's|below is|the following is)[^:]*?:\s*\n",
            r"^.*?(?:Based on (?:the|my) (?:analysis|review|examination))[^:]*?:\s*\n",
        ]
        for pat in preamble_patterns:
            text = re.sub(pat, "", text, count=1, flags=re.IGNORECASE | re.DOTALL)
            break

    # 3. Truncate at first markdown reasoning line (**Step / **Finding / etc.)
    lines = text.split("\n")
    clean_lines = []
    for line in lines:
        if re.match(r"^\s*\*\*", line):
            break
        clean_lines.append(line)
    text = "\n".join(clean_lines)

    return text.strip()


def load_predictions(pred_path: Path) -> tuple[list[dict], list[str], list[str]]:
    """Load prediction data from a directory. Supports both flat and nested layouts."""
    results = []
    predictions = []
    references = []

    summary_files = sorted(pred_path.rglob("summary.json"))

    if not summary_files:
        json_files = sorted(pred_path.glob("sample_*.json"))
        for f in json_files:
            with open(f) as fp:
                data = json.load(fp)
            results.append(data)
            predictions.append(extract_report(data.get("natural_language_report", "")))
            references.append(data.get("ground_truth", ""))
    else:
        for f in summary_files:
            with open(f) as fp:
                data = json.load(fp)
            if not data.get("ground_truth") or not data.get("natural_language_report"):
                continue
            # Try to load structured_report from sibling result.json
            result_path = f.parent / "result.json"
            if result_path.exists():
                with open(result_path) as fp:
                    result_data = json.load(fp)
                data["structured_report"] = result_data.get("structured_report")
            results.append(data)
            predictions.append(extract_report(data.get("natural_language_report", "")))
            references.append(data.get("ground_truth", ""))

    cleaned = sum(1 for p, d in zip(predictions, results)
                  if p != d.get("natural_language_report", ""))
    if cleaned:
        logger.info("Cleaned %d/%d reports (removed thinking/preamble)", cleaned, len(predictions))

    return results, predictions, references


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

ALL_METRICS = ["bleu", "rouge_l", "meteor", "chexbert_f1", "green", "ratescore"]
NLG_METRICS = ["bleu", "rouge_l", "meteor"]
CLINICAL_METRICS = ["chexbert_f1", "green", "ratescore"]


def main():
    parser = argparse.ArgumentParser(description="Evaluate RadClaw outputs (6 metrics)")
    parser.add_argument("--predictions", type=str, required=True,
                        help="Directory with prediction JSONs (supports nested trace dirs)")
    parser.add_argument("--metrics", type=str, default="all",
                        help="Comma-separated metrics: bleu,rouge_l,meteor,chexbert_f1,green,ratescore,evidence,all")
    parser.add_argument("--green_model", type=str, default=None,
                        help="GREEN model name (default: StanfordAIMI/GREEN-RadLlama2-7b)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path (default: <predictions>/evaluation_metrics.json)")
    args = parser.parse_args()

    pred_dir = Path(args.predictions)
    if not pred_dir.exists():
        logger.error("Prediction path does not exist: %s", pred_dir)
        return

    results, predictions, references = load_predictions(pred_dir)
    if not predictions:
        logger.error("No valid predictions found in %s", pred_dir)
        return

    logger.info("Loaded %d prediction-reference pairs", len(predictions))

    metrics_requested = args.metrics.strip()
    if metrics_requested == "all":
        metrics_to_run = list(ALL_METRICS) + ["evidence"]
    elif metrics_requested == "nlg":
        metrics_to_run = list(NLG_METRICS)
    elif metrics_requested == "clinical":
        metrics_to_run = list(CLINICAL_METRICS)
    else:
        metrics_to_run = [m.strip() for m in metrics_requested.split(",")]

    all_metrics = {}

    # --- NLG ---
    if "bleu" in metrics_to_run:
        logger.info("Computing BLEU-4...")
        bleu = compute_bleu(predictions, references)
        all_metrics.update({k: v for k, v in bleu.items() if not k.endswith("_per_sample")})
        if bleu:
            logger.info("  BLEU-4 (corpus): %.4f", bleu.get("bleu4_corpus", 0))
            logger.info("  BLEU-4 (avg):    %.4f", bleu.get("bleu4_avg", 0))
            all_metrics["_bleu4_per_sample"] = bleu.get("bleu4_per_sample", [])

    if "rouge_l" in metrics_to_run:
        logger.info("Computing ROUGE...")
        rouge = compute_rouge_l(predictions, references)
        all_metrics.update({k: v for k, v in rouge.items() if not k.endswith("_per_sample")})
        if rouge:
            logger.info("  ROUGE-1: %.4f", rouge.get("rouge1", 0))
            logger.info("  ROUGE-2: %.4f", rouge.get("rouge2", 0))
            logger.info("  ROUGE-L: %.4f", rouge.get("rouge_l", 0))
            all_metrics["_rouge_l_per_sample"] = rouge.get("rouge_l_per_sample", [])

    if "meteor" in metrics_to_run:
        logger.info("Computing METEOR...")
        meteor = compute_meteor(predictions, references)
        all_metrics.update({k: v for k, v in meteor.items() if not k.endswith("_per_sample")})
        if meteor:
            logger.info("  METEOR:  %.4f", meteor.get("meteor", 0))
            all_metrics["_meteor_per_sample"] = meteor.get("meteor_per_sample", [])

    # --- Clinical ---
    if "chexbert_f1" in metrics_to_run:
        logger.info("Computing CheXbert F1...")
        chexbert = compute_chexbert_f1(predictions, references)
        all_metrics.update({k: v for k, v in chexbert.items() if v is not None and not k.endswith("_per_sample")})
        if chexbert:
            logger.info("  CheXbert F1 micro (5): %.4f", chexbert.get("chexbert_f1_micro_5", 0))
            logger.info("  CheXbert F1 macro (5): %.4f", chexbert.get("chexbert_f1_macro_5", 0))
            logger.info("  CheXbert accuracy (5): %.4f", chexbert.get("chexbert_accuracy_5", 0))
            all_metrics["_chexbert_5_per_sample"] = chexbert.get("chexbert_5_per_sample", [])

    if "green" in metrics_to_run:
        logger.info("Computing GREEN...")
        green = compute_green(predictions, references, model_name=args.green_model,
                              output_dir=str(pred_dir))
        all_metrics.update({k: v for k, v in green.items() if not k.endswith("_per_sample")})
        if green:
            logger.info("  GREEN:   %.4f", green.get("green", 0))
            all_metrics["_green_per_sample"] = green.get("green_per_sample", [])

    if "ratescore" in metrics_to_run:
        logger.info("Computing RaTEScore...")
        rate = compute_ratescore(predictions, references)
        all_metrics.update({k: v for k, v in rate.items() if not k.endswith("_per_sample")})
        if rate:
            logger.info("  RaTEScore: %.4f", rate.get("ratescore", 0))
            all_metrics["_ratescore_per_sample"] = rate.get("ratescore_per_sample", [])

    # --- Evidence Coverage ---
    if "evidence" in metrics_to_run:
        logger.info("Computing evidence coverage...")
        ev = compute_evidence_metrics(results)
        all_metrics.update(ev)
        logger.info("  Image evidence rate:    %.2f%%", ev["image_evidence_rate"] * 100)
        logger.info("  Retrieved evidence rate: %.2f%%", ev["retrieved_evidence_rate"] * 100)
        logger.info("  Impression link rate:   %.2f%%", ev["impression_link_rate"] * 100)

    # --- Pipeline stats ---
    review_passed = sum(1 for r in results if r.get("review_passed"))
    avg_iterations = sum(r.get("iterations", 1) for r in results) / max(len(results), 1)
    all_metrics["review_pass_rate"] = review_passed / max(len(results), 1)
    all_metrics["avg_iterations"] = avg_iterations
    all_metrics["num_samples"] = len(results)
    logger.info("Review pass rate: %.2f%%", all_metrics["review_pass_rate"] * 100)
    logger.info("Avg iterations: %.2f", avg_iterations)

    # --- Summary table ---
    print("\n" + "=" * 60)
    print("  RadClaw Evaluation Results")
    print("=" * 60)
    summary_keys = [
        ("bleu4_corpus", "BLEU-4 (corpus)"),
        ("bleu4_avg",    "BLEU-4 (avg)"),
        ("rouge1",       "ROUGE-1"),
        ("rouge2",       "ROUGE-2"),
        ("rouge_l",      "ROUGE-L"),
        ("meteor",       "METEOR"),
        ("chexbert_f1_micro_5",  "CheXbert F1 micro(5)"),
        ("chexbert_f1_macro_5",  "CheXbert F1 macro(5)"),
        ("chexbert_accuracy_5",  "CheXbert Accuracy(5)"),
        ("green",        "GREEN"),
        ("ratescore",    "RaTEScore"),
    ]
    for key, label in summary_keys:
        if key in all_metrics:
            val = all_metrics[key]
            if isinstance(val, float):
                print(f"  {label:<25s} {val:.4f}")
            else:
                print(f"  {label:<25s} {val}")
    print("=" * 60 + "\n")

    # --- Save ---
    output_path = Path(args.output) if args.output else pred_dir / "evaluation_metrics.json"
    serializable = {}
    for k, v in all_metrics.items():
        if isinstance(v, (int, float, str, bool, list, type(None))):
            serializable[k] = v
        else:
            serializable[k] = str(v)

    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    logger.info("Metrics saved to %s", output_path)

    # --- Save per-sample predictions alongside metrics ---
    per_sample_path = output_path.parent / "predictions_vs_gt.json"
    per_sample_records = []
    for i, (pred, ref, res) in enumerate(zip(predictions, references, results)):
        record = {
            "index": i,
            "sample_index": res.get("sample_index", i),
            "prediction": pred,
            "ground_truth": ref,
            "structured_report": res.get("structured_report"),
        }
        for key in ("_green_per_sample", "_ratescore_per_sample", "_bleu4_per_sample",
                    "_rouge_l_per_sample", "_meteor_per_sample", "_chexbert_5_per_sample"):
            vals = serializable.get(key, [])
            if i < len(vals):
                record[key.lstrip("_").replace("_per_sample", "")] = vals[i]
        per_sample_records.append(record)
    with open(per_sample_path, "w") as f:
        json.dump(per_sample_records, f, indent=2, ensure_ascii=False)
    logger.info("Per-sample predictions saved to %s", per_sample_path)


if __name__ == "__main__":
    main()