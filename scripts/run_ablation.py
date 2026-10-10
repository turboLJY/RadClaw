#!/usr/bin/env python3
"""
Ablation study runner for RadClaw.

Supports four modes that share the same data loading, output format, and
evaluation pipeline — so evaluate.py works on all four output directories
with no changes.

Modes:
  direct         B1 — single LLM call, no tools (src/core/baselines.py)
  visual_only    B2 — Visual Analysis + Report Writing, skip Evidence Retrieval
  retrieval_only B3 — Evidence Retrieval + Report Writing, skip Visual Analysis
  full           Full RadClaw 4-agent pipeline (src/core/orchestrator.py)

Usage:
  python scripts/run_ablation.py --mode direct --indices 1,2,3
  python scripts/run_ablation.py --mode retrieval_only --indices 1,2,3 --output-dir output/ablation_b3
  python scripts/run_ablation.py --mode full   --indices 1,2,3 --output-dir output/ablation_full
"""

import argparse
import asyncio
import base64
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.baselines import run_direct_vlm, run_visual_only, run_retrieval_only
from src.core.orchestrator import run_pipeline
from src.core.types import EventType
from src.data.chexpert_loader import CheXpertLoader
from src.llm.client import LLMClient, ModelConfig, set_shared_client

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

PIPELINE_FN: dict[str, Callable] = {
    "direct": run_direct_vlm,
    "visual_only": run_visual_only,
    "retrieval_only": run_retrieval_only,
    "full": run_pipeline,
}


# ---------------------------------------------------------------------------
# Event collector (mirrors run_trace.py's TraceCollector, standalone copy)
# ---------------------------------------------------------------------------

class TraceCollector:
    def __init__(self):
        self.events: list[dict] = []
        self.agent_traces: list[dict] = []
        self.current_agent: str | None = None
        self.current_agent_events: list[dict] = []
        self.current_iteration: int = 0
        self.result: dict | None = None

    def record(self, event) -> None:
        ts = time.time()
        entry = {"timestamp": ts, "type": event.type.value, "data": event.data}
        self.events.append(entry)

        if event.type == EventType.PIPELINE_UPDATE:
            self.current_iteration = event.data.get("iteration", 0)
        elif event.type == EventType.AGENT_START:
            self.current_agent = event.data.get("agent", "unknown")
            self.current_agent_events = []
        elif event.type == EventType.AGENT_END:
            agent_start_ts = self.current_agent_events[0]["timestamp"] if self.current_agent_events else ts
            self.agent_traces.append({
                "agent": self.current_agent,
                "iteration": self.current_iteration,
                "events": self.current_agent_events,
                "event_count": len(self.current_agent_events),
                "elapsed_seconds": round(ts - agent_start_ts, 2),
            })
            self.current_agent = None
            self.current_agent_events = []
        elif event.type == EventType.DONE and "session_id" in event.data:
            self.result = event.data

        if self.current_agent is not None:
            self.current_agent_events.append(entry)


def _serialize_event_data(data: dict) -> dict:
    out = {}
    for k, v in data.items():
        if isinstance(v, str) and len(v) > 5000:
            out[k] = v[:5000] + f"... [truncated, {len(v)} chars]"
        elif isinstance(v, dict):
            out[k] = _serialize_event_data(v)
        elif isinstance(v, list):
            out[k] = v[:50] if len(v) > 50 else v
        else:
            out[k] = v
    return out


# ---------------------------------------------------------------------------
# Single sample runner
# ---------------------------------------------------------------------------

async def run_one_sample(
    index: int,
    loader: CheXpertLoader,
    settings: dict,
    llm_client: LLMClient,
    output_dir: Path,
    pipeline_fn: Callable,
    mode: str,
) -> dict:
    sample = loader[index]
    exam_info = loader.get_exam_info(index)
    ground_truth = sample["text"]

    logger.info("=" * 60)
    logger.info("[%s] SAMPLE %d", mode.upper(), index)
    logger.info("Ground truth: %s", ground_truth[:150])
    logger.info("=" * 60)

    collector = TraceCollector()
    t0 = time.time()

    async for event in pipeline_fn(
        image=sample["image"],
        exam_info=exam_info,
        settings=settings,
        llm_client=llm_client,
    ):
        collector.record(event)

        if event.type == EventType.AGENT_START:
            logger.info("[AGENT START] %s (iter %d)",
                        event.data.get("agent"), event.data.get("iteration"))
        elif event.type == EventType.AGENT_END:
            logger.info("[AGENT END] %s", event.data.get("agent"))
        elif event.type == EventType.STATUS:
            logger.info("[STATUS] %s", event.data.get("message", ""))

    elapsed = time.time() - t0

    # --- Save outputs ---
    sample_dir = output_dir / f"sample_{index}"
    sample_dir.mkdir(parents=True, exist_ok=True)

    # 1. Full event timeline
    timeline = [
        {"timestamp": e["timestamp"], "type": e["type"],
         "data": _serialize_event_data(e["data"])}
        for e in collector.events
    ]
    with open(sample_dir / "event_timeline.json", "w") as f:
        json.dump(timeline, f, indent=2, ensure_ascii=False)

    # 2. Summary (what evaluate.py reads)
    summary: dict = {
        "sample_index": index,
        "mode": mode,
        "ground_truth": ground_truth,
        "elapsed_seconds": round(elapsed, 1),
        "agent_runs": len(collector.agent_traces),
        "total_events": len(collector.events),
    }
    if collector.result:
        summary["iterations"] = collector.result.get("iterations")
        summary["review_passed"] = collector.result.get("review_passed")
        summary["image_evidence_count"] = collector.result.get("image_evidence_count", 0)
        summary["retrieved_evidence_count"] = collector.result.get("retrieved_evidence_count", 0)
        summary["natural_language_report"] = collector.result.get("natural_language_report", "")

    with open(sample_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # 3. Full result (no base64 images for direct/simple baselines)
    if collector.result:
        result_to_save = {k: v for k, v in collector.result.items() if k != "images"}

        # Save derived images if present (full pipeline mode)
        images_data = collector.result.get("images", {})
        if images_data:
            images_dir = sample_dir / "images"
            images_dir.mkdir(exist_ok=True)
            for img_id, img_data in images_data.items():
                b64 = img_data.get("base64", "")
                if b64:
                    (images_dir / f"{img_id}.png").write_bytes(base64.b64decode(b64))
            image_index = {
                img_id: {k: v for k, v in img_data.items() if k != "base64"}
                for img_id, img_data in images_data.items()
            }
            with open(sample_dir / "image_index.json", "w") as f:
                json.dump(image_index, f, indent=2, ensure_ascii=False)

        with open(sample_dir / "result.json", "w") as f:
            json.dump(result_to_save, f, indent=2, ensure_ascii=False)

    logger.info(
        "[%s] Sample %d done in %.1fs — %d agent run(s), review=%s",
        mode.upper(), index, elapsed, len(collector.agent_traces),
        summary.get("review_passed", "N/A"),
    )
    logger.info("  → %s", sample_dir)
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    parser = argparse.ArgumentParser(
        description="RadClaw ablation study runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mode",
        choices=list(PIPELINE_FN.keys()),
        default="direct",
        help="Ablation mode (default: direct)",
    )
    parser.add_argument(
        "--indices", type=str, default=None,
        help="Comma-separated sample indices (default: all samples)",
    )
    parser.add_argument(
        "--split", type=str, default="findings_section",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: output/ablation_{mode})",
    )
    parser.add_argument(
        "--settings", type=str, default="config/settings.yaml",
    )
    parser.add_argument(
        "--dataset", type=str, default="chexpert",
        choices=[
            "chexpert",
            "chexpert_plus",
            "iu_xray",
            "mimic",
            "mimic_cxr",
            "openi",
            "ct_rate",
            "radgenome",
            "radgenome_chestct",
            "merlin",
            "abdomenatlas",
            "abdomenatlas3",
            "abdomenatlas3_mini",
            "mr_rate",
            "mrrate",
            "radgenome_brain_mri",
            "radgenome_brain",
        ],
        help="Dataset to use (default: chexpert)",
    )
    args = parser.parse_args()

    # If --indices not specified, run all samples
    if args.indices is None:
        indices = None  # resolved after loader is created
    else:
        indices = [int(x.strip()) for x in args.indices.split(",")]
    output_dir = Path(args.output_dir or f"output/ablation_{args.mode}")

    import yaml
    with open(args.settings) as f:
        settings = yaml.safe_load(f) or {}

    config = ModelConfig.from_settings(args.settings)
    llm_client = LLMClient(config)
    set_shared_client(llm_client)

    data_cfg = settings.get("data", {})

    if args.dataset in {"chexpert", "chexpert_plus"}:
        loader = CheXpertLoader(
            data_dir=data_cfg.get(
                "chexpert_path",
                "/cpfs01/user/zhaizhai.cxx/hf/data/Doorad/CheXpert-plus-RRG",
            ),
            split_name=args.split,
        )
    elif args.dataset == "iu_xray":
        from src.data.iu_xray_loader import IUXrayLoader
        loader = IUXrayLoader(
            data_dir=data_cfg.get(
                "iu_xray_path",
                "/cpfs01/user/zhaizhai.cxx/hf/data/X-iZhang/IU-Xray-RRG",
            ),
            split_name=args.split,
        )
    elif args.dataset in {"mimic", "mimic_cxr"}:
        from src.data.mimic_cxr_loader import MIMICCXRLoader
        loader = MIMICCXRLoader(
            data_dir=data_cfg.get(
                "mimic_cxr_path",
                "/cpfs01/user/zhaizhai.cxx/hf/data/Yamini-1628/MIMIC-CXR-RRG",
            ),
            split_name=args.split,
        )
    elif args.dataset == "openi":
        from src.data.openi_loader import OpenILoader
        loader = OpenILoader(
            data_dir=data_cfg.get(
                "openi_path",
                "/cpfs01/user/zhaizhai.cxx/hf/data/ykumards/open-i",
            ),
            split="test",
        )
    elif args.dataset == "ct_rate":
        from src.data.ct_rate_loader import CTRateLoader
        loader = CTRateLoader(
            data_dir=data_cfg.get(
                "ct_rate_path",
                "/mnt/yulan_pretrain_131/clinical_project/hf/ibrahimhamamci/CT-RATE",
            ),
        )
    elif args.dataset in {"radgenome", "radgenome_chestct"}:
        from src.data.radgenome_loader import RadGenomeChestCTLoader
        loader = RadGenomeChestCTLoader(
            data_dir=data_cfg.get(
                "radgenome_path",
                "/mnt/yulan_pretrain_131/clinical_project/hf/RadGenome/RadGenome-ChestCT",
            ),
        )
    elif args.dataset == "merlin":
        from src.data.merlin_loader import MerlinLoader
        loader = MerlinLoader(
            data_dir=data_cfg.get(
                "merlin_path",
                "/mnt/yulan_pretrain_131/clinical_project/hf/merlin",
            ),
        )
    elif args.dataset in {"abdomenatlas", "abdomenatlas3", "abdomenatlas3_mini"}:
        from src.data.abdomenatlas_loader import AbdomenAtlasLoader
        loader = AbdomenAtlasLoader(
            data_dir=data_cfg.get(
                "abdomenatlas_path",
                "/mnt/yulan_pretrain_131/clinical_project/hf/AbdomenAtlas/AbdomenAtlas3.0Mini",
            ),
        )
    elif args.dataset in {"mr_rate", "mrrate"}:
        from src.data.mr_rate_loader import MRRateLoader
        loader = MRRateLoader(
            data_dir=data_cfg.get(
                "mr_rate_path",
                "/mnt/yulan_pretrain_131/clinical_project/hf/Forithmus/MR-RATE",
            ),
        )
    elif args.dataset in {"radgenome_brain_mri", "radgenome_brain"}:
        from src.data.radgenome_brain_mri_loader import RadGenomeBrainMRILoader
        loader = RadGenomeBrainMRILoader(
            data_dir=data_cfg.get(
                "radgenome_brain_mri_path",
                "/mnt/yulan_pretrain_131/clinical_project/hf/JiayuLei/RadGenome-Brain_MRI",
            ),
        )
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    # Default to all samples if --indices not given
    if indices is None:
        indices = list(range(len(loader)))

    output_dir.mkdir(parents=True, exist_ok=True)
    pipeline_fn = PIPELINE_FN[args.mode]

    logger.info("Mode: %s | Samples: %s | Output: %s", args.mode, indices, output_dir)

    all_summaries = []
    total_t0 = time.time()
    skipped = 0

    for idx in indices:
        # Resume: skip if summary.json already exists
        summary_path = output_dir / f"sample_{idx}" / "summary.json"
        if summary_path.exists():
            skipped += 1
            logger.info("Sample %d: already done, skipping", idx)
            with open(summary_path) as f:
                all_summaries.append(json.load(f))
            continue

        try:
            s = await run_one_sample(
                idx, loader, settings, llm_client, output_dir, pipeline_fn, args.mode,
            )
            all_summaries.append(s)
        except Exception:
            logger.exception("Sample %d failed", idx)
            all_summaries.append({"sample_index": idx, "mode": args.mode, "error": True})

    if skipped:
        logger.info("Resumed: skipped %d already-completed samples", skipped)

    total_elapsed = time.time() - total_t0

    batch_summary = {
        "run_time": datetime.now().isoformat(),
        "mode": args.mode,
        "indices": indices,
        "total_elapsed_seconds": round(total_elapsed, 1),
        "samples": all_summaries,
    }
    with open(output_dir / "batch_summary.json", "w") as f:
        json.dump(batch_summary, f, indent=2, ensure_ascii=False)

    # Print table
    print("\n" + "=" * 70)
    print(f"ABLATION [{args.mode.upper()}] — BATCH SUMMARY")
    print("=" * 70)
    print(f"{'Index':>6} {'Time(s)':>8} {'Agents':>7} {'ImgEv':>6} {'RetEv':>6} {'Pass':>6}")
    print("-" * 70)
    for s in all_summaries:
        if s.get("error"):
            print(f"{s['sample_index']:>6} {'FAILED':>8}")
            continue
        print(
            f"{s['sample_index']:>6} "
            f"{s.get('elapsed_seconds', 0):>8.1f} "
            f"{s.get('agent_runs', 0):>7} "
            f"{s.get('image_evidence_count', 0):>6} "
            f"{s.get('retrieved_evidence_count', 0):>6} "
            f"{str(s.get('review_passed', 'N/A')):>6}"
        )
    print("-" * 70)
    print(f"Total: {total_elapsed:.1f}s  |  Output: {output_dir}")
    print()
    print("To evaluate:")
    print(f"  python scripts/evaluate.py --predictions {output_dir} --metrics all")


if __name__ == "__main__":
    asyncio.run(main())
