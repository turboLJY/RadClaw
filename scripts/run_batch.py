#!/usr/bin/env python3
"""Batch process CheXpert samples through the RadClaw pipeline."""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.orchestrator import run_pipeline
from src.core.types import EventType
from src.data.chexpert_loader import CheXpertLoader
from src.llm.client import LLMClient, ModelConfig, set_shared_client

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


async def process_one(loader, index, settings, llm_client, output_dir):
    sample = loader[index]
    exam_info = loader.get_exam_info(index)

    result = None
    async for event in run_pipeline(
        image=sample["image"],
        exam_info=exam_info,
        settings=settings,
        llm_client=llm_client,
    ):
        if event.type == EventType.DONE:
            result = event.data

    if result:
        result["ground_truth"] = sample["text"]
        output_file = output_dir / f"sample_{index}.json"
        with open(output_file, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        logger.info(f"Sample {index}: done ({result.get('iterations', '?')} iterations)")
        return True

    logger.error(f"Sample {index}: failed")
    return False


async def main():
    parser = argparse.ArgumentParser(description="Batch process CheXpert samples")
    parser.add_argument("--split", type=str, default="findings_section")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=-1)
    parser.add_argument("--output-dir", type=str, default="./output/batch")
    parser.add_argument("--settings", type=str, default="config/settings.yaml")
    args = parser.parse_args()

    import yaml
    settings = {}
    if Path(args.settings).exists():
        with open(args.settings) as f:
            settings = yaml.safe_load(f) or {}

    config = ModelConfig.from_settings(args.settings)
    llm_client = LLMClient(config)
    set_shared_client(llm_client)

    data_cfg = settings.get("data", {})
    loader = CheXpertLoader(
        data_dir=data_cfg.get("chexpert_path", ""),
        split_name=args.split,
    )

    end = args.end if args.end > 0 else len(loader)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    success = 0
    total = end - args.start

    for idx in range(args.start, end):
        logger.info(f"Processing sample {idx}/{end-1}...")
        ok = await process_one(loader, idx, settings, llm_client, output_dir)
        if ok:
            success += 1

    logger.info(f"Batch complete: {success}/{total} successful")

    summary = {"split": args.split, "start": args.start, "end": end, "success": success, "total": total}
    with open(output_dir / "batch_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    asyncio.run(main())