#!/usr/bin/env python3
"""Process a single CheXpert sample through the RadClaw pipeline."""

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


async def main():
    parser = argparse.ArgumentParser(description="Run RadClaw on a single CheXpert sample")
    parser.add_argument("--index", type=int, default=0, help="Sample index")
    parser.add_argument("--split", type=str, default="findings_section", help="Dataset split")
    parser.add_argument("--output-dir", type=str, default="./output", help="Output directory")
    parser.add_argument("--settings", type=str, default="config/settings.yaml", help="Settings file")
    args = parser.parse_args()

    import yaml
    settings_path = Path(args.settings)
    settings = {}
    if settings_path.exists():
        with open(settings_path) as f:
            settings = yaml.safe_load(f) or {}

    config = ModelConfig.from_settings(args.settings)
    llm_client = LLMClient(config)
    set_shared_client(llm_client)

    data_cfg = settings.get("data", {})
    loader = CheXpertLoader(
        data_dir=data_cfg.get("chexpert_path", "/cpfs01/user/zhaizhai.cxx/hf/data/Doorad/CheXpert-plus-RRG"),
        split_name=args.split,
    )

    sample = loader[args.index]
    exam_info = loader.get_exam_info(args.index)

    logger.info(f"Processing sample {args.index} from {args.split}")
    logger.info(f"Ground truth: {sample['text'][:200]}...")

    result = None
    async for event in run_pipeline(
        image=sample["image"],
        exam_info=exam_info,
        settings=settings,
        llm_client=llm_client,
    ):
        if event.type == EventType.STATUS:
            logger.info(f"[STATUS] {event.data.get('message', '')}")
        elif event.type == EventType.AGENT_START:
            logger.info(f"[AGENT START] {event.data.get('agent', '')}")
        elif event.type == EventType.AGENT_END:
            logger.info(f"[AGENT END] {event.data.get('agent', '')}")
        elif event.type == EventType.DONE:
            result = event.data

    if result:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / f"sample_{args.index}_{args.split}.json"
        with open(output_file, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        logger.info(f"Results saved to {output_file}")

        if "natural_language_report" in result:
            print("\n" + "=" * 60)
            print("GENERATED REPORT:")
            print("=" * 60)
            print(result["natural_language_report"])

        print("\n" + "=" * 60)
        print("GROUND TRUTH:")
        print("=" * 60)
        print(sample["text"])

        print(f"\nPipeline completed in {result.get('iterations', '?')} iteration(s)")
        print(f"Review passed: {result.get('review_passed', '?')}")
        print(f"Image evidence: {result.get('image_evidence_count', '?')}")
        print(f"Retrieved evidence: {result.get('retrieved_evidence_count', '?')}")
    else:
        logger.error("Pipeline produced no result")


if __name__ == "__main__":
    asyncio.run(main())