#!/usr/bin/env python3
"""Run multiple CheXpert samples and save full agent trajectories."""

import argparse
import asyncio
import base64
import json
import logging
import sys
import time
from datetime import datetime
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


class TraceCollector:
    """Collect all pipeline events and organize by agent."""

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

    def summary(self) -> dict:
        agent_summaries = []
        for trace in self.agent_traces:
            tool_calls = [
                e for e in trace["events"] if e["type"] == "tool_use"
            ]
            tool_results = [
                e for e in trace["events"] if e["type"] == "tool_result"
            ]
            text_deltas = [
                e for e in trace["events"] if e["type"] == "text_delta"
            ]
            full_text = "".join(
                e["data"].get("text", "") for e in text_deltas
            )
            agent_summaries.append({
                "agent": trace["agent"],
                "iteration": trace["iteration"],
                "total_events": trace["event_count"],
                "elapsed_seconds": trace.get("elapsed_seconds", 0),
                "tool_calls": len(tool_calls),
                "tool_results": len(tool_results),
                "text_chunks": len(text_deltas),
                "output_text_length": len(full_text),
                "tool_names_called": [
                    tc["data"].get("tool_name", "") for tc in tool_calls
                ],
            })
        return {
            "total_events": len(self.events),
            "agent_runs": len(self.agent_traces),
            "agents": agent_summaries,
        }


def _serialize_event_data(data: dict) -> dict:
    """Make event data JSON-serializable, truncating large fields."""
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


async def run_one_sample(
    index: int,
    loader: CheXpertLoader,
    settings: dict,
    llm_client: LLMClient,
    output_dir: Path,
) -> dict:
    """Run one sample and save full trace."""
    sample = loader[index]
    exam_info = loader.get_exam_info(index)
    ground_truth = sample["text"]

    logger.info("=" * 60)
    logger.info("SAMPLE %d", index)
    logger.info("Ground truth: %s", ground_truth[:150])
    logger.info("=" * 60)

    collector = TraceCollector()
    t0 = time.time()

    async for event in run_pipeline(
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
    timeline = []
    for e in collector.events:
        timeline.append({
            "timestamp": e["timestamp"],
            "type": e["type"],
            "data": _serialize_event_data(e["data"]),
        })
    with open(sample_dir / "event_timeline.json", "w") as f:
        json.dump(timeline, f, indent=2, ensure_ascii=False)

    # 2. Per-agent traces (full detail)
    for i, trace in enumerate(collector.agent_traces):
        agent_name = trace["agent"]
        iteration = trace["iteration"]
        fname = f"trace_{iteration}_{agent_name}.json"

        tool_calls_detail = []
        text_parts = []
        for e in trace["events"]:
            if e["type"] == "tool_use":
                tool_calls_detail.append({
                    "timestamp": e["timestamp"],
                    "tool_name": e["data"].get("tool_name", ""),
                    "tool_use_id": e["data"].get("tool_use_id", ""),
                    "tool_input": e["data"].get("tool_input", {}),
                })
            elif e["type"] == "tool_result":
                tool_calls_detail.append({
                    "timestamp": e["timestamp"],
                    "type": "result",
                    "tool_name": e["data"].get("tool_name", ""),
                    "tool_use_id": e["data"].get("tool_use_id", ""),
                    "result": e["data"].get("result", ""),
                    "is_error": e["data"].get("is_error", False),
                })
            elif e["type"] == "text_delta":
                text_parts.append(e["data"].get("text", ""))

        agent_trace = {
            "agent": agent_name,
            "iteration": iteration,
            "total_events": trace["event_count"],
            "elapsed_seconds": trace.get("elapsed_seconds", 0),
            "output_text": "".join(text_parts),
            "tool_interactions": tool_calls_detail,
            "all_events": [
                {"timestamp": e["timestamp"], "type": e["type"],
                 "data": _serialize_event_data(e["data"])}
                for e in trace["events"]
            ],
        }
        with open(sample_dir / fname, "w") as f:
            json.dump(agent_trace, f, indent=2, ensure_ascii=False)

    # 3. Summary
    summary = collector.summary()
    summary["sample_index"] = index
    summary["ground_truth"] = ground_truth
    summary["elapsed_seconds"] = round(elapsed, 1)
    if collector.result:
        summary["iterations"] = collector.result.get("iterations")
        summary["review_passed"] = collector.result.get("review_passed")
        summary["image_evidence_count"] = collector.result.get("image_evidence_count")
        summary["retrieved_evidence_count"] = collector.result.get("retrieved_evidence_count")
        summary["natural_language_report"] = collector.result.get("natural_language_report", "")

    # Save each agent's full conversational text for debugging
    agent_texts = {}
    for trace in collector.agent_traces:
        text_parts = [e["data"].get("text", "") for e in trace["events"] if e["type"] == "text_delta"]
        agent_key = f"{trace['agent']}_iter{trace['iteration']}_agent_text"
        agent_texts[agent_key] = "".join(text_parts)
    summary["agent_texts"] = agent_texts

    with open(sample_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # 4. Final pipeline result — extract images first, save separately
    if collector.result:
        images_data = collector.result.pop("images", {})
        if images_data:
            images_dir = sample_dir / "images"
            images_dir.mkdir(exist_ok=True)
            for img_id, img_data in images_data.items():
                b64 = img_data.get("base64", "")
                if b64:
                    img_bytes = base64.b64decode(b64)
                    (images_dir / f"{img_id}.png").write_bytes(img_bytes)
            # Save image index (metadata only, no base64)
            image_index = {
                img_id: {k: v for k, v in img_data.items() if k != "base64"}
                for img_id, img_data in images_data.items()
            }
            with open(sample_dir / "image_index.json", "w") as f:
                json.dump(image_index, f, indent=2, ensure_ascii=False)
            logger.info("Saved %d images to %s/images/", len(images_data), sample_dir)

        with open(sample_dir / "result.json", "w") as f:
            json.dump(collector.result, f, indent=2, ensure_ascii=False)

    logger.info("Sample %d done in %.1fs — saved to %s", index, elapsed, sample_dir)
    logger.info("  Agents: %d runs, Events: %d, Review: %s",
                summary["agent_runs"], summary["total_events"],
                summary.get("review_passed", "?"))

    return summary


async def main():
    parser = argparse.ArgumentParser(description="Run RadClaw on multiple samples with full trace")
    parser.add_argument("--indices", type=str, default="1,2,3",
                        help="Comma-separated sample indices (e.g. '1,2,3')")
    parser.add_argument("--split", type=str, default="findings_section")
    parser.add_argument("--output-dir", type=str, default="./output/traces")
    parser.add_argument("--settings", type=str, default="config/settings.yaml")
    args = parser.parse_args()

    indices = [int(x.strip()) for x in args.indices.split(",")]

    import yaml
    settings_path = Path(args.settings)
    with open(settings_path) as f:
        settings = yaml.safe_load(f) or {}

    config = ModelConfig.from_settings(args.settings)
    llm_client = LLMClient(config)
    set_shared_client(llm_client)

    data_cfg = settings.get("data", {})
    loader = CheXpertLoader(
        data_dir=data_cfg.get("chexpert_path",
                              "/cpfs01/user/zhaizhai.cxx/hf/data/Doorad/CheXpert-plus-RRG"),
        split_name=args.split,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_summaries = []
    total_t0 = time.time()

    for idx in indices:
        try:
            summary = await run_one_sample(idx, loader, settings, llm_client, output_dir)
            all_summaries.append(summary)
        except Exception:
            logger.exception("Sample %d failed", idx)
            all_summaries.append({"sample_index": idx, "error": True})

    total_elapsed = time.time() - total_t0

    # Save batch summary
    batch_summary = {
        "run_time": datetime.now().isoformat(),
        "indices": indices,
        "total_elapsed_seconds": round(total_elapsed, 1),
        "samples": all_summaries,
    }
    with open(output_dir / "batch_summary.json", "w") as f:
        json.dump(batch_summary, f, indent=2, ensure_ascii=False)

    # Print summary table
    print("\n" + "=" * 80)
    print("BATCH SUMMARY")
    print("=" * 80)
    print(f"{'Index':>6} {'Time(s)':>8} {'Agents':>7} {'Events':>7} {'ImgEv':>6} {'RetEv':>6} {'Pass':>6}")
    print("-" * 80)
    for s in all_summaries:
        if s.get("error"):
            print(f"{s['sample_index']:>6} {'FAILED':>8}")
            continue
        print(f"{s['sample_index']:>6} "
              f"{s.get('elapsed_seconds', 0):>8.1f} "
              f"{s.get('agent_runs', 0):>7} "
              f"{s.get('total_events', 0):>7} "
              f"{s.get('image_evidence_count', 0):>6} "
              f"{s.get('retrieved_evidence_count', 0):>6} "
              f"{str(s.get('review_passed', '?')):>6}")
    print("-" * 80)
    print(f"Total: {total_elapsed:.1f}s")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    asyncio.run(main())
