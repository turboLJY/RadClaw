#!/usr/bin/env python3
"""Add relevant evidence content spans to translated annotation results.

The script scans annotation_samples/*/chinese_result.json, visits
structured_report.findings and structured_report.impressions, and asks a
DeepSeek OpenAI-compatible chat-completion model to select the evidence content
single substring most relevant to each report item text. For guideline/pubmed retrieved
evidence, it writes evidence["relevant_segment"] = [start_idx, end_idx].
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections.abc import Iterator, MutableMapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - exercised only when dependency is absent
    OpenAI = None  # type: ignore[assignment]


DEFAULT_ANNOTATION_DIR = Path("annotation_samples")
DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_API_KEY = "sk-62be91fcd39a4752aad8c0fe13b22967"
DEFAULT_MODEL = "deepseek-v4-pro"
TARGET_SECTIONS = ("findings", "impressions")
DEFAULT_SOURCE_TYPES = ("guideline", "pubmed")


@dataclass(frozen=True)
class ResultFile:
    sample_dir: Path
    result_path: Path


@dataclass
class EvidenceRef:
    sample_name: str
    path: str
    evidence: MutableMapping[str, Any]
    report_text: str
    source_type: str
    content: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Use DeepSeek to add relevant_segment character span to each guideline/pubmed "
            "retrieved evidence in annotation_samples/*/chinese_result.json."
        )
    )
    parser.add_argument(
        "--annotation-dir",
        type=Path,
        default=DEFAULT_ANNOTATION_DIR,
        help="Directory containing sample subfolders.",
    )
    parser.add_argument(
        "--sample",
        action="append",
        default=[],
        help="Process only the named sample directory. May be passed multiple times.",
    )
    parser.add_argument(
        "--model",
        choices=("deepseek-v4-flash", "deepseek-v4-pro"),
        default=os.getenv("DEEPSEEK_MODEL", DEFAULT_MODEL),
        help="DeepSeek model used for evidence segment selection.",
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("DEEPSEEK_BASE_URL", DEFAULT_BASE_URL),
        help="DeepSeek OpenAI-compatible API base URL.",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY") or DEFAULT_API_KEY,
        help="DeepSeek API key. Defaults to DEEPSEEK_API_KEY, then OPENAI_API_KEY.",
    )
    parser.add_argument(
        "--source-type",
        action="append",
        default=[],
        help=(
            "Evidence source_type to process. Defaults to guideline and pubmed. "
            "May be passed multiple times."
        ),
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Maximum output tokens for each segment-selection request.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature. Removed automatically for DeepSeek v4 thinking mode.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=5,
        help="API retry count per evidence item.",
    )
    parser.add_argument(
        "--retry-sleep",
        type=float,
        default=10.0,
        help="Initial seconds to sleep between retries; doubles after each failure.",
    )
    parser.add_argument(
        "--request-sleep",
        type=float,
        default=1.0,
        help="Seconds to sleep after each successful API request.",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=("high", "max"),
        default=os.getenv("DEEPSEEK_REASONING_EFFORT", "high"),
        help="DeepSeek thinking-mode reasoning effort.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate relevant_segment when the field already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the processing plan without calling the API or writing files.",
    )
    parser.add_argument(
        "--no-json-response-format",
        action="store_true",
        help="Do not request JSON object mode from the chat-completions API.",
    )
    return parser.parse_args()


def sample_sort_key(path: Path) -> tuple[int, int | str]:
    match = re.search(r"(\d+)$", path.name)
    if match:
        return (0, int(match.group(1)))
    return (1, path.name)


def iter_result_files(annotation_dir: Path, sample_names: Sequence[str]) -> Iterator[ResultFile]:
    if not annotation_dir.exists():
        raise FileNotFoundError(f"Annotation directory not found: {annotation_dir}")

    if sample_names:
        sample_dirs = [annotation_dir / sample_name for sample_name in sample_names]
    else:
        sample_dirs = sorted(
            (path for path in annotation_dir.iterdir() if path.is_dir()),
            key=sample_sort_key,
        )

    for sample_dir in sample_dirs:
        if not sample_dir.is_dir():
            print(f"warning: skipped missing sample directory: {sample_dir}", file=sys.stderr)
            continue
        result_path = sample_dir / "chinese_result.json"
        if result_path.exists():
            yield ResultFile(sample_dir=sample_dir, result_path=result_path)
        else:
            print(f"warning: skipped sample without chinese_result.json: {sample_dir}", file=sys.stderr)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} root must be a JSON object")
    return data


def write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    tmp_path = path.with_name(f".{path.name}.tmp")
    tmp_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    tmp_path.replace(path)


def normalize_source_types(source_types: Sequence[str]) -> set[str]:
    values = source_types or DEFAULT_SOURCE_TYPES
    return {value.strip().lower() for value in values if value.strip()}


def is_valid_existing_segment(value: Any, content: str) -> bool:
    if not isinstance(value, list) or len(value) != 2:
        return False
    start_idx, end_idx = value
    return (
        isinstance(start_idx, int)
        and isinstance(end_idx, int)
        and 0 <= start_idx < end_idx <= len(content)
    )


def collect_evidence_refs(
    result_data: dict[str, Any],
    sample_name: str,
    source_types: set[str],
    overwrite: bool,
) -> list[EvidenceRef]:
    refs: list[EvidenceRef] = []
    structured_report = result_data.get("structured_report")
    if not isinstance(structured_report, dict):
        return refs

    for section_name in TARGET_SECTIONS:
        section = structured_report.get(section_name)
        if not isinstance(section, list):
            continue

        for item_index, item in enumerate(section):
            if not isinstance(item, dict):
                continue
            report_text = item.get("text")
            if not isinstance(report_text, str) or not report_text.strip():
                continue

            support = item.get("support")
            if not isinstance(support, dict):
                continue
            retrieved_evidence = support.get("retrieved_evidence")
            if not isinstance(retrieved_evidence, list):
                continue

            for evidence_index, evidence in enumerate(retrieved_evidence):
                if not isinstance(evidence, dict):
                    continue
                source_type = evidence.get("source_type")
                if not isinstance(source_type, str) or source_type.strip().lower() not in source_types:
                    continue
                content = evidence.get("content")
                if not isinstance(content, str) or not content.strip():
                    continue
                if (
                    not overwrite
                    and is_valid_existing_segment(evidence.get("relevant_segment"), content)
                ):
                    continue

                path = (
                    f"structured_report.{section_name}[{item_index}]"
                    f".support.retrieved_evidence[{evidence_index}]"
                )
                refs.append(
                    EvidenceRef(
                        sample_name=sample_name,
                        path=path,
                        evidence=evidence,
                        report_text=report_text.strip(),
                        source_type=source_type.strip().lower(),
                        content=content,
                    )
                )
    return refs


def build_system_prompt() -> str:
    return (
        "你是严谨的医学证据定位助手。你的任务是在一段检索证据 content 中，找出与影像报告条目 "
        "report_text 最相关、最适合高亮给人工标注者看的一个连续原文片段。\n"
        "要求：\n"
        "1. 只能选择一个片段；该片段必须是 content 中的连续子串，segment_text 必须逐字复制原文，不要改写、翻译或概括。\n"
        "2. 优先选择直接支持或解释 report_text 的单个最小完整片段，通常为 1-3 个完整句子或一个连续段落。\n"
        "3. 如果没有完全直接支持的句子，选择语义最接近、最有助于判断 report_text 的原文片段。\n"
        "4. 不要选择参考文献列表、题录、无关背景或过长上下文，除非它们本身就是最相关证据。\n"
        "5. 输出只能是一个 JSON object，不要 Markdown，不要解释。\n"
        "6. JSON schema 必须是 {\"segment_text\": \"...\", \"start_idx\": 0, \"end_idx\": 0}。"
        "start_idx/end_idx 是 content 的 0-based 字符索引，end_idx 为开区间；如果不能精确计数，"
        "也必须保证 segment_text 是从 content 原文逐字复制的连续子串。"
    )


def build_user_prompt(ref: EvidenceRef) -> str:
    payload = {
        "report_text": ref.report_text,
        "source_type": ref.source_type,
        "content": ref.content,
    }
    return "请定位最相关的一段证据片段。输入 JSON：\n" + json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
    )


def create_openai_client(api_key: str, base_url: str) -> Any:
    if OpenAI is None:
        raise RuntimeError("The openai package is required. Install project dependencies first.")
    if not api_key:
        raise RuntimeError("DeepSeek API key is empty. Set DEEPSEEK_API_KEY or pass --api-key.")
    return OpenAI(api_key=api_key, base_url=base_url)


def enable_deepseek_thinking(request: dict[str, Any], args: argparse.Namespace) -> None:
    request.pop("temperature", None)
    request["reasoning_effort"] = args.reasoning_effort
    request["extra_body"] = {"thinking": {"type": "enabled"}}


def strip_json_fence(text: str) -> str:
    text = text.strip()
    if not text.startswith("```"):
        return text
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def parse_json_object(content: str) -> dict[str, Any]:
    text = strip_json_fence(content)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("LLM response root must be a JSON object")
    return parsed


def nearest_occurrence(content: str, segment_text: str, preferred_start: int | None) -> tuple[int, int] | None:
    if not segment_text:
        return None

    starts: list[int] = []
    start = content.find(segment_text)
    while start != -1:
        starts.append(start)
        start = content.find(segment_text, start + 1)

    if not starts:
        return None
    if preferred_start is None:
        best_start = starts[0]
    else:
        best_start = min(starts, key=lambda value: abs(value - preferred_start))
    return best_start, best_start + len(segment_text)


def response_to_segment(content: str, response_content: str) -> tuple[int, int]:
    parsed = parse_json_object(response_content)
    raw_start = parsed.get("start_idx")
    raw_end = parsed.get("end_idx")
    preferred_start = raw_start if isinstance(raw_start, int) and 0 <= raw_start < len(content) else None

    segment_text = parsed.get("segment_text")
    if isinstance(segment_text, str):
        located = nearest_occurrence(content, segment_text, preferred_start)
        if located is not None:
            return located

    if (
        isinstance(raw_start, int)
        and isinstance(raw_end, int)
        and 0 <= raw_start < raw_end <= len(content)
        and content[raw_start:raw_end].strip()
    ):
        return raw_start, raw_end

    raise ValueError(
        "LLM response did not contain a segment_text substring or valid start_idx/end_idx"
    )


def select_relevant_segment(client: Any, args: argparse.Namespace, ref: EvidenceRef) -> tuple[int, int]:
    request: dict[str, Any] = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": build_system_prompt()},
            {"role": "user", "content": build_user_prompt(ref)},
        ],
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
    }
    if not args.no_json_response_format:
        request["response_format"] = {"type": "json_object"}
    enable_deepseek_thinking(request, args)

    last_error: Exception | None = None
    for attempt in range(1, args.retries + 1):
        try:
            response = client.chat.completions.create(**request)
            content = response.choices[0].message.content
            if not content:
                raise ValueError("empty response content")
            return response_to_segment(ref.content, content)
        except Exception as exc:  # noqa: BLE001 - keep retries around provider/client errors
            last_error = exc
            if attempt >= args.retries:
                break
            time.sleep(args.retry_sleep * (2 ** (attempt - 1)))

    assert last_error is not None
    raise last_error


def collect_all_work(
    result_files: Sequence[ResultFile],
    source_types: set[str],
    overwrite: bool,
) -> list[tuple[ResultFile, dict[str, Any], list[EvidenceRef]]]:
    work: list[tuple[ResultFile, dict[str, Any], list[EvidenceRef]]] = []
    for result_file in result_files:
        data = load_json(result_file.result_path)
        refs = collect_evidence_refs(
            data,
            result_file.sample_dir.name,
            source_types,
            overwrite,
        )
        work.append((result_file, data, refs))
    return work


def print_dry_run(work: Sequence[tuple[ResultFile, dict[str, Any], list[EvidenceRef]]]) -> None:
    total_refs = sum(len(refs) for _, _, refs in work)
    files_with_work = sum(1 for _, _, refs in work if refs)
    print(f"Chinese result files found: {len(work)}")
    print(f"Files to update: {files_with_work}")
    print(f"Evidence items to process: {total_refs}")
    for result_file, _, refs in work:
        if refs:
            print(f"- {result_file.sample_dir.name}: {len(refs)} evidence items")


def run() -> int:
    args = parse_args()
    source_types = normalize_source_types(args.source_type)
    result_files = list(iter_result_files(args.annotation_dir, args.sample))
    work = collect_all_work(result_files, source_types, args.overwrite)

    if args.dry_run:
        print_dry_run(work)
        return 0

    client = create_openai_client(args.api_key, args.base_url)
    segment_cache: dict[tuple[str, str], tuple[int, int]] = {}

    for result_file, data, refs in work:
        if not refs:
            print(f"Skipped {result_file.result_path}")
            continue

        print(f"{result_file.sample_dir.name}: selecting {len(refs)} relevant evidence segments")
        for index, ref in enumerate(refs, start=1):
            cache_key = (ref.report_text, ref.content)
            if cache_key not in segment_cache:
                segment_cache[cache_key] = select_relevant_segment(client, args, ref)
                if args.request_sleep > 0:
                    time.sleep(args.request_sleep)
            start_idx, end_idx = segment_cache[cache_key]
            ref.evidence["relevant_segment"] = [start_idx, end_idx]
            print(f"  {index}/{len(refs)} {ref.path}: [{start_idx}, {end_idx}]")

        write_json_atomic(result_file.result_path, data)
        print(f"Wrote {result_file.result_path}")

    return 0


def main() -> None:
    try:
        raise SystemExit(run())
    except KeyboardInterrupt:
        raise SystemExit(130) from None
    except Exception as exc:  # noqa: BLE001 - command-line script should surface a concise error
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
