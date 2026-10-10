#!/usr/bin/env python3
"""Translate annotation sample reports to Chinese with DeepSeek.

The script reads each immediate child directory under annotation_samples,
translates selected text fields, and writes chinese_result.json and
chinese_summary.json beside the original files. In chinese_summary.json,
the translated reference report is stored as ground_truth_chinese while
the original ground_truth is retained.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections.abc import Iterable, Iterator, MutableMapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - exercised only when dependency is absent
    OpenAI = None  # type: ignore[assignment]


DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-v4-pro"
DEFAULT_API_KEY = "sk-62be91fcd39a4752aad8c0fe13b22967"
DEFAULT_GUIDELINE_TEXT = Path("chest_xray_writing_guideline.txt")
TARGET_RESULT_SECTIONS = ("findings", "impressions")


@dataclass
class FieldRef:
    path: str
    container: MutableMapping[str, Any]
    key: str
    text: str


@dataclass
class SamplePlan:
    sample_dir: Path
    result_refs: list[FieldRef]
    summary_refs: list[FieldRef]
    result_output: Path
    summary_output: Path
    skip_result: bool
    skip_summary: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Translate selected annotation_samples result.json and summary.json fields "
            "to Chinese using a DeepSeek OpenAI-compatible chat completion API."
        )
    )
    parser.add_argument(
        "--annotation-dir",
        type=Path,
        default=Path("annotation_samples"),
        help="Directory containing sample subfolders.",
    )
    parser.add_argument(
        "--guideline-text",
        type=Path,
        default=DEFAULT_GUIDELINE_TEXT,
        help="Chest X-ray report-writing guideline text file used as prompt reference.",
    )
    parser.add_argument(
        "--guideline-max-chars",
        type=int,
        default=0,
        help="Maximum guideline characters to include in the prompt. 0 means full text.",
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("DEEPSEEK_BASE_URL", DEFAULT_BASE_URL),
        help="DeepSeek API base URL.",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("DEEPSEEK_API_KEY", DEFAULT_API_KEY),
        help="DeepSeek API key. Defaults to DEEPSEEK_API_KEY; intentionally empty otherwise.",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("DEEPSEEK_MODEL", DEFAULT_MODEL),
        help="DeepSeek model name. Default follows the user-requested model string.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=20,
        help="Maximum number of unique text strings per API request.",
    )
    parser.add_argument(
        "--batch-max-chars",
        type=int,
        default=6000,
        help="Soft maximum source-text characters per API request batch.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for translation.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Maximum output tokens per translation request.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=5,
        help="API retry count per batch.",
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
        default=10.0,
        help="Seconds to sleep after each successful API request.",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=("high", "max"),
        default=os.getenv("DEEPSEEK_REASONING_EFFORT", "high"),
        help="DeepSeek thinking-mode reasoning effort for deepseek-v4-flash/deepseek-v4-pro.",
    )
    parser.add_argument(
        "--sample",
        action="append",
        default=[],
        help="Process only the named sample directory. May be passed multiple times.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate chinese_result.json/chinese_summary.json if they already exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect files and print the translation plan without calling the API or writing JSON.",
    )
    parser.add_argument(
        "--no-json-response-format",
        action="store_true",
        help="Do not request JSON object mode from the chat-completions API.",
    )
    return parser.parse_args()


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


def is_translatable_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def add_ref(
    refs: list[FieldRef],
    path: str,
    container: MutableMapping[str, Any],
    key: str,
) -> None:
    value = container.get(key)
    if is_translatable_text(value):
        refs.append(FieldRef(path=path, container=container, key=key, text=value))


def add_translated_copy_ref(
    refs: list[FieldRef],
    path: str,
    container: MutableMapping[str, Any],
    source_key: str,
    target_key: str,
) -> None:
    value = container.get(source_key)
    if is_translatable_text(value):
        refs.append(FieldRef(path=path, container=container, key=target_key, text=value))


def collect_result_refs(result_data: dict[str, Any]) -> list[FieldRef]:
    refs: list[FieldRef] = []
    structured_report = result_data.get("structured_report")
    if not isinstance(structured_report, dict):
        return refs

    for section_name in TARGET_RESULT_SECTIONS:
        section = structured_report.get(section_name)
        if not isinstance(section, list):
            continue
        for item_index, item in enumerate(section):
            if not isinstance(item, dict):
                continue

            item_path = f"structured_report.{section_name}[{item_index}]"
            add_ref(refs, f"{item_path}.text", item, "text")

            support = item.get("support")
            if not isinstance(support, dict):
                continue

            image_evidence = support.get("image_evidence")
            if isinstance(image_evidence, list):
                for evidence_index, evidence in enumerate(image_evidence):
                    if not isinstance(evidence, dict):
                        continue
                    add_ref(
                        refs,
                        f"{item_path}.support.image_evidence[{evidence_index}].description",
                        evidence,
                        "description",
                    )

            retrieved_evidence = support.get("retrieved_evidence")
            if isinstance(retrieved_evidence, list):
                for evidence_index, evidence in enumerate(retrieved_evidence):
                    if not isinstance(evidence, dict):
                        continue
                    add_ref(
                        refs,
                        f"{item_path}.support.retrieved_evidence[{evidence_index}].content",
                        evidence,
                        "content",
                    )
    return refs


def collect_summary_refs(summary_data: dict[str, Any]) -> list[FieldRef]:
    refs: list[FieldRef] = []
    add_translated_copy_ref(
        refs,
        "ground_truth_chinese",
        summary_data,
        "ground_truth",
        "ground_truth_chinese",
    )
    return refs


def sample_sort_key(path: Path) -> tuple[int, int | str]:
    match = re.search(r"(\d+)$", path.name)
    if match:
        return (0, int(match.group(1)))
    return (1, path.name)


def iter_sample_dirs(annotation_dir: Path, sample_names: Sequence[str]) -> Iterator[Path]:
    wanted = set(sample_names)
    for child in sorted(annotation_dir.iterdir(), key=sample_sort_key):
        if not child.is_dir():
            continue
        if wanted and child.name not in wanted:
            continue
        yield child


def build_sample_plan(sample_dir: Path, overwrite: bool) -> tuple[SamplePlan, dict[str, Any] | None, dict[str, Any] | None]:
    result_path = sample_dir / "result.json"
    summary_path = sample_dir / "summary.json"
    result_output = sample_dir / "chinese_result.json"
    summary_output = sample_dir / "chinese_summary.json"

    result_data: dict[str, Any] | None = None
    summary_data: dict[str, Any] | None = None
    result_refs: list[FieldRef] = []
    summary_refs: list[FieldRef] = []

    skip_result = result_output.exists() and not overwrite
    skip_summary = summary_output.exists() and not overwrite

    if result_path.exists() and not skip_result:
        result_data = load_json(result_path)
        result_refs = collect_result_refs(result_data)

    if summary_path.exists() and not skip_summary:
        summary_data = load_json(summary_path)
        summary_refs = collect_summary_refs(summary_data)

    return (
        SamplePlan(
            sample_dir=sample_dir,
            result_refs=result_refs,
            summary_refs=summary_refs,
            result_output=result_output,
            summary_output=summary_output,
            skip_result=skip_result,
            skip_summary=skip_summary,
        ),
        result_data,
        summary_data,
    )


def clean_guideline_text(text: str) -> str:
    text = text.replace("\x0c", "\n")
    text = "\n".join(line.rstrip() for line in text.splitlines())
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()


def load_guideline_text(args: argparse.Namespace) -> str:
    guideline_path = args.guideline_text
    if not guideline_path.exists():
        raise FileNotFoundError(f"Guideline text file not found: {guideline_path}")

    text = clean_guideline_text(guideline_path.read_text(encoding="utf-8"))
    if args.guideline_max_chars and len(text) > args.guideline_max_chars:
        text = text[: args.guideline_max_chars].rstrip()
    return text


def build_system_prompt(guideline_text: str) -> str:
    return (
        "你是医学影像报告中英翻译助手，负责将胸部X线报告相关英文文本翻译为中文。\n"
        "要求：\n"
        "1. 只翻译原文已有信息，不添加、删减或推断原文没有的影像表现、诊断、建议、程度或否定信息。\n"
        "2. 使用中文胸部X线影像报告常用规范表述，参考下方《X线报告书写规范 | 胸部》的用语。\n"
        "3. 保留左右侧、部位、数值、单位、编号、标点含义、换行结构以及占位符，例如 XXXX、___、[PHI]。\n"
        "4. FINDINGS 可译为“影像表现”，IMPRESSION/IMPRESSIONS 可译为“诊断意见”。\n"
        "5. 输入会是一个 JSON 对象，键为字段 ID，值为英文原文。只输出一个 JSON 对象，键必须与输入完全一致，值为中文译文；不要输出解释、Markdown 或额外字段。\n\n"
        "参考文档文本：\n"
        f"{guideline_text}"
    )


def build_user_prompt(items: Sequence[tuple[str, str]]) -> str:
    payload = {item_id: source_text for item_id, source_text in items}
    return json.dumps(payload, ensure_ascii=False, indent=2)


def strip_json_fence(text: str) -> str:
    text = text.strip()
    if not text.startswith("```"):
        return text
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def parse_translation_response(content: str) -> dict[str, str]:
    parsed = json.loads(strip_json_fence(content))
    if isinstance(parsed, dict) and isinstance(parsed.get("translations"), dict):
        parsed = parsed["translations"]
    if not isinstance(parsed, dict):
        raise ValueError("translation response is not a JSON object")

    translations: dict[str, str] = {}
    for key, value in parsed.items():
        if isinstance(value, str):
            translations[str(key)] = value
    return translations


def make_batches(
    items: Sequence[tuple[str, str]],
    batch_size: int,
    batch_max_chars: int,
) -> Iterator[list[tuple[str, str]]]:
    if batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if batch_max_chars <= 0:
        batch_max_chars = sys.maxsize

    batch: list[tuple[str, str]] = []
    char_count = 0
    for item in items:
        item_chars = len(item[1])
        if batch and (len(batch) >= batch_size or char_count + item_chars > batch_max_chars):
            yield batch
            batch = []
            char_count = 0
        batch.append(item)
        char_count += item_chars
    if batch:
        yield batch


def create_openai_client(api_key: str, base_url: str) -> Any:
    if OpenAI is None:
        raise RuntimeError("The openai package is required. Install project dependencies first.")
    if not api_key:
        raise RuntimeError("DeepSeek API key is empty. Set DEEPSEEK_API_KEY or pass --api-key.")
    return OpenAI(api_key=api_key, base_url=base_url)


def uses_deepseek_v4_thinking(model: str) -> bool:
    return model.strip().lower() in {"deepseek-v4-flash", "deepseek-v4-pro"}


def enable_deepseek_thinking(request: dict[str, Any], args: argparse.Namespace) -> None:
    if not uses_deepseek_v4_thinking(args.model):
        return

    request.pop("temperature", None)
    request["reasoning_effort"] = args.reasoning_effort
    request["extra_body"] = {"thinking": {"type": "enabled"}}


def complete_json_batch(
    client: Any,
    args: argparse.Namespace,
    system_prompt: str,
    batch: Sequence[tuple[str, str]],
) -> dict[str, str]:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": build_user_prompt(batch)},
    ]
    request: dict[str, Any] = {
        "model": args.model,
        "messages": messages,
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
            values = parse_translation_response(content)
            missing = [item_id for item_id, _ in batch if item_id not in values]
            if missing:
                raise ValueError(f"response missing ids: {', '.join(missing)}")
            return {item_id: values[item_id] for item_id, _ in batch}
        except Exception as exc:  # noqa: BLE001 - keep retries around provider/client errors
            last_error = exc
            if attempt >= args.retries:
                break
            time.sleep(args.retry_sleep * (2 ** (attempt - 1)))

    assert last_error is not None
    raise last_error


def translate_batch(
    client: Any,
    args: argparse.Namespace,
    system_prompt: str,
    batch: Sequence[tuple[str, str]],
) -> dict[str, str]:
    return complete_json_batch(client, args, system_prompt, batch)


def unique_text_items(refs: Iterable[FieldRef]) -> tuple[list[tuple[str, str]], dict[str, str]]:
    items: list[tuple[str, str]] = []
    text_to_id: dict[str, str] = {}
    for ref in refs:
        if ref.text in text_to_id:
            continue
        item_id = f"T{len(items) + 1:04d}"
        text_to_id[ref.text] = item_id
        items.append((item_id, ref.text))
    return items, text_to_id


def apply_translations(refs: Iterable[FieldRef], text_to_translation: dict[str, str]) -> None:
    for ref in refs:
        translation = text_to_translation.get(ref.text)
        if translation is not None:
            ref.container[ref.key] = translation


def translate_refs(
    refs: list[FieldRef],
    client: Any,
    args: argparse.Namespace,
    system_prompt: str,
    progress_label: str,
) -> None:
    items, text_to_id = unique_text_items(refs)
    if not items:
        return

    id_to_translation: dict[str, str] = {}
    batches = list(make_batches(items, args.batch_size, args.batch_max_chars))
    for index, batch in enumerate(batches, start=1):
        print(f"{progress_label}: translating batch {index}/{len(batches)} ({len(batch)} texts)")
        id_to_translation.update(translate_batch(client, args, system_prompt, batch))
        if args.request_sleep > 0:
            time.sleep(args.request_sleep)

    text_to_translation = {
        text: id_to_translation[item_id]
        for text, item_id in text_to_id.items()
        if item_id in id_to_translation
    }
    apply_translations(refs, text_to_translation)


def collect_all_plans(args: argparse.Namespace) -> list[tuple[SamplePlan, dict[str, Any] | None, dict[str, Any] | None]]:
    if not args.annotation_dir.exists():
        raise FileNotFoundError(f"Annotation directory not found: {args.annotation_dir}")

    plans: list[tuple[SamplePlan, dict[str, Any] | None, dict[str, Any] | None]] = []
    for sample_dir in iter_sample_dirs(args.annotation_dir, args.sample):
        plans.append(build_sample_plan(sample_dir, args.overwrite))
    return plans


def print_dry_run(
    plans: Sequence[tuple[SamplePlan, dict[str, Any] | None, dict[str, Any] | None]],
    guideline_text: str,
) -> None:
    result_field_count = sum(len(plan.result_refs) for plan, _, _ in plans)
    summary_field_count = sum(len(plan.summary_refs) for plan, _, _ in plans)
    skipped_results = sum(1 for plan, _, _ in plans if plan.skip_result)
    skipped_summaries = sum(1 for plan, _, _ in plans if plan.skip_summary)

    print(f"Guideline prompt characters: {len(guideline_text)}")
    print(f"Sample directories scanned: {len(plans)}")
    print(f"Result fields to translate: {result_field_count}")
    print(f"Summary fields to translate: {summary_field_count}")
    print(f"Existing chinese_result.json skipped: {skipped_results}")
    print(f"Existing chinese_summary.json skipped: {skipped_summaries}")

    for plan, _, _ in plans:
        if plan.result_refs or plan.summary_refs or plan.skip_result or plan.skip_summary:
            print(
                f"- {plan.sample_dir.name}: "
                f"result_refs={len(plan.result_refs)} "
                f"summary_refs={len(plan.summary_refs)} "
                f"skip_result={plan.skip_result} "
                f"skip_summary={plan.skip_summary}"
            )


def run() -> int:
    args = parse_args()
    guideline_text = load_guideline_text(args)
    plans = collect_all_plans(args)

    if args.dry_run:
        print_dry_run(plans, guideline_text)
        return 0

    client = create_openai_client(args.api_key, args.base_url)
    system_prompt = build_system_prompt(guideline_text)

    for plan, result_data, summary_data in plans:
        if result_data is not None:
            progress_label = f"{plan.sample_dir.name}/result.json"
            translate_refs(
                plan.result_refs,
                client,
                args,
                system_prompt,
                progress_label,
            )
            write_json_atomic(plan.result_output, result_data)
            print(f"Wrote {plan.result_output}")
        elif plan.skip_result:
            print(f"Skipped existing {plan.result_output}")

        if summary_data is not None:
            translate_refs(
                plan.summary_refs,
                client,
                args,
                system_prompt,
                f"{plan.sample_dir.name}/summary.json",
            )
            write_json_atomic(plan.summary_output, summary_data)
            print(f"Wrote {plan.summary_output}")
        elif plan.skip_summary:
            print(f"Skipped existing {plan.summary_output}")

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
