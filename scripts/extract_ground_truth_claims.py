#!/usr/bin/env python3
"""Extract atomic medical claims from translated annotation summaries.

The script scans annotation sample folders, reads each chinese_summary.json,
decomposes ground_truth_chinese into atomic medical claims with DeepSeek, and
writes the claims back to the same JSON file under ground_truth_claims.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections.abc import Iterator, Sequence
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
SECTION_LABEL_RE = re.compile(
    r"(?im)^\s*(影像表现|检查所见|所见|诊断意见|诊断印象|印象|FINDINGS?|IMPRESSIONS?)\s*[:：]\s*"
)
SECTION_LABELS = {
    "影像表现": "FINDINGS",
    "检查所见": "FINDINGS",
    "所见": "FINDINGS",
    "FINDING": "FINDINGS",
    "FINDINGS": "FINDINGS",
    "诊断意见": "IMPRESSION",
    "诊断印象": "IMPRESSION",
    "印象": "IMPRESSION",
    "IMPRESSION": "IMPRESSION",
    "IMPRESSIONS": "IMPRESSION",
}
VALID_SECTIONS = {"FINDINGS", "IMPRESSION"}


@dataclass(frozen=True)
class SampleFile:
    sample_dir: Path
    summary_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Use DeepSeek to split each annotation_samples/*/chinese_summary.json "
            "ground_truth_chinese field into atomic medical claims."
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
        help="DeepSeek model used for claim extraction.",
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("DEEPSEEK_BASE_URL", DEFAULT_BASE_URL),
        help="DeepSeek OpenAI-compatible API base URL.",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("DEEPSEEK_API_KEY", DEFAULT_API_KEY),
        help="DeepSeek API key. Defaults to DEEPSEEK_API_KEY, then OPENAI_API_KEY.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Maximum output tokens for each claim-extraction request.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=5,
        help="API retry count per sample.",
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
        help="Regenerate ground_truth_claims when the field already exists.",
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


def iter_sample_files(annotation_dir: Path, sample_names: Sequence[str]) -> Iterator[SampleFile]:
    if not annotation_dir.exists():
        raise FileNotFoundError(f"Annotation directory not found: {annotation_dir}")

    if sample_names:
        sample_dirs = [annotation_dir / sample_name for sample_name in sample_names]
    else:
        sample_dirs = sorted(path for path in annotation_dir.iterdir() if path.is_dir())

    for sample_dir in sample_dirs:
        if not sample_dir.is_dir():
            print(f"warning: skipped missing sample directory: {sample_dir}", file=sys.stderr)
            continue
        summary_path = sample_dir / "chinese_summary.json"
        if summary_path.exists():
            yield SampleFile(sample_dir=sample_dir, summary_path=summary_path)
        else:
            print(f"warning: skipped sample without chinese_summary.json: {sample_dir}", file=sys.stderr)


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


def normalize_section_label(label: str) -> str:
    normalized = label.strip().upper()
    return SECTION_LABELS.get(label.strip(), SECTION_LABELS.get(normalized, normalized))


def split_ground_truth_sections(text: str) -> dict[str, str]:
    sections = {"FINDINGS": "", "IMPRESSION": ""}
    matches = list(SECTION_LABEL_RE.finditer(text))
    if not matches:
        sections["FINDINGS"] = text.strip()
        return sections

    for index, match in enumerate(matches):
        section = normalize_section_label(match.group(1))
        if section not in VALID_SECTIONS:
            continue
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        content = text[start:end].strip()
        if content:
            sections[section] = "\n\n".join(part for part in (sections[section], content) if part)
    return sections


def build_system_prompt() -> str:
    return (
        "你是严谨的放射科医学文本标注助手。你的任务是把中文影像报告拆解为原子医学 claim。"
        "原子 claim 是一个可以独立核查的医学结论，必须尽量保留原文中的解剖部位、侧别、程度、"
        "否定、可能性、鉴别诊断和因果/解释关系。不要加入原文没有的信息，不要翻译成英文，"
        "不要输出非医学寒暄或说明。"
    )


def build_user_prompt(sections: dict[str, str]) -> str:
    payload = {
        "FINDINGS": sections.get("FINDINGS", ""),
        "IMPRESSION": sections.get("IMPRESSION", ""),
    }
    return (
        "请将下面 JSON 中的 FINDINGS 与 IMPRESSION 文本分别拆解为原子医学 claim。\n"
        "输出要求：\n"
        "1. 只输出一个 JSON object，不要 Markdown。\n"
        "2. JSON schema 必须是 {\"claims\": [{\"section\": \"FINDINGS|IMPRESSION\", \"claim\": \"...\"}]}。\n"
        "3. section 必须严格使用 FINDINGS 或 IMPRESSION，表示 claim 来自哪个输入字段。\n"
        "4. 一个 claim 只表达一个医学结论；并列结论、列表项和否定发现要拆开。\n"
        "5. 修饰信息必须留在同一个 claim 中，例如部位、侧别、程度、否定、可疑/可能、鉴别诊断。\n"
        "6. 如果同一结论同时出现在 FINDINGS 和 IMPRESSION 中，分别保留，不要跨 section 去重。\n"
        "7. 如果某个 section 为空，不要为它生成 claim。\n\n"
        f"输入 JSON：\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )


def create_openai_client(api_key: str, base_url: str) -> Any:
    if OpenAI is None:
        raise RuntimeError("The openai package is required. Install project dependencies first.")
    if not api_key:
        raise RuntimeError("DeepSeek API key is empty. Set DEEPSEEK_API_KEY or pass --api-key.")
    return OpenAI(api_key=api_key, base_url=base_url)


def enable_deepseek_thinking(request: dict[str, Any], args: argparse.Namespace) -> None:
    request["reasoning_effort"] = args.reasoning_effort
    request["extra_body"] = {"thinking": {"type": "enabled"}}


def parse_json_object(content: str) -> dict[str, Any]:
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, flags=re.DOTALL)
        if not match:
            raise
        parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("LLM response root must be a JSON object")
    return parsed


def normalize_claims_response(content: str) -> list[dict[str, str]]:
    parsed = parse_json_object(content)
    raw_claims = parsed.get("claims")
    if not isinstance(raw_claims, list):
        raise ValueError("LLM response must contain a claims list")

    claims: list[dict[str, str]] = []
    for index, raw_claim in enumerate(raw_claims):
        if not isinstance(raw_claim, dict):
            raise ValueError(f"claims[{index}] must be an object")
        section = raw_claim.get("section")
        claim = raw_claim.get("claim")
        if not isinstance(section, str) or section not in VALID_SECTIONS:
            raise ValueError(f"claims[{index}].section must be FINDINGS or IMPRESSION")
        if not isinstance(claim, str) or not claim.strip():
            raise ValueError(f"claims[{index}].claim must be a non-empty string")
        claims.append({"section": section, "claim": claim.strip()})
    return claims


def extract_claims(
    client: Any,
    args: argparse.Namespace,
    sections: dict[str, str],
) -> list[dict[str, str]]:
    request: dict[str, Any] = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": build_system_prompt()},
            {"role": "user", "content": build_user_prompt(sections)},
        ],
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
            return normalize_claims_response(content)
        except Exception as exc:  # noqa: BLE001 - keep retries around provider/client errors
            last_error = exc
            if attempt >= args.retries:
                break
            time.sleep(args.retry_sleep * (2 ** (attempt - 1)))

    assert last_error is not None
    raise last_error


def should_process(data: dict[str, Any], overwrite: bool) -> bool:
    ground_truth_chinese = data.get("ground_truth_chinese")
    if not isinstance(ground_truth_chinese, str) or not ground_truth_chinese.strip():
        return False
    return overwrite or "ground_truth_claims" not in data


def print_dry_run(sample_files: Sequence[SampleFile], overwrite: bool) -> None:
    process_count = 0
    skip_existing_count = 0
    skip_missing_text_count = 0

    for sample_file in sample_files:
        data = load_json(sample_file.summary_path)
        ground_truth_chinese = data.get("ground_truth_chinese")
        if not isinstance(ground_truth_chinese, str) or not ground_truth_chinese.strip():
            skip_missing_text_count += 1
            continue
        if "ground_truth_claims" in data and not overwrite:
            skip_existing_count += 1
            continue
        process_count += 1

    print(f"Summary files found: {len(sample_files)}")
    print(f"Files to process: {process_count}")
    print(f"Existing ground_truth_claims skipped: {skip_existing_count}")
    print(f"Missing/empty ground_truth_chinese skipped: {skip_missing_text_count}")


def run() -> int:
    args = parse_args()
    sample_files = list(iter_sample_files(args.annotation_dir, args.sample))

    if args.dry_run:
        print_dry_run(sample_files, args.overwrite)
        return 0

    client = create_openai_client(args.api_key, args.base_url)

    for sample_file in sample_files:
        data = load_json(sample_file.summary_path)
        if not should_process(data, args.overwrite):
            print(f"Skipped {sample_file.summary_path}")
            continue

        ground_truth_chinese = data["ground_truth_chinese"]
        sections = split_ground_truth_sections(ground_truth_chinese)
        print(f"{sample_file.sample_dir.name}: extracting ground_truth_claims")
        data["ground_truth_claims"] = extract_claims(client, args, sections)
        write_json_atomic(sample_file.summary_path, data)
        print(f"Wrote {sample_file.summary_path}")

        if args.request_sleep > 0:
            time.sleep(args.request_sleep)

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
