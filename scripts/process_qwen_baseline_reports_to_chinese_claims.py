#!/usr/bin/env python3
"""Extract Qwen baseline report claims and translate them to Chinese.

The script scans annotation_baseline_reports sample folders, reads
qwen3.6-flash.txt, extracts atomic English medical claims from Findings and
Impression-like sections, translates each claim with the same chest X-ray
translation prompt used by translate_annotation_samples_to_chinese.py, and
writes qwen3.6-flash_chinese_result.json beside the source report.
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
from types import SimpleNamespace
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - exercised only when dependency is absent
    OpenAI = None  # type: ignore[assignment]

from translate_annotation_samples_to_chinese import (
    DEFAULT_API_KEY,
    DEFAULT_BASE_URL,
    DEFAULT_GUIDELINE_TEXT,
    DEFAULT_MODEL,
    build_system_prompt as build_translation_system_prompt,
    clean_guideline_text,
    make_batches,
    translate_batch,
    write_json_atomic,
)


DEFAULT_ANNOTATION_DIR = Path("annotation_baseline_reports")
DEFAULT_REPORT_NAME = "qwen3.6-flash.txt"
DEFAULT_OUTPUT_NAME = "qwen3.6-flash_chinese_result.json"
VALID_SECTIONS = {"FINDINGS", "IMPRESSION"}

HEADING_RE = re.compile(
    r"^\s*(?:[*-]\s*)?(?:#{1,6}\s*)?(?:\*\*)?\s*"
    r"(?P<label>[^:\n*]+?)"
    r"(?:\*\*)?\s*:\s*(?P<rest>.*)$"
)

FINDINGS_HEADINGS = {
    "finding",
    "findings",
    "detailed findings",
}

IMPRESSION_HEADINGS = {
    "conclusion",
    "conclusions",
    "impression",
    "impressions",
    "overall impression",
    "summary",
    "summary impression",
    "summary impression (based on description)",
}

FINDING_SUBSECTION_HEADINGS = {
    "abdomen",
    "airway",
    "artifacts",
    "bones",
    "bones and soft tissue",
    "bones and soft tissues",
    "bones/soft tissue",
    "cardiac device",
    "cardiac silhouette",
    "cardiomediastinal silhouette",
    "cardiomediastinum",
    "cardiovascular",
    "cardiovascular and mediastinum",
    "cardiovascular findings",
    "central catheter",
    "costophrenic angles",
    "diaphragm",
    "effusions",
    "general",
    "heart and mediastinum",
    "hila",
    "left lung",
    "lungs",
    "lung",
    "lungs and pleura",
    "mediastinum",
    "mediastinum and heart",
    "medical devices and surgical hardware",
    "medical devices/implants",
    "musculoskeletal and soft tissues",
    "opacities",
    "osseous structures",
    "other",
    "pleura",
    "prosthetic valve",
    "pulmonary findings",
    "right lung",
    "soft tissues",
    "sternotomy wires",
    "subdiaphragmatic region",
    "support devices",
    "tubing",
    "vascularity",
}

SKIP_HEADINGS = {
    "exam",
    "examination",
    "general overview",
    "image type",
    "image type and technique",
    "indication",
    "overview",
    "technique",
    "technique and view",
}


@dataclass(frozen=True)
class ReportFile:
    sample_dir: Path
    report_path: Path
    output_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract atomic English claims from annotation_baseline_reports/*/"
            "qwen3.6-flash.txt and translate each claim to Chinese."
        )
    )
    parser.add_argument(
        "--annotation-dir",
        type=Path,
        default=DEFAULT_ANNOTATION_DIR,
        help="Directory containing baseline report sample subfolders.",
    )
    parser.add_argument(
        "--report-name",
        default=DEFAULT_REPORT_NAME,
        help="Report filename to read inside each sample directory.",
    )
    parser.add_argument(
        "--output-name",
        default=DEFAULT_OUTPUT_NAME,
        help="Output JSON filename to write inside each sample directory.",
    )
    parser.add_argument(
        "--guideline-text",
        type=Path,
        default=DEFAULT_GUIDELINE_TEXT,
        help="Chest X-ray report-writing guideline text file used as translation reference.",
    )
    parser.add_argument(
        "--guideline-max-chars",
        type=int,
        default=0,
        help="Maximum guideline characters to include in the translation prompt. 0 means full text.",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("DEEPSEEK_MODEL", DEFAULT_MODEL),
        help="DeepSeek model used for claim extraction and translation.",
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("DEEPSEEK_BASE_URL", DEFAULT_BASE_URL),
        help="DeepSeek OpenAI-compatible API base URL.",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("DEEPSEEK_API_KEY", DEFAULT_API_KEY),
        help="DeepSeek API key.",
    )
    parser.add_argument(
        "--api-timeout",
        type=float,
        default=180.0,
        help="Per-request API timeout in seconds.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=20,
        help="Maximum number of unique claim strings per translation request.",
    )
    parser.add_argument(
        "--batch-max-chars",
        type=int,
        default=6000,
        help="Soft maximum source claim characters per translation request batch.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Maximum output tokens per API request.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=5,
        help="API retry count per request.",
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
        "--sample",
        action="append",
        default=[],
        help="Process only the named sample directory. May be passed multiple times.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate output files if they already exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the processing plan without calling the API or writing JSON.",
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


def iter_report_files(args: argparse.Namespace) -> Iterator[ReportFile]:
    if not args.annotation_dir.exists():
        raise FileNotFoundError(f"Annotation directory not found: {args.annotation_dir}")

    if args.sample:
        sample_dirs = [args.annotation_dir / sample_name for sample_name in args.sample]
    else:
        sample_dirs = sorted(
            (path for path in args.annotation_dir.iterdir() if path.is_dir()),
            key=sample_sort_key,
        )

    for sample_dir in sample_dirs:
        if not sample_dir.is_dir():
            print(f"warning: skipped missing sample directory: {sample_dir}", file=sys.stderr)
            continue
        report_path = sample_dir / args.report_name
        if not report_path.exists():
            print(f"warning: skipped sample without {args.report_name}: {sample_dir}", file=sys.stderr)
            continue
        yield ReportFile(
            sample_dir=sample_dir,
            report_path=report_path,
            output_path=sample_dir / args.output_name,
        )


def strip_markdown(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^\s*[*-]\s*", "", text)
    text = text.replace("**", "")
    text = text.replace("__", "")
    return text.strip()


def normalize_heading(label: str) -> str:
    label = strip_markdown(label).lower()
    label = re.sub(r"\s+", " ", label)
    return label.strip()


def clean_section_line(line: str) -> str:
    line = line.strip()
    line = re.sub(r"^\s*[*-]\s*", "", line)
    line = line.replace("**", "")
    line = line.replace("__", "")
    return line.strip()


def section_for_heading(label: str) -> str | None:
    normalized = normalize_heading(label)
    if normalized in FINDINGS_HEADINGS:
        return "FINDINGS"
    if normalized in IMPRESSION_HEADINGS:
        return "IMPRESSION"
    if normalized in FINDING_SUBSECTION_HEADINGS:
        return "FINDINGS"
    if normalized in SKIP_HEADINGS:
        return None
    return None


def append_line(sections: dict[str, list[str]], section: str | None, line: str) -> None:
    if section not in VALID_SECTIONS:
        return
    cleaned = clean_section_line(line)
    if cleaned:
        sections[section].append(cleaned)


def split_report_sections(text: str) -> dict[str, str]:
    sections: dict[str, list[str]] = {"FINDINGS": [], "IMPRESSION": []}
    current_section: str | None = None
    saw_section_heading = False

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            if current_section in VALID_SECTIONS and sections[current_section]:
                sections[current_section].append("")
            continue

        match = HEADING_RE.match(line)
        if match:
            label = match.group("label")
            rest = strip_markdown(match.group("rest"))
            next_section = section_for_heading(label)
            normalized = normalize_heading(label)

            if normalized in SKIP_HEADINGS and not saw_section_heading:
                current_section = None
                continue

            if next_section in VALID_SECTIONS:
                current_section = next_section
                saw_section_heading = True
                if normalized in FINDING_SUBSECTION_HEADINGS and rest:
                    append_line(sections, current_section, f"{strip_markdown(label)}: {rest}")
                elif rest:
                    append_line(sections, current_section, rest)
                continue

            append_line(sections, current_section, line)
            continue

        append_line(sections, current_section, line)

    return {
        section: "\n".join(lines).strip()
        for section, lines in sections.items()
    }


def build_claim_extraction_system_prompt() -> str:
    return (
        "You are a meticulous radiology text annotation assistant. Your task is to split "
        "English chest X-ray report text into atomic medical claims. An atomic claim is "
        "one independently checkable medical assertion. Preserve anatomy, laterality, "
        "severity, negation, uncertainty, differential diagnoses, device/hardware details, "
        "and causal or explanatory relationships when they are present in the source. "
        "Do not add, infer, translate, summarize away, or normalize beyond what is needed "
        "to make a concise standalone English claim."
    )


def build_claim_extraction_user_prompt(sections: dict[str, str]) -> str:
    payload = {
        "FINDINGS": sections.get("FINDINGS", ""),
        "IMPRESSION": sections.get("IMPRESSION", ""),
    }
    return (
        "Split the FINDINGS and IMPRESSION text in the JSON below into atomic medical "
        "claims.\n"
        "Output requirements:\n"
        "1. Output exactly one JSON object and no Markdown.\n"
        "2. The JSON schema must be "
        "{\"claims\": [{\"section\": \"FINDINGS|IMPRESSION\", \"claim_text\": \"...\"}]}.\n"
        "3. section must be exactly FINDINGS or IMPRESSION and must reflect the source "
        "input field.\n"
        "4. Keep claim_text in English.\n"
        "5. Split conjoined observations and conjoined negated findings into separate "
        "claims when each can be checked independently.\n"
        "6. Keep modifiers in the same claim, including anatomy, side, degree, negation, "
        "uncertainty, and differential diagnoses.\n"
        "7. Exclude non-medical introduction, technique/view/acquisition statements, and "
        "patient-facing explanatory parentheticals unless they are required for the claim.\n"
        "8. If a section is empty, do not create claims for that section.\n"
        "9. If the same claim appears in both FINDINGS and IMPRESSION, keep both entries "
        "with their respective section values.\n\n"
        f"Input JSON:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )


def create_openai_client(api_key: str, base_url: str, timeout: float) -> Any:
    if OpenAI is None:
        raise RuntimeError("The openai package is required. Install project dependencies first.")
    if not api_key:
        raise RuntimeError("DeepSeek API key is empty. Set DEEPSEEK_API_KEY or pass --api-key.")
    return OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)


def uses_deepseek_v4_thinking(model: str) -> bool:
    return model.strip().lower() in {"deepseek-v4-flash", "deepseek-v4-pro"}


def enable_deepseek_thinking(request: dict[str, Any], args: argparse.Namespace) -> None:
    if not uses_deepseek_v4_thinking(args.model):
        return

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
    try:
        parsed = json.loads(strip_json_fence(content))
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, flags=re.DOTALL)
        if not match:
            raise
        parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("LLM response root must be a JSON object")
    return parsed


def normalize_claims_response(content: str) -> dict[str, list[str]]:
    parsed = parse_json_object(content)
    raw_claims = parsed.get("claims")
    if not isinstance(raw_claims, list):
        raise ValueError("LLM response must contain a claims list")

    section_claims: dict[str, list[str]] = {"FINDINGS": [], "IMPRESSION": []}
    seen: set[tuple[str, str]] = set()
    for index, raw_claim in enumerate(raw_claims):
        if not isinstance(raw_claim, dict):
            raise ValueError(f"claims[{index}] must be an object")
        section = raw_claim.get("section")
        claim = raw_claim.get("claim_text", raw_claim.get("claim"))
        if not isinstance(section, str) or section not in VALID_SECTIONS:
            raise ValueError(f"claims[{index}].section must be FINDINGS or IMPRESSION")
        if not isinstance(claim, str) or not claim.strip():
            raise ValueError(f"claims[{index}].claim_text must be a non-empty string")

        claim_text = clean_section_line(claim)
        key = (section, claim_text.lower())
        if key in seen:
            continue
        seen.add(key)
        section_claims[section].append(claim_text)
    return section_claims


def extract_claims(
    client: Any,
    args: argparse.Namespace,
    sections: dict[str, str],
) -> dict[str, list[str]]:
    request: dict[str, Any] = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": build_claim_extraction_system_prompt()},
            {"role": "user", "content": build_claim_extraction_user_prompt(sections)},
        ],
        "temperature": 0.0,
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


def load_guideline_text(path: Path, max_chars: int) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Guideline text file not found: {path}")
    text = clean_guideline_text(path.read_text(encoding="utf-8"))
    if max_chars and len(text) > max_chars:
        text = text[:max_chars].rstrip()
    return text


def translate_claim_texts(
    client: Any,
    args: argparse.Namespace,
    claim_texts: Sequence[str],
) -> dict[str, str]:
    unique_texts: list[str] = []
    seen: set[str] = set()
    for claim_text in claim_texts:
        if claim_text not in seen:
            seen.add(claim_text)
            unique_texts.append(claim_text)

    if not unique_texts:
        return {}

    guideline_text = load_guideline_text(args.guideline_text, args.guideline_max_chars)
    system_prompt = build_translation_system_prompt(guideline_text)
    translation_args = SimpleNamespace(
        model=args.model,
        temperature=0.0,
        max_tokens=args.max_tokens,
        no_json_response_format=args.no_json_response_format,
        retries=args.retries,
        retry_sleep=args.retry_sleep,
        reasoning_effort=args.reasoning_effort,
    )

    id_to_text = {
        f"T{index + 1:04d}": claim_text
        for index, claim_text in enumerate(unique_texts)
    }
    id_to_translation: dict[str, str] = {}
    items = list(id_to_text.items())
    batches = list(make_batches(items, args.batch_size, args.batch_max_chars))
    for index, batch in enumerate(batches, start=1):
        print(f"translation: batch {index}/{len(batches)} ({len(batch)} claims)")
        id_to_translation.update(translate_batch(client, translation_args, system_prompt, batch))
        if args.request_sleep > 0 and index < len(batches):
            time.sleep(args.request_sleep)

    return {
        claim_text: id_to_translation[item_id]
        for item_id, claim_text in id_to_text.items()
    }


def build_output_data(
    natural_language_report: str,
    claims: dict[str, list[str]],
    translations: dict[str, str],
) -> dict[str, Any]:
    return {
        "natural_language_report": natural_language_report,
        "findings": [
            {"claim_text": claim_text, "chinese_text": translations.get(claim_text, "")}
            for claim_text in claims.get("FINDINGS", [])
        ],
        "impression": [
            {"claim_text": claim_text, "chinese_text": translations.get(claim_text, "")}
            for claim_text in claims.get("IMPRESSION", [])
        ],
    }


def print_dry_run(report_files: Sequence[ReportFile], overwrite: bool) -> None:
    skipped = 0
    for report_file in report_files:
        if report_file.output_path.exists() and not overwrite:
            skipped += 1
            print(f"- {report_file.sample_dir.name}: skipped existing output")
            continue
        report = report_file.report_path.read_text(encoding="utf-8")
        sections = split_report_sections(report)
        print(
            f"- {report_file.sample_dir.name}: "
            f"findings_chars={len(sections['FINDINGS'])} "
            f"impression_chars={len(sections['IMPRESSION'])}"
        )
    print(f"Reports found: {len(report_files)}")
    print(f"Existing outputs skipped: {skipped}")


def run() -> int:
    args = parse_args()
    report_files = list(iter_report_files(args))

    if args.dry_run:
        print_dry_run(report_files, args.overwrite)
        return 0

    client = create_openai_client(args.api_key, args.base_url, args.api_timeout)

    for index, report_file in enumerate(report_files, start=1):
        if report_file.output_path.exists() and not args.overwrite:
            print(f"Skipped existing {report_file.output_path}")
            continue

        report = report_file.report_path.read_text(encoding="utf-8")
        sections = split_report_sections(report)
        if not sections["FINDINGS"] and not sections["IMPRESSION"]:
            print(f"warning: no Findings/Impression text found: {report_file.report_path}", file=sys.stderr)

        print(f"{report_file.sample_dir.name}: extracting claims ({index}/{len(report_files)})")
        claims = extract_claims(client, args, sections)
        claim_texts = [claim for section_claims in claims.values() for claim in section_claims]

        if args.request_sleep > 0:
            time.sleep(args.request_sleep)

        print(f"{report_file.sample_dir.name}: translating {len(claim_texts)} claims")
        translations = translate_claim_texts(client, args, claim_texts)
        output_data = build_output_data(report, claims, translations)
        write_json_atomic(report_file.output_path, output_data)
        print(f"Wrote {report_file.output_path}")

        if args.request_sleep > 0 and index < len(report_files):
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
