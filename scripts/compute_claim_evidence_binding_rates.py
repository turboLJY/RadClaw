#!/usr/bin/env python3
"""Compute evidence-binding rates for grouped generated claim types."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


CLAIM_TYPE_GROUP_MAP = {
    "positive_finding": "observation_claim",
    "negative_finding": "observation_claim",
    "normality_claim": "normality_claim",
    "attribute_claim": "attribute_claim",
    "measurement_claim": "attribute_claim",
    "comparison_claim": "comparison_claim",
    "diagnostic_inference": "diagnostic_claim",
    "etiology_claim": "diagnostic_claim",
    "impression_summary": "diagnostic_claim",
    "recommendation_claim": "recommendation_claim",
}

GROUP_ORDER = (
    "observation_claim",
    "normality_claim",
    "attribute_claim",
    "comparison_claim",
    "diagnostic_claim",
    "recommendation_claim",
)

SUPPORT_EVIDENCE_KEYS = (
    "image_evidence",
    "visual_evidence",
    "retrieved_evidence",
    "medical_evidence",
    "similar_case_evidence",
    "support_evidence",
    "supporting_evidence",
    "evidence",
    "evidence_ids",
)

EXCLUDED_SUPPORT_KEYS = {
    "conflict_evidence",
    "conflicts",
    "conflict_flags",
}


@dataclass
class StructuredItem:
    item_id: str
    section: str
    text: str
    support: Any
    linked_findings: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scan sample_* folders under one or more output directories, map "
            "generated_claims to structured_report findings/impressions, and compute "
            "evidence-binding rates for grouped claim types."
        )
    )
    parser.add_argument(
        "target_dirs",
        nargs="+",
        type=Path,
        help="Output directories containing sample_* folders, e.g. output/mimic/full_radclaw.",
    )
    parser.add_argument(
        "--sample-prefix",
        default="sample_",
        help="Only scan immediate child directories with this prefix.",
    )
    parser.add_argument(
        "--hallucination-filename",
        default="hallucination_detection.json",
        help="Per-sample hallucination detection JSON filename.",
    )
    parser.add_argument(
        "--result-filename",
        default="result.json",
        help="Per-sample result JSON filename containing structured_report.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Write machine-readable JSON instead of text.",
    )
    return parser.parse_args()


def warn(message: str) -> None:
    print(f"warning: {message}", file=sys.stderr)


def sample_sort_key(path: Path) -> tuple[int, int | str]:
    match = re.search(r"(\d+)$", path.name)
    if match:
        return (0, int(match.group(1)))
    return (1, path.name)


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        warn(f"{path}: file does not exist")
        return None
    except json.JSONDecodeError as exc:
        warn(f"{path}: invalid JSON: {exc.msg}")
        return None
    except OSError as exc:
        warn(f"{path}: cannot read file: {exc}")
        return None

    if not isinstance(data, dict):
        warn(f"{path}: root JSON value is not an object")
        return None
    return data


def iter_sample_dirs(target_dir: Path, sample_prefix: str) -> list[Path]:
    if not target_dir.exists():
        warn(f"{target_dir}: directory does not exist")
        return []
    if not target_dir.is_dir():
        warn(f"{target_dir}: not a directory")
        return []
    return sorted(
        (
            path
            for path in target_dir.iterdir()
            if path.is_dir() and path.name.startswith(sample_prefix)
        ),
        key=sample_sort_key,
    )


def is_non_empty(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, dict):
        return any(is_non_empty(item) for item in value.values())
    if isinstance(value, (list, tuple, set)):
        return any(is_non_empty(item) for item in value)
    return bool(value)


def has_bound_support(support: Any) -> bool:
    """Return True when a support object contains non-empty evidence."""
    if not is_non_empty(support):
        return False
    if not isinstance(support, dict):
        return True

    for key in SUPPORT_EVIDENCE_KEYS:
        if is_non_empty(support.get(key)):
            return True

    for key, value in support.items():
        if key in EXCLUDED_SUPPORT_KEYS:
            continue
        if is_non_empty(value):
            return True
    return False


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    text = value if isinstance(value, str) else str(value)
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def significant_tokens(text: str) -> set[str]:
    stopwords = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "in",
        "is",
        "it",
        "no",
        "of",
        "on",
        "or",
        "the",
        "to",
        "with",
    }
    return {token for token in text.split() if len(token) > 2 and token not in stopwords}


def claim_id_to_structured_id(claim_id: Any) -> str | None:
    if not isinstance(claim_id, str):
        return None
    match = re.match(r"^[A-Za-z]+_([FI])_(\d+)$", claim_id.strip())
    if not match:
        return None
    prefix, number = match.groups()
    return f"{prefix.upper()}{int(number):03d}"


def coerce_str_list(value: Any) -> list[str]:
    if value in (None, "", [], {}):
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, str) and item.strip()]
    return []


def extract_structured_items(result_data: dict[str, Any]) -> tuple[list[StructuredItem], dict[str, bool]]:
    structured_report = result_data.get("structured_report")
    if not isinstance(structured_report, dict):
        return [], {}

    items: list[StructuredItem] = []
    finding_support: dict[str, bool] = {}

    for section, key_names, id_names in (
        ("findings", ("findings", "Findings"), ("finding_id", "id")),
        ("impressions", ("impressions", "Impressions", "impression", "Impression"), ("impression_id", "id")),
    ):
        section_value = None
        for key in key_names:
            if key in structured_report:
                section_value = structured_report[key]
                break
        if isinstance(section_value, dict):
            section_value = section_value.get("items")
        if not isinstance(section_value, list):
            continue

        for index, entry in enumerate(section_value, start=1):
            if not isinstance(entry, dict):
                continue
            item_id = ""
            for id_name in id_names:
                value = entry.get(id_name)
                if isinstance(value, str) and value.strip():
                    item_id = value.strip()
                    break
            if not item_id:
                item_id = f"{'F' if section == 'findings' else 'I'}{index:03d}"

            item = StructuredItem(
                item_id=item_id,
                section=section,
                text=entry.get("text") if isinstance(entry.get("text"), str) else "",
                support=entry.get("support"),
                linked_findings=coerce_str_list(entry.get("linked_findings")),
            )
            items.append(item)
            if section == "findings":
                finding_support[item_id] = has_bound_support(item.support)

    item_support: dict[str, bool] = {}
    for item in items:
        bound = has_bound_support(item.support)
        if item.section == "impressions" and not bound:
            bound = any(finding_support.get(finding_id, False) for finding_id in item.linked_findings)
        item_support[item.item_id] = bound
    return items, item_support


def candidate_items_for_claim(claim: dict[str, Any], items: list[StructuredItem]) -> list[StructuredItem]:
    section = normalize_text(claim.get("section"))
    if section.startswith("finding"):
        candidates = [item for item in items if item.section == "findings"]
    elif section.startswith("impression"):
        candidates = [item for item in items if item.section == "impressions"]
    else:
        candidates = items
    return candidates or items


def match_claim_to_item(
    claim: dict[str, Any],
    items: list[StructuredItem],
) -> tuple[StructuredItem | None, str]:
    candidates = candidate_items_for_claim(claim, items)
    claim_text = normalize_text(claim.get("raw_text") or claim.get("claim_text") or claim.get("text"))

    if claim_text:
        exact_matches = [
            item
            for item in candidates
            if item.text and claim_text in normalize_text(item.text)
        ]
        if exact_matches:
            return min(exact_matches, key=lambda item: len(normalize_text(item.text))), "text"

        claim_tokens = significant_tokens(claim_text)
        if len(claim_tokens) >= 2:
            scored_matches: list[tuple[float, StructuredItem]] = []
            for item in candidates:
                item_tokens = significant_tokens(normalize_text(item.text))
                if not item_tokens:
                    continue
                score = len(claim_tokens & item_tokens) / len(claim_tokens)
                if score >= 0.75:
                    scored_matches.append((score, item))
            if scored_matches:
                scored_matches.sort(key=lambda pair: (pair[0], -len(normalize_text(pair[1].text))), reverse=True)
                return scored_matches[0][1], "token_overlap"

    structured_id = claim_id_to_structured_id(claim.get("claim_id"))
    if structured_id is not None:
        for item in candidates:
            if item.item_id == structured_id:
                return item, "id"

    return None, "unmatched"


def empty_group_stats() -> dict[str, dict[str, Any]]:
    return {
        group: {
            "total_claims": 0,
            "bound_claims": 0,
            "unbound_claims": 0,
            "unmatched_claims": 0,
            "error_claims": 0,
            "bound_error_claims": 0,
            "unbound_error_claims": 0,
            "missing_taxonomy_claims": 0,
            "bound_missing_taxonomy_claims": 0,
            "unbound_missing_taxonomy_claims": 0,
            "binding_rate": None,
            "hallucination_rate": None,
            "bound_hallucination_rate": None,
            "unbound_hallucination_rate": None,
        }
        for group in GROUP_ORDER
    }


def extract_taxonomy_result_map(hallucination_data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    raw_response = hallucination_data.get("taxonomy_raw_response")
    if not isinstance(raw_response, dict):
        return {}

    claim_level_results = raw_response.get("claim_level_results")
    if not isinstance(claim_level_results, list):
        return {}

    result_by_claim_id: dict[str, dict[str, Any]] = {}
    for item in claim_level_results:
        if not isinstance(item, dict):
            continue
        claim_id = item.get("claim_id")
        if not isinstance(claim_id, str) or not claim_id.strip():
            continue
        result_by_claim_id[claim_id.strip()] = {
            "is_error": item.get("is_error") is True,
            "support_status": normalize_support_status(item.get("support_status")),
        }
    return result_by_claim_id


def normalize_support_status(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        return "<missing>"
    return value.strip().lower()


def empty_support_status_counters() -> dict[str, Counter[str]]:
    return {group: Counter() for group in GROUP_ORDER}


def add_support_status_result(
    counts: dict[str, Counter[str]],
    errors: dict[str, Counter[str]],
    group: str,
    support_status: str,
    *,
    is_error: bool,
) -> None:
    counts[group][support_status] += 1
    if is_error:
        errors[group][support_status] += 1


def finalize_support_status_stats(
    counts: dict[str, Counter[str]],
    errors: dict[str, Counter[str]],
    group_stats: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    finalized: dict[str, dict[str, Any]] = {}
    for group in GROUP_ORDER:
        bound_total = group_stats[group]["bound_claims"]
        statuses: dict[str, dict[str, Any]] = {}
        for status in sorted(counts[group]):
            status_count = counts[group][status]
            error_count = errors[group][status]
            statuses[status] = {
                "count": status_count,
                "error_claims": error_count,
                "proportion_among_bound": None if bound_total == 0 else round(status_count / bound_total, 6),
                "hallucination_rate": None if status_count == 0 else round(error_count / status_count, 6),
            }
        finalized[group] = {
            "bound_claims": bound_total,
            "support_statuses": statuses,
        }
    return finalized


def add_claim_result(
    stats: dict[str, dict[str, Any]],
    group: str,
    *,
    bound: bool,
    matched: bool,
    is_error: bool,
    has_taxonomy_result: bool,
) -> None:
    group_stats = stats[group]
    group_stats["total_claims"] += 1
    if bound:
        group_stats["bound_claims"] += 1
        if is_error:
            group_stats["bound_error_claims"] += 1
        if not has_taxonomy_result:
            group_stats["bound_missing_taxonomy_claims"] += 1
    else:
        group_stats["unbound_claims"] += 1
        if is_error:
            group_stats["unbound_error_claims"] += 1
        if not has_taxonomy_result:
            group_stats["unbound_missing_taxonomy_claims"] += 1
    if is_error:
        group_stats["error_claims"] += 1
    if not has_taxonomy_result:
        group_stats["missing_taxonomy_claims"] += 1
    if not matched:
        group_stats["unmatched_claims"] += 1


def finalize_group_stats(stats: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    finalized: dict[str, dict[str, Any]] = {}
    for group in GROUP_ORDER:
        values = dict(stats[group])
        total = values["total_claims"]
        bound_total = values["bound_claims"]
        unbound_total = values["unbound_claims"]
        values["binding_rate"] = None if total == 0 else round(values["bound_claims"] / total, 6)
        values["hallucination_rate"] = None if total == 0 else round(values["error_claims"] / total, 6)
        values["bound_hallucination_rate"] = (
            None if bound_total == 0 else round(values["bound_error_claims"] / bound_total, 6)
        )
        values["unbound_hallucination_rate"] = (
            None if unbound_total == 0 else round(values["unbound_error_claims"] / unbound_total, 6)
        )
        finalized[group] = values
    return finalized


def collect_binding_rates(
    target_dirs: list[Path],
    *,
    sample_prefix: str,
    hallucination_filename: str,
    result_filename: str,
) -> dict[str, Any]:
    overall_stats = empty_group_stats()
    overall_support_status_counts = empty_support_status_counters()
    overall_support_status_errors = empty_support_status_counters()
    per_target_stats: dict[str, dict[str, dict[str, Any]]] = {}
    per_target_support_status_stats: dict[str, dict[str, dict[str, Any]]] = {}
    ignored_claim_types: Counter[str] = Counter()
    match_methods: Counter[str] = Counter()
    scanned_samples = 0
    samples_with_hallucination_json = 0
    samples_with_result_json = 0
    considered_claims = 0
    claims_with_taxonomy_result = 0
    claims_missing_taxonomy_result = 0

    for target_dir in target_dirs:
        target_stats = empty_group_stats()
        target_support_status_counts = empty_support_status_counters()
        target_support_status_errors = empty_support_status_counters()
        for sample_dir in iter_sample_dirs(target_dir, sample_prefix):
            scanned_samples += 1
            hallucination_path = sample_dir / hallucination_filename
            if not hallucination_path.exists():
                continue

            samples_with_hallucination_json += 1
            hallucination_data = load_json(hallucination_path)
            if hallucination_data is None:
                continue

            taxonomy_result_map = extract_taxonomy_result_map(hallucination_data)

            result_data = load_json(sample_dir / result_filename)
            if result_data is None:
                items: list[StructuredItem] = []
                item_support: dict[str, bool] = {}
            else:
                samples_with_result_json += 1
                items, item_support = extract_structured_items(result_data)

            claims = hallucination_data.get("generated_claims")
            if not isinstance(claims, list):
                warn(f"{hallucination_path}: generated_claims is not a list")
                continue

            for index, claim in enumerate(claims):
                if not isinstance(claim, dict):
                    warn(f"{hallucination_path}: generated_claims[{index}] is not an object")
                    continue

                claim_type = claim.get("claim_type")
                if not isinstance(claim_type, str):
                    ignored_claim_types["<missing>"] += 1
                    continue
                group = CLAIM_TYPE_GROUP_MAP.get(claim_type.strip())
                if group is None:
                    ignored_claim_types[claim_type.strip() or "<empty>"] += 1
                    continue

                considered_claims += 1
                matched_item, match_method = match_claim_to_item(claim, items)
                match_methods[match_method] += 1
                matched = matched_item is not None
                bound = bool(matched_item and item_support.get(matched_item.item_id, False))

                claim_id = claim.get("claim_id")
                claim_id_key = claim_id.strip() if isinstance(claim_id, str) else ""
                taxonomy_result = taxonomy_result_map.get(claim_id_key, {})
                has_taxonomy_result = claim_id_key in taxonomy_result_map
                is_error = taxonomy_result.get("is_error") is True
                support_status = taxonomy_result.get("support_status", "<missing>")
                if has_taxonomy_result:
                    claims_with_taxonomy_result += 1
                else:
                    claims_missing_taxonomy_result += 1

                if bound:
                    add_support_status_result(
                        overall_support_status_counts,
                        overall_support_status_errors,
                        group,
                        support_status,
                        is_error=is_error,
                    )
                    add_support_status_result(
                        target_support_status_counts,
                        target_support_status_errors,
                        group,
                        support_status,
                        is_error=is_error,
                    )

                add_claim_result(
                    overall_stats,
                    group,
                    bound=bound,
                    matched=matched,
                    is_error=is_error,
                    has_taxonomy_result=has_taxonomy_result,
                )
                add_claim_result(
                    target_stats,
                    group,
                    bound=bound,
                    matched=matched,
                    is_error=is_error,
                    has_taxonomy_result=has_taxonomy_result,
                )

        finalized_target_stats = finalize_group_stats(target_stats)
        per_target_stats[str(target_dir)] = finalized_target_stats
        per_target_support_status_stats[str(target_dir)] = finalize_support_status_stats(
            target_support_status_counts,
            target_support_status_errors,
            finalized_target_stats,
        )

    finalized_overall_stats = finalize_group_stats(overall_stats)

    return {
        "target_dirs": [str(path) for path in target_dirs],
        "scanned_samples": scanned_samples,
        "samples_with_hallucination_json": samples_with_hallucination_json,
        "samples_with_result_json": samples_with_result_json,
        "considered_claims": considered_claims,
        "claims_with_taxonomy_result": claims_with_taxonomy_result,
        "claims_missing_taxonomy_result": claims_missing_taxonomy_result,
        "ignored_claim_types": dict(sorted(ignored_claim_types.items())),
        "match_methods": dict(sorted(match_methods.items())),
        "claim_type_group_map": CLAIM_TYPE_GROUP_MAP,
        "overall": finalized_overall_stats,
        "per_target": per_target_stats,
        "overall_bound_support_status": finalize_support_status_stats(
            overall_support_status_counts,
            overall_support_status_errors,
            finalized_overall_stats,
        ),
        "per_target_bound_support_status": per_target_support_status_stats,
    }


def format_rate(value: float | None) -> str:
    return "NA" if value is None else f"{value:.6f}"


def print_support_status_table(
    stats: dict[str, dict[str, Any]],
    *,
    indent: str = "  ",
) -> None:
    print(
        f"{indent}group,support_status,bound_total,status_count,proportion_among_bound,"
        "error,hallucination_rate"
    )
    for group, group_values in stats.items():
        bound_total = group_values["bound_claims"]
        statuses = group_values["support_statuses"]
        if not statuses:
            print(f"{indent}{group},<none>,{bound_total},0,NA,0,NA")
            continue
        for status, values in statuses.items():
            proportion = format_rate(values["proportion_among_bound"])
            hallucination_rate = format_rate(values["hallucination_rate"])
            print(
                f"{indent}{group},{status},{bound_total},{values['count']},"
                f"{proportion},{values['error_claims']},{hallucination_rate}"
            )


def print_text(summary: dict[str, Any]) -> None:
    print("Scanned target dirs:")
    for target_dir in summary["target_dirs"]:
        print(f"  {target_dir}")

    print()
    print(f"sample folders scanned: {summary['scanned_samples']}")
    print(f"samples with hallucination JSON: {summary['samples_with_hallucination_json']}")
    print(f"samples with result JSON: {summary['samples_with_result_json']}")
    print(f"considered mapped-type generated claims: {summary['considered_claims']}")
    print(f"claims with taxonomy result: {summary['claims_with_taxonomy_result']}")
    print(f"claims missing taxonomy result: {summary['claims_missing_taxonomy_result']}")
    print(f"match methods: {summary['match_methods']}")
    print(f"ignored claim types: {summary['ignored_claim_types']}")

    print()
    print("overall evidence-binding and hallucination rates:")
    print(
        "  group,total,bound,unbound,unmatched,binding_rate,"
        "error,bound_error,unbound_error,missing_taxonomy,"
        "hallucination_rate,bound_hallucination_rate,unbound_hallucination_rate"
    )
    for group, values in summary["overall"].items():
        binding_rate = format_rate(values["binding_rate"])
        hallucination_rate = format_rate(values["hallucination_rate"])
        bound_hallucination_rate = format_rate(values["bound_hallucination_rate"])
        unbound_hallucination_rate = format_rate(values["unbound_hallucination_rate"])
        print(
            f"  {group},{values['total_claims']},{values['bound_claims']},"
            f"{values['unbound_claims']},{values['unmatched_claims']},{binding_rate},"
            f"{values['error_claims']},{values['bound_error_claims']},"
            f"{values['unbound_error_claims']},{values['missing_taxonomy_claims']},"
            f"{hallucination_rate},{bound_hallucination_rate},{unbound_hallucination_rate}"
        )

    print()
    print("overall bound-claim support-status proportions and hallucination rates:")
    print_support_status_table(summary["overall_bound_support_status"], indent="  ")

    print()
    print("per-target evidence-binding and hallucination rates:")
    for target_dir, stats in summary["per_target"].items():
        print(f"  {target_dir}")
        print(
            "    group,total,bound,unbound,unmatched,binding_rate,"
            "error,bound_error,unbound_error,missing_taxonomy,"
            "hallucination_rate,bound_hallucination_rate,unbound_hallucination_rate"
        )
        for group, values in stats.items():
            binding_rate = format_rate(values["binding_rate"])
            hallucination_rate = format_rate(values["hallucination_rate"])
            bound_hallucination_rate = format_rate(values["bound_hallucination_rate"])
            unbound_hallucination_rate = format_rate(values["unbound_hallucination_rate"])
            print(
                f"    {group},{values['total_claims']},{values['bound_claims']},"
                f"{values['unbound_claims']},{values['unmatched_claims']},{binding_rate},"
                f"{values['error_claims']},{values['bound_error_claims']},"
                f"{values['unbound_error_claims']},{values['missing_taxonomy_claims']},"
                f"{hallucination_rate},{bound_hallucination_rate},{unbound_hallucination_rate}"
            )

    print()
    print("per-target bound-claim support-status proportions and hallucination rates:")
    for target_dir, stats in summary["per_target_bound_support_status"].items():
        print(f"  {target_dir}")
        print_support_status_table(stats, indent="    ")


def main() -> int:
    args = parse_args()
    summary = collect_binding_rates(
        args.target_dirs,
        sample_prefix=args.sample_prefix,
        hallucination_filename=args.hallucination_filename,
        result_filename=args.result_filename,
    )
    if args.json:
        json.dump(summary, sys.stdout, ensure_ascii=False, indent=2)
        print()
    else:
        print_text(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
