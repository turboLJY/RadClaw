from __future__ import annotations

import argparse
import base64
import json
import math
import mimetypes
import os
import re
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


HALLUCINATION_TYPES = {
    "H1": "H1_fabricated_finding",
    "H2": "H2_omitted_finding",
    "H3": "H3_negation_error",
    "H4": "H4_location_or_laterality_error",
    "H5": "H5_attribute_error",
    "H6": "H6_comparison_error",
    "H7": "H7_unsupported_diagnostic_inference",
    "H8": "H8_findings_impression_inconsistency",
    "H9": "H9_certainty_calibration_error",
    "H10": "H10_unsupported_recommendation",
}

ALLOWED_CLAIM_TYPES = {
    "positive_finding",
    "negative_finding",
    "attribute_claim",
    "measurement_claim",
    "comparison_claim",
    "impression_summary",
    "diagnostic_inference",
    "etiology_claim",
    "certainty_claim",
    "recommendation_claim",
    "normality_claim",
}

SUPPORTED_IMAGE_EXTENSIONS = {
    ".bmp",
    ".gif",
    ".jpeg",
    ".jpg",
    ".nii",
    ".nii.gz",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}

TEXT_KEYS = (
    "raw_text",
    "text",
    "description",
    "content",
    "finding",
    "observation",
    "impression",
    "summary",
    "sentence",
)

EVIDENCE_KEYS = (
    "linked_evidence",
    "evidence",
    "visual_evidence",
    "image_evidence",
    "support_evidence",
    "supporting_evidence",
    "evidence_ids",
)

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "s",
    "seen",
    "show",
    "shows",
    "the",
    "there",
    "this",
    "to",
    "was",
    "were",
    "with",
    "within",
}

NEGATION_PATTERNS = (
    r"\bno\b",
    r"\bwithout\b",
    r"\babsent\b",
    r"\bnegative for\b",
    r"\bfree of\b",
    r"\bno evidence of\b",
    r"\bnot seen\b",
    r"\bnot identified\b",
    r"\bnot detected\b",
    r"\bno acute\b",
)

CERTAINTY_PATTERNS = {
    "cannot_exclude": (r"\bcannot exclude\b", r"\bcan not exclude\b"),
    "possible": (r"\bpossible\b", r"\bpossibly\b", r"\bmay represent\b", r"\bmay be\b"),
    "probable": (r"\bprobable\b", r"\bprobably\b", r"\blikely\b"),
    "suspicious": (r"\bsuspicious for\b", r"\bconcerning for\b", r"\bsuggestive of\b"),
}

SEVERITY_TERMS = (
    "trace",
    "minimal",
    "mild",
    "moderate",
    "severe",
    "small",
    "tiny",
    "large",
    "marked",
    "extensive",
    "diffuse",
    "focal",
)

TEMPORAL_TERMS = (
    "new",
    "stable",
    "unchanged",
    "increased",
    "increase",
    "decreased",
    "decrease",
    "improved",
    "improving",
    "worsened",
    "worsening",
    "resolved",
    "resolution",
    "interval",
    "progression",
)

DESCRIPTOR_TERMS = (
    "acute",
    "chronic",
    "calcified",
    "solid",
    "cystic",
    "ground-glass",
    "ground glass",
    "spiculated",
    "linear",
    "patchy",
    "nodular",
    "lobulated",
    "enhancing",
    "nonenhancing",
    "hyperdense",
    "hypodense",
    "hyperintense",
    "hypointense",
)

FINDING_TERMS = sorted(
    {
        "airspace opacity",
        "abnormality",
        "aneurysm",
        "arachnoid cyst",
        "atelectasis",
        "atrophy",
        "bone lesion",
        "bowel obstruction",
        "cardiomegaly",
        "cavitation",
        "consolidation",
        "cyst",
        "diffusion restriction",
        "edema",
        "effusion",
        "emphysema",
        "emphysematous changes",
        "enhancement",
        "fracture",
        "fractures",
        "gliosis",
        "ground glass opacity",
        "hemorrhage",
        "hernia",
        "hydrocephalus",
        "hyperinflation",
        "infarct",
        "infiltrate",
        "ischemia",
        "lesion",
        "mass",
        "midline shift",
        "mucosal thickening",
        "nodule",
        "nodules",
        "opacity",
        "opacities",
        "pneumonia",
        "pneumothorax",
        "pulmonary edema",
        "pleural effusion",
        "pleural effusions",
        "postoperative change",
        "rib cage deformity",
        "rib fracture",
        "rib fractures",
        "stenosis",
        "thickening",
        "tumor",
    },
    key=len,
    reverse=True,
)

ANATOMY_TERMS = sorted(
    {
        "abdomen": "abdomen",
        "adrenal": "adrenal gland",
        "appendix": "appendix",
        "basal ganglia": "basal ganglia",
        "bladder": "bladder",
        "brain": "brain",
        "brainstem": "brainstem",
        "cardiac silhouette": "heart",
        "cerebellum": "cerebellum",
        "colon": "colon",
        "corpus callosum": "corpus callosum",
        "diaphragm": "diaphragm",
        "ethmoid": "ethmoid sinus",
        "frontal lobe": "frontal lobe",
        "gallbladder": "gallbladder",
        "heart": "heart",
        "hilum": "hilum",
        "kidney": "kidney",
        "left lower lobe": "left lower lobe",
        "left upper lobe": "left upper lobe",
        "liver": "liver",
        "lung": "lung",
        "lungs": "lung",
        "mastoid": "mastoid",
        "maxillary sinus": "maxillary sinus",
        "mediastinum": "mediastinum",
        "occipital lobe": "occipital lobe",
        "orbit": "orbit",
        "pancreas": "pancreas",
        "paranasal sinus": "paranasal sinus",
        "parietal lobe": "parietal lobe",
        "pleura": "pleural space",
        "pleural": "pleural space",
        "prostate": "prostate",
        "right lower lobe": "right lower lobe",
        "right middle lobe": "right middle lobe",
        "right upper lobe": "right upper lobe",
        "rib cage": "rib cage",
        "ribs": "ribs",
        "rib": "rib",
        "sinus": "sinus",
        "spleen": "spleen",
        "spine": "spine",
        "temporal lobe": "temporal lobe",
        "thalamus": "thalamus",
        "uterus": "uterus",
        "ventricle": "ventricle",
    }.items(),
    key=lambda item: len(item[0]),
    reverse=True,
)


RADIOLOGY_SYNONYM_GROUPS = {
    "pleural effusion": (
        "pleural effusion", "pleural fluid", "pleural fluid collection", "effusion", "costophrenic blunting",
        "blunted costophrenic angle", "blunting of the costophrenic angle",
    ),
    "pneumothorax": ("pneumothorax", "ptx", "pleural air"),
    "cardiomegaly": (
        "cardiomegaly", "enlarged heart", "heart enlarged", "enlarged cardiac silhouette",
        "cardiac enlargement", "borderline enlarged heart", "borderline cardiomegaly",
        "mild enlargement of cardiac silhouette", "increased cardiac silhouette", "heart size enlarged",
    ),
    "pulmonary edema": (
        "pulmonary edema", "interstitial edema", "vascular congestion", "pulmonary vascular congestion",
        "congestive change", "cephalization", "fluid overload", "alveolar edema",
    ),
    "atelectasis": ("atelectasis", "subsegmental atelectasis", "discoid atelectasis", "linear atelectasis", "plate-like atelectasis", "volume loss"),
    "consolidation": ("consolidation", "airspace consolidation", "air-space consolidation", "lobar consolidation"),
    "airspace opacity": (
        "airspace opacity", "air-space opacity", "airspace disease", "air-space disease", "opacity",
        "opacities", "pulmonary opacity", "parenchymal opacity", "infiltrate", "infiltration",
    ),
    "ground glass opacity": (
        "ground glass opacity", "ground-glass opacity", "groundglass opacity", "ggo", "ground glass opacities",
        "ground-glass opacities", "groundglass opacities",
    ),
    "pneumonia": ("pneumonia", "infectious pneumonia", "bacterial pneumonia", "infection", "infectious process", "pneumonic infiltrate"),
    "emphysema": ("emphysema", "emphysematous change", "emphysematous changes", "bullous emphysema", "bullous emphysematous change", "copd change"),
    "hyperinflation": ("hyperinflation", "hyperinflated lungs", "pulmonary hyperinflation", "hyperexpanded lungs"),
    "nodule": ("nodule", "nodules", "pulmonary nodule", "lung nodule", "nodular opacity", "nodular opacities"),
    "mass": ("mass", "masses", "pulmonary mass", "lung mass", "masslike opacity", "mass-like opacity"),
    "lesion": ("lesion", "lesions", "focal lesion", "abnormality", "focal abnormality"),
    "fracture": ("fracture", "fractures", "fx", "rib fracture", "rib fractures", "osseous fracture", "bony fracture"),
    "rib deformity": ("rib deformity", "rib cage deformity", "rib cage deformities", "chronic rib deformity", "healed rib fracture", "old rib fracture"),
    "hernia": ("hernia", "hiatal hernia", "diaphragmatic hernia"),
    "mucosal thickening": ("mucosal thickening", "sinus mucosal thickening", "mucoperiosteal thickening"),
    "hydrocephalus": ("hydrocephalus", "ventriculomegaly", "ventricular enlargement", "enlarged ventricles"),
    "midline shift": ("midline shift", "shift of midline", "subfalcine shift", "mass effect with shift"),
    "hemorrhage": ("hemorrhage", "haemorrhage", "bleed", "intracranial hemorrhage", "ich", "intracranial bleed"),
    "infarct": ("infarct", "infarction", "stroke", "ischemic infarct", "acute infarct"),
    "ischemia": ("ischemia", "ischaemia", "ischemic change", "ischemic changes"),
    "diffusion restriction": ("diffusion restriction", "restricted diffusion", "dwi hyperintensity", "reduced diffusion"),
    "edema": ("edema", "oedema", "swelling", "vasogenic edema", "marrow edema"),
    "enhancement": ("enhancement", "contrast enhancement", "enhancing lesion", "abnormal enhancement"),
    "stenosis": ("stenosis", "narrowing", "stricture", "luminal narrowing"),
    "cyst": ("cyst", "cystic lesion", "simple cyst", "arachnoid cyst"),
    "gliosis": ("gliosis", "gliotic change", "gliotic changes", "encephalomalacia and gliosis"),
    "tumor": ("tumor", "tumour", "neoplasm", "malignancy", "cancer", "metastasis", "metastatic disease"),
    "postoperative change": ("postoperative change", "postoperative changes", "postsurgical change", "postsurgical changes", "post surgical change", "surgical change"),
    "normality": (
        "normal", "unremarkable", "within normal limits", "no acute cardiopulmonary abnormality",
        "no acute abnormality", "clear", "patent",
    ),
}

ANATOMY_SYNONYM_GROUPS = {
    "heart": ("heart", "cardiac silhouette", "cardiomediastinal silhouette", "cardiomediastinum"),
    "mediastinum": ("mediastinum", "mediastinal", "hilar contours", "hilum", "hila"),
    "lung": ("lung", "lungs", "pulmonary", "lung fields", "hemithorax"),
    "pleural space": ("pleura", "pleural", "pleural space", "costophrenic angle", "costophrenic angles"),
    "right upper lobe": ("right upper lobe", "rul"),
    "right middle lobe": ("right middle lobe", "rml"),
    "right lower lobe": ("right lower lobe", "rll", "right base", "right basilar"),
    "left upper lobe": ("left upper lobe", "lul"),
    "left lower lobe": ("left lower lobe", "lll", "left base", "left basilar"),
    "rib cage": ("rib cage", "ribs", "rib", "hemithoracic cage"),
    "diaphragm": ("diaphragm", "hemidiaphragm", "hemidiaphragms"),
    "brain": ("brain", "cerebral parenchyma", "intracranial"),
    "ventricle": ("ventricle", "ventricles", "ventricular system"),
    "sinus": ("sinus", "sinuses", "paranasal sinus", "paranasal sinuses"),
}

SYNONYM_TO_CANONICAL: dict[str, str] = {}
for canonical, variants in RADIOLOGY_SYNONYM_GROUPS.items():
    for variant in variants:
        SYNONYM_TO_CANONICAL[variant] = canonical

ANATOMY_TO_CANONICAL: dict[str, str] = {}
for canonical, variants in ANATOMY_SYNONYM_GROUPS.items():
    for variant in variants:
        ANATOMY_TO_CANONICAL[variant] = canonical

SORTED_SYNONYMS = sorted(SYNONYM_TO_CANONICAL.items(), key=lambda item: len(item[0]), reverse=True)
SORTED_ANATOMY_SYNONYMS = sorted(ANATOMY_TO_CANONICAL.items(), key=lambda item: len(item[0]), reverse=True)

STAGE_B_FINDING_TYPES = {
    "positive_finding",
    "negative_finding",
    "attribute_claim",
    "measurement_claim",
    "comparison_claim",
    "normality_claim",
    "diagnostic_inference",
    "etiology_claim",
    "certainty_claim",
}

DIAGNOSTIC_OVERINTERPRETATION_TERMS = {
    "pneumonia",
    "infection",
    "malignancy",
    "cancer",
    "metastasis",
    "metastatic",
    "tumor",
    "neoplasm",
    "abscess",
    "infarct",
    "ischemia",
}

NONSPECIFIC_FINDING_TERMS = {
    "airspace opacity",
    "ground glass opacity",
    "opacity",
    "lesion",
    "nodule",
    "mass",
    "consolidation",
    "edema",
}

CERTAINTY_SCALE = {
    "absent": 0,
    "negative": 0,
    "cannot_exclude": 1,
    "possible": 2,
    "uncertain": 2,
    "suspicious": 3,
    "probable": 4,
    "likely": 4,
    "definite": 5,
    "consistent with": 5,
}




@dataclass
class Claim:
    claim_id: str
    case_id: str
    source: str
    section: str
    raw_text: str
    claim_type: str
    core_finding: str | None
    polarity: str
    anatomy: str | None
    laterality: str
    location_detail: str | None
    severity: str | None
    measurement: str | None
    descriptor: str | None
    temporal: str | None
    certainty: str
    recommendation: str | None
    linked_evidence: list[Any] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Match:
    claim: Claim | None
    score: float


@dataclass
class DetectorConfig:
    min_match_score: float = 0.45
    omission_match_score: float = 0.42
    impression_match_score: float = 0.38
    strict_negative_claims: bool = False
    include_negative_omissions: bool = False


@dataclass
class LLMConfig:
    model: str
    api_key: str
    base_url: str | None = None
    temperature: float = 0.0
    max_tokens: int = 8192
    timeout: float = 120.0
    max_retries: int = 2
    use_json_mode: bool = True
    include_images: bool = True
    image_stage: str = "all"
    max_llm_images: int = 4
    image_detail: str = "low"
    max_report_chars: int = 20000
    max_evidence_chars: int = 12000


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        text = value
    else:
        text = str(value)
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text)
    return text.strip()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def lower_contains(text: str, patterns: tuple[str, ...] | list[str]) -> bool:
    text_lower = text.lower()
    return any(re.search(pattern, text_lower) for pattern in patterns)


def split_report_sections(text: str) -> dict[str, str]:
    text = normalize_text(text)
    if not text:
        return {"Findings": "", "Impression": ""}

    pattern = re.compile(
        r"\b(findings?|impressions?|impression|conclusions?|conclusion)\s*:",
        re.IGNORECASE,
    )
    matches = list(pattern.finditer(text))
    if not matches:
        return {"Findings": text, "Impression": ""}

    sections = {"Findings": "", "Impression": ""}
    for idx, match in enumerate(matches):
        label = match.group(1).lower()
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        content = text[start:end].strip()
        if label.startswith("finding"):
            sections["Findings"] = join_text(sections["Findings"], content)
        else:
            sections["Impression"] = join_text(sections["Impression"], content)
    return sections


def join_text(*parts: str) -> str:
    clean = [normalize_text(part) for part in parts if normalize_text(part)]
    return "\n".join(clean)


def first_present(mapping: dict[str, Any], keys: tuple[str, ...] | list[str]) -> Any:
    for key in keys:
        if key in mapping and mapping[key] not in (None, "", [], {}):
            return mapping[key]
    return None


def get_nested_section(structured_report: Any, names: tuple[str, ...]) -> Any:
    if not isinstance(structured_report, dict):
        return None
    value = first_present(structured_report, names)
    if value is not None:
        return value
    sections = structured_report.get("sections")
    if isinstance(sections, dict):
        return first_present(sections, names)
    return None


def entry_to_text(entry: Any) -> str:
    if isinstance(entry, str):
        return normalize_text(entry)
    if isinstance(entry, dict):
        for key in TEXT_KEYS:
            value = entry.get(key)
            if isinstance(value, str) and normalize_text(value):
                return normalize_text(value)

        selected: list[str] = []
        for key in (
            "polarity",
            "severity",
            "laterality",
            "location",
            "location_detail",
            "anatomy",
            "core_finding",
            "descriptor",
            "measurement",
        ):
            value = entry.get(key)
            if isinstance(value, str) and normalize_text(value):
                selected.append(normalize_text(value))
        if selected:
            return normalize_text(" ".join(dict.fromkeys(selected)))

        fallback: list[str] = []
        for key, value in entry.items():
            key_lower = key.lower()
            if any(marker in key_lower for marker in ("id", "path", "image", "evidence")):
                continue
            if isinstance(value, str) and normalize_text(value):
                fallback.append(normalize_text(value))
        return normalize_text(". ".join(fallback))
    if isinstance(entry, list):
        return join_text(*(entry_to_text(item) for item in entry))
    return normalize_text(entry)


def extract_linked_evidence(entry: Any) -> list[Any]:
    if not isinstance(entry, dict):
        return []
    evidence: list[Any] = []
    for key in EVIDENCE_KEYS:
        value = entry.get(key)
        if value in (None, "", [], {}):
            continue
        if isinstance(value, list):
            evidence.extend(value)
        else:
            evidence.append(value)
    return evidence


def coerce_section_text_and_entries(value: Any) -> tuple[str, list[Any]]:
    if value is None:
        return "", []
    if isinstance(value, list):
        return join_text(*(entry_to_text(item) for item in value)), value
    if isinstance(value, dict):
        text = entry_to_text(value)
        entries = value.get("items") if isinstance(value.get("items"), list) else [value]
        return text, entries
    text = normalize_text(value)
    return text, []


def extract_generated_sections(result: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
    structured = result.get("structured_report")
    findings_value: Any = None
    impression_value: Any = None

    if isinstance(structured, dict):
        findings_value = get_nested_section(structured, ("findings", "Findings", "finding"))
        impression_value = get_nested_section(
            structured,
            ("impressions", "impression", "Impressions", "Impression"),
        )
    elif isinstance(structured, str):
        split = split_report_sections(structured)
        findings_value = split["Findings"]
        impression_value = split["Impression"]

    fallback_report = first_present(
        result,
        (
            "natural_language_report",
            "generated_report",
            "report",
            "final_report",
        ),
    )
    if fallback_report is None:
        fallback_report = first_present(
            summary,
            (
                "natural_language_report",
                "generated_report",
                "report",
                "final_report",
            ),
        )

    if (findings_value is None and impression_value is None) and fallback_report is not None:
        if isinstance(fallback_report, dict):
            findings_value = first_present(fallback_report, ("findings", "Findings"))
            impression_value = first_present(
                fallback_report,
                ("impression", "impressions", "Impression", "Impressions"),
            )
        else:
            split = split_report_sections(normalize_text(fallback_report))
            findings_value = split["Findings"]
            impression_value = split["Impression"]

    findings_text, findings_entries = coerce_section_text_and_entries(findings_value)
    impression_text, impression_entries = coerce_section_text_and_entries(impression_value)

    return {
        "Findings": {"text": findings_text, "entries": findings_entries},
        "Impression": {"text": impression_text, "entries": impression_entries},
    }


def extract_reference_sections(summary: dict[str, Any], result: dict[str, Any]) -> dict[str, str]:
    ground_truth = first_present(summary, ("ground_truth", "reference_report", "reference"))
    if ground_truth is None:
        ground_truth = first_present(result, ("ground_truth", "reference_report", "reference"))

    if isinstance(ground_truth, dict):
        return {
            "Findings": normalize_text(first_present(ground_truth, ("findings", "Findings"))),
            "Impression": normalize_text(
                first_present(
                    ground_truth,
                    ("impression", "impressions", "Impression", "Impressions"),
                )
            ),
        }
    return split_report_sections(normalize_text(ground_truth))


def split_claims(text: str) -> list[str]:
    text = normalize_text(text)
    if not text:
        return []
    text = re.sub(r"[\u2022\u2023\u25e6\u2043\u2219]", "\n", text)
    text = re.sub(r"\n\s*[-*]\s+", "\n", text)
    text = re.sub(r"\n\s*\d+[\).]\s+", "\n", text)
    text = re.sub(
        r"\b(findings?|impressions?|impression|conclusions?|conclusion)\s*:",
        "\n",
        text,
        flags=re.IGNORECASE,
    )

    chunks: list[str] = []
    for line in text.splitlines():
        line = normalize_text(line)
        if not line:
            continue
        parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])|;\s+", line)
        chunks.extend(parts)

    claims: list[str] = []
    for chunk in chunks:
        cleaned = re.sub(r"^\s*[-*]?\s*\d*[\).]?\s*", "", chunk).strip(" -")
        cleaned = normalize_text(cleaned)
        if len(cleaned) < 3:
            continue
        if cleaned.lower() in {"findings", "impression", "impressions", "conclusion"}:
            continue
        claims.append(cleaned)
    return claims


def token_normalize(token: str) -> str:
    token = token.lower()
    token = re.sub(r"[^a-z0-9.-]", "", token)
    token = token.strip(".-")
    if len(token) > 4 and token.endswith("ies"):
        token = token[:-3] + "y"
    elif len(token) > 4 and token.endswith("es") and not token.endswith("ses"):
        token = token[:-2]
    elif len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        token = token[:-1]
    return token


def text_tokens(text: str) -> set[str]:
    tokens = {token_normalize(token) for token in re.findall(r"[A-Za-z0-9.-]+", text)}
    return {token for token in tokens if token and token not in STOPWORDS and not token.isdigit()}


def find_first_term(text_lower: str, terms: tuple[str, ...] | list[str]) -> str | None:
    for term in terms:
        if re.search(rf"\b{re.escape(term)}\b", text_lower):
            return term
    return None


def extract_laterality(text_lower: str) -> str:
    if re.search(r"\b(bilateral|bilaterally|both)\b", text_lower):
        return "bilateral"
    if re.search(r"\bright\b", text_lower):
        return "right"
    if re.search(r"\bleft\b", text_lower):
        return "left"
    if re.search(r"\bmidline\b", text_lower):
        return "midline"
    return "none"


def extract_anatomy(text_lower: str) -> str | None:
    for term, canonical in ANATOMY_TERMS:
        if re.search(rf"\b{re.escape(term)}\b", text_lower):
            return canonical
    return None


def extract_location_detail(text_lower: str) -> str | None:
    patterns = (
        r"\b(right|left)\s+(upper|middle|lower)\s+lobe\b",
        r"\b(upper|middle|lower)\s+lobe\b",
        r"\b(right|left)\s+(frontal|parietal|temporal|occipital)\s+lobe\b",
        r"\b(frontal|parietal|temporal|occipital)\s+lobe\b",
        r"\b(apex|base|basilar|perihilar|retrocardiac|subpleural|central|peripheral)\b",
    )
    for pattern in patterns:
        match = re.search(pattern, text_lower)
        if match:
            return match.group(0)
    return None


def extract_measurement(text: str) -> str | None:
    match = re.search(
        r"\b\d+(?:\.\d+)?\s*(?:x|by)\s*\d+(?:\.\d+)?(?:\s*(?:x|by)\s*\d+(?:\.\d+)?)?\s*(?:mm|cm|mL|ml)?\b",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        return normalize_text(match.group(0))
    match = re.search(r"\b\d+(?:\.\d+)?\s*(?:mm|cm|mL|ml|cc|%)\b", text, flags=re.IGNORECASE)
    if match:
        return normalize_text(match.group(0))
    return None


def extract_certainty(text_lower: str) -> str:
    for certainty, patterns in CERTAINTY_PATTERNS.items():
        if lower_contains(text_lower, patterns):
            return certainty
    return "definite"


def has_normality_language(text_lower: str) -> bool:
    return bool(
        re.search(
            r"\b(normal|unremarkable|within normal limits|clear|patent|sharp costophrenic|no acute cardiopulmonary abnormality)\b",
            text_lower,
        )
    )


def starts_with_primary_negation(text_lower: str) -> bool:
    return bool(
        re.match(r"^\s*(no|without|negative for|absence of|absent|not)\b", text_lower)
        or re.match(r"^\s*there\s+(is|are)\s+no\b", text_lower)
    )


def detect_polarity(text_lower: str, core_finding: str | None, certainty: str) -> str:
    if certainty != "definite":
        return "uncertain"
    if starts_with_primary_negation(text_lower):
        return "negative"
    if core_finding:
        position = text_lower.find(core_finding)
        if position >= 0:
            before_core = text_lower[max(0, position - 48):position]
            if lower_contains(
                before_core,
                (
                    r"\bno\b",
                    r"\bwithout\b",
                    r"\bnegative for\b",
                    r"\babsent\b",
                    r"\bnot identified\b",
                    r"\bnot detected\b",
                    r"\bnot seen\b",
                ),
            ):
                return "negative"
    if re.search(r"\b(no evidence of|negative for|not identified|not detected|not seen)\b", text_lower):
        return "negative"
    if re.search(r"\b(clear|unremarkable|within normal limits|patent)\b", text_lower):
        return "negative"
    if has_normality_language(text_lower) and not re.search(
        r"\b(enlarged|borderline|fracture|fractures|mass|nodule|effusion|opacity|consolidation|pneumothorax|emphysema|edema|lesion|tumor)\b",
        text_lower,
    ):
        return "negative"
    return "positive"


def map_structured_claim_type(raw_type: str, claim: Claim) -> str:
    value = normalize_text(raw_type).lower().replace(" ", "_").replace("-", "_")
    text_lower = claim.raw_text.lower()
    if value in ALLOWED_CLAIM_TYPES:
        return value
    if value in {"normal", "normality", "normal_finding"}:
        claim.polarity = "negative"
        return "normality_claim"
    if value in {"abnormal", "abnormality", "finding", "positive", "present"}:
        if not starts_with_primary_negation(text_lower):
            claim.polarity = "positive"
        if claim.claim_type in {"measurement_claim", "comparison_claim", "certainty_claim"}:
            return claim.claim_type
        return "positive_finding"
    if value in {"diagnosis", "diagnostic", "impression"}:
        if claim.polarity == "negative":
            return "negative_finding"
        if claim.claim_type in {"recommendation_claim", "comparison_claim", "measurement_claim"}:
            return claim.claim_type
        return "diagnostic_inference"
    if value in {"recommendation", "recommend"}:
        if re.search(r"\b(recommend(?:ed|s|ing|ation)?|follow[- ]?up|advise(?:d|s)?|consider|should|suggest(?:ed|s|ing)?|clinical correlation|correlat(?:e|ion))\b", text_lower):
            return "recommendation_claim"
        return claim.claim_type if claim.claim_type in ALLOWED_CLAIM_TYPES else "positive_finding"
    if value in {"negative", "absent"}:
        claim.polarity = "negative"
        return "negative_finding"
    return claim.claim_type if claim.claim_type in ALLOWED_CLAIM_TYPES else "positive_finding"


def apply_structured_polarity(claim: Claim, raw_value: str) -> None:
    value = normalize_text(raw_value).lower().replace(" ", "_").replace("-", "_")
    if value in {"absent", "negative", "normal", "no", "none"}:
        claim.polarity = "negative"
    elif value in {"possible", "probable", "uncertain", "suspicious"}:
        claim.polarity = "uncertain"
    elif value in {"present", "positive", "abnormal", "yes"}:
        claim.polarity = "positive"


def extract_core_finding(text_lower: str) -> str | None:
    for term in FINDING_TERMS:
        if re.search(rf"\b{re.escape(term)}\b", text_lower):
            return term

    tokens = [
        token
        for token in text_tokens(text_lower)
        if token
        not in {
            "normal",
            "unremarkable",
            "acute",
            "mild",
            "moderate",
            "severe",
            "right",
            "left",
            "bilateral",
            "possible",
            "likely",
        }
    ]
    if not tokens:
        return None
    return " ".join(sorted(tokens)[:3])


def classify_claim_type(text_lower: str, polarity: str, measurement: str | None) -> str:
    if re.search(r"\b(recommend(?:ed|s|ing|ation)?|follow[- ]?up|advise(?:d|s)?|consider|should|suggest(?:ed|s|ing)?|clinical correlation|correlat(?:e|ion))\b", text_lower):
        return "recommendation_claim"
    if find_first_term(text_lower, TEMPORAL_TERMS):
        return "comparison_claim"
    if measurement:
        return "measurement_claim"
    if re.search(
        r"\b(compatible with|consistent with|concerning for|suggestive of|suspicious for|represents|favor|favored|secondary to|due to)\b",
        text_lower,
    ):
        if re.search(r"\b(due to|secondary to|etiology|cause)\b", text_lower):
            return "etiology_claim"
        return "diagnostic_inference"
    if polarity == "negative":
        if has_normality_language(text_lower):
            return "normality_claim"
        return "negative_finding"
    if re.search(r"\b(possible|probable|likely|suspicious|cannot exclude|may represent|may be)\b", text_lower):
        return "certainty_claim"
    return "positive_finding"


def extract_claim_fields(
    *,
    raw_text: str,
    case_id: str,
    source: str,
    section: str,
    claim_id: str,
    entry: Any = None,
) -> Claim:
    text_lower = raw_text.lower()
    certainty = extract_certainty(text_lower)
    core_finding = extract_core_finding(text_lower)
    polarity = detect_polarity(text_lower, core_finding, certainty)
    measurement = extract_measurement(raw_text)
    claim_type = classify_claim_type(text_lower, polarity, measurement)

    claim = Claim(
        claim_id=claim_id,
        case_id=case_id,
        source=source,
        section=section,
        raw_text=raw_text,
        claim_type=claim_type,
        core_finding=core_finding,
        polarity=polarity,
        anatomy=extract_anatomy(text_lower),
        laterality=extract_laterality(text_lower),
        location_detail=extract_location_detail(text_lower),
        severity=find_first_term(text_lower, SEVERITY_TERMS),
        measurement=measurement,
        descriptor=find_first_term(text_lower, DESCRIPTOR_TERMS),
        temporal=find_first_term(text_lower, TEMPORAL_TERMS),
        certainty=certainty,
        recommendation=raw_text if claim_type == "recommendation_claim" else None,
        linked_evidence=extract_linked_evidence(entry),
    )

    if isinstance(entry, dict):
        apply_structured_overrides(claim, entry)
    return claim


def apply_structured_overrides(claim: Claim, entry: dict[str, Any]) -> None:
    raw_type = first_present(entry, ("claim_type", "type", "category"))
    if isinstance(raw_type, str) and normalize_text(raw_type):
        claim.claim_type = map_structured_claim_type(raw_type, claim)

    raw_polarity = first_present(entry, ("polarity", "status", "presence"))
    if isinstance(raw_polarity, str) and normalize_text(raw_polarity):
        apply_structured_polarity(claim, raw_polarity)

    string_fields = {
        "core_finding": ("core_finding", "finding_name"),
        "anatomy": ("anatomy", "organ"),
        "laterality": ("laterality",),
        "location_detail": ("location_detail", "location"),
        "severity": ("severity",),
        "measurement": ("measurement", "size"),
        "descriptor": ("descriptor",),
        "temporal": ("temporal", "comparison"),
        "certainty": ("certainty",),
        "recommendation": ("recommendation",),
    }
    for field_name, keys in string_fields.items():
        value = first_present(entry, keys)
        if not isinstance(value, str) or not normalize_text(value):
            continue
        cleaned = normalize_text(value).lower()
        if field_name == "core_finding" and len(cleaned) > 80:
            continue
        setattr(claim, field_name, cleaned)

    if claim.claim_type not in ALLOWED_CLAIM_TYPES:
        claim.claim_type = classify_claim_type(claim.raw_text.lower(), claim.polarity, claim.measurement)
    if claim.claim_type in {"negative_finding", "normality_claim"}:
        claim.polarity = "negative"
    if claim.claim_type == "recommendation_claim":
        claim.recommendation = claim.raw_text
    elif not first_present(entry, ("recommendation",)):
        claim.recommendation = None

def build_claims_from_entries(
    *,
    entries: list[Any],
    section_text: str,
    case_id: str,
    source: str,
    section: str,
    prefix: str,
) -> list[Claim]:
    claims: list[Claim] = []
    counter = 1

    if entries:
        for entry in entries:
            text = entry_to_text(entry)
            if not text:
                continue
            fragments = split_claims(text)
            if not fragments:
                fragments = [text]
            for fragment in fragments:
                claim_id = f"{prefix}_{counter:03d}"
                claims.append(
                    extract_claim_fields(
                        raw_text=fragment,
                        case_id=case_id,
                        source=source,
                        section=section,
                        claim_id=claim_id,
                        entry=entry,
                    )
                )
                counter += 1
        return claims

    for fragment in split_claims(section_text):
        claim_id = f"{prefix}_{counter:03d}"
        claims.append(
            extract_claim_fields(
                raw_text=fragment,
                case_id=case_id,
                source=source,
                section=section,
                claim_id=claim_id,
            )
        )
        counter += 1
    return claims


def case_id_from(sample_dir: Path, summary: dict[str, Any], result: dict[str, Any]) -> str:
    for container in (summary, result):
        value = first_present(container, ("case_id", "sample_id", "study_id", "id"))
        if value is not None:
            return str(value)
    return sample_dir.name


def build_case_claims(
    sample_dir: Path,
    summary: dict[str, Any],
    result: dict[str, Any],
) -> tuple[str, dict[str, str], dict[str, str], list[Claim], list[Claim]]:
    case_id = case_id_from(sample_dir, summary, result)
    generated_sections = extract_generated_sections(result, summary)
    reference_sections = extract_reference_sections(summary, result)

    generated_claims: list[Claim] = []
    generated_claims.extend(
        build_claims_from_entries(
            entries=generated_sections["Findings"]["entries"],
            section_text=generated_sections["Findings"]["text"],
            case_id=case_id,
            source="generated",
            section="Findings",
            prefix="G_F",
        )
    )
    generated_claims.extend(
        build_claims_from_entries(
            entries=generated_sections["Impression"]["entries"],
            section_text=generated_sections["Impression"]["text"],
            case_id=case_id,
            source="generated",
            section="Impression",
            prefix="G_I",
        )
    )

    reference_claims: list[Claim] = []
    reference_claims.extend(
        build_claims_from_entries(
            entries=[],
            section_text=reference_sections["Findings"],
            case_id=case_id,
            source="reference",
            section="Findings",
            prefix="R_F",
        )
    )
    reference_claims.extend(
        build_claims_from_entries(
            entries=[],
            section_text=reference_sections["Impression"],
            case_id=case_id,
            source="reference",
            section="Impression",
            prefix="R_I",
        )
    )

    generated_report = {
        "findings": generated_sections["Findings"]["text"],
        "impression": generated_sections["Impression"]["text"],
    }
    reference_report = {
        "findings": reference_sections["Findings"],
        "impression": reference_sections["Impression"],
    }
    return case_id, generated_report, reference_report, generated_claims, reference_claims


def claim_similarity(left: Claim, right: Claim, *, ignore_polarity: bool = False) -> float:
    left_tokens = text_tokens(left.raw_text)
    right_tokens = text_tokens(right.raw_text)
    union = left_tokens | right_tokens
    jaccard = len(left_tokens & right_tokens) / len(union) if union else 0.0

    score = 0.35 * jaccard
    if left.core_finding and right.core_finding:
        if left.core_finding == right.core_finding:
            score += 0.35
        elif token_overlap(left.core_finding, right.core_finding) >= 0.5:
            score += 0.2
    if left.anatomy and right.anatomy:
        if left.anatomy == right.anatomy:
            score += 0.15
        elif token_overlap(left.anatomy, right.anatomy) >= 0.5:
            score += 0.08
    if left.location_detail and right.location_detail and left.location_detail == right.location_detail:
        score += 0.08
    if not ignore_polarity and left.polarity == right.polarity:
        score += 0.05
    if left.laterality != "none" and right.laterality != "none":
        if left.laterality == right.laterality:
            score += 0.05
        elif {left.laterality, right.laterality} <= {"left", "right"}:
            score -= 0.12
    return max(0.0, min(1.0, score))


def token_overlap(left: str, right: str) -> float:
    left_tokens = text_tokens(left)
    right_tokens = text_tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / min(len(left_tokens), len(right_tokens))


def best_match(claim: Claim, candidates: list[Claim], *, ignore_polarity: bool = False) -> Match:
    best = Match(None, 0.0)
    for candidate in candidates:
        score = claim_similarity(claim, candidate, ignore_polarity=ignore_polarity)
        if score > best.score:
            best = Match(candidate, score)
    return best


def has_polarity_conflict(left: Claim, right: Claim) -> bool:
    if left.polarity == right.polarity:
        return False
    if "uncertain" in {left.polarity, right.polarity}:
        return False
    return same_entity(left, right)


def same_entity(left: Claim, right: Claim) -> bool:
    core_ok = bool(left.core_finding and right.core_finding and token_overlap(left.core_finding, right.core_finding) >= 0.5)
    anatomy_ok = not left.anatomy or not right.anatomy or token_overlap(left.anatomy, right.anatomy) >= 0.5
    return core_ok and anatomy_ok


def location_conflict(left: Claim, right: Claim) -> bool:
    if not same_entity(left, right):
        return False
    if left.laterality != "none" and right.laterality != "none":
        if {left.laterality, right.laterality} <= {"left", "right"} and left.laterality != right.laterality:
            return True
    if left.location_detail and right.location_detail and left.location_detail != right.location_detail:
        return token_overlap(left.location_detail, right.location_detail) < 0.5
    return False


def attribute_conflict(left: Claim, right: Claim) -> bool:
    if not same_entity(left, right):
        return False
    for field_name in ("severity", "descriptor"):
        left_value = getattr(left, field_name)
        right_value = getattr(right, field_name)
        if left_value and right_value and left_value != right_value:
            return True
    if left.measurement and right.measurement and normalize_measurement(left.measurement) != normalize_measurement(right.measurement):
        return True
    return False


def temporal_conflict(left: Claim, right: Claim) -> bool:
    if not same_entity(left, right):
        return False
    if left.temporal and right.temporal and left.temporal != right.temporal:
        opposite_groups = (
            {"increased", "increase", "worsened", "worsening", "progression"},
            {"decreased", "decrease", "improved", "improving", "resolved", "resolution"},
            {"stable", "unchanged"},
            {"new"},
        )
        for group in opposite_groups:
            if left.temporal in group and right.temporal in group:
                return False
        return True
    return False


def certainty_conflict(left: Claim, right: Claim) -> bool:
    if not same_entity(left, right):
        return False
    uncertain_values = {"possible", "probable", "suspicious", "cannot_exclude", "uncertain"}
    if left.certainty == right.certainty:
        return False
    return left.certainty == "definite" and right.certainty in uncertain_values


def normalize_measurement(value: str) -> str:
    return re.sub(r"\s+", "", value.lower())


def evaluate_generated_claim(
    claim: Claim,
    reference_claims: list[Claim],
    generated_findings: list[Claim],
    config: DetectorConfig,
) -> dict[str, Any]:
    match = best_match(claim, reference_claims)
    loose_match = best_match(claim, reference_claims, ignore_polarity=True)
    hallucination_types: list[str] = []
    reasons: list[str] = []
    support_status = "not_entailed"
    matched = match if match.score >= config.min_match_score else Match(None, 0.0)

    conflict_candidate = loose_match.claim if loose_match.score >= config.min_match_score else None
    if matched.claim:
        support_status = "entailed"
        conflict_candidate = matched.claim

    if conflict_candidate and has_polarity_conflict(claim, conflict_candidate):
        support_status = "contradicted"
        hallucination_types.append(HALLUCINATION_TYPES["H3"])
        reasons.append("Generated claim polarity conflicts with the matched reference claim.")
    elif conflict_candidate and location_conflict(claim, conflict_candidate):
        support_status = "contradicted"
        hallucination_types.append(HALLUCINATION_TYPES["H4"])
        reasons.append("Generated claim location or laterality conflicts with the matched reference claim.")
    elif conflict_candidate and temporal_conflict(claim, conflict_candidate):
        support_status = "contradicted"
        hallucination_types.append(HALLUCINATION_TYPES["H6"])
        reasons.append("Generated comparison or temporal wording conflicts with the reference claim.")
    elif conflict_candidate and attribute_conflict(claim, conflict_candidate):
        support_status = "contradicted"
        hallucination_types.append(HALLUCINATION_TYPES["H5"])
        reasons.append("Generated severity, measurement, or descriptor conflicts with the reference claim.")
    elif conflict_candidate and certainty_conflict(claim, conflict_candidate):
        support_status = "contradicted"
        hallucination_types.append(HALLUCINATION_TYPES["H9"])
        reasons.append("Generated claim is more certain than the matched reference claim.")
    elif not matched.claim:
        hallucination_types.extend(classify_unmatched_generated_claim(claim, config))
        if hallucination_types:
            reasons.append("No sufficiently similar supporting reference claim was found.")
        else:
            support_status = "unknown"
            reasons.append("No matching reference claim found; conservative settings do not mark this claim erroneous.")

    if claim.section == "Impression":
        inconsistency_reason = impression_inconsistency_reason(claim, generated_findings, config)
        if inconsistency_reason:
            if HALLUCINATION_TYPES["H8"] not in hallucination_types:
                hallucination_types.append(HALLUCINATION_TYPES["H8"])
            reasons.append(inconsistency_reason)
            if support_status == "entailed":
                support_status = "partially_entailed"

    primary_type = hallucination_types[0] if hallucination_types else None
    matched_claim = matched.claim or (conflict_candidate if conflict_candidate else None)
    matched_score = matched.score if matched.claim else (loose_match.score if conflict_candidate else 0.0)
    return {
        "claim_id": claim.claim_id,
        "section": claim.section,
        "raw_text": claim.raw_text,
        "claim_type": claim.claim_type,
        "support_status": support_status,
        "is_error": bool(hallucination_types),
        "hallucination_types": hallucination_types,
        "primary_hallucination_type": primary_type,
        "matched_reference_claim_id": matched_claim.claim_id if matched_claim else None,
        "matched_reference_text": matched_claim.raw_text if matched_claim else None,
        "match_score": round(matched_score, 4),
        "reasons": reasons,
    }


def classify_unmatched_generated_claim(claim: Claim, config: DetectorConfig) -> list[str]:
    if claim.claim_type == "recommendation_claim":
        return [HALLUCINATION_TYPES["H10"]]
    if claim.claim_type in {"diagnostic_inference", "etiology_claim"}:
        return [HALLUCINATION_TYPES["H7"]]
    if claim.claim_type == "comparison_claim":
        return [HALLUCINATION_TYPES["H6"]]
    if claim.claim_type == "certainty_claim" and claim.certainty == "definite":
        return [HALLUCINATION_TYPES["H9"]]
    if claim.polarity == "positive" and claim.claim_type in {
        "positive_finding",
        "attribute_claim",
        "measurement_claim",
        "impression_summary",
        "certainty_claim",
    }:
        return [HALLUCINATION_TYPES["H1"]]
    if config.strict_negative_claims and claim.claim_type in {"negative_finding", "normality_claim"}:
        return [HALLUCINATION_TYPES["H1"]]
    return []


def impression_inconsistency_reason(
    claim: Claim,
    generated_findings: list[Claim],
    config: DetectorConfig,
) -> str | None:
    if not generated_findings:
        return None
    if claim.claim_type in {"negative_finding", "normality_claim"} and claim.polarity == "negative":
        positive_findings = [item for item in generated_findings if item.polarity == "positive"]
        conflict = best_match(claim, positive_findings, ignore_polarity=True)
        if conflict.claim and conflict.score >= config.impression_match_score and has_polarity_conflict(claim, conflict.claim):
            return "Generated impression negates a finding described as positive in generated findings."
        return None

    if claim.claim_type not in {
        "positive_finding",
        "diagnostic_inference",
        "etiology_claim",
        "comparison_claim",
        "measurement_claim",
        "certainty_claim",
        "recommendation_claim",
    }:
        return None

    match = best_match(claim, generated_findings, ignore_polarity=True)
    if match.claim and match.score >= config.impression_match_score:
        if has_polarity_conflict(claim, match.claim):
            return "Generated impression polarity conflicts with generated findings."
        return None
    return "Generated impression claim is not supported by generated findings."


def is_omission_candidate(claim: Claim, config: DetectorConfig) -> bool:
    if claim.claim_type in {"positive_finding", "diagnostic_inference", "etiology_claim", "comparison_claim", "measurement_claim"}:
        return claim.polarity != "negative"
    if claim.claim_type == "recommendation_claim":
        return True
    if config.include_negative_omissions and claim.claim_type in {"negative_finding", "normality_claim"}:
        return True
    return False


def detect_omissions(
    reference_claims: list[Claim],
    generated_claims: list[Claim],
    config: DetectorConfig,
) -> list[dict[str, Any]]:
    omissions: list[dict[str, Any]] = []
    for reference_claim in reference_claims:
        if not is_omission_candidate(reference_claim, config):
            continue
        match = best_match(reference_claim, generated_claims, ignore_polarity=True)
        if match.claim and match.score >= config.omission_match_score and not has_polarity_conflict(reference_claim, match.claim):
            continue
        omissions.append(
            {
                "reference_claim_id": reference_claim.claim_id,
                "section": reference_claim.section,
                "raw_text": reference_claim.raw_text,
                "claim_type": reference_claim.claim_type,
                "hallucination_type": HALLUCINATION_TYPES["H2"],
                "matched_generated_claim_id": match.claim.claim_id if match.claim else None,
                "matched_generated_text": match.claim.raw_text if match.claim else None,
                "match_score": round(match.score, 4),
                "reason": "Clinically relevant reference claim was not matched by the generated report.",
            }
        )
    return omissions


def load_image_index(sample_dir: Path) -> dict[str, Any]:
    indexes: dict[str, Any] = {}
    for path in (sample_dir.parent / "image_index.json", sample_dir / "image_index.json"):
        if not path.exists() or not path.is_file():
            continue
        try:
            data = load_json(path)
        except Exception:
            continue
        indexes.update(data)
    return indexes


def image_id_from_path(path: Path) -> str:
    if path.name.lower().endswith(".nii.gz"):
        return path.name[:-7]
    return path.stem


def collect_image_manifest(sample_dir: Path, max_images: int) -> list[dict[str, Any]]:
    images_dir = sample_dir / "images"
    image_index = load_image_index(sample_dir)
    if not images_dir.exists() or not images_dir.is_dir() or max_images == 0:
        return []

    manifest: list[dict[str, Any]] = []
    for path in sorted(images_dir.rglob("*")):
        if not path.is_file():
            continue
        suffix = "".join(path.suffixes[-2:]).lower() if path.name.lower().endswith(".nii.gz") else path.suffix.lower()
        if suffix not in SUPPORTED_IMAGE_EXTENSIONS:
            continue
        image_id = image_id_from_path(path)
        indexed = image_index.get(image_id, {}) if isinstance(image_index, dict) else {}
        description = indexed.get("description") if isinstance(indexed, dict) else None
        stat = path.stat()
        item: dict[str, Any] = {
            "image_id": image_id,
            "path": str(path.relative_to(sample_dir)),
            "bytes": stat.st_size,
            "description": normalize_text(description),
        }
        if isinstance(indexed, dict):
            if indexed.get("tool_chain") is not None:
                item["tool_chain"] = indexed.get("tool_chain")
            if indexed.get("size") is not None:
                item["size"] = indexed.get("size")
            if indexed.get("metadata") is not None:
                item["metadata"] = indexed.get("metadata")
        manifest.append(item)
        if len(manifest) >= max_images:
            break
    return manifest

def log_progress(message: str, *, enabled: bool = True) -> None:
    if not enabled:
        return
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


SUPPORT_STATUSES = {
    "entailed",
    "partially_entailed",
    "not_entailed",
    "contradicted",
    "unverifiable",
}

RECOMMENDATION_STATUSES = {
    "supported",
    "partially_supported",
    "unsupported",
    "contradicted",
    "unverifiable",
}

CLINICAL_SEVERITIES = {"critical", "major", "minor", "negligible", None}

PRIMARY_PRIORITY = [
    HALLUCINATION_TYPES["H3"],
    HALLUCINATION_TYPES["H8"],
    HALLUCINATION_TYPES["H1"],
    HALLUCINATION_TYPES["H2"],
    HALLUCINATION_TYPES["H4"],
    HALLUCINATION_TYPES["H6"],
    HALLUCINATION_TYPES["H5"],
    HALLUCINATION_TYPES["H7"],
    HALLUCINATION_TYPES["H9"],
    HALLUCINATION_TYPES["H10"],
]

VALID_H_TYPES = set(HALLUCINATION_TYPES.values())

CLAIM_EXTRACTION_PROMPT = """You are an expert radiology information extraction system.

Extract atomic clinical claims from the following radiology report section.

Rules:
1. Each claim should contain one clinical assertion only.
2. Split compound sentences into separate claims.
3. Preserve negation, uncertainty, laterality, location, severity, measurement, and comparison.
4. Classify each claim into one of:
   positive_finding, negative_finding, attribute_claim, measurement_claim,
   comparison_claim, impression_summary, diagnostic_inference,
   etiology_claim, certainty_claim, recommendation_claim, normality_claim.
5. Return valid JSON only.

Input section:
Section name: {section_name}
Text:
{section_text}

Return this JSON object only:
{{
  "claims": [
    {{
      "raw_text": "...",
      "claim_type": "...",
      "core_finding": "...",
      "polarity": "positive | negative | uncertain",
      "anatomy": "...",
      "laterality": "left | right | bilateral | midline | none | unknown",
      "location_detail": "...",
      "severity": "...",
      "measurement": "...",
      "descriptor": "...",
      "temporal": "...",
      "certainty": "...",
      "recommendation": "..."
    }}
  ]
}}
"""

SUPPORT_VERIFICATION_PROMPT = """You are an expert radiology report evaluator.

Your task is to determine whether a generated clinical claim is supported by the provided reference/evidence.

Generated claim:
{claim}

Reference/evidence:
{premise}

Definitions:
- entailed: the evidence fully supports the claim.
- partially_entailed: the evidence supports part of the claim but not all details, certainty, location, measurement, or interpretation.
- not_entailed: the evidence does not support the claim.
- contradicted: the evidence explicitly contradicts the claim.
- unverifiable: the evidence is insufficient to judge.

Return valid JSON only:
{{
  "support_status": "entailed | partially_entailed | not_entailed | contradicted | unverifiable",
  "supporting_evidence": ["..."],
  "brief_reason": "..."
}}
"""

INTERNAL_CONSISTENCY_PROMPT = """You are an expert radiology report evaluator.

Determine whether the Impression claim is supported by the generated Findings.

Generated Findings claims:
{findings_claims}

Generated Impression claim:
{impression_claim}

Return valid JSON only:
{{
  "support_status": "entailed | partially_entailed | not_entailed | contradicted | unverifiable",
  "contradicting_finding_claims": ["..."],
  "brief_reason": "..."
}}
"""

RECOMMENDATION_SUPPORT_PROMPT = """You are an expert radiology report evaluator.

Determine whether the generated recommendation is clinically supported.

Recommendation claim:
{recommendation_claim}

Available findings:
{findings_claims}

Patient context:
{patient_context}

Guideline or medical evidence:
{medical_evidence}

Return valid JSON only:
{{
  "recommendation_status": "supported | partially_supported | unsupported | contradicted | unverifiable",
  "supporting_evidence": ["..."],
  "brief_reason": "..."
}}
"""

TAXONOMY_CLASSIFICATION_PROMPT = """You are an expert radiology report hallucination evaluator.

Classify claim-level medical hallucinations/errors for one radiology report generation case.
Use the H1-H10 taxonomy below and the provided LLM stage outputs.

Hallucination taxonomy:
- H1_fabricated_finding: generated abnormality/finding is not supported by reference or evidence.
- H2_omitted_finding: clinically relevant positive reference finding is not mentioned by generated report. H2 is an omission event, not a generated-claim label.
- H3_negation_error: generated and reference claims share the same finding/anatomy but opposite polarity.
- H4_location_or_laterality_error: correct finding but wrong organ, side, lobe, or clinically meaningful location.
- H5_attribute_error: correct finding but wrong size, number, severity, extent, morphology, density, signal, margin, distribution, or measurement.
- H6_comparison_error: wrong temporal comparison such as new/stable/increased/decreased/resolved.
- H7_unsupported_diagnostic_inference: diagnosis, etiology, causal claim, or overinterpretation is not supported by findings/evidence.
- H8_findings_impression_inconsistency: generated Impression contradicts or is not supported by generated Findings.
- H9_certainty_calibration_error: certainty is too strong or too weak relative to reference/evidence.
- H10_unsupported_recommendation: recommendation is unsupported by findings, patient context, guidelines, or accepted practice.

Primary type priority:
1. H3_negation_error
2. H8_findings_impression_inconsistency
3. H1_fabricated_finding
4. H2_omitted_finding
5. H4_location_or_laterality_error
6. H6_comparison_error
7. H5_attribute_error
8. H7_unsupported_diagnostic_inference
9. H9_certainty_calibration_error
10. H10_unsupported_recommendation

Rules:
- Return one claim_level_results item for every generated claim id, exactly once.
- Put omissions only in omission_events, not as H2 labels on generated claims.
- Do not double count a laterality/location mismatch as H1 plus H2; use H4.
- Do not double count overinterpretation of a matched finding as H1; use H7.
- Use Stage A support status to decide whether a generated claim is supported, but assign H-types using the taxonomy rules.
- Stage B CRIMSON-style findings results are deterministic pre-classification results. Preserve their generated-claim H-types and H2 omission events unless the relevant generated claim is unverifiable.
- Image evidence contains image_id, description, and attached image blocks. Use both the image pixels and the image descriptions when available.
- Use internal consistency results for H8.
- Use recommendation support results for H10.
- Use clinical severity labels only from: critical, major, minor, negligible. Use null if uncertain.
- If support_status is unverifiable and there is no contradiction or clear taxonomy error, do not count it as an error.

Case input JSON:
{case_payload}

Return valid JSON only with this schema:
{{
  "claim_level_results": [
    {{
      "claim_id": "G_F_001",
      "section": "Findings | Impression",
      "claim_text": "...",
      "claim_type": "...",
      "support_status": "entailed | partially_entailed | not_entailed | contradicted | unverifiable",
      "is_error": true,
      "primary_hallucination_type": "H1_fabricated_finding | ... | null",
      "hallucination_types": ["H1_fabricated_finding"],
      "matched_reference_claims": ["R_F_001"],
      "evidence_used": ["..."],
      "clinical_severity": "critical | major | minor | negligible | null",
      "brief_reason": "...",
      "error_details": {{}}
    }}
  ],
  "omission_events": [
    {{
      "reference_claim_id": "R_F_001",
      "reference_claim_text": "...",
      "hallucination_type": "H2_omitted_finding",
      "matched_generated_claims": ["G_F_001"],
      "clinical_severity": "critical | major | minor | negligible | null",
      "brief_reason": "..."
    }}
  ]
}}
"""


def compact_json(value: Any, max_chars: int = 12000) -> str:
    return json.dumps(truncate_value(value, max_chars), ensure_ascii=False, indent=2)


def truncate_value(value: Any, max_chars: int) -> Any:
    if isinstance(value, str):
        text = normalize_text(value)
        if len(text) > max_chars:
            return text[:max_chars] + "... [truncated]"
        return text
    if isinstance(value, list):
        return [truncate_value(item, max_chars) for item in value]
    if isinstance(value, dict):
        return {str(key): truncate_value(item, max_chars) for key, item in value.items()}
    return value


def parse_json_response(raw: str) -> Any:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    candidates: list[str] = []
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        end = text.rfind(closer)
        if start != -1 and end > start:
            candidates.append(text[start:end + 1])
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    raise ValueError(f"LLM did not return valid JSON: {raw[:500]}")


class OpenAIBackbone:
    def __init__(self, config: LLMConfig):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("The openai package is required for LLM-backed hallucination detection.") from exc

        kwargs: dict[str, Any] = {
            "api_key": config.api_key,
            "timeout": config.timeout,
            "max_retries": config.max_retries,
        }
        if config.base_url:
            kwargs["base_url"] = config.base_url
        self.client = OpenAI(**kwargs)
        self.config = config

    def call_json(self, prompt: str, *, image_paths: list[Path] | None = None) -> Any:
        messages = [
            {"role": "system", "content": "You are a careful radiology evaluation system. Return valid JSON only."},
            {"role": "user", "content": self._content(prompt, image_paths or [])},
        ]
        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        if self.config.use_json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        last_error: Exception | None = None
        for attempt in range(self.config.max_retries + 1):
            try:
                response = self.client.chat.completions.create(**kwargs)
                content = response.choices[0].message.content
                if not content:
                    raise ValueError("LLM returned an empty message")
                return parse_json_response(content)
            except Exception as exc:  # noqa: BLE001 - surface final retry context.
                last_error = exc
                if attempt >= self.config.max_retries:
                    break
                time.sleep(min(2 ** attempt, 8))
        raise RuntimeError(f"LLM JSON call failed after retries: {last_error}")

    def _content(self, prompt: str, image_paths: list[Any]) -> str | list[dict[str, Any]]:
        if not image_paths:
            return prompt
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for index, image_item in enumerate(image_paths, start=1):
            image_path = image_item.get("path") if isinstance(image_item, dict) else image_item
            if not isinstance(image_path, Path):
                image_path = Path(image_path)
            if isinstance(image_item, dict):
                description = normalize_text(image_item.get("description"))
                image_id = normalize_text(image_item.get("image_id")) or image_path.stem
                rel_path = normalize_text(image_item.get("relative_path")) or image_path.name
            else:
                description = ""
                image_id = image_path.stem
                rel_path = image_path.name
            content.append({
                "type": "text",
                "text": f"Image {index}: image_id={image_id}; path={rel_path}; description={description or 'No description provided.'}",
            })
            block = image_path_to_content_block(image_path, detail=self.config.image_detail)
            if block is not None:
                content.append(block)
        return content


def image_path_to_content_block(path: Path, *, detail: str) -> dict[str, Any] | None:
    suffix = "".join(path.suffixes[-2:]).lower() if path.name.lower().endswith(".nii.gz") else path.suffix.lower()
    if suffix not in {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".webp"}:
        return None
    mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:{mime_type};base64,{data}",
            "detail": detail,
        },
    }


def llm_image_paths(sample_dir: Path, image_manifest: list[dict[str, Any]], max_images: int) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for item in image_manifest:
        rel_path = item.get("path")
        if not isinstance(rel_path, str):
            continue
        path = sample_dir / rel_path
        suffix = "".join(path.suffixes[-2:]).lower() if path.name.lower().endswith(".nii.gz") else path.suffix.lower()
        if suffix not in {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".webp"}:
            continue
        if path.exists() and path.is_file():
            llm_item = dict(item)
            llm_item["path"] = path
            llm_item["relative_path"] = rel_path
            items.append(llm_item)
        if len(items) >= max_images:
            break
    return items

def coerce_list(value: Any) -> list[Any]:
    if value in (None, "", [], {}):
        return []
    if isinstance(value, list):
        return value
    return [value]


def collect_optional_evidence(
    summary: dict[str, Any],
    result: dict[str, Any],
    image_manifest: list[dict[str, Any]],
    *,
    max_chars: int,
) -> dict[str, Any]:
    evidence: dict[str, Any] = {
        "image_evidence": [
            {
                "image_id": item.get("image_id"),
                "path": item.get("path"),
                "description": item.get("description"),
                "tool_chain": item.get("tool_chain"),
                "size": item.get("size"),
                "metadata": item.get("metadata"),
            }
            for item in image_manifest
        ],
        "retrieved_medical_evidence": [],
        "similar_case_evidence": [],
        "prior_report": None,
        "patient_context": {},
        "image_manifest": image_manifest,
    }
    for key in ("retrieved_medical_evidence", "similar_case_evidence"):
        evidence[key].extend(coerce_list(summary.get(key)))
        evidence[key].extend(coerce_list(result.get(key)))
    for key in ("prior_report", "patient_context"):
        evidence[key] = summary.get(key) or result.get(key) or evidence[key]
    return truncate_value(evidence, max_chars)

def collect_structured_evidence(value: Any) -> list[Any]:
    collected: list[Any] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key in EVIDENCE_KEYS:
                collected.extend(coerce_list(item))
            elif key in {"findings", "impressions", "sections"} or isinstance(item, (dict, list)):
                collected.extend(collect_structured_evidence(item))
    elif isinstance(value, list):
        for item in value:
            collected.extend(collect_structured_evidence(item))
    return collected[:100]


def claim_from_llm_item(
    item: dict[str, Any],
    *,
    claim_id: str,
    case_id: str,
    source: str,
    section: str,
) -> Claim:
    raw_text = normalize_text(item.get("raw_text") or item.get("claim_text") or item.get("text"))
    if not raw_text:
        raise ValueError(f"LLM claim item is missing raw_text: {item}")
    claim = extract_claim_fields(
        raw_text=raw_text,
        case_id=case_id,
        source=source,
        section=section,
        claim_id=claim_id,
    )
    for field_name in (
        "core_finding",
        "anatomy",
        "laterality",
        "location_detail",
        "severity",
        "measurement",
        "descriptor",
        "temporal",
        "certainty",
        "recommendation",
    ):
        value = item.get(field_name)
        if isinstance(value, str) and normalize_text(value):
            setattr(claim, field_name, normalize_text(value))
        elif value is None:
            setattr(claim, field_name, None if field_name != "laterality" else claim.laterality)
    claim_type = normalize_text(item.get("claim_type")).lower()
    if claim_type in ALLOWED_CLAIM_TYPES:
        claim.claim_type = claim_type
    polarity = normalize_text(item.get("polarity")).lower()
    if polarity in {"positive", "negative", "uncertain"}:
        claim.polarity = polarity
    if claim.claim_type in {"negative_finding", "normality_claim"}:
        claim.polarity = "negative"
    if claim.claim_type == "recommendation_claim" and not claim.recommendation:
        claim.recommendation = claim.raw_text
    return claim


def llm_extract_section_claims(
    llm: OpenAIBackbone,
    *,
    case_id: str,
    source: str,
    section: str,
    text: str,
    prefix: str,
) -> list[Claim]:
    section_text = normalize_text(text)
    if not section_text:
        return []
    prompt = CLAIM_EXTRACTION_PROMPT.format(section_name=section, section_text=section_text)
    response = llm.call_json(prompt)
    raw_claims = response.get("claims") if isinstance(response, dict) else response
    if not isinstance(raw_claims, list):
        raise ValueError(f"Claim extraction returned no claims list for {source} {section}: {response}")
    claims: list[Claim] = []
    for index, item in enumerate(raw_claims, start=1):
        if not isinstance(item, dict):
            continue
        claims.append(
            claim_from_llm_item(
                item,
                claim_id=f"{prefix}_{index:03d}",
                case_id=case_id,
                source=source,
                section=section,
            )
        )
    if not claims and section_text:
        raise ValueError(f"LLM returned zero claims for non-empty {source} {section}")
    return claims


def llm_extract_all_claims(
    llm: OpenAIBackbone,
    *,
    case_id: str,
    generated_report: dict[str, str],
    reference_report: dict[str, str],
) -> tuple[list[Claim], list[Claim]]:
    generated_claims: list[Claim] = []
    reference_claims: list[Claim] = []
    generated_claims.extend(
        llm_extract_section_claims(
            llm,
            case_id=case_id,
            source="generated",
            section="Findings",
            text=generated_report.get("findings", ""),
            prefix="G_F",
        )
    )
    generated_claims.extend(
        llm_extract_section_claims(
            llm,
            case_id=case_id,
            source="generated",
            section="Impression",
            text=generated_report.get("impression", ""),
            prefix="G_I",
        )
    )
    reference_claims.extend(
        llm_extract_section_claims(
            llm,
            case_id=case_id,
            source="reference",
            section="Findings",
            text=reference_report.get("findings", ""),
            prefix="R_F",
        )
    )
    reference_claims.extend(
        llm_extract_section_claims(
            llm,
            case_id=case_id,
            source="reference",
            section="Impression",
            text=reference_report.get("impression", ""),
            prefix="R_I",
        )
    )
    return generated_claims, reference_claims


def claim_prompt_dict(claim: Claim) -> dict[str, Any]:
    return {
        "claim_id": claim.claim_id,
        "section": claim.section,
        "raw_text": claim.raw_text,
        "claim_type": claim.claim_type,
        "core_finding": claim.core_finding,
        "polarity": claim.polarity,
        "anatomy": claim.anatomy,
        "laterality": claim.laterality,
        "location_detail": claim.location_detail,
        "severity": claim.severity,
        "measurement": claim.measurement,
        "descriptor": claim.descriptor,
        "temporal": claim.temporal,
        "certainty": claim.certainty,
        "recommendation": claim.recommendation,
    }


def sanitize_support_result(response: Any, claim_id: str) -> dict[str, Any]:
    if not isinstance(response, dict):
        raise ValueError(f"Support verification for {claim_id} returned non-object JSON")
    status = normalize_text(response.get("support_status")).lower()
    if status not in SUPPORT_STATUSES:
        status = "unverifiable"
    supporting_evidence = response.get("supporting_evidence")
    if not isinstance(supporting_evidence, list):
        supporting_evidence = coerce_list(supporting_evidence)
    return {
        "claim_id": claim_id,
        "support_status": status,
        "supporting_evidence": [normalize_text(item) for item in supporting_evidence if normalize_text(item)],
        "brief_reason": normalize_text(response.get("brief_reason")),
    }


def llm_verify_support(
    llm: OpenAIBackbone,
    *,
    generated_claims: list[Claim],
    reference_claims: list[Claim],
    generated_report: dict[str, str],
    reference_report: dict[str, str],
    evidence: dict[str, Any],
    image_paths: list[Path],
) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    reference_payload = [claim_prompt_dict(claim) for claim in reference_claims]
    for claim in generated_claims:
        premise = {
            "reference_claims": reference_payload,
            "reference_report": reference_report,
            "generated_report_context": generated_report,
            "image_evidence": evidence.get("image_evidence", []),
            "retrieved_medical_evidence": evidence.get("retrieved_medical_evidence", []),
            "similar_case_evidence": evidence.get("similar_case_evidence", []),
            "prior_report": evidence.get("prior_report"),
            "patient_context": evidence.get("patient_context", {}),
            "image_manifest": evidence.get("image_manifest", []),
        }
        prompt = SUPPORT_VERIFICATION_PROMPT.format(
            claim=compact_json(claim_prompt_dict(claim)),
            premise=compact_json(premise),
        )
        attach_images = image_paths if llm.config.include_images and llm.config.image_stage == "all" else []
        response = llm.call_json(prompt, image_paths=attach_images)
        results[claim.claim_id] = sanitize_support_result(response, claim.claim_id)
    return results


def llm_internal_consistency(
    llm: OpenAIBackbone,
    *,
    generated_claims: list[Claim],
) -> dict[str, dict[str, Any]]:
    findings_claims = [claim_prompt_dict(claim) for claim in generated_claims if claim.section == "Findings"]
    impression_claims = [claim for claim in generated_claims if claim.section == "Impression"]
    results: dict[str, dict[str, Any]] = {}
    for claim in impression_claims:
        prompt = INTERNAL_CONSISTENCY_PROMPT.format(
            findings_claims=compact_json(findings_claims),
            impression_claim=compact_json(claim_prompt_dict(claim)),
        )
        response = llm.call_json(prompt)
        status = normalize_text(response.get("support_status") if isinstance(response, dict) else "").lower()
        if status not in SUPPORT_STATUSES:
            status = "unverifiable"
        contradicting = response.get("contradicting_finding_claims", []) if isinstance(response, dict) else []
        results[claim.claim_id] = {
            "claim_id": claim.claim_id,
            "support_status": status,
            "contradicting_finding_claims": coerce_list(contradicting),
            "brief_reason": normalize_text(response.get("brief_reason") if isinstance(response, dict) else ""),
        }
    return results


def llm_recommendation_support(
    llm: OpenAIBackbone,
    *,
    generated_claims: list[Claim],
    reference_claims: list[Claim],
    evidence: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    findings_claims = [claim_prompt_dict(claim) for claim in generated_claims if claim.section == "Findings"]
    findings_claims.extend(claim_prompt_dict(claim) for claim in reference_claims if claim.section == "Findings")
    recommendations = [claim for claim in generated_claims if claim.claim_type == "recommendation_claim"]
    results: dict[str, dict[str, Any]] = {}
    medical_evidence = {
        "retrieved_medical_evidence": evidence.get("retrieved_medical_evidence", []),
        "similar_case_evidence": evidence.get("similar_case_evidence", []),
        "image_evidence": evidence.get("image_evidence", []),
    }
    for claim in recommendations:
        prompt = RECOMMENDATION_SUPPORT_PROMPT.format(
            recommendation_claim=compact_json(claim_prompt_dict(claim)),
            findings_claims=compact_json(findings_claims),
            patient_context=compact_json(evidence.get("patient_context", {})),
            medical_evidence=compact_json(medical_evidence),
        )
        response = llm.call_json(prompt)
        status = normalize_text(response.get("recommendation_status") if isinstance(response, dict) else "").lower()
        if status not in RECOMMENDATION_STATUSES:
            status = "unverifiable"
        supporting = response.get("supporting_evidence", []) if isinstance(response, dict) else []
        results[claim.claim_id] = {
            "claim_id": claim.claim_id,
            "recommendation_status": status,
            "supporting_evidence": coerce_list(supporting),
            "brief_reason": normalize_text(response.get("brief_reason") if isinstance(response, dict) else ""),
        }
    return results


def choose_primary_type(hallucination_types: list[str], *, claim_type: str | None = None) -> str | None:
    if not hallucination_types:
        return None
    if claim_type == "recommendation_claim" and HALLUCINATION_TYPES["H10"] in hallucination_types:
        return HALLUCINATION_TYPES["H10"]
    for candidate in PRIMARY_PRIORITY:
        if candidate in hallucination_types:
            return candidate
    return hallucination_types[0]


def sanitize_hallucination_types(values: Any) -> list[str]:
    types: list[str] = []
    for item in coerce_list(values):
        value = normalize_text(item)
        if value in VALID_H_TYPES and value != HALLUCINATION_TYPES["H2"] and value not in types:
            types.append(value)
    return types



def normalize_for_synonym(value: Any) -> str:
    text = normalize_text(value).lower()
    text = text.replace("-", " ")
    text = re.sub(r"[^a-z0-9%./ ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def canonicalize_with_map(value: Any, mapping: dict[str, str], sorted_items: list[tuple[str, str]]) -> str | None:
    text = normalize_for_synonym(value)
    if not text:
        return None
    if text in mapping:
        return mapping[text]
    normalized_tokens = " ".join(token_normalize(token) for token in text.split())
    if normalized_tokens in mapping:
        return mapping[normalized_tokens]
    for variant, canonical in sorted_items:
        pattern = rf"(?<![a-z0-9]){re.escape(variant)}(?![a-z0-9])"
        if re.search(pattern, text):
            return canonical
    return normalized_tokens or text


def canonical_finding(claim: Claim) -> str | None:
    values = [claim.core_finding, claim.raw_text]
    for value in values:
        canonical = canonicalize_with_map(value, SYNONYM_TO_CANONICAL, SORTED_SYNONYMS)
        if canonical and canonical != "normality":
            return canonical
    return None


def canonical_anatomy(value: Any) -> str | None:
    return canonicalize_with_map(value, ANATOMY_TO_CANONICAL, SORTED_ANATOMY_SYNONYMS)


def claim_anatomy_key(claim: Claim) -> str | None:
    return canonical_anatomy(claim.location_detail) or canonical_anatomy(claim.anatomy) or canonical_anatomy(claim.raw_text)


def anatomy_compatible(left: Claim, right: Claim) -> bool:
    left_key = claim_anatomy_key(left)
    right_key = claim_anatomy_key(right)
    if not left_key or not right_key:
        return True
    if left_key == right_key:
        return True
    broad_pairs = {
        ("right upper lobe", "lung"), ("right middle lobe", "lung"), ("right lower lobe", "lung"),
        ("left upper lobe", "lung"), ("left lower lobe", "lung"),
        ("pleural space", "lung"),
        ("rib cage", "lung"),
    }
    return (left_key, right_key) in broad_pairs or (right_key, left_key) in broad_pairs


def crimson_match_score(left: Claim, right: Claim) -> float:
    left_core = canonical_finding(left)
    right_core = canonical_finding(right)
    score = 0.0
    if left_core and right_core:
        if left_core == right_core:
            score += 0.72
        else:
            overlap = token_overlap(left_core, right_core)
            if overlap >= 0.75:
                score += 0.55
            elif overlap >= 0.5:
                score += 0.42
    raw_overlap = token_overlap(left.raw_text, right.raw_text)
    score += min(raw_overlap * 0.25, 0.18)
    if anatomy_compatible(left, right):
        score += 0.12
    left_anat = claim_anatomy_key(left)
    right_anat = claim_anatomy_key(right)
    if left_anat and right_anat and left_anat == right_anat:
        score += 0.08
    return min(score, 1.0)


def crimson_candidate_match(left: Claim, right: Claim, threshold: float) -> bool:
    left_core = canonical_finding(left)
    right_core = canonical_finding(right)
    if left_core and right_core and left_core == right_core:
        return True
    return crimson_match_score(left, right) >= threshold


def match_findings_crimson(
    generated_findings: list[Claim],
    reference_findings: list[Claim],
    config: DetectorConfig,
) -> list[tuple[Claim, Claim, float]]:
    candidates: list[tuple[float, Claim, Claim]] = []
    threshold = max(0.35, min(config.min_match_score, 0.5))
    for gen in generated_findings:
        for ref in reference_findings:
            score = crimson_match_score(gen, ref)
            if crimson_candidate_match(gen, ref, threshold):
                candidates.append((score, gen, ref))
    candidates.sort(key=lambda item: item[0], reverse=True)
    used_gen: set[str] = set()
    used_ref: set[str] = set()
    pairs: list[tuple[Claim, Claim, float]] = []
    for score, gen, ref in candidates:
        if gen.claim_id in used_gen or ref.claim_id in used_ref:
            continue
        used_gen.add(gen.claim_id)
        used_ref.add(ref.claim_id)
        pairs.append((gen, ref, score))
    return pairs


def crimson_location_conflict(gen: Claim, ref: Claim) -> bool:
    if not crimson_candidate_match(gen, ref, 0.35):
        return False
    if gen.laterality != "none" and ref.laterality != "none" and gen.laterality != ref.laterality:
        if {gen.laterality, ref.laterality} <= {"left", "right", "bilateral"}:
            return True
    gen_anat = claim_anatomy_key(gen)
    ref_anat = claim_anatomy_key(ref)
    if gen_anat and ref_anat and gen_anat != ref_anat and not anatomy_compatible(gen, ref):
        return True
    if gen.location_detail and ref.location_detail:
        gen_loc = normalize_for_synonym(gen.location_detail)
        ref_loc = normalize_for_synonym(ref.location_detail)
        if gen_loc != ref_loc and token_overlap(gen_loc, ref_loc) < 0.5:
            return True
    return False


def certainty_value(claim: Claim) -> int:
    certainty = normalize_for_synonym(claim.certainty)
    if certainty in CERTAINTY_SCALE:
        return CERTAINTY_SCALE[certainty]
    text = claim.raw_text.lower()
    if starts_with_primary_negation(text) or claim.polarity == "negative":
        return 0
    for key, value in CERTAINTY_SCALE.items():
        if key and key in text:
            return value
    return 5 if claim.polarity == "positive" else 2


def certainty_miscalibrated(gen: Claim, ref: Claim) -> bool:
    if not crimson_candidate_match(gen, ref, 0.35):
        return False
    return abs(certainty_value(gen) - certainty_value(ref)) >= 2


def overinterpreted_finding(gen: Claim, ref: Claim) -> bool:
    gen_text = normalize_for_synonym(" ".join([gen.raw_text, gen.core_finding or ""]))
    ref_core = canonical_finding(ref)
    gen_core = canonical_finding(gen)
    has_diagnostic_language = (
        gen.claim_type in {"diagnostic_inference", "etiology_claim"}
        or any(term in gen_text for term in DIAGNOSTIC_OVERINTERPRETATION_TERMS)
    )
    if not has_diagnostic_language:
        return False
    if gen_core in {"pneumonia", "tumor", "infarct", "ischemia"} and ref_core in NONSPECIFIC_FINDING_TERMS:
        return True
    if any(term in gen_text for term in {"malignant", "malignancy", "cancer", "metastatic", "metastasis"}) and ref_core in {"nodule", "mass", "lesion", "airspace opacity", "ground glass opacity"}:
        return True
    if "pneumonia" in gen_text and ref_core in {"airspace opacity", "ground glass opacity", "opacity", "consolidation"}:
        return True
    return False


def clinically_relevant_reference_claim(claim: Claim, config: DetectorConfig) -> bool:
    if claim.claim_type in {"positive_finding", "diagnostic_inference", "etiology_claim", "comparison_claim", "measurement_claim", "certainty_claim"}:
        return claim.polarity != "negative"
    if config.include_negative_omissions and claim.claim_type in {"negative_finding", "normality_claim"}:
        return True
    return False


def make_stage_b_result(
    claim: Claim,
    *,
    support: dict[str, Any] | None,
    h_types: list[str],
    matched_ref: Claim | None = None,
    match_score: float | None = None,
    reason: str = "",
    error_details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    support_status = (support or {}).get("support_status") or ("contradicted" if h_types else "entailed")
    primary = choose_primary_type(h_types, claim_type=claim.claim_type)
    return {
        "claim_id": claim.claim_id,
        "section": claim.section,
        "raw_text": claim.raw_text,
        "claim_text": claim.raw_text,
        "claim_type": claim.claim_type,
        "support_status": support_status,
        "is_error": bool(h_types),
        "primary_hallucination_type": primary,
        "hallucination_types": h_types,
        "matched_reference_claims": [matched_ref.claim_id] if matched_ref else [],
        "matched_reference_claim_id": matched_ref.claim_id if matched_ref else None,
        "matched_reference_text": matched_ref.raw_text if matched_ref else None,
        "match_score": round(match_score, 4) if match_score is not None else None,
        "evidence_used": (support or {}).get("supporting_evidence", []),
        "clinical_severity": None,
        "brief_reason": reason,
        "reasons": [reason] if reason else [],
        "error_details": error_details or {},
        "stage_b": True,
    }


def run_stage_b_findings(
    generated_claims: list[Claim],
    reference_claims: list[Claim],
    support_results: dict[str, dict[str, Any]],
    config: DetectorConfig,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    generated_findings = [claim for claim in generated_claims if claim.section == "Findings" and claim.claim_type in STAGE_B_FINDING_TYPES]
    reference_findings = [claim for claim in reference_claims if claim.section == "Findings" and claim.claim_type in STAGE_B_FINDING_TYPES]
    pairs = match_findings_crimson(generated_findings, reference_findings, config)
    matched_gen = {gen.claim_id for gen, _, _ in pairs}
    matched_ref = {ref.claim_id for _, ref, _ in pairs}
    results: dict[str, dict[str, Any]] = {}

    for gen, ref, score in pairs:
        h_types: list[str] = []
        details: dict[str, Any] = {}
        reasons: list[str] = []
        if has_polarity_conflict(gen, ref) or (gen.polarity != ref.polarity and "uncertain" not in {gen.polarity, ref.polarity}):
            h_types.append(HALLUCINATION_TYPES["H3"])
            details["polarity"] = {"generated_value": gen.polarity, "reference_value": ref.polarity}
            reasons.append("Generated and reference finding match but polarity is opposite.")
        elif crimson_location_conflict(gen, ref):
            h_types.append(HALLUCINATION_TYPES["H4"])
            details["location"] = {
                "generated_laterality": gen.laterality,
                "reference_laterality": ref.laterality,
                "generated_anatomy": gen.anatomy,
                "reference_anatomy": ref.anatomy,
                "generated_location_detail": gen.location_detail,
                "reference_location_detail": ref.location_detail,
            }
            reasons.append("Matched finding has clinically meaningful location or laterality mismatch.")
        if attribute_conflict(gen, ref):
            h_types.append(HALLUCINATION_TYPES["H5"])
            details["attribute"] = {
                "generated_severity": gen.severity,
                "reference_severity": ref.severity,
                "generated_measurement": gen.measurement,
                "reference_measurement": ref.measurement,
                "generated_descriptor": gen.descriptor,
                "reference_descriptor": ref.descriptor,
            }
            reasons.append("Matched finding has severity, measurement, or descriptor mismatch.")
        if temporal_conflict(gen, ref):
            h_types.append(HALLUCINATION_TYPES["H6"])
            details["temporal"] = {"generated_value": gen.temporal, "reference_value": ref.temporal}
            reasons.append("Matched finding has temporal comparison mismatch.")
        if overinterpreted_finding(gen, ref):
            h_types.append(HALLUCINATION_TYPES["H7"])
            details["overinterpretation"] = {"generated_core": gen.core_finding, "reference_core": ref.core_finding}
            reasons.append("Generated finding is a stronger diagnostic interpretation than the matched reference finding.")
        if certainty_miscalibrated(gen, ref):
            h_types.append(HALLUCINATION_TYPES["H9"])
            details["certainty"] = {"generated_value": gen.certainty, "reference_value": ref.certainty}
            reasons.append("Generated certainty is clinically miscalibrated relative to the matched reference finding.")
        h_types = list(dict.fromkeys(h_types))
        results[gen.claim_id] = make_stage_b_result(
            gen,
            support=support_results.get(gen.claim_id),
            h_types=h_types,
            matched_ref=ref,
            match_score=score,
            reason=" ".join(reasons) if reasons else "Generated finding is supported by the matched reference finding.",
            error_details=details,
        )

    for gen in generated_findings:
        if gen.claim_id in matched_gen:
            continue
        support = support_results.get(gen.claim_id, {})
        support_status = support.get("support_status")
        h_types: list[str] = []
        reason = "No matched reference finding was found."
        if gen.claim_type in {"diagnostic_inference", "etiology_claim"} and support_status in {"partially_entailed", "not_entailed", "contradicted"}:
            h_types.append(HALLUCINATION_TYPES["H7"])
            reason = "Unmatched generated finding makes an unsupported diagnostic or etiologic inference."
        elif gen.claim_type == "comparison_claim" and support_status in {"not_entailed", "contradicted"}:
            h_types.append(HALLUCINATION_TYPES["H6"])
            reason = "Generated comparison claim has no matched reference support."
        elif gen.polarity == "positive" and support_status in {"not_entailed", "contradicted"}:
            h_types.append(HALLUCINATION_TYPES["H1"])
            reason = "Generated positive finding has no matched reference/evidence-supported finding."
        elif config.strict_negative_claims and gen.claim_type in {"negative_finding", "normality_claim"} and support_status in {"not_entailed", "contradicted"}:
            h_types.append(HALLUCINATION_TYPES["H1"])
            reason = "Generated negative/normal finding is unsupported under strict-negative mode."
        results[gen.claim_id] = make_stage_b_result(
            gen,
            support=support,
            h_types=h_types,
            matched_ref=None,
            match_score=None,
            reason=reason,
        )

    omissions: list[dict[str, Any]] = []
    for ref in reference_findings:
        if ref.claim_id in matched_ref:
            continue
        if not clinically_relevant_reference_claim(ref, config):
            continue
        omissions.append(
            {
                "reference_claim_id": ref.claim_id,
                "section": ref.section,
                "raw_text": ref.raw_text,
                "reference_claim_text": ref.raw_text,
                "claim_type": ref.claim_type,
                "hallucination_type": HALLUCINATION_TYPES["H2"],
                "matched_generated_claims": [],
                "clinical_severity": None,
                "brief_reason": "Clinically relevant reference finding has no matched generated finding under CRIMSON-style matching.",
                "reason": "Clinically relevant reference finding has no matched generated finding under CRIMSON-style matching.",
                "stage_b": True,
            }
        )
    return results, omissions

def merge_hallucination_types(base: list[str], extra: list[str]) -> list[str]:
    merged = list(base)
    for item in extra:
        if item in VALID_H_TYPES and item != HALLUCINATION_TYPES["H2"] and item not in merged:
            merged.append(item)
    return merged


def apply_unverifiable_suppression(result: dict[str, Any]) -> dict[str, Any]:
    if result.get("support_status") != "unverifiable":
        return result
    suppressed = sanitize_hallucination_types(result.get("hallucination_types", []))
    if suppressed:
        result["unverifiable_hallucination_types_suppressed"] = suppressed
    result["is_error"] = False
    result["hallucination_types"] = []
    result["primary_hallucination_type"] = None
    result["brief_reason"] = join_text(
        result.get("brief_reason", ""),
        "Support status is unverifiable; this claim is excluded from error counts.",
    )
    result["reasons"] = [result["brief_reason"]] if result.get("brief_reason") else []
    return result


def sanitize_claim_results(
    response: Any,
    *,
    generated_claims: list[Claim],
    reference_claims: list[Claim],
    support_results: dict[str, dict[str, Any]],
    internal_results: dict[str, dict[str, Any]],
    recommendation_results: dict[str, dict[str, Any]],
    stage_b_results: dict[str, dict[str, Any]],
    stage_b_omission_events: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not isinstance(response, dict):
        raise ValueError("Taxonomy classification returned non-object JSON")
    raw_results = response.get("claim_level_results")
    if not isinstance(raw_results, list):
        raise ValueError("Taxonomy classification missing claim_level_results")

    generated_by_id = {claim.claim_id: claim for claim in generated_claims}
    reference_by_id = {claim.claim_id: claim for claim in reference_claims}
    results_by_id: dict[str, dict[str, Any]] = {}

    for item in raw_results:
        if not isinstance(item, dict):
            continue
        claim_id = normalize_text(item.get("claim_id"))
        if claim_id not in generated_by_id:
            continue
        claim = generated_by_id[claim_id]
        support = support_results.get(claim_id, {})
        stage_b = stage_b_results.get(claim_id)
        h_types = sanitize_hallucination_types(item.get("hallucination_types", []))
        if stage_b:
            h_types = merge_hallucination_types(h_types, sanitize_hallucination_types(stage_b.get("hallucination_types", [])))

        internal = internal_results.get(claim_id)
        if internal and internal.get("support_status") in {"not_entailed", "contradicted"}:
            if HALLUCINATION_TYPES["H8"] not in h_types:
                h_types.append(HALLUCINATION_TYPES["H8"])

        recommendation = recommendation_results.get(claim_id)
        if recommendation and recommendation.get("recommendation_status") in {"unsupported", "contradicted"}:
            if HALLUCINATION_TYPES["H10"] not in h_types:
                h_types.append(HALLUCINATION_TYPES["H10"])

        support_status = normalize_text(item.get("support_status")).lower() or support.get("support_status", "unverifiable")
        if stage_b and stage_b.get("support_status") and support_status not in SUPPORT_STATUSES:
            support_status = stage_b.get("support_status")
        if support_status not in SUPPORT_STATUSES:
            support_status = support.get("support_status", "unverifiable")
        clinical_severity = normalize_text(item.get("clinical_severity")).lower() or None
        if clinical_severity not in CLINICAL_SEVERITIES:
            clinical_severity = None
        matched_ids = [normalize_text(value) for value in coerce_list(item.get("matched_reference_claims"))]
        if stage_b:
            matched_ids.extend(normalize_text(value) for value in coerce_list(stage_b.get("matched_reference_claims")))
        matched_ids = list(dict.fromkeys(value for value in matched_ids if value in reference_by_id))
        primary = normalize_text(item.get("primary_hallucination_type")) or choose_primary_type(h_types, claim_type=claim.claim_type)
        if primary not in h_types:
            primary = choose_primary_type(h_types, claim_type=claim.claim_type)
        evidence_used = item.get("evidence_used")
        if not isinstance(evidence_used, list):
            evidence_used = coerce_list(evidence_used)
        if stage_b:
            evidence_used.extend(coerce_list(stage_b.get("evidence_used")))
        brief_reason = normalize_text(item.get("brief_reason")) or support.get("brief_reason", "")
        if stage_b and stage_b.get("brief_reason"):
            brief_reason = join_text(brief_reason, f"Stage B: {stage_b['brief_reason']}")
        if internal and internal.get("brief_reason") and HALLUCINATION_TYPES["H8"] in h_types:
            brief_reason = join_text(brief_reason, f"Internal consistency: {internal['brief_reason']}")
        if recommendation and recommendation.get("brief_reason") and HALLUCINATION_TYPES["H10"] in h_types:
            brief_reason = join_text(brief_reason, f"Recommendation support: {recommendation['brief_reason']}")
        error_details = item.get("error_details") if isinstance(item.get("error_details"), dict) else {}
        if stage_b and isinstance(stage_b.get("error_details"), dict):
            error_details = {**error_details, "stage_b": stage_b.get("error_details")}

        results_by_id[claim_id] = apply_unverifiable_suppression({
            "claim_id": claim_id,
            "section": claim.section,
            "raw_text": claim.raw_text,
            "claim_text": claim.raw_text,
            "claim_type": claim.claim_type,
            "support_status": support_status,
            "is_error": bool(h_types),
            "primary_hallucination_type": primary,
            "hallucination_types": h_types,
            "matched_reference_claims": matched_ids,
            "matched_reference_claim_id": matched_ids[0] if matched_ids else None,
            "matched_reference_text": reference_by_id[matched_ids[0]].raw_text if matched_ids else None,
            "evidence_used": [normalize_text(value) for value in evidence_used if normalize_text(value)],
            "clinical_severity": clinical_severity,
            "brief_reason": brief_reason,
            "reasons": [brief_reason] if brief_reason else [],
            "error_details": error_details,
            "stage_a_support": support,
            "stage_b_findings": stage_b,
            "internal_consistency": internal,
            "recommendation_support": recommendation,
        })

    missing = sorted(set(generated_by_id) - set(results_by_id), key=natural_claim_id_key)
    if missing:
        raise ValueError(f"Taxonomy classification omitted generated claim ids: {missing}")

    omission_events = sanitize_omission_events(response.get("omission_events", []), reference_by_id)
    seen_refs = {event["reference_claim_id"] for event in omission_events}
    for event in stage_b_omission_events:
        ref_id = event.get("reference_claim_id")
        if ref_id in reference_by_id and ref_id not in seen_refs:
            omission_events.append(event)
            seen_refs.add(ref_id)
    return [results_by_id[claim.claim_id] for claim in generated_claims], omission_events

def natural_claim_id_key(value: str) -> list[Any]:
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", value)]


def sanitize_omission_events(raw_events: Any, reference_by_id: dict[str, Claim]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for item in coerce_list(raw_events):
        if not isinstance(item, dict):
            continue
        ref_id = normalize_text(item.get("reference_claim_id"))
        if ref_id not in reference_by_id:
            continue
        severity = normalize_text(item.get("clinical_severity")).lower() or None
        if severity not in CLINICAL_SEVERITIES:
            severity = None
        matched = [normalize_text(value) for value in coerce_list(item.get("matched_generated_claims"))]
        reason = normalize_text(item.get("brief_reason"))
        events.append(
            {
                "reference_claim_id": ref_id,
                "section": reference_by_id[ref_id].section,
                "raw_text": reference_by_id[ref_id].raw_text,
                "reference_claim_text": reference_by_id[ref_id].raw_text,
                "claim_type": reference_by_id[ref_id].claim_type,
                "hallucination_type": HALLUCINATION_TYPES["H2"],
                "matched_generated_claims": matched,
                "clinical_severity": severity,
                "brief_reason": reason,
                "reason": reason,
            }
        )
    return events


def compute_case_metrics(
    generated_claims: list[Claim],
    reference_claims: list[Claim],
    claim_results: list[dict[str, Any]],
    omission_events: list[dict[str, Any]],
) -> dict[str, Any]:
    num_generated = len(generated_claims)
    num_reference = len(reference_claims)
    unverifiable_claims = [item for item in claim_results if item.get("support_status") == "unverifiable"]
    num_unverifiable = len(unverifiable_claims)
    erroneous_generated = sum(
        1 for item in claim_results
        if item.get("is_error") and item.get("support_status") != "unverifiable"
    )
    num_omissions = len(omission_events)

    multi_counts = Counter()
    primary_counts = Counter()
    section_errors = Counter()
    section_unverifiable = Counter()
    section_totals = Counter(claim.section for claim in generated_claims)
    for item in claim_results:
        if item.get("support_status") == "unverifiable":
            section_unverifiable.update([item.get("section", "unknown")])
            continue
        h_types = sanitize_hallucination_types(item.get("hallucination_types", []))
        multi_counts.update(h_types)
        primary = item.get("primary_hallucination_type")
        if primary in VALID_H_TYPES:
            primary_counts.update([primary])
        if item.get("is_error"):
            section_errors.update([item.get("section", "unknown")])
    multi_counts.update(event["hallucination_type"] for event in omission_events)
    primary_counts.update(event["hallucination_type"] for event in omission_events)

    generated_error_rate = erroneous_generated / num_generated if num_generated else math.nan
    omission_rate = num_omissions / num_reference if num_reference else math.nan
    total_event_rate = (erroneous_generated + num_omissions) / (num_generated + num_reference) if (num_generated + num_reference) else math.nan
    unverifiable_rate = num_unverifiable / num_generated if num_generated else math.nan

    section_error_rates = {
        section: round(section_errors[section] / total, 6) if total else None
        for section, total in sorted(section_totals.items())
    }
    section_unverifiable_rates = {
        section: round(section_unverifiable[section] / total, 6) if total else None
        for section, total in sorted(section_totals.items())
    }
    return {
        "num_generated_claims": num_generated,
        "num_reference_claims": num_reference,
        "num_erroneous_generated_claims": erroneous_generated,
        "num_omission_events": num_omissions,
        "num_unverifiable_generated_claims": num_unverifiable,
        "unverifiable_claim_ids": [item.get("claim_id") for item in unverifiable_claims],
        "hallucination_rate": None if math.isnan(generated_error_rate) else round(generated_error_rate, 6),
        "generated_claim_error_rate": None if math.isnan(generated_error_rate) else round(generated_error_rate, 6),
        "omission_rate": None if math.isnan(omission_rate) else round(omission_rate, 6),
        "total_hallucination_event_rate": None if math.isnan(total_event_rate) else round(total_event_rate, 6),
        "unverifiable_rate": None if math.isnan(unverifiable_rate) else round(unverifiable_rate, 6),
        "hallucination_type_counts": dict(sorted(multi_counts.items())),
        "primary_hallucination_type_counts": dict(sorted(primary_counts.items())),
        "multi_label_hallucination_type_counts": dict(sorted(multi_counts.items())),
        "section_error_rates": section_error_rates,
        "section_unverifiable_rates": section_unverifiable_rates,
    }

def llm_classify_taxonomy(
    llm: OpenAIBackbone,
    *,
    case_id: str,
    generated_report: dict[str, str],
    reference_report: dict[str, str],
    generated_claims: list[Claim],
    reference_claims: list[Claim],
    evidence: dict[str, Any],
    support_results: dict[str, dict[str, Any]],
    internal_results: dict[str, dict[str, Any]],
    recommendation_results: dict[str, dict[str, Any]],
    stage_b_results: dict[str, dict[str, Any]],
    stage_b_omission_events: list[dict[str, Any]],
    config: DetectorConfig,
    image_paths: list[Path],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    payload = {
        "case_id": case_id,
        "generated_report": generated_report,
        "reference_report": reference_report,
        "generated_claims": [claim_prompt_dict(claim) for claim in generated_claims],
        "reference_claims": [claim_prompt_dict(claim) for claim in reference_claims],
        "stage_a_support_results": support_results,
        "internal_consistency_results": internal_results,
        "recommendation_support_results": recommendation_results,
        "stage_b_findings_results": stage_b_results,
        "stage_b_omission_events": stage_b_omission_events,
        "evidence": evidence,
        "config": {
            "strict_negative_claims": config.strict_negative_claims,
            "include_negative_omissions": config.include_negative_omissions,
        },
    }
    prompt = TAXONOMY_CLASSIFICATION_PROMPT.format(case_payload=compact_json(payload, max_chars=llm.config.max_report_chars))
    attach_images = image_paths if llm.config.include_images and llm.config.image_stage in {"taxonomy", "all"} else []
    response = llm.call_json(prompt, image_paths=attach_images)
    claim_results, omissions = sanitize_claim_results(
        response,
        generated_claims=generated_claims,
        reference_claims=reference_claims,
        support_results=support_results,
        internal_results=internal_results,
        recommendation_results=recommendation_results,
        stage_b_results=stage_b_results,
        stage_b_omission_events=stage_b_omission_events,
    )
    return claim_results, omissions, response


def evaluate_case(
    sample_dir: Path,
    config: DetectorConfig,
    llm: OpenAIBackbone,
    *,
    include_image_manifest: bool = True,
    max_images: int = 64,
    progress_enabled: bool = True,
) -> dict[str, Any]:
    result_path = sample_dir / "result.json"
    summary_path = sample_dir / "summary.json"
    log_progress(f"{sample_dir.name}: loading result.json and summary.json", enabled=progress_enabled)
    result = load_json(result_path)
    summary = load_json(summary_path)
    case_id = case_id_from(sample_dir, summary, result)
    log_progress(f"{case_id}: Stage 0 parsing reports and evidence", enabled=progress_enabled)
    generated_sections = extract_generated_sections(result, summary)
    reference_sections = extract_reference_sections(summary, result)
    generated_report = {
        "findings": truncate_value(generated_sections["Findings"]["text"], llm.config.max_report_chars),
        "impression": truncate_value(generated_sections["Impression"]["text"], llm.config.max_report_chars),
    }
    reference_report = {
        "findings": truncate_value(reference_sections["Findings"], llm.config.max_report_chars),
        "impression": truncate_value(reference_sections["Impression"], llm.config.max_report_chars),
    }
    image_manifest = collect_image_manifest(sample_dir, max_images) if include_image_manifest else []
    described_images = sum(1 for item in image_manifest if item.get("description"))
    log_progress(
        f"{case_id}: Stage 0 found {len(image_manifest)} image files ({described_images} with descriptions)",
        enabled=progress_enabled,
    )
    evidence = collect_optional_evidence(
        summary,
        result,
        image_manifest,
        max_chars=llm.config.max_evidence_chars,
    )
    image_paths = llm_image_paths(sample_dir, image_manifest, llm.config.max_llm_images)
    log_progress(
        f"{case_id}: using {len(image_paths)} image(s) for multimodal prompts" if llm.config.include_images else f"{case_id}: image upload disabled",
        enabled=progress_enabled,
    )

    log_progress(f"{case_id}: Stage 0 LLM claim extraction started", enabled=progress_enabled)
    generated_claims, reference_claims = llm_extract_all_claims(
        llm,
        case_id=case_id,
        generated_report=generated_report,
        reference_report=reference_report,
    )
    log_progress(
        f"{case_id}: Stage 0 extracted {len(generated_claims)} generated claims and {len(reference_claims)} reference claims",
        enabled=progress_enabled,
    )

    log_progress(f"{case_id}: Stage A support verification started", enabled=progress_enabled)
    support_results = llm_verify_support(
        llm,
        generated_claims=generated_claims,
        reference_claims=reference_claims,
        generated_report=generated_report,
        reference_report=reference_report,
        evidence=evidence,
        image_paths=image_paths,
    )
    support_status_counts = Counter(item.get("support_status") for item in support_results.values())
    log_progress(f"{case_id}: Stage A complete {dict(support_status_counts)}", enabled=progress_enabled)

    log_progress(f"{case_id}: Stage B CRIMSON-style findings matching started", enabled=progress_enabled)
    stage_b_results, stage_b_omission_events = run_stage_b_findings(
        generated_claims,
        reference_claims,
        support_results,
        config,
    )
    stage_b_error_count = sum(1 for item in stage_b_results.values() if item.get("is_error"))
    log_progress(
        f"{case_id}: Stage B complete ({len(stage_b_results)} finding results, {stage_b_error_count} errors, {len(stage_b_omission_events)} omissions)",
        enabled=progress_enabled,
    )

    log_progress(f"{case_id}: Stage C internal consistency check started", enabled=progress_enabled)
    internal_results = llm_internal_consistency(llm, generated_claims=generated_claims)
    log_progress(f"{case_id}: Stage C internal consistency complete ({len(internal_results)} impression claims)", enabled=progress_enabled)

    log_progress(f"{case_id}: Stage C recommendation support check started", enabled=progress_enabled)
    recommendation_results = llm_recommendation_support(
        llm,
        generated_claims=generated_claims,
        reference_claims=reference_claims,
        evidence=evidence,
    )
    log_progress(f"{case_id}: Stage C recommendation support complete ({len(recommendation_results)} recommendation claims)", enabled=progress_enabled)

    log_progress(f"{case_id}: Stage D final H1-H10 taxonomy classification started", enabled=progress_enabled)
    claim_results, omission_events, taxonomy_raw = llm_classify_taxonomy(
        llm,
        case_id=case_id,
        generated_report=generated_report,
        reference_report=reference_report,
        generated_claims=generated_claims,
        reference_claims=reference_claims,
        evidence=evidence,
        support_results=support_results,
        internal_results=internal_results,
        recommendation_results=recommendation_results,
        stage_b_results=stage_b_results,
        stage_b_omission_events=stage_b_omission_events,
        config=config,
        image_paths=image_paths,
    )
    metrics = compute_case_metrics(generated_claims, reference_claims, claim_results, omission_events)
    log_progress(
        f"{case_id}: Stage D complete (errors={metrics['num_erroneous_generated_claims']}, omissions={metrics['num_omission_events']}, unverifiable={metrics['num_unverifiable_generated_claims']})",
        enabled=progress_enabled,
    )
    return {
        "case_id": case_id,
        "sample_dir": str(sample_dir),
        "generated_report": generated_report,
        "reference_report": reference_report,
        "image_evidence": image_manifest,
        "llm_backend": {
            "provider": "openai_compatible",
            "model": llm.config.model,
            "base_url": llm.config.base_url,
            "include_images": llm.config.include_images,
            "image_stage": llm.config.image_stage if llm.config.include_images else "none",
            "max_llm_images": llm.config.max_llm_images if llm.config.include_images else 0,
        },
        "generated_claims": [claim.to_dict() for claim in generated_claims],
        "reference_claims": [claim.to_dict() for claim in reference_claims],
        "stage_a_support_results": support_results,
        "internal_consistency_results": internal_results,
        "recommendation_support_results": recommendation_results,
        "stage_b_findings_results": stage_b_results,
        "stage_b_omission_events": stage_b_omission_events,
        "claim_level_results": claim_results,
        "omission_events": omission_events,
        "taxonomy_raw_response": taxonomy_raw,
        "case_metrics": metrics,
    }

def natural_key(path: Path) -> list[Any]:
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", path.name)]


def iter_case_dirs(target_dir: Path, case_glob: str) -> list[Path]:
    candidates = [path for path in target_dir.glob(case_glob) if path.is_dir()]
    return sorted(candidates, key=natural_key)


def resolve_llm_config(args: argparse.Namespace) -> LLMConfig:
    api_key = args.api_key or os.environ.get(args.api_key_env) or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        env_names = ", ".join(dict.fromkeys([args.api_key_env, "OPENAI_API_KEY"]))
        raise ValueError(f"Missing API key. Pass --api-key or set one of: {env_names}.")
    model = args.model or os.environ.get("HALLUCINATION_MODEL") or os.environ.get("OPENAI_MODEL")
    if not model:
        raise ValueError("Missing model. Pass --model or set HALLUCINATION_MODEL / OPENAI_MODEL.")
    base_url = args.base_url or os.environ.get("OPENAI_BASE_URL")
    return LLMConfig(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        max_retries=args.max_retries,
        use_json_mode=not args.no_json_mode,
        include_images=args.include_images,
        image_stage=args.image_stage,
        max_llm_images=args.max_llm_images,
        image_detail=args.image_detail,
        max_report_chars=args.max_report_chars,
        max_evidence_chars=args.max_evidence_chars,
    )


def process_target(args: argparse.Namespace) -> dict[str, Any]:
    target_dir = Path(args.target_dir)
    if not target_dir.exists():
        raise FileNotFoundError(f"Target directory does not exist: {target_dir}")

    config = DetectorConfig(
        min_match_score=args.min_match_score,
        omission_match_score=args.omission_match_score,
        impression_match_score=args.impression_match_score,
        strict_negative_claims=args.strict_negative_claims,
        include_negative_omissions=args.include_negative_omissions,
    )
    llm_config = resolve_llm_config(args)
    llm = OpenAIBackbone(llm_config)

    case_dirs = iter_case_dirs(target_dir, args.case_glob)
    if args.limit is not None:
        case_dirs = case_dirs[: args.limit]
    progress_enabled = not args.quiet
    log_progress(
        f"Starting hallucination detection: target={target_dir}, cases={len(case_dirs)}, model={llm_config.model}, images={'on' if llm_config.include_images else 'off'}",
        enabled=progress_enabled,
    )
    processed = 0
    skipped = 0
    failed: list[dict[str, str]] = []
    aggregate_counts = Counter()
    aggregate_primary_counts = Counter()
    aggregate_metrics = {
        "num_generated_claims": 0,
        "num_reference_claims": 0,
        "num_erroneous_generated_claims": 0,
        "num_omission_events": 0,
        "num_unverifiable_generated_claims": 0,
    }

    for index, sample_dir in enumerate(case_dirs, start=1):
        output_path = sample_dir / args.output_name
        log_progress(f"Case {index}/{len(case_dirs)} {sample_dir.name}: start", enabled=progress_enabled)
        if output_path.exists() and not args.overwrite:
            skipped += 1
            log_progress(f"Case {index}/{len(case_dirs)} {sample_dir.name}: skipped existing {args.output_name}", enabled=progress_enabled)
            continue
        if not (sample_dir / "result.json").exists() or not (sample_dir / "summary.json").exists():
            failed.append({"sample_dir": str(sample_dir), "error": "missing result.json or summary.json"})
            log_progress(f"Case {index}/{len(case_dirs)} {sample_dir.name}: missing result.json or summary.json", enabled=progress_enabled)
            continue
        started_at = time.time()
        try:
            case_output = evaluate_case(
                sample_dir,
                config,
                llm,
                include_image_manifest=not args.no_image_manifest,
                max_images=args.max_images,
                progress_enabled=progress_enabled,
            )
            with output_path.open("w", encoding="utf-8") as f:
                json.dump(case_output, f, ensure_ascii=False, indent=2)
                f.write("\n")
            processed += 1
            metrics = case_output["case_metrics"]
            for key in aggregate_metrics:
                aggregate_metrics[key] += metrics[key]
            aggregate_counts.update(metrics.get("hallucination_type_counts", {}))
            aggregate_primary_counts.update(metrics.get("primary_hallucination_type_counts", {}))
            log_progress(
                f"Case {index}/{len(case_dirs)} {sample_dir.name}: done in {time.time() - started_at:.1f}s; wrote {output_path}",
                enabled=progress_enabled,
            )
        except Exception as exc:  # noqa: BLE001 - keep batch processing robust.
            failed.append({"sample_dir": str(sample_dir), "error": repr(exc)})
            log_progress(f"Case {index}/{len(case_dirs)} {sample_dir.name}: failed with {exc!r}", enabled=progress_enabled)
            if args.fail_fast:
                break

    generated_rate = (
        aggregate_metrics["num_erroneous_generated_claims"] / aggregate_metrics["num_generated_claims"]
        if aggregate_metrics["num_generated_claims"]
        else math.nan
    )
    omission_rate = (
        aggregate_metrics["num_omission_events"] / aggregate_metrics["num_reference_claims"]
        if aggregate_metrics["num_reference_claims"]
        else math.nan
    )
    total_rate = (
        (aggregate_metrics["num_erroneous_generated_claims"] + aggregate_metrics["num_omission_events"])
        / (aggregate_metrics["num_generated_claims"] + aggregate_metrics["num_reference_claims"])
        if (aggregate_metrics["num_generated_claims"] + aggregate_metrics["num_reference_claims"])
        else math.nan
    )
    unverifiable_rate = (
        aggregate_metrics["num_unverifiable_generated_claims"] / aggregate_metrics["num_generated_claims"]
        if aggregate_metrics["num_generated_claims"]
        else math.nan
    )
    batch_summary = {
        "target_dir": str(target_dir),
        "case_glob": args.case_glob,
        "output_name": args.output_name,
        "num_case_dirs": len(case_dirs),
        "num_processed": processed,
        "num_skipped": skipped,
        "num_failed": len(failed),
        "failed": failed,
        "aggregate_metrics": {
            **aggregate_metrics,
            "hallucination_rate": None if math.isnan(generated_rate) else round(generated_rate, 6),
            "generated_claim_error_rate": None if math.isnan(generated_rate) else round(generated_rate, 6),
            "omission_rate": None if math.isnan(omission_rate) else round(omission_rate, 6),
            "total_hallucination_event_rate": None if math.isnan(total_rate) else round(total_rate, 6),
            "unverifiable_rate": None if math.isnan(unverifiable_rate) else round(unverifiable_rate, 6),
            "hallucination_type_counts": dict(sorted(aggregate_counts.items())),
            "primary_hallucination_type_counts": dict(sorted(aggregate_primary_counts.items())),
        },
        "config": {
            "llm_backend": "openai_compatible",
            "model": llm_config.model,
            "base_url": llm_config.base_url,
            "temperature": llm_config.temperature,
            "max_tokens": llm_config.max_tokens,
            "use_json_mode": llm_config.use_json_mode,
            "include_images": llm_config.include_images,
            "image_stage": llm_config.image_stage if llm_config.include_images else "none",
            "max_llm_images": llm_config.max_llm_images if llm_config.include_images else 0,
            "include_image_manifest": not args.no_image_manifest,
            "max_images": args.max_images,
            "strict_negative_claims": config.strict_negative_claims,
            "include_negative_omissions": config.include_negative_omissions,
        },
    }

    if args.write_batch_summary:
        summary_path = target_dir / args.batch_summary_name
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(batch_summary, f, ensure_ascii=False, indent=2)
            f.write("\n")
        log_progress(f"Batch summary written to {summary_path}", enabled=progress_enabled)
    log_progress(
        f"Finished hallucination detection: processed={processed}, skipped={skipped}, failed={len(failed)}",
        enabled=progress_enabled,
    )
    return batch_summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="LLM-backed claim-level medical hallucination detection for RadClaw outputs.",
    )
    parser.add_argument(
        "--target-dir",
        required=True,
        help="Directory containing sample_* case folders, for example output/mimic/full_radclaw.",
    )
    parser.add_argument("--case-glob", default="sample_*", help="Glob used to find case folders.")
    parser.add_argument("--limit", type=int, default=None, help="Optional maximum number of case folders to process.")
    parser.add_argument(
        "--output-name",
        default="hallucination_detection.json",
        help="Per-case JSON filename written inside each sample folder.",
    )

    llm_group = parser.add_argument_group("LLM backbone")
    llm_group.add_argument("--model", default=None, help="OpenAI-compatible chat or vision model name. Required unless HALLUCINATION_MODEL/OPENAI_MODEL is set.")
    llm_group.add_argument("--api-key", default=None, help="API key. Prefer an environment variable for shared systems.")
    llm_group.add_argument("--api-key-env", default="OPENAI_API_KEY", help="Environment variable to read the API key from.")
    llm_group.add_argument("--base-url", default=None, help="Optional OpenAI-compatible base URL, e.g. a vLLM server /v1 endpoint.")
    llm_group.add_argument("--temperature", type=float, default=0.0)
    llm_group.add_argument("--max-tokens", type=int, default=32768)
    llm_group.add_argument("--timeout", type=float, default=120.0)
    llm_group.add_argument("--max-retries", type=int, default=5)
    llm_group.add_argument("--no-json-mode", action="store_true", help="Disable response_format={type: json_object} for APIs that do not support it.")

    image_group = parser.add_argument_group("Image evidence")
    image_group.add_argument("--no-image-manifest", action="store_true", help="Do not include image file manifests from sample/images in output/prompt text.")
    image_group.add_argument("--max-images", type=int, default=64, help="Maximum image files to record in the manifest.")
    image_group.add_argument("--include-images", action=argparse.BooleanOptionalAction, default=True, help="Send raster images to a multimodal model as image evidence. Use --no-include-images for text-only models.")
    image_group.add_argument("--image-stage", choices=("taxonomy", "all"), default="all", help="When --include-images is set, attach images to only final taxonomy prompt or all support prompts.")
    image_group.add_argument("--max-llm-images", type=int, default=8, help="Maximum raster images to send to the LLM per case.")
    image_group.add_argument("--image-detail", choices=("low", "high", "auto"), default="low")

    parser.add_argument("--max-report-chars", type=int, default=20000)
    parser.add_argument("--max-evidence-chars", type=int, default=12000)
    parser.add_argument("--min-match-score", type=float, default=0.45, help="Retained for config traceability; LLM performs semantic matching.")
    parser.add_argument("--omission-match-score", type=float, default=0.42, help="Retained for config traceability; LLM performs semantic matching.")
    parser.add_argument("--impression-match-score", type=float, default=0.38, help="Retained for config traceability; LLM performs semantic matching.")
    parser.add_argument("--strict-negative-claims", action="store_true", help="Tell the taxonomy LLM to count unsupported generated negative/normal claims as errors.")
    parser.add_argument("--include-negative-omissions", action="store_true", help="Tell the taxonomy LLM to also emit H2 events for clinically important negative/normal reference claims.")
    parser.add_argument("--overwrite", action="store_true", help="Re-run cases even when the per-case output JSON already exists.")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-case and per-stage progress output.")
    parser.add_argument(
        "--write-batch-summary",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write target-level hallucination_batch_summary.json.",
    )
    parser.add_argument("--batch-summary-name", default="hallucination_batch_summary.json")
    return parser
def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        summary = process_target(args)
    except Exception as exc:  # noqa: BLE001 - CLI should report configuration/runtime errors cleanly.
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    aggregate = summary["aggregate_metrics"]
    print(
        "Processed {processed}/{total} cases; failed={failed}; "
        "generated_errors={errors}; omissions={omissions}; hallucination_rate={rate}".format(
            processed=summary["num_processed"],
            total=summary["num_case_dirs"],
            failed=summary["num_failed"],
            errors=aggregate["num_erroneous_generated_claims"],
            omissions=aggregate["num_omission_events"],
            rate=aggregate["hallucination_rate"],
        )
    )
    return 0 if summary["num_failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
