#!/usr/bin/env python3
"""Fetch SNOMED CT radiology-related concepts via Snowstorm FHIR API.

Usage:
    python scripts/ingest/download_snomed.py [--output data/raw/snomed_parsed.json]

Fetches subsets of SNOMED CT relevant to radiology via the public
Snowstorm FHIR terminology server. No license needed for API access.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
import urllib.parse
from pathlib import Path


FHIR_TX_URL = "https://tx.fhir.org/r4"
EDITION = "900000000000207008"

RADIOLOGY_QUERIES = [
    "radiological finding",
    "radiographic finding",
    "chest x-ray finding",
    "CT finding",
    "MRI finding",
    "ultrasound finding",
    "mammographic finding",
    "pleural effusion",
    "pneumothorax",
    "pneumonia",
    "atelectasis",
    "pulmonary edema",
    "cardiomegaly",
    "mass",
    "nodule",
    "fracture",
    "consolidation",
    "opacity",
    "calcification",
    "lymphadenopathy",
    "hepatomegaly",
    "splenomegaly",
    "ascites",
    "hernia",
    "stenosis",
    "aneurysm",
    "embolism",
    "thrombosis",
    "hemorrhage",
    "infarction",
    "edema",
    "fibrosis",
    "necrosis",
    "abscess",
    "cyst",
    "tumor",
    "metastasis",
    "brain lesion",
    "liver lesion",
    "kidney lesion",
    "bone lesion",
    "lung lesion",
    "breast lesion",
    "prostate lesion",
    "thyroid lesion",
    "ovarian lesion",
    "pancreatic lesion",
    "adrenal lesion",
]

ANATOMY_QUERIES = [
    "thoracic structure",
    "lung structure",
    "heart structure",
    "liver structure",
    "kidney structure",
    "brain structure",
    "spine structure",
    "bone structure",
    "breast structure",
    "abdominal structure",
    "pelvic structure",
    "vascular structure",
]


def _lookup_designations(code: str) -> list[str]:
    """Fetch synonyms/designations for a SNOMED code via $lookup."""
    url = (
        f"{FHIR_TX_URL}/CodeSystem/$lookup"
        f"?system=http://snomed.info/sct&code={code}"
    )
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/fhir+json"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return []

    designations = []
    for param in data.get("parameter", []):
        if param.get("name") == "designation":
            for part in param.get("part", []):
                if part.get("name") == "value":
                    val = part.get("valueString", "").strip()
                    if val:
                        designations.append(val)
    return designations


def fetch_concepts(query: str, max_count: int = 100, fetch_synonyms: bool = True) -> list[dict]:
    params = urllib.parse.urlencode({
        "filter": query,
        "count": str(max_count),
        "url": "http://snomed.info/sct?fhir_vs",
    })
    url = f"{FHIR_TX_URL}/ValueSet/$expand?{params}"

    try:
        req = urllib.request.Request(url, headers={"Accept": "application/fhir+json"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        print(f"  [error] {query}: {e}")
        return []

    results = []
    expansion = data.get("expansion", {})
    for item in expansion.get("contains", []):
        code = item.get("code", "")
        display = item.get("display", "")
        if not (code and display):
            continue

        synonyms = []
        if fetch_synonyms:
            raw = _lookup_designations(code)
            synonyms = [s for s in raw if s.lower() != display.lower()]
            time.sleep(0.15)

        embed_parts = [display]
        if synonyms:
            embed_parts.append("synonyms: " + ", ".join(synonyms[:5]))
        embed_text = ". ".join(embed_parts)

        results.append({
            "sctid": code,
            "term": display,
            "definition": display,
            "synonyms": synonyms,
            "semantic_type": "",
            "embed_text": embed_text,
        })

    return results


def main():
    parser = argparse.ArgumentParser(description="Fetch SNOMED CT radiology concepts")
    parser.add_argument("--output", default="data/raw/snomed_parsed.json")
    parser.add_argument("--max-per-query", type=int, default=100)
    args = parser.parse_args()

    all_queries = RADIOLOGY_QUERIES + ANATOMY_QUERIES
    seen_codes: set[str] = set()
    all_entries: list[dict] = []

    print(f"Fetching SNOMED concepts for {len(all_queries)} queries ...")
    for i, query in enumerate(all_queries, 1):
        print(f"  [{i}/{len(all_queries)}] {query} ...", end=" ", flush=True)
        concepts = fetch_concepts(query, args.max_per_query)
        new = 0
        for c in concepts:
            if c["sctid"] not in seen_codes:
                seen_codes.add(c["sctid"])
                all_entries.append(c)
                new += 1
        print(f"{len(concepts)} found, {new} new")
        time.sleep(0.5)

    print(f"\nTotal unique concepts: {len(all_entries)}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_entries, f, ensure_ascii=False, indent=1)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()