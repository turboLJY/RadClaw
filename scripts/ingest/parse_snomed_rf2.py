#!/usr/bin/env python3
"""Parse SNOMED CT RF2 Snapshot files into structured JSON for radiology use.

Reads directly from the zip archive. Filters to radiology-relevant semantic types.

Usage:
    python scripts/ingest/parse_snomed_rf2.py \
        --input data/raw/SnomedCT_ManagedServiceUS_PRODUCTION_US1000124_20260301T120000Z.zip \
        --output data/raw/snomed_parsed.json \
        --terminology-output data/knowledge/terminology/snomed.json
"""

from __future__ import annotations

import argparse
import io
import json
import re
import zipfile
from collections import defaultdict
from pathlib import Path

# RF2 type IDs
FSN_TYPE   = "900000000000003001"   # Fully Specified Name
SYN_TYPE   = "900000000000013009"   # Synonym
DEF_TYPE   = "900000000000550004"   # Text Definition

# Semantic tags to include without keyword filter
INCLUDE_TAGS = {
    "finding",
    "body structure",
    "morphologic abnormality",
    "observable entity",
    "situation",              # e.g. "Patient on oxygen therapy"
    "clinical finding",       # alias used in some editions
}

# Disorder tag included only when term contains imaging keywords
IMAGING_RE = re.compile(
    r"pulmonar|thorac|pleural|lung|bronch|trachea|mediastin|cardiac|heart|aort|vascular|"
    r"hepat|liver|biliar|pancrea|renal|kidney|adrenal|abdom|pelvic|intestin|bowel|"
    r"cerebr|brain|intracranial|stroke|infarct|hemorrhag|haemorrhag|tumor|tumour|"
    r"neoplasm|cancer|carcinoma|sarcoma|lymphoma|metastas|lesion|mass|nodule|"
    r"fracture|dislocation|spinal|vertebr|skeletal|bone|"
    r"pneumo|effusion|consolidat|atelectas|emphysema|fibrosis|edema|oedema|"
    r"thrombos|embol|aneurysm|stenosis|occlusion|calcif",
    re.IGNORECASE,
)


def _find_terminology_dir(z: zipfile.ZipFile) -> str:
    for name in z.namelist():
        if "Snapshot/Terminology/" in name and name.endswith("sct2_Concept_Snapshot"):
            return name.rsplit("sct2_Concept_Snapshot", 1)[0]
    # fallback: search for concept file
    for name in z.namelist():
        if "sct2_Concept_Snapshot" in name and not name.startswith("__"):
            return name.rsplit("sct2_Concept_Snapshot", 1)[0]
    raise FileNotFoundError("Cannot find Snapshot/Terminology in zip")


def _read_tsv(z: zipfile.ZipFile, path: str):
    """Yield rows (as lists) from a tab-separated file inside zip, skipping header."""
    with z.open(path) as raw:
        reader = io.TextIOWrapper(raw, encoding="utf-8")
        next(reader)  # skip header
        for line in reader:
            yield line.rstrip("\n").split("\t")


def _extract_semantic_tag(fsn: str) -> str:
    if fsn.endswith(")") and "(" in fsn:
        return fsn.rsplit("(", 1)[1][:-1].lower()
    return ""


def parse(zip_path: str) -> list[dict]:
    z = zipfile.ZipFile(zip_path)
    term_dir = _find_terminology_dir(z)

    # ---------- Step 1: Active concept IDs ----------
    print("Loading active concepts ...")
    active_ids: set[str] = set()
    concept_file = term_dir + "sct2_Concept_Snapshot_US1000124_20260301.txt"
    for row in _read_tsv(z, concept_file):
        if len(row) < 3:
            continue
        if row[2] == "1":          # active
            active_ids.add(row[0])
    print(f"  {len(active_ids):,} active concepts")

    # ---------- Step 2: FSN + synonyms from Description ----------
    print("Loading descriptions ...")
    fsn: dict[str, str] = {}            # conceptId -> FSN text
    synonyms: dict[str, list[str]] = defaultdict(list)
    desc_file = term_dir + "sct2_Description_Snapshot-en_US1000124_20260301.txt"
    for row in _read_tsv(z, desc_file):
        if len(row) < 8:
            continue
        if row[2] != "1":       # inactive description
            continue
        cid   = row[4]
        ttype = row[6]
        term  = row[7]
        if cid not in active_ids:
            continue
        if ttype == FSN_TYPE:
            fsn[cid] = term
        elif ttype == SYN_TYPE:
            synonyms[cid].append(term)
    print(f"  FSNs: {len(fsn):,}")

    # ---------- Step 3: Text definitions ----------
    print("Loading text definitions ...")
    definitions: dict[str, str] = {}
    def_file = term_dir + "sct2_TextDefinition_Snapshot-en_US1000124_20260301.txt"
    for row in _read_tsv(z, def_file):
        if len(row) < 8:
            continue
        if row[2] != "1":
            continue
        cid  = row[4]
        term = row[7]
        if cid in active_ids and cid not in definitions:
            definitions[cid] = term
    print(f"  Definitions: {len(definitions):,}")

    # ---------- Step 4: Filter and build entries ----------
    print("Filtering to radiology-relevant concepts ...")
    entries: list[dict] = []
    skipped = 0

    for cid, fsn_text in fsn.items():
        tag = _extract_semantic_tag(fsn_text)
        preferred = fsn_text.rsplit("(", 1)[0].strip() if "(" in fsn_text else fsn_text

        if tag in INCLUDE_TAGS:
            pass  # include
        elif tag == "disorder":
            if not IMAGING_RE.search(preferred):
                skipped += 1
                continue
        else:
            skipped += 1
            continue

        syns = synonyms.get(cid, [])
        definition = definitions.get(cid, "")

        # Build embed_text
        embed_parts = [preferred]
        if definition:
            embed_parts.append(definition[:200])
        elif syns:
            embed_parts.append("Synonyms: " + ", ".join(syns[:4]))
        embed_text = ": ".join(embed_parts[:2])

        entries.append({
            "sctid": cid,
            "term": preferred,
            "fsn": fsn_text,
            "synonyms": syns,
            "definition": definition,
            "semantic_type": tag,
            "embed_text": embed_text,
        })

    print(f"  Included: {len(entries):,}  Skipped: {skipped:,}")
    return entries


def main():
    ap = argparse.ArgumentParser(description="Parse SNOMED CT RF2 Snapshot")
    ap.add_argument(
        "--input",
        default="data/raw/SnomedCT_ManagedServiceUS_PRODUCTION_US1000124_20260301T120000Z.zip",
    )
    ap.add_argument("--output", default="data/raw/snomed_parsed.json")
    ap.add_argument("--terminology-output", default="data/knowledge/terminology/snomed.json")
    args = ap.parse_args()

    entries = parse(args.input)

    # Write raw parsed
    raw_path = Path(args.output)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)
    print(f"\nWrote {raw_path} ({raw_path.stat().st_size // 1024:,} KB, {len(entries):,} entries)")

    # Write terminology knowledge file
    term_path = Path(args.terminology_output)
    term_path.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "system": "SNOMED CT",
        "title": "SNOMED Clinical Terms — US Edition 2026-03-01 (Radiology Subset)",
        "source": "SNOMED International / National Library of Medicine",
        "version": "20260301",
        "total_entries": len(entries),
        "entries": entries,
    }
    with open(term_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"Wrote {term_path} ({term_path.stat().st_size // 1024:,} KB)")

    # Sample output
    print("\nSample entries:")
    for e in entries[:5]:
        print(f"  [{e['semantic_type']:25s}] {e['sctid']}  {e['term']}")
        if e["definition"]:
            print(f"    def: {e['definition'][:80]}...")


if __name__ == "__main__":
    main()
