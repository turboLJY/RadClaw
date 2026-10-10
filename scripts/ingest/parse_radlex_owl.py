#!/usr/bin/env python3
"""Parse RadLex OWL → JSON using lxml (faster and more robust than owlready2).

Usage:
    python scripts/ingest/parse_radlex_owl.py [--owl-path data/raw/RadLex.owl] [--output data/raw/radlex_parsed.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from lxml import etree


NS = {
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
    "owl": "http://www.w3.org/2002/07/owl#",
    "skos": "http://www.w3.org/2004/02/skos/core#",
    "obo": "http://purl.obolibrary.org/obo/",
    "radlex": "http://www.radlex.org/RID/",
    "oboInOwl": "http://www.geneontology.org/formats/oboInOwl#",
}

RADLEX_NS = "http://www.radlex.org/RID/"


def parse_owl(owl_path: str) -> list[dict]:
    print(f"Parsing {owl_path} with lxml ...")
    tree = etree.parse(owl_path)
    root = tree.getroot()

    classes = root.findall(".//owl:Class", NS)
    print(f"  Found {len(classes)} owl:Class elements")

    entries = []
    for cls in classes:
        about = cls.get(f"{{{NS['rdf']}}}about", "")
        if "RID" not in about:
            continue

        rid = about.rsplit("/", 1)[-1].rsplit("#", 1)[-1]
        if not rid.startswith("RID"):
            continue

        label = ""
        for label_el in cls.findall("rdfs:label", NS):
            lang = label_el.get(f"{{{NS['rdf'].replace('22-rdf-syntax-ns', 'XML/1998/namespace').replace('http://www.w3.org/1999/02/22-rdf-syntax-ns#', 'http://www.w3.org/XML/1998/namespace')}}}lang", "")
            if not lang:
                lang = label_el.get("{http://www.w3.org/XML/1998/namespace}lang", "")
            if lang == "en" and label_el.text:
                label = label_el.text.strip()
                break
        if not label:
            for label_el in cls.findall("rdfs:label", NS):
                if label_el.text:
                    label = label_el.text.strip()
                    break

        if not label:
            label = rid

        definition = ""
        def_el = cls.find(f"{{{RADLEX_NS}}}Definition", NS)
        if def_el is not None and def_el.text:
            definition = def_el.text.strip()
        if not definition:
            comment_el = cls.find(f"{{{RADLEX_NS}}}Comment", NS)
            if comment_el is not None and comment_el.text:
                definition = comment_el.text.strip()

        synonyms = []
        for syn_el in cls.findall(f"{{{RADLEX_NS}}}Synonym", NS):
            if syn_el.text and syn_el.text.strip():
                synonyms.append(syn_el.text.strip())
        for syn_el in cls.findall(f"{{{RADLEX_NS}}}Acronym", NS):
            if syn_el.text and syn_el.text.strip():
                synonyms.append(syn_el.text.strip())

        parent_code = ""
        subclass_el = cls.find("rdfs:subClassOf", NS)
        if subclass_el is not None:
            parent_ref = subclass_el.get(f"{{{NS['rdf']}}}resource", "")
            if "RID" in parent_ref:
                parent_code = parent_ref.rsplit("/", 1)[-1].rsplit("#", 1)[-1]

        parts = [label]
        if definition:
            parts.append(definition)
        embed_text = ": ".join(parts[:2])
        if synonyms:
            embed_text += ". synonyms: " + ", ".join(synonyms[:5])

        entries.append({
            "code": rid,
            "preferred_name": label,
            "definition": definition,
            "synonyms": synonyms,
            "category": "",
            "parent_code": parent_code,
            "embed_text": embed_text,
        })

    print(f"  Extracted {len(entries)} RadLex entries with RID codes.")
    with_def = sum(1 for e in entries if e["definition"])
    with_syn = sum(1 for e in entries if e["synonyms"])
    print(f"  With definitions: {with_def}, with synonyms: {with_syn}")

    code_to_name = {e["code"]: e["preferred_name"] for e in entries}
    enriched = 0
    for e in entries:
        pc = e.get("parent_code", "")
        if pc and pc in code_to_name:
            parent_name = code_to_name[pc]
            if parent_name.lower() not in e["embed_text"].lower():
                e["embed_text"] += f". parent: {parent_name}"
                enriched += 1
    print(f"  Enriched {enriched} entries with parent concept name in embed_text")

    return entries


def main():
    parser = argparse.ArgumentParser(description="Parse RadLex OWL to JSON")
    parser.add_argument("--owl-path", default="data/raw/RadLex.owl")
    parser.add_argument("--output", default="data/raw/radlex_parsed.json")
    args = parser.parse_args()

    if not Path(args.owl_path).exists():
        print(f"Error: OWL file not found at {args.owl_path}", file=sys.stderr)
        sys.exit(1)

    entries = parse_owl(args.owl_path)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=1)
    print(f"  Wrote {args.output} ({Path(args.output).stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()