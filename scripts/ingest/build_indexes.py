#!/usr/bin/env python3
"""Build FAISS indexes from parsed JSON data.

Usage:
    python scripts/ingest/build_indexes.py [--source radlex] [--parsed-dir data/raw] [--output-dir data/indexes]

Supports: radlex, snomed, guidelines (or 'all').
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path


def _configure_hf_mirror() -> None:
    if os.environ.get("HF_ENDPOINT"):
        return
    try:
        urllib.request.urlopen("https://huggingface.co", timeout=5)
    except Exception:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        print("[info] Using HF mirror: https://hf-mirror.com")


def build_radlex_index(parsed_dir: str, output_dir: str, device: str, batch_size: int):
    from src.kb.embedder import MedicalEmbedder

    parsed_path = Path(parsed_dir) / "radlex_parsed.json"
    if not parsed_path.exists():
        print(f"Error: {parsed_path} not found. Run parse_radlex_owl.py first.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading parsed RadLex from {parsed_path} ...")
    with open(parsed_path, "r", encoding="utf-8") as f:
        entries = json.load(f)
    print(f"  {len(entries)} entries loaded.")

    texts = [e["embed_text"] for e in entries]
    metadata = [
        {
            "code": e["code"],
            "preferred_name": e["preferred_name"],
            "definition": e.get("definition", ""),
            "synonyms": e.get("synonyms", []),
            "category": e.get("category", ""),
            "parent_code": e.get("parent_code", ""),
        }
        for e in entries
    ]

    print("Loading embedding model ...")
    embedder = MedicalEmbedder(device=device)

    print(f"Embedding {len(texts)} texts (batch_size={batch_size}) ...")
    embeddings = embedder.embed_texts(texts, batch_size=batch_size, show_progress=True)
    print(f"  Embedding shape: {embeddings.shape}")

    from src.kb.vector_store import VectorStore

    store = VectorStore.build(embeddings, metadata, output_dir, "radlex")
    print(f"  RadLex index built: {store.size} vectors.")

    from src.kb.bm25_index import BM25Index
    BM25Index.build(texts, metadata, output_dir, "radlex")
    print("  RadLex BM25 index built.")


def build_snomed_index(parsed_dir: str, output_dir: str, device: str, batch_size: int):
    parsed_path = Path(parsed_dir) / "snomed_parsed.json"
    if not parsed_path.exists():
        print(f"[skip] {parsed_path} not found.")
        return

    from src.kb.embedder import MedicalEmbedder

    print(f"Loading parsed SNOMED from {parsed_path} ...")
    with open(parsed_path, "r", encoding="utf-8") as f:
        entries = json.load(f)
    print(f"  {len(entries)} entries loaded.")

    texts = [e.get("embed_text", e.get("term", "")) for e in entries]
    metadata = [
        {
            "sctid": e.get("sctid", ""),
            "term": e.get("term", ""),
            "definition": e.get("definition", ""),
            "synonyms": e.get("synonyms", []),
            "semantic_type": e.get("semantic_type", ""),
        }
        for e in entries
    ]

    embedder = MedicalEmbedder(device=device)
    print(f"Embedding {len(texts)} texts ...")
    embeddings = embedder.embed_texts(texts, batch_size=batch_size, show_progress=True)

    from src.kb.vector_store import VectorStore

    VectorStore.build(embeddings, metadata, output_dir, "snomed")
    print("  SNOMED index built.")

    from src.kb.bm25_index import BM25Index
    BM25Index.build(texts, metadata, output_dir, "snomed")
    print("  SNOMED BM25 index built.")


def _load_guideline_dir(dirpath: Path, pattern: str, seen_keys: set) -> list[dict]:
    entries = []
    for jf in sorted(dirpath.glob(pattern)):
        with open(jf, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict) or "entries" not in data:
            if isinstance(data, list):
                entries.extend(data)
            continue

        key = (data.get("title", ""), data.get("version", ""))
        if key in seen_keys:
            continue
        seen_keys.add(key)

        envelope_modality = data.get("modality", [])
        envelope_disease_type = data.get("disease_type", [])
        envelope_source = data.get("source", "")
        envelope_system = data.get("system", "")
        specialty = jf.parent.name

        for entry in data["entries"]:
            if "modality" not in entry:
                entry["modality"] = envelope_modality
            if "disease_type" not in entry:
                entry["disease_type"] = envelope_disease_type
            if "source" not in entry and envelope_source:
                entry["source"] = envelope_source
            if "system" not in entry and envelope_system:
                entry["system"] = envelope_system
            entry.setdefault("specialty", specialty)
            entries.append(entry)

        print(f"    {jf.relative_to(dirpath)}: {len(data['entries'])} entries")
    return entries


def build_guidelines_index(parsed_dir: str, output_dir: str, device: str, batch_size: int):
    data_root = Path(parsed_dir).parent / "knowledge"
    lit_dir = data_root / "literature" / "guidelines"
    legacy_dir = data_root / "guidelines"

    if not lit_dir.exists() and not legacy_dir.exists():
        print("[skip] No guidelines directories found.")
        return

    from src.kb.embedder import MedicalEmbedder

    seen_keys: set[tuple] = set()
    entries: list[dict] = []

    if lit_dir.exists():
        print(f"Scanning {lit_dir} ...")
        entries.extend(_load_guideline_dir(lit_dir, "**/*.json", seen_keys))

    if legacy_dir.exists():
        print(f"Scanning legacy {legacy_dir} ...")
        entries.extend(_load_guideline_dir(legacy_dir, "*.json", seen_keys))

    if not entries:
        print("[skip] No guideline entries found.")
        return

    print(f"  Total: {len(entries)} guideline entries")

    texts = [e.get("embed_text", e.get("title", "") + ": " + e.get("content", "")) for e in entries]
    metadata = entries

    embedder = MedicalEmbedder(device=device)
    print(f"Embedding {len(texts)} guideline texts ...")
    embeddings = embedder.embed_texts(texts, batch_size=batch_size, show_progress=True)

    from src.kb.vector_store import VectorStore

    VectorStore.build(embeddings, metadata, output_dir, "guidelines")
    print("  Guidelines index built.")

    from src.kb.bm25_index import BM25Index
    BM25Index.build(texts, metadata, output_dir, "guidelines")
    print("  Guidelines BM25 index built.")


def main():
    parser = argparse.ArgumentParser(description="Build FAISS indexes")
    parser.add_argument(
        "--source",
        default="all",
        choices=["radlex", "snomed", "guidelines", "all"],
    )
    parser.add_argument("--parsed-dir", default="data/raw")
    parser.add_argument("--output-dir", default="data/indexes")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    _configure_hf_mirror()

    if args.source in ("radlex", "all"):
        build_radlex_index(args.parsed_dir, args.output_dir, args.device, args.batch_size)
    if args.source in ("snomed", "all"):
        build_snomed_index(args.parsed_dir, args.output_dir, args.device, args.batch_size)
    if args.source in ("guidelines", "all"):
        build_guidelines_index(args.parsed_dir, args.output_dir, args.device, args.batch_size)


if __name__ == "__main__":
    main()
