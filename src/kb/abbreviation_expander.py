"""Abbreviation/acronym expansion for radiology queries.

Expands common radiology abbreviations in queries before vector search
to improve recall. E.g., "CXR consolidation" → "chest X-ray (CXR) consolidation".
"""

from __future__ import annotations

import json
import re
from pathlib import Path


_DEFAULT_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "knowledge" / "abbreviations.json"


class AbbreviationExpander:
    def __init__(self, path: str | Path = _DEFAULT_PATH) -> None:
        self._abbr_to_full: dict[str, str] = {}
        self._full_to_abbr: dict[str, str] = {}
        p = Path(path)
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                raw = json.load(f)
            for abbr, full in raw.items():
                self._abbr_to_full[abbr.upper()] = full
                self._full_to_abbr[full.lower()] = abbr.upper()

    @property
    def size(self) -> int:
        return len(self._abbr_to_full)

    def expand(self, abbr: str) -> str | None:
        return self._abbr_to_full.get(abbr.upper())

    def abbreviate(self, full_form: str) -> str | None:
        return self._full_to_abbr.get(full_form.lower())

    def expand_query(self, query: str) -> str:
        if not self._abbr_to_full:
            return query

        tokens = re.split(r'(\s+)', query)
        expanded = []
        for token in tokens:
            stripped = token.strip()
            upper = stripped.upper()
            if upper in self._abbr_to_full and (stripped.isupper() or stripped == upper):
                full = self._abbr_to_full[upper]
                expanded.append(f"{full} ({stripped})")
            else:
                expanded.append(token)
        return "".join(expanded)