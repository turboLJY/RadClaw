"""Async PubMed / PMC E-utilities client with rate limiting and caching."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any
from xml.etree import ElementTree

from src.kb.base import KBSearchResult

logger = logging.getLogger(__name__)

_EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
_RATE_LIMIT_NO_KEY = 3
_RATE_LIMIT_WITH_KEY = 10


@dataclass
class PubMedArticle:
    pmid: str
    title: str
    abstract: str
    authors: list[str] = field(default_factory=list)
    journal: str = ""
    year: str = ""
    doi: str = ""
    mesh_terms: list[str] = field(default_factory=list)
    pmcid: str = ""


class PubMedClient:
    def __init__(
        self,
        api_key: str | None = None,
        max_results_per_query: int = 5,
    ) -> None:
        self._api_key = api_key
        self._max_results = max_results_per_query
        self._rate_limit = _RATE_LIMIT_WITH_KEY if api_key else _RATE_LIMIT_NO_KEY
        self._last_request_time = 0.0
        self._cache: dict[str, Any] = {}

    @property
    def enabled(self) -> bool:
        return True

    def _rate_wait(self) -> None:
        interval = 1.0 / self._rate_limit
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < interval:
            time.sleep(interval - elapsed)
        self._last_request_time = time.monotonic()

    def _build_url(self, endpoint: str, params: dict) -> str:
        if self._api_key:
            params["api_key"] = self._api_key
        return f"{_EUTILS_BASE}/{endpoint}?{urllib.parse.urlencode(params)}"

    def _fetch(self, url: str) -> bytes:
        self._rate_wait()
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.read()

    def _esearch(self, query: str, max_results: int) -> list[str]:
        cache_key = f"esearch:{query}:{max_results}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        url = self._build_url("esearch.fcgi", {
            "db": "pubmed",
            "term": query,
            "retmax": str(max_results),
            "retmode": "json",
            "sort": "relevance",
        })

        try:
            data = json.loads(self._fetch(url))
            ids = data.get("esearchresult", {}).get("idlist", [])
            self._cache[cache_key] = ids
            return ids
        except Exception as e:
            logger.warning("PubMed esearch failed: %s", e)
            return []

    def _efetch(self, pmids: list[str]) -> list[PubMedArticle]:
        if not pmids:
            return []

        cache_key = f"efetch:{','.join(sorted(pmids))}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        url = self._build_url("efetch.fcgi", {
            "db": "pubmed",
            "id": ",".join(pmids),
            "rettype": "xml",
            "retmode": "xml",
        })

        try:
            xml_bytes = self._fetch(url)
            articles = self._parse_pubmed_xml(xml_bytes)
            self._cache[cache_key] = articles
            return articles
        except Exception as e:
            logger.warning("PubMed efetch failed: %s", e)
            return []

    @staticmethod
    def _parse_pubmed_xml(xml_bytes: bytes) -> list[PubMedArticle]:
        root = ElementTree.fromstring(xml_bytes)
        articles = []

        for article_el in root.findall(".//PubmedArticle"):
            pmid = ""
            pmid_el = article_el.find(".//PMID")
            if pmid_el is not None and pmid_el.text:
                pmid = pmid_el.text

            title = ""
            title_el = article_el.find(".//ArticleTitle")
            if title_el is not None:
                title = "".join(title_el.itertext())

            abstract = ""
            abstract_el = article_el.find(".//AbstractText")
            if abstract_el is not None:
                abstract = "".join(abstract_el.itertext())

            authors = []
            for author_el in article_el.findall(".//Author"):
                last = author_el.findtext("LastName", "")
                first = author_el.findtext("ForeName", "")
                if last:
                    authors.append(f"{last} {first}".strip())

            journal = article_el.findtext(".//Journal/Title", "")
            year = article_el.findtext(".//PubDate/Year", "")
            if not year:
                medline_date = article_el.findtext(".//PubDate/MedlineDate", "")
                if medline_date:
                    year = medline_date[:4]

            doi = ""
            for id_el in article_el.findall(".//ArticleId"):
                if id_el.get("IdType") == "doi" and id_el.text:
                    doi = id_el.text
                    break

            pmcid = ""
            for id_el in article_el.findall(".//ArticleId"):
                if id_el.get("IdType") == "pmc" and id_el.text:
                    pmcid = id_el.text
                    break

            mesh_terms = []
            for mesh_el in article_el.findall(".//MeshHeading/DescriptorName"):
                if mesh_el.text:
                    mesh_terms.append(mesh_el.text)

            articles.append(PubMedArticle(
                pmid=pmid,
                title=title,
                abstract=abstract,
                authors=authors[:5],
                journal=journal,
                year=year,
                doi=doi,
                mesh_terms=mesh_terms,
                pmcid=pmcid,
            ))

        return articles

    def search_sync(self, query: str, max_results: int | None = None) -> list[KBSearchResult]:
        mr = max_results or self._max_results
        pmids = self._esearch(query, mr)
        if not pmids:
            return []

        articles = self._efetch(pmids)
        results = []
        for art in articles:
            content_parts = []
            if art.title:
                content_parts.append(art.title)
            if art.abstract:
                content_parts.append(art.abstract[:500])

            citation = ""
            if art.authors:
                citation = art.authors[0]
                if len(art.authors) > 1:
                    citation += " et al."
            if art.journal:
                citation += f" {art.journal}"
            if art.year:
                citation += f" ({art.year})"

            results.append(KBSearchResult(
                source_id=f"PMID:{art.pmid}",
                source_system="pubmed",
                preferred_term=art.title,
                content=". ".join(content_parts),
                score=0.8,
                metadata={
                    "pmid": art.pmid,
                    "pmcid": art.pmcid,
                    "authors": art.authors,
                    "journal": art.journal,
                    "year": art.year,
                    "doi": art.doi,
                    "citation": citation,
                    "mesh_terms": art.mesh_terms,
                    "has_full_text": bool(art.pmcid),
                },
            ))

        return results

    # ------------------------------------------------------------------
    # PMC full-text retrieval (on-demand)
    # ------------------------------------------------------------------

    def _fetch_pmc_xml(self, pmcid: str) -> str:
        pmc_num = pmcid.replace("PMC", "")
        url = self._build_url("efetch.fcgi", {
            "db": "pmc",
            "id": pmc_num,
            "rettype": "xml",
            "retmode": "xml",
        })
        return self._fetch(url).decode("utf-8")

    @staticmethod
    def _parse_pmc_jats(xml_str: str) -> dict:
        """Parse PMC JATS XML into structured sections."""
        root = ElementTree.fromstring(xml_str)

        def _text(el) -> str:
            if el is None:
                return ""
            return "".join(el.itertext()).strip()

        title = _text(root.find(".//article-title"))

        abstract_parts = []
        for abs_el in root.findall(".//abstract//p"):
            t = _text(abs_el)
            if t:
                abstract_parts.append(t)
        abstract = " ".join(abstract_parts)

        sections = []
        for sec in root.findall(".//body/sec"):
            sec_title = _text(sec.find("title"))
            paragraphs = []
            for p in sec.findall(".//p"):
                t = _text(p)
                if t:
                    paragraphs.append(t)
            if paragraphs:
                sections.append({
                    "title": sec_title,
                    "text": " ".join(paragraphs),
                })

        tables = []
        for tw in root.findall(".//table-wrap"):
            caption = _text(tw.find(".//caption"))
            rows = []
            for tr in tw.findall(".//tr"):
                cells = [_text(c) for c in tr.findall("*")]
                if any(cells):
                    rows.append(cells)
            if rows:
                tables.append({"caption": caption, "rows": rows})

        refs = []
        for ref in root.findall(".//ref-list/ref"):
            citation = _text(ref)
            if citation:
                refs.append(citation)

        full_text = " ".join(
            s["text"] for s in sections
        )

        return {
            "title": title,
            "abstract": abstract,
            "sections": sections,
            "tables": tables,
            "references": refs,
            "full_text": full_text,
        }

    def fetch_full_text_sync(self, pmcid: str) -> dict | None:
        """Fetch and parse PMC full text. Returns None if unavailable."""
        if not pmcid:
            return None
        cache_key = f"pmc_full:{pmcid}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        try:
            xml_str = self._fetch_pmc_xml(pmcid)
            result = self._parse_pmc_jats(xml_str)
            result["pmcid"] = pmcid
            self._cache[cache_key] = result
            return result
        except Exception as e:
            logger.warning("PMC full-text fetch failed for %s: %s", pmcid, e)
            return None

    async def fetch_full_text(self, pmcid: str) -> dict | None:
        return await asyncio.to_thread(self.fetch_full_text_sync, pmcid)

    async def search(self, query: str, max_results: int | None = None) -> list[KBSearchResult]:
        return await asyncio.to_thread(self.search_sync, query, max_results)
