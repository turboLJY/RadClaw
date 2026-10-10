"""
Evidence retrieval tools for the RadClaw evidence agent.

Provides tools for standardizing observations, concept lookup,
literature search, case retrieval, evidence scoring, conflict
detection, and evidence summarization.
"""

from __future__ import annotations

import logging

from src.core.tool import ToolRegistry

logger = logging.getLogger(__name__)


def _build_radlex_source(indexes_dir: str, embedder):
    """Try to build a RadLexSource; return None on failure."""
    try:
        from src.kb.radlex_source import RadLexSource

        source = RadLexSource(indexes_dir=indexes_dir, embedder=embedder)
        if source.is_available():
            logger.info("RadLex vector source available (%d vectors)", source._store.size)
            return source
        logger.info("RadLex vector source not available, using legacy matching")
    except Exception:
        logger.debug("RadLex vector source init failed", exc_info=True)
    return None


def _build_snomed_source(indexes_dir: str, embedder, settings: dict | None = None):
    """Try to build a SnomedSource; return None on failure."""
    try:
        from src.kb.snomed_source import SnomedSource

        snomed_cfg = (settings or {}).get("snomed", {})
        source = SnomedSource(
            indexes_dir=indexes_dir,
            embedder=embedder,
            snowstorm_url=snomed_cfg.get("snowstorm_url", "https://tx.fhir.org/r4"),
            edition=snomed_cfg.get("edition", "900000000000207008"),
            mode=snomed_cfg.get("mode", "api"),
        )
        if source.is_available():
            logger.info("SNOMED source available (mode=%s)", snomed_cfg.get("mode", "api"))
            return source
    except Exception:
        logger.debug("SnomedSource init failed", exc_info=True)
    return None


def _build_embedder(settings: dict | None = None):
    """Try to create a MedicalEmbedder; return None on failure."""
    try:
        from src.kb.embedder import MedicalEmbedder

        if settings:
            emb_cfg = settings.get("embedding", {})
            return MedicalEmbedder(
                model_name=emb_cfg.get(
                    "model_name",
                    "pritamdeka/S-PubMedBert-MS-MARCO",
                ),
                device=emb_cfg.get("device", "cpu"),
            )
        return MedicalEmbedder()
    except Exception:
        logger.debug("MedicalEmbedder init failed", exc_info=True)
    return None


def _build_guidelines_store(indexes_dir: str, knowledge_dir: str, embedder):
    """Try to build a GuidelinesStore; return None on failure."""
    try:
        from src.kb.guidelines_store import GuidelinesStore

        store = GuidelinesStore(
            indexes_dir=indexes_dir,
            knowledge_dir=knowledge_dir,
            embedder=embedder,
        )
        if store.is_available():
            logger.info("Guidelines vector store available")
            return store
    except Exception:
        logger.debug("GuidelinesStore init failed", exc_info=True)
    return None


def _build_pubmed_client(settings: dict | None = None):
    """Try to build a PubMedClient; return None if disabled."""
    try:
        pubmed_cfg = (settings or {}).get("pubmed", {})
        if not pubmed_cfg.get("enabled", True):
            return None
        from src.kb.pubmed_client import PubMedClient

        return PubMedClient(
            api_key=pubmed_cfg.get("api_key"),
            max_results_per_query=pubmed_cfg.get("max_results_per_query", 5),
        )
    except Exception:
        logger.debug("PubMedClient init failed", exc_info=True)
    return None


def build_evidence_registry(
    knowledge_dir: str = "./data/knowledge",
    indexes_dir: str | None = None,
    settings: dict | None = None,
) -> ToolRegistry:
    """Import and register all 7 evidence retrieval tools."""
    from src.tools.evidence.standardize_obs import StandardizeObservationTool
    from src.tools.evidence.concept_lookup import ConceptLookupTool
    from src.tools.evidence.literature_search import LiteratureSearchTool
    from src.tools.evidence.case_retrieval import CaseRetrievalTool
    from src.tools.evidence.evidence_scoring import EvidenceScoringTool
    from src.tools.evidence.conflict_detection import ConflictDetectionTool
    from src.tools.evidence.evidence_summary import EvidenceSummaryTool

    embedder = _build_embedder(settings)
    radlex_source = None
    snomed_source = None
    pubmed_client = None

    if indexes_dir and embedder:
        radlex_source = _build_radlex_source(indexes_dir, embedder)
    if indexes_dir:
        snomed_source = _build_snomed_source(indexes_dir or "", embedder, settings)

    pubmed_client = _build_pubmed_client(settings)

    remote_cfg = (settings or {}).get("remote_kb", {})
    remote_base_url: str | None = remote_cfg.get("base_url") or None
    remote_timeout: float = float(remote_cfg.get("timeout", 10.0))

    registry = ToolRegistry()

    registry.register(StandardizeObservationTool(
        knowledge_dir=knowledge_dir,
        radlex_source=radlex_source,
        snomed_source=snomed_source,
    ))
    registry.register(ConceptLookupTool(
        knowledge_dir=knowledge_dir,
        radlex_source=radlex_source,
        snomed_source=snomed_source,
    ))
    registry.register(LiteratureSearchTool(
        remote_base_url=remote_base_url,
        timeout=remote_timeout,
        pubmed_client=pubmed_client,
    ))
    registry.register(CaseRetrievalTool(
        remote_base_url=remote_base_url,
        timeout=remote_timeout,
    ))
    registry.register(EvidenceScoringTool(knowledge_dir=knowledge_dir, embedder=embedder))
    registry.register(ConflictDetectionTool(knowledge_dir=knowledge_dir))
    registry.register(EvidenceSummaryTool(knowledge_dir=knowledge_dir))

    return registry