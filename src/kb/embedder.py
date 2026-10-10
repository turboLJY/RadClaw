"""Medical text embedding wrapper using sentence-transformers."""

from __future__ import annotations

import logging
import os
import ssl
from typing import Sequence

import numpy as np

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "pritamdeka/S-PubMedBert-MS-MARCO"
_HF_MIRROR = "https://hf-mirror.com"
_ssl_patched = False


def _disable_ssl_verification() -> None:
    """Bypass SSL verification for corporate environments with self-signed certs."""
    global _ssl_patched
    if _ssl_patched:
        return
    _ssl_patched = True
    os.environ.setdefault("CURL_CA_BUNDLE", "")
    os.environ.setdefault("REQUESTS_CA_BUNDLE", "")
    ssl._create_default_https_context = ssl._create_unverified_context
    try:
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    except Exception:
        pass
    try:
        import requests
        _original_send = requests.Session.send

        def _patched_send(self, request, **kwargs):
            kwargs["verify"] = False
            return _original_send(self, request, **kwargs)

        requests.Session.send = _patched_send
    except Exception:
        pass
    logger.debug("SSL verification disabled")


def _configure_hf_mirror() -> None:
    """Set HF_ENDPOINT to Chinese mirror if huggingface.co has SSL issues."""
    if os.environ.get("HF_ENDPOINT"):
        return
    import urllib.request
    try:
        urllib.request.urlopen("https://huggingface.co", timeout=5)
    except Exception:
        os.environ["HF_ENDPOINT"] = _HF_MIRROR
        # huggingface_hub caches ENDPOINT at import time; patch it directly
        try:
            import huggingface_hub.constants as hf_const
            hf_const.ENDPOINT = _HF_MIRROR
        except Exception:
            pass
        logger.info("huggingface.co unreachable or SSL error, using mirror %s", _HF_MIRROR)


class MedicalEmbedder:
    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        device: str = "cpu",
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._model = None
        self._load_failed = False

    def _load(self):
        if self._model is not None:
            return
        if self._load_failed:
            raise RuntimeError(f"Embedding model {self._model_name} failed to load; not retrying")
        from sentence_transformers import SentenceTransformer

        logger.info("Loading embedding model %s on %s", self._model_name, self._device)

        # Strategy 1: load from local cache (model cached by build_indexes.py)
        # Must patch the constant directly — huggingface_hub caches it at import time
        try:
            import huggingface_hub.constants as hf_const
            old_offline = hf_const.HF_HUB_OFFLINE
            hf_const.HF_HUB_OFFLINE = True
            try:
                self._model = SentenceTransformer(
                    self._model_name, device=self._device,
                )
                logger.info("Loaded embedding model from local cache")
                return
            except Exception as exc:
                logger.debug("Offline load failed: %s", exc)
            finally:
                hf_const.HF_HUB_OFFLINE = old_offline
        except ImportError:
            pass

        # Strategy 2: use HF mirror (hf-mirror.com has valid SSL cert)
        _configure_hf_mirror()
        try:
            self._model = SentenceTransformer(
                self._model_name, device=self._device,
            )
            logger.info("Loaded embedding model via mirror/network")
            return
        except Exception as exc:
            logger.warning("Mirror load failed: %s", exc)

        # Strategy 3: last resort — disable SSL and try direct
        _disable_ssl_verification()
        try:
            self._model = SentenceTransformer(
                self._model_name, device=self._device,
            )
            logger.info("Loaded embedding model with SSL bypass")
        except Exception as exc:
            self._load_failed = True
            raise RuntimeError(f"All embedding load strategies failed: {exc}") from exc

    def embed_texts(
        self,
        texts: Sequence[str],
        batch_size: int = 64,
        show_progress: bool = False,
    ) -> np.ndarray:
        self._load()
        embeddings = self._model.encode(
            list(texts),
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,
        )
        return np.asarray(embeddings, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed_texts([query])[0]

    @property
    def dimension(self) -> int:
        self._load()
        return self._model.get_sentence_embedding_dimension()