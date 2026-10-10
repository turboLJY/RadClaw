"""
Session image store for derived images created by visual tools.
Stores PIL images in memory with unique IDs for reference in evidence chains.
"""

from __future__ import annotations

import base64
import io
import logging
from PIL import Image

logger = logging.getLogger(__name__)


class SessionImageStore:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self._images: dict[str, tuple[Image.Image, str]] = {}
        self._tool_chains: dict[str, list[str]] = {}
        self._metadata: dict[str, dict] = {}
        self._counter = 0

    def save(self, image: Image.Image, description: str, tool_name: str = "", metadata: dict | None = None) -> str:
        self._counter += 1
        image_id = f"IMG_{self.session_id[:8]}_{self._counter:03d}"
        self._images[image_id] = (image, description)
        self._tool_chains[image_id] = [tool_name] if tool_name else []
        self._metadata[image_id] = metadata or {}
        return image_id

    def get(self, image_id: str) -> tuple[Image.Image, str] | None:
        return self._images.get(image_id)

    def add_tool_to_chain(self, image_id: str, tool_name: str) -> None:
        if image_id in self._tool_chains:
            self._tool_chains[image_id].append(tool_name)

    def get_tool_chain(self, image_id: str) -> list[str]:
        return self._tool_chains.get(image_id, [])

    def to_base64(self, image_id: str, fmt: str = "PNG") -> str | None:
        entry = self._images.get(image_id)
        if not entry:
            return None
        img, _ = entry
        buffer = io.BytesIO()
        img.save(buffer, format=fmt)
        return base64.b64encode(buffer.getvalue()).decode()

    def list_all(self) -> list[dict]:
        return [
            {
                "image_id": k,
                "description": v[1],
                "size": v[0].size,
                "tool_chain": self._tool_chains.get(k, []),
                "metadata": self._metadata.get(k, {}),
            }
            for k, v in self._images.items()
        ]

    def count(self) -> int:
        return len(self._images)