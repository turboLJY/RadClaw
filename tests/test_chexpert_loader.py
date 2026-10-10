"""Test CheXpert dataset loader."""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_loader_import():
    from src.data.chexpert_loader import CheXpertLoader
    assert CheXpertLoader is not None


def test_image_store():
    from PIL import Image
    from src.data.image_store import SessionImageStore

    store = SessionImageStore("test123")
    img = Image.new("RGB", (100, 100), color="red")

    img_id = store.save(img, "test image", "test_tool")
    assert img_id.startswith("IMG_")
    assert store.count() == 1

    retrieved = store.get(img_id)
    assert retrieved is not None
    assert retrieved[1] == "test image"

    b64 = store.to_base64(img_id)
    assert b64 is not None
    assert len(b64) > 0

    chain = store.get_tool_chain(img_id)
    assert chain == ["test_tool"]

    all_imgs = store.list_all()
    assert len(all_imgs) == 1
    assert all_imgs[0]["image_id"] == img_id


def test_image_store_missing():
    from src.data.image_store import SessionImageStore

    store = SessionImageStore("test")
    assert store.get("nonexistent") is None
    assert store.to_base64("nonexistent") is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])