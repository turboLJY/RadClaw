"""
FastAPI + WebSocket interface for RadClaw.
Provides real-time streaming of the multi-agent pipeline.
"""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import yaml
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.core.orchestrator import run_pipeline
from src.core.types import EventType
from src.data.chexpert_loader import CheXpertLoader
from src.llm.client import LLMClient, ModelConfig, set_shared_client

logger = logging.getLogger(__name__)

_settings: dict = {}
_llm_client: LLMClient | None = None
_loader: CheXpertLoader | None = None


def _load_settings(path: str = "config/settings.yaml") -> dict:
    p = Path(path)
    if p.exists():
        with open(p) as f:
            return yaml.safe_load(f) or {}
    return {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _settings, _llm_client, _loader

    _settings = _load_settings()
    config = ModelConfig.from_settings()
    _llm_client = LLMClient(config)
    set_shared_client(_llm_client)

    logger.info("RadClaw server started")
    yield
    logger.info("RadClaw server shutting down")


def create_app() -> FastAPI:
    app = FastAPI(title="RadClaw", lifespan=lifespan)

    cors_origins = _settings.get("server", {}).get("cors_origins", "*")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[cors_origins] if cors_origins != "*" else ["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    async def health():
        return {"status": "ok", "service": "radclaw"}

    @app.get("/api/dataset/info")
    async def dataset_info():
        global _loader
        if _loader is None:
            data_cfg = _settings.get("data", {})
            _loader = CheXpertLoader(
                data_dir=data_cfg.get("chexpert_path", ""),
                split_name=data_cfg.get("default_split", "findings_section"),
            )
        return {
            "split": _loader.split_name,
            "count": len(_loader),
        }

    @app.post("/api/process/{index}")
    async def process_sample(index: int):
        global _loader
        if _loader is None:
            data_cfg = _settings.get("data", {})
            _loader = CheXpertLoader(
                data_dir=data_cfg.get("chexpert_path", ""),
                split_name=data_cfg.get("default_split", "findings_section"),
            )

        sample = _loader[index]
        exam_info = _loader.get_exam_info(index)

        events = []
        async for event in run_pipeline(
            image=sample["image"],
            exam_info=exam_info,
            settings=_settings,
            llm_client=_llm_client,
        ):
            events.append(event.to_dict())
            if event.type == EventType.DONE:
                return JSONResponse(content=event.data)

        return JSONResponse(content={"error": "Pipeline did not complete"}, status_code=500)

    @app.websocket("/ws/pipeline")
    async def pipeline_websocket(ws: WebSocket):
        await ws.accept()
        try:
            while True:
                msg = json.loads(await ws.receive_text())
                index = msg.get("index", 0)

                global _loader
                if _loader is None:
                    data_cfg = _settings.get("data", {})
                    _loader = CheXpertLoader(
                        data_dir=data_cfg.get("chexpert_path", ""),
                        split_name=data_cfg.get("default_split", "findings_section"),
                    )

                sample = _loader[index]
                exam_info = _loader.get_exam_info(index)

                async for event in run_pipeline(
                    image=sample["image"],
                    exam_info=exam_info,
                    settings=_settings,
                    llm_client=_llm_client,
                ):
                    await ws.send_json(event.to_dict())

        except WebSocketDisconnect:
            logger.info("WebSocket disconnected")
        except Exception as e:
            logger.error(f"WebSocket error: {e}")

    return app