"""RadClaw entry point."""

import logging
import uvicorn
import yaml
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)

from src.web.router import create_app

app = create_app()

if __name__ == "__main__":
    settings = {}
    settings_path = Path("config/settings.yaml")
    if settings_path.exists():
        with open(settings_path) as f:
            settings = yaml.safe_load(f) or {}

    server = settings.get("server", {})
    host = server.get("host", "0.0.0.0")
    port = int(server.get("port", 8080))

    uvicorn.run(app, host=host, port=port, log_level="info")