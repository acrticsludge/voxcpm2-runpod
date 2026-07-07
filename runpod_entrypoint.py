import logging
import os
import sys

try:
    import runpod  # type: ignore
except ImportError:  # pragma: no cover - used for local testing
    runpod = None

from config import get_config
from handler import handler
from model import service

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), stream=sys.stdout)
logger = logging.getLogger("voxcpm.entrypoint")


def start_server() -> None:
    if runpod is None:
        raise SystemExit("runpod is required to start the serverless worker")

    config = get_config()
    logger.setLevel(getattr(logging, config["log_level"], logging.INFO))
    logger.info(
        "startup",
        extra={
            "event": "startup",
            "model_id": config["model_id"],
            "cache_dir": config["cache_dir"],
            "preload_model_on_startup": config["preload_model_on_startup"],
        },
    )
    service.preload_on_startup()
    runpod.serverless.start({"handler": handler})


if __name__ == "__main__":
    start_server()
