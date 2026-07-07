import logging
import os
from pathlib import Path
from typing import Any, Dict

DEFAULT_MODEL_ID = "openbmb/VoxCPM2"
DEFAULT_CACHE_DIR = "/tmp/voxcpm-cache"
DEFAULT_HF_HOME = "/tmp/huggingface"
DEFAULT_LOG_LEVEL = "INFO"


def get_config() -> Dict[str, Any]:
    cache_dir = os.getenv("CACHE_DIR") or os.getenv("HF_HOME") or DEFAULT_CACHE_DIR
    hf_home = os.getenv("HF_HOME") or os.getenv("CACHE_DIR") or DEFAULT_HF_HOME
    model_id = os.getenv("MODEL_ID", DEFAULT_MODEL_ID)
    token = os.getenv("HF_TOKEN")
    log_level = os.getenv("LOG_LEVEL", DEFAULT_LOG_LEVEL).upper()
    preload_on_startup = os.getenv("PRELOAD_MODEL_ON_STARTUP", "false").lower() == "true"

    os.environ.setdefault("HF_HOME", hf_home)
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    Path(hf_home).mkdir(parents=True, exist_ok=True)

    return {
        "model_id": model_id,
        "cache_dir": cache_dir,
        "hf_home": hf_home,
        "hf_token": token,
        "log_level": log_level,
        "debug": os.getenv("DEBUG", "false").lower() == "true",
        "preload_model_on_startup": preload_on_startup,
    }


def get_log_level() -> int:
    config = get_config()
    return getattr(logging, config["log_level"], logging.INFO)
