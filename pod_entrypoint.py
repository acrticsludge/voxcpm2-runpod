import json
import logging
import os
import sys
from typing import Any, Dict

from flask import Flask, jsonify, request

from config import get_config
from handler import build_error_response, handler, health_check, validate_request
from model import service

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), stream=sys.stdout)
logger = logging.getLogger("voxcpm.pod")

app = Flask(__name__)


@app.get("/health")
def health_route() -> Any:
    return jsonify(health_check())


@app.post("/run")
def run_route() -> Any:
    payload = request.get_json(silent=True) or {}
    result = handler({"id": "pod-request", "input": payload})
    status_code = 200
    if isinstance(result, dict) and result.get("error"):
        status_code = int(result.get("statusCode", 400))
    return jsonify(result), status_code


@app.post("/infer")
def infer_route() -> Any:
    payload = request.get_json(silent=True) or {}
    result = handle_pod_payload(payload)
    status_code = 200
    if isinstance(result, dict) and result.get("error"):
        status_code = int(result.get("statusCode", 400))
    return jsonify(result), status_code


def handle_pod_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(payload, dict) and payload.get("health") is True:
        return health_check()

    request_payload = payload.get("input") if isinstance(payload, dict) and isinstance(payload.get("input"), dict) else payload
    request_obj, error = validate_request(request_payload)
    if error is not None:
        return build_error_response(error)

    try:
        service.load()
        return handler({"id": "pod-request", "input": payload})
    except Exception as exc:
        logger.exception("pod.request.failed", extra={"event": "pod.request.failed", "error": str(exc)})
        return build_error_response(f"Inference failed: {exc}", 500)


if __name__ == "__main__":
    config = get_config()
    logger.setLevel(getattr(logging, config["log_level"], logging.INFO))
    logger.info(
        "pod.startup",
        extra={
            "event": "pod.startup",
            "model_id": config["model_id"],
            "cache_dir": config["cache_dir"],
            "preload_model_on_startup": config["preload_model_on_startup"],
        },
    )
    service.preload_on_startup()
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8000")), debug=False)
