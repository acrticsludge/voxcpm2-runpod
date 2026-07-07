import base64
import json
import logging
import os
import sys
import tempfile
import traceback
from typing import Any, Dict, Optional, Tuple

import numpy as np
import soundfile as sf

try:
    import runpod  # type: ignore
except ImportError:  # pragma: no cover - used for local testing
    runpod = None

from config import get_config
from model import service
from utils import create_temp_wav_path, decode_base64_audio, encode_wav_to_base64

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), stream=sys.stdout)
logger = logging.getLogger("voxcpm.handler")


def build_error_response(message: str, status_code: int = 400) -> Dict[str, Any]:
    return {"error": message, "statusCode": status_code}


def validate_request(payload: Any) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    if not isinstance(payload, dict):
        return None, "Request body must be a JSON object"

    input_payload = payload.get("input")
    if input_payload is None:
        input_payload = payload

    if not isinstance(input_payload, dict):
        return None, "Request input must be an object"

    text = input_payload.get("text")
    if not isinstance(text, str) or not text.strip():
        return None, "Input text is required"

    speaker_audio = input_payload.get("speakerAudio")
    reference_wav_path = None
    if speaker_audio is not None:
        if not isinstance(speaker_audio, str) or not speaker_audio.strip():
            return None, "speakerAudio must be a non-empty base64 string when provided"
        try:
            temp_path = create_temp_wav_path(prefix="speaker")
            reference_wav_path = decode_base64_audio(speaker_audio, temp_path)
        except ValueError as exc:
            return None, str(exc)

    cfg_value = input_payload.get("cfgValue", 3.0)
    if not isinstance(cfg_value, (int, float)):
        return None, "cfgValue must be numeric"

    do_normalize = bool(input_payload.get("doNormalize", False))
    denoise = bool(input_payload.get("denoise", False))
    control_instruction = input_payload.get("controlInstruction", "")
    if not isinstance(control_instruction, str):
        return None, "controlInstruction must be a string"

    return {
        "text": text,
        "reference_wav_path": reference_wav_path,
        "cfg_value": float(cfg_value),
        "do_normalize": do_normalize,
        "denoise": denoise,
        "control_instruction": control_instruction,
    }, None


def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    logger.info("request.received", extra={"event": "request.received", "job_id": job.get("id")})
    payload = job.get("input") or {}
    request, error = validate_request(payload)
    if error is not None:
        logger.warning("request.invalid", extra={"event": "request.invalid", "error": error})
        return build_error_response(error)

    try:
        model = service.load()
        logger.info("request.processing", extra={"event": "request.processing", "text_length": len(request["text"])})
        audio, sample_rate = service.generate(
            text=request["text"],
            reference_wav_path=request["reference_wav_path"],
            cfg_value=request["cfg_value"],
            normalize=request["do_normalize"],
            denoise=request["denoise"],
            control_instruction=request["control_instruction"],
        )

        output_path = create_temp_wav_path(prefix="output")
        sf.write(output_path, audio.astype(np.float32), sample_rate)
        audio_b64 = encode_wav_to_base64(output_path)
        return {"audio": audio_b64}
    except Exception as exc:
        logger.exception("request.failed", extra={"event": "request.failed", "error": str(exc)})
        return build_error_response(f"Inference failed: {exc}", 500)


if __name__ == "__main__":
    if runpod is None:
        raise SystemExit("runpod is required to start the serverless worker")

    config = get_config()
    logger.setLevel(getattr(logging, config["log_level"], logging.INFO))
    logger.info("startup", extra={"event": "startup", "model_id": config["model_id"], "cache_dir": config["cache_dir"]})
    runpod.serverless.start({"handler": handler})
