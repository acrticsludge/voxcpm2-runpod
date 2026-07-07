import logging
import os
import threading
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from config import get_config
from voxcpm import VoxCPM

logger = logging.getLogger("voxcpm.model")


class VoxCPMService:
    def __init__(self) -> None:
        self.config = get_config()
        self._model: Optional[VoxCPM] = None
        self._lock = threading.RLock()
        self._init_started = False

    def detect_device(self) -> str:
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def get_gpu_status(self) -> Dict[str, Any]:
        if not torch.cuda.is_available():
            return {"available": False, "device_count": 0, "device_name": None}

        device_count = torch.cuda.device_count()
        device_name = torch.cuda.get_device_name(0) if device_count else None
        try:
            free_memory, total_memory = torch.cuda.mem_get_info()
            memory = {
                "free_bytes": int(free_memory),
                "total_bytes": int(total_memory),
                "free_gb": round(free_memory / (1024**3), 2),
                "total_gb": round(total_memory / (1024**3), 2),
            }
        except Exception:
            memory = None

        return {
            "available": True,
            "device_count": int(device_count),
            "device_name": device_name,
            "memory": memory,
        }

    def get_status(self) -> Dict[str, Any]:
        return {
            "loaded": self._model is not None,
            "initializing": self._init_started,
            "device": self.detect_device(),
            "gpu": self.get_gpu_status(),
            "model_id": self.config["model_id"],
            "cache_dir": self.config["cache_dir"],
        }

    def preload_on_startup(self) -> Dict[str, Any]:
        if not self.config.get("preload_model_on_startup", True):
            return self.get_status()

        try:
            self.load()
            return self.get_status()
        except Exception as exc:
            logger.exception(
                "startup.validation.failed",
                extra={"event": "startup.validation.failed", "error": str(exc)},
            )
            return {**self.get_status(), "startup_error": str(exc)}

    def load(self, force: bool = False) -> VoxCPM:
        if self._model is not None and not force:
            return self._model

        with self._lock:
            if self._model is not None and not force:
                return self._model
            if self._init_started and not force:
                raise RuntimeError("Model is already being initialized")

            self._init_started = True
            device = self.detect_device()
            logger.info(
                "model.load.start",
                extra={
                    "event": "model.load.start",
                    "device": device,
                    "model_id": self.config["model_id"],
                    "cache_dir": self.config["cache_dir"],
                },
            )

            start_time = time.perf_counter()
            try:
                if device == "cuda":
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True

                if self.config["hf_token"]:
                    os.environ["HF_TOKEN"] = self.config["hf_token"]

                optimize = device != "cpu"
                self._model = VoxCPM.from_pretrained(
                    self.config["model_id"],
                    load_denoiser=False,
                    cache_dir=self.config["cache_dir"],
                    local_files_only=False,
                    optimize=optimize,
                    device=device,
                )
                elapsed = time.perf_counter() - start_time
                logger.info(
                    "model.load.complete",
                    extra={
                        "event": "model.load.complete",
                        "device": device,
                        "elapsed_seconds": round(elapsed, 3),
                    },
                )
                return self._model
            except Exception as exc:
                logger.exception(
                    "model.load.failed",
                    extra={"event": "model.load.failed", "error": str(exc)},
                )
                raise
            finally:
                self._init_started = False

    def generate(
        self,
        text: str,
        reference_wav_path: Optional[str] = None,
        cfg_value: float = 2.0,
        normalize: bool = False,
        denoise: bool = False,
        control_instruction: str = "",
    ) -> Tuple[np.ndarray, int]:
        model = self.load()
        with self._lock:
            start_time = time.perf_counter()
            try:
                if control_instruction:
                    text = f"({control_instruction}){text}"

                audio = model.generate(
                    text=text,
                    reference_wav_path=reference_wav_path,
                    cfg_value=cfg_value,
                    normalize=normalize,
                    denoise=denoise,
                    inference_timesteps=10,
                )
                sample_rate = int(getattr(model.tts_model, "sample_rate", 24000))
                duration = time.perf_counter() - start_time
                logger.info(
                    "inference.complete",
                    extra={
                        "event": "inference.complete",
                        "elapsed_seconds": round(duration, 3),
                        "sample_rate": sample_rate,
                        "audio_length": int(len(audio)),
                    },
                )
                return audio, sample_rate
            except torch.cuda.OutOfMemoryError as exc:
                logger.exception(
                    "inference.out_of_memory",
                    extra={"event": "inference.out_of_memory", "error": str(exc)},
                )
                raise RuntimeError("GPU ran out of memory during inference") from exc
            except Exception as exc:
                logger.exception(
                    "inference.failed",
                    extra={"event": "inference.failed", "error": str(exc)},
                )
                raise
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()


service = VoxCPMService()
