import base64
import binascii
import logging
import os
import tempfile
from typing import Optional

import soundfile as sf

logger = logging.getLogger("voxcpm.utils")


def decode_base64_audio(payload: str, output_path: str) -> str:
    if not isinstance(payload, str) or not payload.strip():
        raise ValueError("speakerAudio must be a non-empty base64 string")

    try:
        raw_bytes = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("speakerAudio must be valid base64-encoded WAV audio") from exc

    if not raw_bytes:
        raise ValueError("Decoded audio is empty")

    with open(output_path, "wb") as handle:
        handle.write(raw_bytes)

    try:
        with sf.SoundFile(output_path, "r") as audio_file:
            if audio_file.frames <= 0:
                raise ValueError("Decoded audio is empty")
    except RuntimeError as exc:
        raise ValueError("speakerAudio must decode to a valid WAV file") from exc

    return output_path


def encode_wav_to_base64(wav_path: str) -> str:
    with open(wav_path, "rb") as handle:
        encoded = base64.b64encode(handle.read()).decode("ascii")
    return encoded


def create_temp_wav_path(prefix: str = "voxcpm") -> str:
    temp_dir = tempfile.gettempdir()
    os.makedirs(temp_dir, exist_ok=True)
    return os.path.join(temp_dir, f"{prefix}-{os.getpid()}-{os.urandom(4).hex()}.wav")
