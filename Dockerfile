FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_ID=openbmb/VoxCPM2 \
    CACHE_DIR=/cache/voxcpm \
    HF_HOME=/cache/huggingface \
    LOG_LEVEL=INFO

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-dev build-essential git curl libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./

RUN python3 -m pip install --upgrade pip setuptools wheel \
    && python3 -m pip install --extra-index-url https://download.pytorch.org/whl/cu124 -r requirements.txt

COPY . /app

RUN mkdir -p /cache/voxcpm /cache/huggingface \
    && python3 -m compileall /app

EXPOSE 8000

CMD ["python3", "handler.py"]
