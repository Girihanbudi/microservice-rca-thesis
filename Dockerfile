# ~2.5–3.0GB base instead of 7–9GB
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Minimal OS packages + Python
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3-pip python3-dev python3-venv ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Optional: use a venv to keep things tidy (doesn't grow size much)
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app
COPY requirements.txt .

# Pin torch/vision/audio to cu118 wheels only (no CPU/ROCm extras)
# Make sure your requirements.txt does NOT also pull torch again!
RUN python -m pip install --upgrade pip \
 && python -m pip install --no-cache-dir \
      --extra-index-url https://download.pytorch.org/whl/cu118 \
      torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 \
 && python -m pip install --no-cache-dir -r requirements.txt

# Copy only what you need (use .dockerignore!)
COPY . .

CMD ["python", "-u", "worker.py"]
