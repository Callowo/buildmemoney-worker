# Build Me Money AI — RunPod Worker
# Base: RunPod's official PyTorch image (CUDA 11.8, Python 3.10, Ubuntu 22.04)
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

# ── System packages ───────────────────────────────────────────────────────────
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        ffmpeg \
        fonts-liberation \
        fontconfig \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        wget \
        unzip && \
    fc-cache -f -v && \
    rm -rf /var/lib/apt/lists/*

# ── Clone SadTalker ───────────────────────────────────────────────────────────
WORKDIR /app
RUN git clone https://github.com/OpenTalker/SadTalker.git

# ── Install SadTalker Python dependencies ─────────────────────────────────────
WORKDIR /app/SadTalker
RUN pip install --no-cache-dir -r requirements.txt

# ── Worker Python dependencies ────────────────────────────────────────────────
# NOTE: SadTalker model weights are downloaded at container startup by handler.py
# using the HF_TOKEN environment variable. This avoids build-time auth issues.
WORKDIR /app
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# ── Worker entry point ────────────────────────────────────────────────────────
COPY handler.py .

CMD ["python", "-u", "handler.py"]
