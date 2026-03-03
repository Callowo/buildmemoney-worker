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

# ── Download SadTalker + GFPGAN model weights at build time ───────────────────
# Baked into image so cold starts never need to download models (~1.5 GB total)
RUN mkdir -p /app/SadTalker/checkpoints /app/SadTalker/gfpgan/weights && \
    wget -q --show-progress -O /app/SadTalker/checkpoints/SadTalker_V0.0.2_256.safetensors \
        "https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/SadTalker_V0.0.2_256.safetensors" && \
    wget -q --show-progress -O /app/SadTalker/checkpoints/SadTalker_V0.0.2_512.safetensors \
        "https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/SadTalker_V0.0.2_512.safetensors" && \
    wget -q --show-progress -O /app/SadTalker/checkpoints/mapping_00109-model.pth.tar \
        "https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/mapping_00109-model.pth.tar" && \
    wget -q --show-progress -O /app/SadTalker/checkpoints/mapping_00229-model.pth.tar \
        "https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/mapping_00229-model.pth.tar" && \
    wget -q --show-progress -O /app/SadTalker/gfpgan/weights/GFPGANv1.4.pth \
        "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth" && \
    wget -q --show-progress -O /app/SadTalker/gfpgan/weights/detection_Resnet50_Final.pth \
        "https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth" && \
    wget -q --show-progress -O /app/SadTalker/gfpgan/weights/parsing_parsenet.pth \
        "https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth"

# ── Worker Python dependencies ────────────────────────────────────────────────
WORKDIR /app
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# ── Worker entry point ────────────────────────────────────────────────────────
COPY handler.py .

CMD ["python", "-u", "handler.py"]
