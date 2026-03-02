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

# ── Download SadTalker model checkpoints ──────────────────────────────────────
# Models are baked into the image so the worker starts instantly (no cold-download)
RUN mkdir -p checkpoints && \
    wget -q -O checkpoints/SadTalker_V0.0.2_256.safetensors \
        "https://huggingface.co/vinthony/SadTalker/resolve/main/SadTalker_V0.0.2_256.safetensors" && \
    wget -q -O checkpoints/SadTalker_V0.0.2_512.safetensors \
        "https://huggingface.co/vinthony/SadTalker/resolve/main/SadTalker_V0.0.2_512.safetensors" && \
    wget -q -O checkpoints/mapping_00109-model.pth.tar \
        "https://huggingface.co/vinthony/SadTalker/resolve/main/mapping_00109-model.pth.tar" && \
    wget -q -O checkpoints/mapping_00229-model.pth.tar \
        "https://huggingface.co/vinthony/SadTalker/resolve/main/mapping_00229-model.pth.tar"

RUN mkdir -p gfpgan/weights && \
    wget -q -O gfpgan/weights/GFPGANv1.4.pth \
        "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth" && \
    wget -q -O gfpgan/weights/detection_Resnet50_Final.pth \
        "https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth" && \
    wget -q -O gfpgan/weights/parsing_parsenet.pth \
        "https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth"

# ── Worker Python dependencies ────────────────────────────────────────────────
WORKDIR /app
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# ── Worker entry point ────────────────────────────────────────────────────────
COPY handler.py .

CMD ["python", "-u", "handler.py"]
