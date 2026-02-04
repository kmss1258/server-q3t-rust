#!/bin/bash
# Build qwen3-tts Docker image (cpu/gpu selectable)
#
# Usage: ./build-docker.sh [image-name] [features] [mode] [entrypoint]
#   image-name: Docker image name (default: qwen3-tts)
#   features:   Cargo features (default: flash-attn,cli,server for gpu; cli,server for cpu)
#   mode:       gpu|cpu (default: gpu)
#   entrypoint: serve|generate_audio (default: serve)

set -e

IMAGE_NAME="${1:-qwen3-tts}"
MODE="${3:-gpu}"
if [ -n "$2" ]; then
    FEATURES="$2"
else
    if [ "$MODE" = "cpu" ]; then
        FEATURES="cli,server"
    else
        FEATURES="flash-attn,cli,server"
    fi
fi
ENTRYPOINT_BIN="${4:-serve}"
CONTAINER_NAME="qwen3-tts-builder-$$"

if [ "$MODE" = "cpu" ]; then
    BASE_IMAGE="ubuntu:22.04"
    GPU_ARGS=""
else
    BASE_IMAGE="nvcr.io/nvidia/pytorch:25.11-py3"
    GPU_ARGS="--gpus all"
fi

echo "=== Building $IMAGE_NAME ($MODE) with features: $FEATURES ==="

cleanup() {
    echo "Cleaning up..."
    docker rm -f "$CONTAINER_NAME" 2>/dev/null || true
}
trap cleanup EXIT

echo "Starting build container ($MODE)..."
docker run -d --name "$CONTAINER_NAME" $GPU_ARGS \
    "$BASE_IMAGE" sleep infinity

docker exec "$CONTAINER_NAME" bash -c 'mkdir -p /workspace'

echo "Copying source..."
docker cp . "$CONTAINER_NAME":/workspace/project

echo "Installing build dependencies..."
docker exec "$CONTAINER_NAME" bash -c \
    'apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates build-essential pkg-config ffmpeg python3 python3-pip \
        && rm -rf /var/lib/apt/lists/*'

echo "Installing Rust toolchain..."
docker exec "$CONTAINER_NAME" bash -c \
    'curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y'

echo "Building with features: $FEATURES"
docker exec "$CONTAINER_NAME" bash -c \
    "source ~/.cargo/env && cd /workspace/project && cargo build --release --features '$FEATURES'"

echo "Installing binaries..."
docker exec "$CONTAINER_NAME" bash -c \
    'cp /workspace/project/target/release/generate_audio /usr/local/bin/ 2>/dev/null || true && \
     cp /workspace/project/target/release/serve /usr/local/bin/ 2>/dev/null || true && \
     mkdir -p /examples/data && \
     cp /workspace/project/examples/data/clone_2.wav /examples/data/ 2>/dev/null || true && \
     cp /workspace/project/scripts/transcribe.py /usr/local/bin/ 2>/dev/null || true && \
     mkdir -p /output && \
     rm -rf /workspace/project'

echo "Installing Python tools (whisper, optional flash-attn)..."
PY_PKGS="openai-whisper scipy"
if [[ "$FEATURES" == *"flash-attn"* ]]; then
    PY_PKGS="$PY_PKGS flash-attn"
fi
docker exec "$CONTAINER_NAME" bash -c \
    "pip install --no-cache-dir $PY_PKGS"

echo "Creating image..."
docker commit \
    --change 'WORKDIR /output' \
    --change 'ENTRYPOINT ["'"$ENTRYPOINT_BIN"'"]' \
    --change 'CMD ["--help"]' \
    "$CONTAINER_NAME" "$IMAGE_NAME"

echo ""
echo "=== Done! ==="
echo "Image: $IMAGE_NAME"
echo "Entrypoint: $ENTRYPOINT_BIN"
echo "Mode: $MODE"
echo ""
echo "Usage:"
if [ "$ENTRYPOINT_BIN" = "serve" ]; then
    if [ "$MODE" = "gpu" ]; then
        echo "  docker run --gpus all -e BASE_MODEL_DIR=/models/base -e VOICE_DESIGN_MODEL_DIR=/models/voicedesign \\"
        echo "    -v /path/to/models:/models -p 8000:8000 $IMAGE_NAME"
    else
        echo "  docker run -e BASE_MODEL_DIR=/models/base -e VOICE_DESIGN_MODEL_DIR=/models/voicedesign \\"
        echo "    -v /path/to/models:/models -p 8000:8000 $IMAGE_NAME"
    fi
else
    echo "  docker run --gpus all -v /path/to/models:/models -v /path/to/output:/output $IMAGE_NAME \\"
    echo "    --model-dir /models/1.7b-customvoice --speaker ryan --text \"Hello\" --device cuda --output /output/out.wav"
fi
