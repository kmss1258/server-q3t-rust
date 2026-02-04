#!/usr/bin/env bash
# Download required Qwen3-TTS model files (curl, resumable)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

MODEL_ROOT="${MODEL_ROOT:-$REPO_ROOT/models}"
TOKENIZER_DIR="$MODEL_ROOT/tokenizer"
SPEECH_TOKENIZER_DIR="$MODEL_ROOT/speech_tokenizer"
BASE_DIR="$MODEL_ROOT/Qwen3-TTS-12Hz-1.7B-Base"
VOICE_DESIGN_DIR="$MODEL_ROOT/Qwen3-TTS-12Hz-1.7B-VoiceDesign"

mkdir -p "$TOKENIZER_DIR" "$SPEECH_TOKENIZER_DIR" "$BASE_DIR" "$VOICE_DESIGN_DIR"

echo "=== Downloading Text Tokenizer ==="
curl -L -C - --progress-bar \
  "https://huggingface.co/Qwen/Qwen2-0.5B/resolve/main/tokenizer.json" \
  -o "$TOKENIZER_DIR/tokenizer.json"

echo ""
echo "=== Downloading Speech Tokenizer ==="
curl -L -C - --progress-bar \
  "https://huggingface.co/Qwen/Qwen3-TTS-Tokenizer-12Hz/resolve/main/config.json" \
  -o "$SPEECH_TOKENIZER_DIR/config.json"
curl -L -C - --progress-bar \
  "https://huggingface.co/Qwen/Qwen3-TTS-Tokenizer-12Hz/resolve/main/preprocessor_config.json" \
  -o "$SPEECH_TOKENIZER_DIR/preprocessor_config.json"
curl -L -C - --progress-bar \
  "https://huggingface.co/Qwen/Qwen3-TTS-Tokenizer-12Hz/resolve/main/model.safetensors" \
  -o "$SPEECH_TOKENIZER_DIR/model.safetensors"

echo ""
echo "=== Downloading 1.7B Base ==="
curl -L -C - --progress-bar \
  "https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base/resolve/main/config.json" \
  -o "$BASE_DIR/config.json"
curl -L -C - --progress-bar \
  "https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base/resolve/main/generation_config.json" \
  -o "$BASE_DIR/generation_config.json"
curl -L -C - --progress-bar \
  "https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base/resolve/main/preprocessor_config.json" \
  -o "$BASE_DIR/preprocessor_config.json"
curl -L -C - --progress-bar \
  "https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base/resolve/main/tokenizer_config.json" \
  -o "$BASE_DIR/tokenizer_config.json"
curl -L -C - --progress-bar \
  "https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base/resolve/main/model.safetensors" \
  -o "$BASE_DIR/model.safetensors"

echo ""
echo "=== Downloading 1.7B VoiceDesign ==="
curl -L -C - --progress-bar \
  "https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign/resolve/main/config.json" \
  -o "$VOICE_DESIGN_DIR/config.json"
curl -L -C - --progress-bar \
  "https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign/resolve/main/generation_config.json" \
  -o "$VOICE_DESIGN_DIR/generation_config.json"
curl -L -C - --progress-bar \
  "https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign/resolve/main/preprocessor_config.json" \
  -o "$VOICE_DESIGN_DIR/preprocessor_config.json"
curl -L -C - --progress-bar \
  "https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign/resolve/main/tokenizer_config.json" \
  -o "$VOICE_DESIGN_DIR/tokenizer_config.json"
curl -L -C - --progress-bar \
  "https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign/resolve/main/model.safetensors" \
  -o "$VOICE_DESIGN_DIR/model.safetensors"

echo ""
echo "Done. Models saved in: $MODEL_ROOT"
