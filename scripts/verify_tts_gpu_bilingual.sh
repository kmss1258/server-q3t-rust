#!/usr/bin/env bash
set -euo pipefail

MAX_ATTEMPTS="${MAX_ATTEMPTS:-3}"
URL="${URL:-http://localhost:19160}"
CONTAINER_NAME="${CONTAINER_NAME:-qwen3-tts-serve}"
USE_DIRECT="${USE_DIRECT:-0}"

run_case() {
  local label="$1"
  local language="$2"
  local ref_text="$3"
  local instruct="$4"
  local text="$5"

  echo "=== ${label} ==="
  URL="${URL}" \
  CONTAINER_NAME="${CONTAINER_NAME}" \
  USE_DIRECT="${USE_DIRECT}" \
  LANGUAGE="${language}" \
  REF_TEXT="${ref_text}" \
  INSTRUCT="${instruct}" \
  TEXT="${text}" \
  MAX_ATTEMPTS="${MAX_ATTEMPTS}" \
  python3 scripts/verify_tts_gpu_loop.py
}

run_case \
  "English" \
  "english" \
  "The sun set behind the mountains, painting the sky in shades of gold and violet. Birds sang their evening songs as the world grew quiet." \
  "A calm male narrator with deep voice" \
  "Hello, this is a voice cloning test on GPU. The voice should sound similar."

run_case \
  "Korean" \
  "korean" \
  "오늘은 날씨가 맑고 바람이 선선합니다. 우리는 공원에서 산책을 했습니다." \
  "차분한 남성 화자" \
  "안녕하세요. 이것은 그래픽 처리 장치에서 음성 복제 테스트입니다. 목소리가 비슷해야 합니다."
