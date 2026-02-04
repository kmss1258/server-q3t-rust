#!/usr/bin/env bash
set -euo pipefail

URL="${URL:-http://localhost:19160}"
REF_TEXT="${REF_TEXT:-The sun set behind the mountains, painting the sky in shades of gold and violet. Birds sang their evening songs as the world grew quiet.}"
INSTRUCT="${INSTRUCT:-A calm male narrator with deep voice}"
TEXT="${TEXT:-Hello, this is a voice cloning test on GPU. The voice should sound similar.}"
OUT_DIR="${OUT_DIR:-/tmp}"
CONTAINER_NAME="${CONTAINER_NAME:-qwen3-tts-serve}"

VD_MP3="${OUT_DIR}/vd.mp3"
REF_WAV="${OUT_DIR}/ref.wav"
PROMPT_ICL="${OUT_DIR}/prompt_icl.json"
PROMPT_XV="${OUT_DIR}/prompt_xv.json"
OUT_ICL="${OUT_DIR}/out_icl.mp3"
OUT_XV="${OUT_DIR}/out_xv.mp3"

echo "[1/5] VoiceDesign -> ref.wav"
python3 - <<'PY' >"${OUT_DIR}/payload_vd.json"
import json
payload = {
  'text': """${REF_TEXT}""",
  'instruct': """${INSTRUCT}""",
  'language': 'english',
  'options': {'duration_seconds': 20}
}
print(json.dumps(payload))
PY

curl -s -X POST "$URL/v1/audio/voice-design" \
  -H "Content-Type: application/json" \
  -d "@${OUT_DIR}/payload_vd.json" \
  -o "$VD_MP3"
ffmpeg -y -i "$VD_MP3" -ac 1 -ar 24000 -c:a pcm_s16le "$REF_WAV" >/dev/null 2>&1

echo "[2/5] Create prompts (ICL + x_vector_only)"
curl -s -X POST "$URL/v1/audio/voice-clone/prompt" \
  -F "ref_audio=@${REF_WAV}" \
  -F "ref_text=${REF_TEXT}" \
  -o "$PROMPT_ICL"

curl -s -X POST "$URL/v1/audio/voice-clone/prompt" \
  -F "ref_audio=@${REF_WAV}" \
  -F "x_vector_only=1" \
  -o "$PROMPT_XV"

echo "[3/5] VoiceClone synth"
python3 - <<'PY' >"${OUT_DIR}/payload_icl.json"
import json
with open("${PROMPT_ICL}") as f:
    prompt = json.load(f)['prompt']
payload = {
  'text': """${TEXT}""",
  'language': 'english',
  'options': {'duration_seconds': 20},
  'prompt': prompt
}
print(json.dumps(payload))
PY

python3 - <<'PY' >"${OUT_DIR}/payload_xv.json"
import json
with open("${PROMPT_XV}") as f:
    prompt = json.load(f)['prompt']
payload = {
  'text': """${TEXT}""",
  'language': 'english',
  'options': {'duration_seconds': 20},
  'prompt': prompt
}
print(json.dumps(payload))
PY

curl -s -X POST "$URL/v1/audio/voice-clone" \
  -H "Content-Type: application/json" \
  -d "@${OUT_DIR}/payload_icl.json" \
  -o "$OUT_ICL"

curl -s -X POST "$URL/v1/audio/voice-clone" \
  -H "Content-Type: application/json" \
  -d "@${OUT_DIR}/payload_xv.json" \
  -o "$OUT_XV"

echo "[4/5] Duration check"
ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$OUT_ICL"
ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$OUT_XV"

echo "[5/5] STT (Whisper in container)"
docker cp "$OUT_ICL" "${CONTAINER_NAME}:/tmp/out_icl.mp3"
docker cp "$OUT_XV" "${CONTAINER_NAME}:/tmp/out_xv.mp3"

docker exec -i "$CONTAINER_NAME" python3 - <<'PY'
import whisper, warnings
warnings.filterwarnings('ignore')
model = whisper.load_model('tiny')
for name in ['out_icl.mp3', 'out_xv.mp3']:
    result = model.transcribe(f'/tmp/{name}', fp16=False)
    print(f"{name}: {result['text']}")
PY

echo "Done. Outputs:"
echo "- ${OUT_ICL}"
echo "- ${OUT_XV}"
