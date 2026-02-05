# TTS Verification Playbook

This document captures the current verification workflow for Qwen3-TTS voice cloning
and VoiceDesign. Use it to reproduce issues and confirm fixes. Keep this updated.

## Goals

- Validate VoiceDesign output.
- Validate VoiceClone in both ICL and x_vector_only modes.
- Compare CPU vs GPU behavior.
- Record evidence via durations and STT transcripts.

## Environment

- GPU server URL: http://localhost:19160
- Local CPU server URL: http://localhost:8010 (example)

## Reference Text

Use a consistent ref text for ICL:

"The sun set behind the mountains, painting the sky in shades of gold and violet.\
 Birds sang their evening songs as the world grew quiet."

## Step 1: VoiceDesign -> Reference WAV

```bash
URL="http://localhost:19160"

python3 - <<'PY' >/tmp/tts_payload.json
import json
payload = {
  'text': 'The sun set behind the mountains, painting the sky in shades of gold and violet. Birds sang their evening songs as the world grew quiet.',
  'instruct': 'A calm male narrator with deep voice',
  'language': 'english',
  'options': {'duration_seconds': 20}
}
print(json.dumps(payload))
PY

curl -s -X POST "$URL/v1/audio/voice-design" \
  -H "Content-Type: application/json" \
  -d @/tmp/tts_payload.json \
  -o /tmp/vd.mp3

ffmpeg -y -i /tmp/vd.mp3 -ac 1 -ar 24000 -c:a pcm_s16le /tmp/ref.wav >/dev/null 2>&1
```

## Step 2: Create Prompts

```bash
URL="http://localhost:19160"
REF_TEXT="The sun set behind the mountains, painting the sky in shades of gold and violet. Birds sang their evening songs as the world grew quiet."

# ICL prompt
curl -s -X POST "$URL/v1/audio/voice-clone/prompt" \
  -F "ref_audio=@/tmp/ref.wav" \
  -F "ref_text=${REF_TEXT}" \
  -o /tmp/prompt_icl.json

# x_vector_only prompt
curl -s -X POST "$URL/v1/audio/voice-clone/prompt" \
  -F "ref_audio=@/tmp/ref.wav" \
  -F "x_vector_only=1" \
  -o /tmp/prompt_xv.json
```

## Step 3: VoiceClone Synthesis (ICL + x_vector_only)

```bash
TEXT="Hello, this is a voice cloning test on GPU. The voice should sound similar."

python3 - <<'PY' >/tmp/payload_icl.json
import json
with open('/tmp/prompt_icl.json') as f:
    prompt = json.load(f)['prompt']
payload = {
  'text': 'Hello, this is a voice cloning test on GPU. The voice should sound similar.',
  'language': 'english',
  'options': {'duration_seconds': 20},
  'prompt': prompt
}
print(json.dumps(payload))
PY

python3 - <<'PY' >/tmp/payload_xv.json
import json
with open('/tmp/prompt_xv.json') as f:
    prompt = json.load(f)['prompt']
payload = {
  'text': 'Hello, this is a voice cloning test on GPU. The voice should sound similar.',
  'language': 'english',
  'options': {'duration_seconds': 20},
  'prompt': prompt
}
print(json.dumps(payload))
PY

curl -s -X POST "$URL/v1/audio/voice-clone/prompted" \
  -H "Content-Type: application/json" \
  -d @/tmp/payload_icl.json \
  -o /tmp/out_icl.mp3

curl -s -X POST "$URL/v1/audio/voice-clone/prompted" \
  -H "Content-Type: application/json" \
  -d @/tmp/payload_xv.json \
  -o /tmp/out_xv.mp3
```

## Step 4: Duration Check

```bash
ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 /tmp/out_icl.mp3
ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 /tmp/out_xv.mp3
```

## Step 5: STT Verification (Whisper in container)

```bash
docker cp /tmp/out_icl.mp3 qwen3-tts-serve:/tmp/
docker cp /tmp/out_xv.mp3 qwen3-tts-serve:/tmp/

docker exec -i qwen3-tts-serve python3 - <<'PY'
import whisper, warnings
warnings.filterwarnings('ignore')
model = whisper.load_model('tiny')
for name in ['out_icl.mp3', 'out_xv.mp3']:
    result = model.transcribe(f'/tmp/{name}', fp16=False)
    print(f"{name}: {result['text']}")
PY
```

## Step 6: Automated Script (GPU)

```bash
./scripts/verify_tts_gpu.sh
```

Optional overrides:

```bash
URL="http://localhost:19160" \
REF_TEXT="The sun set behind the mountains..." \
INSTRUCT="A calm male narrator with deep voice" \
TEXT="Hello, this is a voice cloning test on GPU. The voice should sound similar." \
OUT_DIR="/tmp" \
CONTAINER_NAME="qwen3-tts-serve" \
./scripts/verify_tts_gpu.sh
```

## Expected Outcomes

- VoiceDesign: intelligible and matches input text.
- VoiceClone x_vector_only: matches target text.
- VoiceClone ICL: should match target text. Any garbling indicates GPU-only ICL issue.

## GPU Debug Flags

- Disable fused RMSNorm (runtime):
  - `DISABLE_FUSED_RMSNORM=1`
  - Must be set on container start, not build time.

## Build Notes

- GPU build without flash-attn:
  - features: `cli,server,cuda`
- GPU build with flash-attn:
  - features: `flash-attn,cli,server`
