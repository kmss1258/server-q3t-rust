# HTTP API

All endpoints return `audio/mpeg` (MP3) unless noted.

## Endpoints

| Method | Path | Description | Request type |
| --- | --- | --- | --- |
| POST | `/v1/audio/voice-design` | VoiceDesign synthesis from text + instruct | JSON |
| POST | `/v1/audio/voice-clone` | Direct voice clone (ref_audio + optional ref_text) | multipart/form-data |
| POST | `/v1/audio/voice-clone/prompt` | Build reusable prompt (speaker + ICL codes) | multipart/form-data |
| POST | `/v1/audio/voice-clone/prompted` | Voice clone using a saved prompt | JSON |

## Flow (Design -> Clone)

```
client
  | POST /v1/audio/voice-design (text + instruct)  --> mp3
  | mp3 -> wav (ffmpeg)
  | POST /v1/audio/voice-clone (ref_audio + ref_text + text) --> mp3
```

## Request schemas

### VoiceDesign

```json
{
  "text": "Hello",
  "instruct": "A soft-spoken, airy, high-tone female narrator in her 20s, whispering gently with calm breathiness.",
  "language": "korean",
  "options": {
    "duration_seconds": 15,
    "temperature": 0.7,
    "top_k": 50,
    "top_p": 0.9,
    "repetition_penalty": 1.05,
    "seed": 42
  }
}
```

### Voice Clone (direct, multipart/form-data)

- `ref_audio`: WAV file (required)
- `ref_text`: transcript text (optional, recommended for ICL)
- `text`: target text (required)
- `language`: optional
- `x_vector_only`: `true`/`1` to skip ICL codes
- `options`: JSON string (optional) with `duration_seconds`, `temperature`, etc.

### Voice Clone Prompt (multipart/form-data)

- `ref_audio`: WAV file (required)
- `ref_text`: transcript text (optional, recommended for ICL)
- `x_vector_only`: `true`/`1` to skip ICL codes

### Voice Clone (prompted)

```json
{
  "text": "Hello",
  "language": "korean",
  "options": { "duration_seconds": 15 },
  "prompt": {
    "speaker_embedding": [0.0, 0.1, 0.2],
    "ref_codes": [[2149, 10, 11], [2149, 12, 13]],
    "ref_text_ids": [1, 2, 3]
  }
}
```

## Example requests

### VoiceDesign -> mp3

```bash
curl -X POST http://localhost:8010/v1/audio/voice-design \
  -H "Content-Type: application/json" \
  -d '{"text":"오빠.. 나 당장 갈 것 같아.. 빨리 와줬으면 좋겠어. 부탁이야.","instruct":"A soft-spoken, airy, high-tone female narrator in her 20s, whispering gently with calm breathiness.","language":"korean","options":{"duration_seconds":15}}' \
  -o voice_design.mp3
```

### Voice Clone (direct) -> mp3

```bash
curl -X POST http://localhost:8010/v1/audio/voice-clone \
  -F "ref_audio=@voice_design.wav" \
  -F "ref_text=오빠.. 나 당장 갈 것 같아.. 빨리 와줬으면 좋겠어. 부탁이야." \
  -F "text=오빠.. 나 당장 갈 것 같아.. 빨리 와줬으면 좋겠어. 부탁이야." \
  -F "language=korean" \
  -o voice_clone.mp3
```

### Prompt 생성

```bash
curl -X POST http://localhost:8010/v1/audio/voice-clone/prompt \
  -F "ref_audio=@voice_design.wav" \
  -F "ref_text=오빠.. 나 당장 갈 것 같아.. 빨리 와줬으면 좋겠어. 부탁이야." \
  -o prompt.json
```

### Voice Clone (prompted) -> mp3

```bash
curl -X POST http://localhost:8010/v1/audio/voice-clone/prompted \
  -H "Content-Type: application/json" \
  -d "$(jq -c --arg text '오빠.. 나 당장 갈 것 같아.. 빨리 와줬으면 좋겠어. 부탁이야.' '{text:$text, language:"korean", options:{duration_seconds:15}, prompt:.prompt}' prompt.json)" \
  -o voice_clone.mp3
```
