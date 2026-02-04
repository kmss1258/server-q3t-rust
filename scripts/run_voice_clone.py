#!/usr/bin/env python3
import json
import subprocess
from pathlib import Path
import urllib.request


BASE_URL = "http://n.kami.live:19160"
TEXT = "오빠.. 나 당장 갈 것 같아.. 빨리 와줬으면 좋겠어. 부탁이야."
INSTRUCT = "A soft-spoken, airy, high-tone female narrator in her 20s, whispering gently with calm breathiness."

OUT_DIR = Path(".")
VOICE_DESIGN_MP3 = OUT_DIR / "voice_design.mp3"
VOICE_DESIGN_WAV = OUT_DIR / "voice_design.wav"
PROMPT_JSON = OUT_DIR / "prompt.json"
VOICE_CLONE_MP3 = OUT_DIR / "voice_clone.mp3"


def post_json(path: str, payload: dict, out_path: Path) -> None:
    url = f"{BASE_URL}{path}"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as resp:
        out_path.write_bytes(resp.read())


def post_multipart_prompt(wav_path: Path, ref_text: str, out_path: Path) -> None:
    url = f"{BASE_URL}/v1/audio/voice-clone/prompt"
    curl_cmd = [
        "curl",
        "-sS",
        "-X",
        "POST",
        url,
        "-F",
        f"ref_audio=@{wav_path}",
        "-F",
        f"ref_text={ref_text}",
        "-o",
        str(out_path),
    ]
    subprocess.run(curl_cmd, check=True)


def main() -> None:
    voice_design_payload = {
        "text": TEXT,
        "instruct": INSTRUCT,
        "language": "korean",
        "options": {"duration_seconds": 15},
    }
    post_json("/v1/audio/voice-design", voice_design_payload, VOICE_DESIGN_MP3)

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(VOICE_DESIGN_MP3),
            "-ac",
            "1",
            "-ar",
            "24000",
            "-c:a",
            "pcm_s16le",
            str(VOICE_DESIGN_WAV),
        ],
        check=True,
    )

    post_multipart_prompt(VOICE_DESIGN_WAV, TEXT, PROMPT_JSON)

    prompt = json.loads(PROMPT_JSON.read_text(encoding="utf-8"))
    clone_payload = {
        "text": TEXT,
        "language": "korean",
        "options": {"duration_seconds": 15},
        "prompt": prompt["prompt"],
    }
    post_json("/v1/audio/voice-clone", clone_payload, VOICE_CLONE_MP3)


if __name__ == "__main__":
    main()
