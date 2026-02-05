#!/usr/bin/env python3
import os
import re
import sys
import time
import subprocess


def normalize(text: str) -> str:
    return re.sub(r"[^\w]+", "", text.lower())


def main() -> int:
    max_attempts = int(os.getenv("MAX_ATTEMPTS", "20"))
    sleep_seconds = float(os.getenv("SLEEP_SECONDS", "2"))

    target_text = os.getenv("TARGET_TEXT")
    if not target_text:
        target_text = os.getenv("TEXT")
    if not target_text:
        target_text = (
            "Hello, this is a voice cloning test on GPU. The voice should sound similar."
        )
    if not target_text:
        target_text = os.getenv(
            "REF_TEXT",
            "The sun set behind the mountains, painting the sky in shades of gold and violet. "
            "Birds sang their evening songs as the world grew quiet.",
        )

    env = os.environ.copy()

    for attempt in range(1, max_attempts + 1):
        print(f"[attempt {attempt}/{max_attempts}] Running verify_tts_gpu.sh")
        proc = subprocess.run(
            ["bash", "scripts/verify_tts_gpu.sh"],
            capture_output=True,
            text=True,
            env=env,
        )

        if proc.stdout:
            print(proc.stdout, end="")
        if proc.returncode != 0:
            if proc.stderr:
                print(proc.stderr, file=sys.stderr, end="")
            print("verify_tts_gpu.sh failed; stopping.", file=sys.stderr)
            return proc.returncode

        icl_match = re.search(r"out_icl\.mp3:\s*(.*)", proc.stdout)
        xv_match = re.search(r"out_xv\.mp3:\s*(.*)", proc.stdout)

        icl_text = icl_match.group(1).strip() if icl_match else ""
        xv_text = xv_match.group(1).strip() if xv_match else ""

        if not icl_text:
            print("ICL STT text not found; retrying.")
        else:
            if normalize(icl_text) == normalize(target_text):
                print("ICL STT matches target text.")
                if xv_text:
                    print(f"x_vector_only STT: {xv_text}")
                return 0

            print("ICL STT mismatch; retrying.")
            print(f"Target: {target_text}")
            print(f"ICL:    {icl_text}")
            if xv_text:
                print(f"xv:     {xv_text}")

        if attempt < max_attempts:
            time.sleep(sleep_seconds)

    print("Max attempts reached without an ICL match.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
