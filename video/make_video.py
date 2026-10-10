"""Assemble the demo video: one frame per scene, narrated with macOS `say`, stitched with ffmpeg.

usage: python video/make_video.py [--tts elevenlabs] [--voice "Ava (Premium)"]
"""
import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
ELEVEN_VOICE = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")  # "Rachel", a calm narrator
ELEVEN_MODEL = os.getenv("ELEVENLABS_MODEL", "eleven_multilingual_v2")

HERE = Path(__file__).parent
FRAMES, BUILD = HERE / "frames", HERE / "build"
PAD = 0.6  # seconds of silence after each line


def duration(path: Path) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True, check=True)
    return float(out.stdout.strip())


def tts_eleven(text: str, out: Path) -> Path:
    """ElevenLabs narration, cached by text so re-renders don't spend characters again."""
    mp3 = out.with_name(f"el_{hashlib.sha1((ELEVEN_VOICE + ELEVEN_MODEL + text).encode()).hexdigest()[:12]}.mp3")
    if not mp3.exists():
        r = requests.post(f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVEN_VOICE}?output_format=mp3_44100_128",
                          headers={"xi-api-key": os.environ["ELEVENLABS_API_KEY"]}, timeout=120,
                          json={"text": text, "model_id": ELEVEN_MODEL,
                                "voice_settings": {"stability": 0.45, "similarity_boost": 0.8, "style": 0.15}})
        r.raise_for_status()
        mp3.write_bytes(r.content)
    return mp3


def main(voice: str, tts: str):
    scenes = json.loads((HERE / "narration.json").read_text())["scenes"]
    assert not any("[RESULT]" in s["text"] for s in scenes), "fill in the [RESULT] line from analyze.py first"
    BUILD.mkdir(exist_ok=True)
    clips = []
    for i, s in enumerate(scenes):
        if tts == "elevenlabs":
            audio = tts_eleven(s["text"], BUILD / f"{i:02d}.mp3")
        else:
            audio = BUILD / f"{i:02d}.aiff"
            subprocess.run(["say", "-v", voice, "-r", "178", "-o", str(audio), s["text"]], check=True)
        clip = BUILD / f"{i:02d}.mp4"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-loop", "1", "-i", str(FRAMES / s["image"]), "-i", str(audio),
                        "-af", f"apad=pad_dur={PAD}", "-t", f"{duration(audio) + PAD:.2f}",
                        "-vf", "scale=1920:1080,format=yuv420p", "-c:v", "libx264", "-tune", "stillimage", "-r", "30",
                        "-c:a", "aac", "-b:a", "160k", "-ar", "44100", str(clip)], check=True)
        clips.append(clip)
        print(f"scene {i:2d} {s['image']:16s} {duration(clip):5.1f}s")
    (BUILD / "list.txt").write_text("".join(f"file '{c.name}'\n" for c in clips))
    out = HERE / "agentrank_demo.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(BUILD / "list.txt"),
                    "-c", "copy", str(out)], check=True)
    print(f"wrote {out} ({duration(out):.0f}s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", default="Ava (Premium)")
    ap.add_argument("--tts", choices=["say", "elevenlabs"], default="say")
    a = ap.parse_args()
    main(a.voice, a.tts)
