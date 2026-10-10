"""Assemble the demo video: one frame per scene, narrated with macOS `say`, stitched with ffmpeg.

usage: python video/make_video.py [--voice "Ava (Premium)"]
"""
import argparse
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).parent
FRAMES, BUILD = HERE / "frames", HERE / "build"
PAD = 0.6  # seconds of silence after each line


def duration(path: Path) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True, check=True)
    return float(out.stdout.strip())


def main(voice: str):
    scenes = json.loads((HERE / "narration.json").read_text())["scenes"]
    assert not any("[RESULT]" in s["text"] for s in scenes), "fill in the [RESULT] line from analyze.py first"
    BUILD.mkdir(exist_ok=True)
    clips = []
    for i, s in enumerate(scenes):
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
    main(ap.parse_args().voice)
