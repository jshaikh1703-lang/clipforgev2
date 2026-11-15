"""Twitch VOD or clip -> clips.json (Twitch uses audio/chat peaks when captions are unavailable)

    python twitch_ingest.py "https://www.twitch.tv/videos/2650011234" --min-score 55
Needs TwitchDownloaderCLI on PATH for chat replay (optional): https://github.com/lay295/TwitchDownloader
"""
import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

import ijson
import yt_dlp

from agents import MoCapAgent, ShoCapAgent
from common import (
    audio_loudness, hms, merge_chunk_clips, ram_guard, run_pipeline,
    shift_signal_timestamps, slice_segments, transcribe_audio,
)
from database import ClipForgeStore, default_store_path


def twitch_id(url: str) -> str:
    m = re.search(r"twitch\.tv/videos/(\d+)", url) or re.search(r"(?:/clip/|clips\.twitch\.tv/)([\w-]+)", url)
    if not m:
        sys.exit("Not a Twitch VOD or clip URL.")
    return m.group(1)


def download_chat(url: str, tid: str, work: Path) -> list[float]:
    """Download chat with TwitchDownloaderCLI, then stream-parse it (the JSON can be 100+ MB)."""
    if not shutil.which("TwitchDownloaderCLI"):
        print("[chat] TwitchDownloaderCLI not found, continuing without chat signals")
        return []
    out = work / "chat.json"
    r = subprocess.run(["TwitchDownloaderCLI", "chatdownload", "--id", tid, "-o", str(out)])
    if r.returncode != 0 or not out.exists():
        print("[chat] download failed, continuing without chat signals")
        return []
    with open(out, "rb") as f:
        offsets = [float(c["content_offset_seconds"]) for c in ijson.items(f, "comments.item")
                   if "content_offset_seconds" in c]
    out.unlink()
    return offsets


def main(progress_callback: Callable[[str, float | None], None] | None = None) -> dict[str, Any]:
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("--min-score", type=float, default=55)
    ap.add_argument("--trends")
    ap.add_argument("--workdir", default="work")
    ap.add_argument("--no-chat", action="store_true")
    ap.add_argument("--api-keys", default=None, help="comma-separated provider API keys")
    ap.add_argument("--target-count", type=int, default=10)
    ap.add_argument("--task-id", default="")
    a = ap.parse_args()

    def report_progress(message: str, percent: float | None = None) -> None:
        print(message, flush=True)
        if progress_callback:
            progress_callback(message, percent)

    tid = twitch_id(a.url)
    work = Path(a.workdir) / f"twitch_{tid}"
    work.mkdir(parents=True, exist_ok=True)
    store = ClipForgeStore(default_store_path())
    store.initialize(Path(__file__).with_name("styles"))

    download_progress_range = [20.0, 35.0]

    def yt_dlp_progress(status: dict) -> None:
        if status.get("status") == "finished":
            report_progress("[yt-dlp] Audio download finished", download_progress_range[1])
        elif status.get("status") == "downloading":
            downloaded = status.get("downloaded_bytes", 0)
            total = status.get("total_bytes") or status.get("total_bytes_estimate")
            percent = downloaded / total * 100 if total else 0
            scaled_percent = (download_progress_range[0] +
                              percent * (download_progress_range[1] -
                                         download_progress_range[0]) / 100)
            report_progress(f"[yt-dlp] Audio download {percent:.1f}%", scaled_percent)

    base = {"quiet": True, "no_warnings": True, "progress_hooks": [yt_dlp_progress]}
    report_progress("[twitch] Fetching video metadata", 0)
    with yt_dlp.YoutubeDL(base) as y:
        info = y.extract_info(a.url, download=False)
    duration = float(info["duration"])
    print(f"[meta] {info.get('title')} ({duration / 3600:.2f} h)")

    report_progress("[twitch] Fetching chat replay", 10)
    chat = [] if a.no_chat else download_chat(a.url, tid, work)
    ram_guard("chat parsed")

    report_progress("[twitch] Downloading audio for analysis", 20)
    opts = {**base, "format": "Audio_Only/audio_only/bestaudio/worst", "outtmpl": str(work / "audio.%(ext)s")}
    with yt_dlp.YoutubeDL(opts) as y:
        y.download([a.url])
    audio = next(iter(sorted(work.glob("audio.*"))))
    loud = audio_loudness(audio)
    ram_guard("loudness computed")

    report_progress("[twitch] Transcribing audio with Whisper", 40)
    segments = transcribe_audio(audio)
    audio.unlink()
    report_progress("[twitch] Preparing five low-quality video ranges for MoCap", 65)
    sensitivity = store.get_agent_configs()["MoCap"].get("motion_sensitivity", "normal")
    report_progress("[twitch] Analyzing video in five 20% chunks", 70)
    store.record_agent_event("MoCap", "running", "Analyzing low-resolution video and chat",
                             task_id=a.task_id or None)
    mocap_agent = MoCapAgent()
    short_video_signals: list[str] = []
    needs_full_context = duration / 5 < 25

    def analyze_chunk(chunk_video: Path, chunk_work: Path, index: int,
                      start_s: float, end_s: float) -> list[dict]:
        chunk_duration = end_s - start_s
        chunk_segments = slice_segments(segments, start_s, end_s)
        chunk_chat = [offset - start_s for offset in chat if start_s <= offset < end_s]
        loud_start, loud_end = int(start_s), min(len(loud), int(end_s + 0.999))
        chunk_loudness = loud[loud_start:loud_end]
        mocap_signals = mocap_agent.analyze(
            chunk_video, chunk_chat, chunk_duration, sensitivity)
        if needs_full_context:
            (chunk_work / "mocap_signals.json").write_text(
                json.dumps(mocap_signals, indent=2), encoding="utf-8")
            short_video_signals.extend(shift_signal_timestamps(mocap_signals, start_s))
            return []

        clips = run_pipeline(
            source_url=a.url, work=chunk_work, duration_s=chunk_duration,
            segments=chunk_segments, chat_offsets=chunk_chat, loudness=chunk_loudness,
            min_score=a.min_score, trends_file=a.trends, api_keys_str=a.api_keys,
            target_count=a.target_count, additional_signals=mocap_signals,
        )
        for clip in clips:
            clip["start_s"] += start_s
            clip["end_s"] += start_s
            clip["start"] = hms(clip["start_s"])
            clip["end"] = hms(clip["end_s"])
        return clips

    chunk_clips = mocap_agent.analyze_video(
        a.url, work, duration, analyze_chunk, report_progress)
    if needs_full_context:
        clips = run_pipeline(
            source_url=a.url, work=work, duration_s=duration, segments=segments,
            chat_offsets=chat, loudness=loud, min_score=a.min_score, trends_file=a.trends,
            api_keys_str=a.api_keys, target_count=a.target_count,
            additional_signals=short_video_signals,
        )
    else:
        clips = merge_chunk_clips(chunk_clips, a.url, a.target_count)
    (work / "clips.json").write_text(json.dumps(clips, indent=2), encoding="utf-8")
    timestamps = [
        {"clip_id": clip["id"], "start_s": clip["start_s"], "end_s": clip["end_s"]}
        for clip in clips
    ]
    store.record_agent_event("MoCap", "completed",
                             f"Analyzed five chunks and found {len(clips)} candidate moments",
                             task_id=a.task_id or None, progress=100,
                             result={"timestamps": timestamps})
    store.record_agent_event("ShoCap", "running",
                             f"Preparing previews for {len(timestamps)} timestamped moments",
                             task_id=a.task_id or None, result={"timestamps": timestamps})
    report_progress("[twitch] Downloading selected clip previews", 92)
    extracted = ShoCapAgent.prepare_previews(a.url, work, clips)
    store.record_agent_event(
        "ShoCap", "completed", f"Prepared {len(extracted)} clip previews",
        task_id=a.task_id or None, progress=100,
        result={"preview_count": len(extracted), "clip_ids": [clip["id"] for clip in clips]})
    report_progress("[twitch] Ingest complete", 100)
    return {"video_id": work.name, "clips": clips}


if __name__ == "__main__":
    main()
