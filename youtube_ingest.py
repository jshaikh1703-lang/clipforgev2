"""YouTube livestream VOD -> clips.json   (captions first, Whisper only as a fallback)

    python youtube_ingest.py "https://www.youtube.com/watch?v=VIDEO_ID" --min-score 55 --trends trends.txt
"""
import argparse
import glob
import json
import sys
from pathlib import Path
from typing import Any, Callable

import yt_dlp

from agents import MoCapAgent, ShoCapAgent
from common import (
    Segment, audio_loudness, hms, merge_chunk_clips, ram_guard, run_pipeline,
    shift_signal_timestamps, slice_segments, transcribe_audio,
)
from database import ClipForgeStore, default_store_path


def parse_json3(path: str) -> list[Segment]:
    data = json.load(open(path, encoding="utf-8"))
    out = []
    for ev in data.get("events", []):
        event_start = ev.get("tStartMs", 0) / 1000
        for item in ev.get("segs", []):
            text = item.get("utf8", "").replace("\n", " ").strip()
            if not text:
                continue
            start = event_start + item.get("tOffsetMs", 0) / 1000
            duration = item.get("dDurationMs", ev.get("dDurationMs", 0)) / 1000
            words = text.split()
            word_duration = duration / max(1, len(words))
            timestamps = [{"start": start + i * word_duration, "end": start + (i + 1) * word_duration,
                           "text": word} for i, word in enumerate(words)] if duration > 0 else None
            out.append(Segment(start, text, start + duration if duration > 0 else None, timestamps))
    return out


def parse_live_chat(path: str) -> list[float]:
    """Return message offsets (seconds). Reads line by line, so RAM stays flat."""
    offsets = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                ms = json.loads(line)["replayChatItemAction"]["videoOffsetTimeMsec"]
                if int(ms) >= 0:
                    offsets.append(int(ms) / 1000)
            except (KeyError, ValueError, TypeError):
                continue
    return offsets


def main(progress_callback: Callable[[str, float | None], None] | None = None) -> dict[str, Any]:
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("--min-score", type=float, default=55)
    ap.add_argument("--trends", help="text file of trending topics/queries, one per line")
    ap.add_argument("--workdir", default="work")
    ap.add_argument("--no-chat", action="store_true", help="skip live chat replay (slow on long streams)")
    ap.add_argument("--no-audio", action="store_true", help="skip audio download and loudness peaks")
    ap.add_argument("--api-keys", default=None, help="comma-separated provider API keys")
    ap.add_argument("--target-count", type=int, default=10)
    ap.add_argument("--task-id", default="")
    a = ap.parse_args()

    def report_progress(message: str, percent: float | None = None) -> None:
        print(message, flush=True)
        if progress_callback:
            progress_callback(message, percent)

    download_progress_range = [0.0, 10.0]

    def yt_dlp_progress(status: dict) -> None:
        if status.get("status") == "finished":
            report_progress("[yt-dlp] Download finished", download_progress_range[1])
            return
        if status.get("status") != "downloading":
            return

        downloaded = status.get("downloaded_bytes", 0)
        total = status.get("total_bytes") or status.get("total_bytes_estimate")
        percent = downloaded / total * 100 if total else None
        if total:
            message = f"[yt-dlp] {percent:.1f}% ({downloaded / 2**20:.1f}/{total / 2**20:.1f} MiB)"
        else:
            message = f"[yt-dlp] Downloaded {downloaded / 2**20:.1f} MiB"
        eta = status.get("eta")
        if eta is not None:
            message += f", ETA {eta}s"
        scaled_percent = (download_progress_range[0] +
                          (percent or 0) * (download_progress_range[1] -
                                            download_progress_range[0]) / 100)
        report_progress(message, scaled_percent)

    base = {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "extractor_args": {"youtube": {"player_client": ["android", "web"]}},
        "progress_hooks": [yt_dlp_progress],
    }
    report_progress("[youtube] Fetching video metadata")
    with yt_dlp.YoutubeDL(base) as y:
        info = y.extract_info(a.url, download=False)
    if info.get("is_live"):
        sys.exit("This stream is still live. Wait until it ends and YouTube finishes processing the VOD.")
    vid, duration = info["id"], float(info["duration"])
    work = Path(a.workdir) / vid
    work.mkdir(parents=True, exist_ok=True)
    store = ClipForgeStore(default_store_path())
    store.initialize(Path(__file__).with_name("styles"))
    print(f"[meta] {info['title']} ({duration / 3600:.1f} h)")

    # 1) captions (no video download)
    report_progress("[youtube] Fetching captions", 10.0)
    download_progress_range[:] = [10.0, 15.0]
    files = glob.glob(str(work / f"{vid}.en*.json3"))
    segments = []
    if files:
        print(f"[captions] using existing: {files[0]}")
        segments = parse_json3(files[0])
    
    if not segments:
        try:
            caps = {**base, "skip_download": True, "writeautomaticsub": True, "writesubtitles": True,
                    "subtitleslangs": ["en.*"], "subtitlesformat": "json3", "outtmpl": str(work / "%(id)s.%(ext)s"),
                    "ignoreerrors": True, "sleep_interval_subtitles": 3}
            with yt_dlp.YoutubeDL(caps) as y:
                y.download([a.url])
            files = glob.glob(str(work / f"{vid}.en*.json3"))
            segments = parse_json3(files[0]) if files else []
        except Exception as e:
            print(f"[captions] download failed or rate limited (429): {e}")
    ram_guard("captions parsed")

    # 2) live chat replay (optional, best-effort)
    chat: list[float] = []
    if not a.no_chat:
        report_progress("[youtube] Fetching live chat replay", 20.0)
        download_progress_range[:] = [20.0, 25.0]
        try:
            lc = {**base, "skip_download": True, "writesubtitles": True, "subtitleslangs": ["live_chat"],
                  "subtitlesformat": "json", "outtmpl": str(work / "%(id)s.%(ext)s")}
            with yt_dlp.YoutubeDL(lc) as y:
                y.download([a.url])
            cf = glob.glob(str(work / f"{vid}.live_chat.json"))
            if cf:
                chat = parse_live_chat(cf[0])
                Path(cf[0]).unlink()
        except Exception as e:  # chat replay is often unavailable
            print(f"[chat] skipped: {e}")
    ram_guard("chat parsed")

    # 3) audio only (for loudness peaks, and for Whisper if captions are missing)
    loud: list[float] = []
    audio = None
    if not a.no_audio or not segments:
        report_progress("[youtube] Downloading audio for analysis")
        download_progress_range[:] = [25.0, 35.0]
        opts = {**base, "format": "bestaudio/best", "outtmpl": str(work / "audio.%(ext)s")}
        with yt_dlp.YoutubeDL(opts) as y:
            y.download([a.url])
        audio = next(iter(sorted(work.glob("audio.*"))), None)
    if audio and not a.no_audio:
        report_progress("[youtube] Computing audio peaks", 35.0)
        loud = audio_loudness(audio)
        ram_guard("loudness computed")
        report_progress("[youtube] Audio peak analysis complete", 40.0)
    if not segments:
        if audio:
            report_progress("[youtube] Transcribing audio with Whisper", 45.0)
            segments = transcribe_audio(audio)
        else:
            report_progress("[youtube] Captions and audio unavailable; selecting clips from peaks", 55.0)

    if audio and audio.exists():
        audio.unlink()
    report_progress("[youtube] Preparing five low-quality video ranges for MoCap", 65.0)
    sensitivity = store.get_agent_configs()["MoCap"].get("motion_sensitivity", "normal")
    report_progress("[youtube] Analyzing video in five 20% chunks", 70.0)
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
    report_progress("[youtube] Downloading selected clip previews", 92.0)
    extracted = ShoCapAgent.prepare_previews(a.url, work, clips)
    store.record_agent_event(
        "ShoCap", "completed", f"Prepared {len(extracted)} clip previews",
        task_id=a.task_id or None, progress=100,
        result={"preview_count": len(extracted), "clip_ids": [clip["id"] for clip in clips]})
    report_progress("[youtube] Ingest complete", 100.0)
    return {"video_id": vid, "clips": clips}


if __name__ == "__main__":
    main()
