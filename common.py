from __future__ import annotations

"""Shared pipeline for the ClipForge ingest scripts (YouTube and Twitch).

Flow: timestamped transcript + chat/audio signals -> multi-provider scoring -> local scoring -> clips.json
"""
import json
import math
import os
import re
import subprocess
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import Callable

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) ClipForge/1.0"
_opener = urllib.request.build_opener()
_opener.addheaders = [("User-Agent", USER_AGENT)]
urllib.request.install_opener(_opener)

# Weights from the dashboard's Viral Probability Index.
WEIGHTS = {"hook": 0.30, "pacing": 0.25, "trend": 0.25, "loop": 0.20}
GEMINI_MODELS = ("gemini-3.8-flash", "gemini-3.5-flash-lite", "gemini-2.5-flash-lite")
GROQ_MODELS = ("openai/gpt-oss-120b", "llama-3.1-8b-instant")
OPENROUTER_MODEL = "meta-llama/llama-3.3-70b-instruct:free"
RAM_LIMIT_MB = int(os.environ.get("RAM_LIMIT_MB", "250"))
ACTIVE_REFERENCE_BLUEPRINT: dict | None = None


def default_api_keys() -> str:
    """Resolve provider keys from the environment, then the project .env file."""
    api_keys = os.environ.get("CLIPFORGE_API_KEYS", "").strip()
    if not api_keys:
        env_path = Path(__file__).with_name(".env")
        if env_path.is_file():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                name, separator, value = line.partition("=")
                if separator and name.strip() == "CLIPFORGE_API_KEYS":
                    value = value.strip()
                    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
                        value = value[1:-1]
                    else:
                        value = value.split(" #", 1)[0].strip()
                    api_keys = value
                    break
    return api_keys or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY", "")


@dataclass
class Segment:
    start: float  # seconds from stream start
    text: str
    end: float | None = None
    words: list[dict] | None = None


def slice_segments(segments: list[Segment], start_s: float, end_s: float) -> list[Segment]:
    """Return transcript segments and word timestamps localized to a time range."""
    sliced = []
    for segment in segments:
        segment_end = segment.end if segment.end is not None else segment.start
        if (segment_end < start_s or segment.start >= end_s
                or (segment_end == start_s and segment.start < start_s)):
            continue
        words = []
        for word in segment.words or []:
            try:
                word_start, word_end = float(word["start"]), float(word["end"])
            except (KeyError, TypeError, ValueError):
                continue
            if word_end > start_s and word_start < end_s:
                words.append({**word, "start": max(start_s, word_start) - start_s,
                              "end": min(end_s, word_end) - start_s})
        sliced.append(Segment(
            max(segment.start, start_s) - start_s, segment.text,
            min(segment_end, end_s) - start_s if segment.end is not None else None,
            words or None,
        ))
    return sliced


def hms(sec: float) -> str:
    sec = int(sec)
    return f"{sec // 3600:02d}:{sec % 3600 // 60:02d}:{sec % 60:02d}"


def shift_signal_timestamps(signals: list[str], offset_s: float) -> list[str]:
    timestamp = re.compile(r"\[(\d{2}):(\d{2}):(\d{2})\]")

    def shift(match: re.Match[str]) -> str:
        hours, minutes, seconds = (int(value) for value in match.groups())
        return f"[{hms(hours * 3600 + minutes * 60 + seconds + offset_s)}]"

    return [timestamp.sub(shift, signal) for signal in signals]


def ram_mb() -> float:
    import psutil
    return psutil.Process().memory_info().rss / 2**20


def ram_guard(stage: str, limit_mb: int | None = None) -> None:
    used, limit = ram_mb(), limit_mb or RAM_LIMIT_MB
    print(f"[ram] {stage}: {used:.0f} MB (limit {limit} MB)")
    if used > limit:
        raise MemoryError(f"RAM guard tripped at '{stage}': {used:.0f} MB > {limit} MB")


# ---------- transcript ----------
def transcript_to_text(segments: list[Segment], bucket_s: float = 8.0) -> str:
    """Merge tiny caption fragments into ~8 s lines: '[01:02:03] text'."""
    lines, cur_t, cur = [], None, []
    for seg in sorted(segments, key=lambda s: s.start):
        t = seg.text.strip()
        if not t:
            continue
        if cur_t is None or seg.start - cur_t >= bucket_s:
            if cur:
                lines.append(f"[{hms(cur_t)}] {' '.join(cur)}")
            cur_t, cur = seg.start, []
        cur.append(t)
    if cur:
        lines.append(f"[{hms(cur_t)}] {' '.join(cur)}")
    return "\n".join(lines)


def transcribe_audio(audio_path: Path) -> list[Segment]:
    """Local Whisper (faster-whisper, int8, CPU). Slow on 6 h of audio and uses more RAM than the ingest stage."""
    from faster_whisper import WhisperModel
    model = WhisperModel(os.environ.get("WHISPER_MODEL", "small"), device="cpu", compute_type="int8")
    segs, info = model.transcribe(str(audio_path), language=os.environ.get("WHISPER_LANG", "en"),
                                  vad_filter=True, word_timestamps=True)
    out, next_report = [], 0
    for s in segs:
        words = [{"start": w.start, "end": w.end, "text": w.word.strip()}
                 for w in (s.words or []) if w.word.strip()]
        out.append(Segment(s.start, s.text, s.end, words))
        if s.start >= next_report:
            print(f"[whisper] {hms(s.start)} / {hms(info.duration)}")
            next_report = s.start + 600
    return out


# ---------- signals ----------
def find_spikes(values: list[float], top: int = 30, min_gap_s: int = 30, win: int = 5) -> list[tuple[int, float]]:
    """Smooth per-second values, then return the strongest peaks (t, score), at least min_gap_s apart."""
    n = len(values)
    if n < win * 2:
        return []
    cs = [0.0]
    for v in values:
        cs.append(cs[-1] + v)
    sm = [(cs[min(n, i + win)] - cs[i]) / win for i in range(n)]
    mean = sum(sm) / n
    std = math.sqrt(sum((x - mean) ** 2 for x in sm) / n) or 1.0
    cand = sorted(((x, i) for i, x in enumerate(sm) if x > mean + 2 * std), reverse=True)
    picked: list[tuple[int, float]] = []
    for x, i in cand:
        if all(abs(i - j) >= min_gap_s for j, _ in picked):
            picked.append((i, round((x - mean) / std, 1)))
        if len(picked) >= top:
            break
    return sorted(picked)


def per_second_counts(offsets_s: list[float], duration_s: float) -> list[float]:
    counts = [0.0] * (int(duration_s) + 1)
    for t in offsets_s:
        if 0 <= t <= duration_s:
            counts[int(t)] += 1
    return counts


def audio_loudness(audio_path: Path) -> list[float]:
    """Stream audio through ffmpeg and return loudness (dB-ish) per second. Constant RAM."""
    import numpy as np
    cmd = ["ffmpeg", "-v", "error", "-i", str(audio_path), "-f", "s16le", "-ac", "1", "-ar", "16000", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    out, chunk = [], 32000  # 1 s of 16 kHz int16
    while True:
        buf = proc.stdout.read(chunk)
        if not buf:
            break
        x = np.frombuffer(buf[: len(buf) // 2 * 2], dtype=np.int16).astype(np.float32)
        out.append(20 * math.log10(float(np.sqrt(np.mean(x * x))) + 1.0))
    proc.wait()
    return out


def download_source_video(source_url: str, work: Path) -> Path:
    """Download a bounded-resolution source for analysis and clip extraction."""
    import yt_dlp

    existing = sorted(path for path in work.glob("source.*")
                      if path.suffix.lower() in {".mp4", ".mkv", ".webm"})
    if existing:
        return existing[0]
    options = {
        "format": "bv*[height<=360]+ba/b[height<=360]",
        "merge_output_format": "mp4",
        "outtmpl": str(work / "source.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
    }
    with yt_dlp.YoutubeDL(options) as downloader:
        downloader.download([source_url])
    downloaded = sorted(path for path in work.glob("source.*")
                        if path.suffix.lower() in {".mp4", ".mkv", ".webm"})
    if not downloaded:
        raise RuntimeError("The 360p source download did not produce a video file.")
    return downloaded[0]


def extract_clip_slices(source_path: Path, work: Path, clips: list[dict]) -> list[Path]:
    extracted = []
    for clip in clips:
        start_s, end_s = float(clip["start_s"]), float(clip["end_s"])
        if start_s < 0 or end_s <= start_s:
            raise ValueError(f"Clip {clip.get('id')} has invalid extraction timestamps.")
        clip_path = work / (
            f"clip_{clip['id']}_{round(start_s * 1000)}_{round(end_s * 1000)}.mp4")
        if not clip_path.exists():
            temporary_path = clip_path.with_name(clip_path.stem + ".partial.mp4")
            try:
                subprocess.run([
                    "ffmpeg", "-y", "-v", "error", "-ss", f"{start_s:.3f}",
                    "-i", str(source_path), "-t", f"{end_s - start_s:.3f}",
                    "-map", "0:v:0", "-map", "0:a?", "-c:v", "libx264",
                    "-preset", "veryfast", "-crf", "22", "-threads", "2",
                    "-c:a", "aac", "-b:a", "128k", str(temporary_path),
                ], check=True, timeout=900)
                temporary_path.replace(clip_path)
            finally:
                if temporary_path.exists():
                    temporary_path.unlink()
        extracted.append(clip_path)
    return extracted


def extract_clip_slices_and_cleanup_source(source_path: Path, work: Path,
                                           clips: list[dict]) -> list[Path]:
    extracted = extract_clip_slices(source_path, work, clips)
    source_path.unlink()
    return extracted


def analyze_video_in_chunks(source_path: Path, work: Path, duration_s: float,
                            analyze_chunk: Callable[[Path, Path, int, float, float], list[dict]],
                            progress_callback: Callable[[str, float | None], None] | None = None
                            ) -> list[dict]:
    """Analyze five sequential video sections, persisting results and deleting each section."""
    if not math.isfinite(duration_s) or duration_s <= 0:
        raise ValueError("Video duration must be a positive finite number.")

    chunk_count = 5
    chunk_root = work / "analysis_chunks"
    chunk_root.mkdir(parents=True, exist_ok=True)
    all_clips: list[dict] = []
    for index in range(chunk_count):
        start_s = duration_s * index / chunk_count
        end_s = duration_s * (index + 1) / chunk_count
        chunk_work = chunk_root / f"chunk_{index + 1:02d}"
        chunk_work.mkdir(parents=True, exist_ok=True)
        chunk_video = chunk_work / "analysis_chunk.mp4"
        try:
            if progress_callback:
                progress_callback(
                    f"Processing video segment {index * 20}-{(index + 1) * 20}% "
                    f"(chunk {index + 1}/{chunk_count})",
                    70.0 + index * 4.0,
                )
            subprocess.run([
                "ffmpeg", "-y", "-v", "error", "-ss", f"{start_s:.3f}",
                "-i", str(source_path), "-t", f"{end_s - start_s:.3f}",
                "-map", "0:v:0", "-an", "-c:v", "libx264", "-preset", "ultrafast",
                "-threads", "2", str(chunk_video),
            ], check=True)
            clips = analyze_chunk(chunk_video, chunk_work, index, start_s, end_s)
            if not isinstance(clips, list) or any(not isinstance(clip, dict) for clip in clips):
                raise RuntimeError(f"Analysis of chunk {index + 1} returned invalid clip results.")

            (chunk_work / "results.json").write_text(
                json.dumps(clips, indent=2), encoding="utf-8")
            all_clips.extend(clips)
            results_path = work / "chunk_results.json"
            temporary_results_path = results_path.with_suffix(".json.tmp")
            temporary_results_path.write_text(json.dumps(all_clips, indent=2), encoding="utf-8")
            temporary_results_path.replace(results_path)
        finally:
            if chunk_video.exists():
                chunk_video.unlink()

        if progress_callback:
            progress_callback(
                f"Analyzed video segment {index * 20}-{(index + 1) * 20}% "
                f"(chunk {index + 1}/{chunk_count}); results saved and chunk removed",
                70.0 + (index + 1) * 4.0,
            )
    return all_clips


def merge_chunk_clips(clips: list[dict], source_url: str, target_count: int) -> list[dict]:
    """Rank chunk results globally, remove overlaps, and assign stable clip IDs."""
    ranked = sorted(clips, key=lambda clip: -float(clip.get("viral_probability", 0)))
    selected: list[dict] = []
    for clip in ranked:
        start_s, end_s = float(clip["start_s"]), float(clip["end_s"])
        if start_s < 0 or end_s <= start_s:
            continue
        if any(start_s < current["end_s"] and end_s > current["start_s"] for current in selected):
            continue
        selected.append(dict(clip))
        if len(selected) >= max(1, min(10, int(target_count))):
            break

    for index, clip in enumerate(selected, 1):
        clip["id"] = index
        clip["start"] = hms(clip["start_s"])
        clip["end"] = hms(clip["end_s"])
        clip["duration_s"] = round(clip["end_s"] - clip["start_s"])
        clip["download_cmd"] = (
            f'yt-dlp "{source_url}" --download-sections "*{clip["start"]}-{clip["end"]}" '
            f'--force-keyframes-at-cuts -f "bv*[height<=360]+ba/b[height<=360]" '
            f'-o "work/clip_{index}.%(ext)s"')
    return selected


# ---------- Gemini ----------
PROMPT = Template("""You are a short-form video editor.
Below are a ${hours} hour stream's transcript lines and audio/chat signal peaks.
Pick up to ${n} distinct, self-contained moments, each 25-55 seconds long, for YouTube Shorts / Reels / TikTok.
Rank the clips by Viral Probability. Start with a hook in the first 3 seconds, end on the payoff or a clean loop,
and do not overlap. Use second offsets from stream start. Do not invent transcript content.
Rate each 0-100 on: hook, pacing, trend, and loop.
Return ONLY a JSON array of up to ${n} objects:
[{"start_s": 0, "end_s": 0, "hook_text": "max 6 words, caps", "scores": {"hook": 0, "pacing": 0, "trend": 0, "loop": 0},
  "why": "2 sentences citing specific transcript/signal evidence", "titles": ["", "", ""],
  "caption": "one Instagram caption", "hashtags": ["#", "#", "#", "#", "#"]}]

TRENDING TOPICS:
${trends}

SIGNAL PEAKS:
${signals}

REFERENCE VIRAL BLUEPRINT:
${reference_blueprint}

TRANSCRIPT:
${transcript}
""")


def score_with_ai_router(transcript: str, trends: str, api_keys_str: str | None = None, target_count: int = 10,
                         *, signals: str = "", duration_s: float = 0,
                         reference_blueprint: dict | None = None) -> list[dict]:
    """Try compatible free-tier providers/models in key order; TEST never makes a network call."""
    target_count = max(1, min(10, int(target_count)))
    keys_value = (api_keys_str or "").strip() or default_api_keys()
    keys = [key.strip() for key in keys_value.split(",") if key.strip()]
    if "TEST" in keys:
        return [
            {"start_s": 10 + i * 55, "end_s": 40 + i * 55, "hook_text": f"TEST CLIP {i + 1}",
             "scores": {"hook": 95, "pacing": 95, "trend": 95, "loop": 95},
             "why": "Sample test clip; no external API was called.",
             "titles": [f"Test Clip {i + 1}"], "caption": "Test clip", "hashtags": ["#test"]}
            for i in range(target_count) if 40 + i * 55 <= max(40, duration_s)
        ]
    if not keys:
        raise RuntimeError("Provide one or more Gemini (AIza...), Groq (gsk_...), or OpenRouter (sk-or-...) API keys.")

    prompt = PROMPT.substitute(hours=f"{duration_s / 3600:.1f}", n=target_count,
                               trends=trends or "(none provided)", signals=signals or "(none)",
                               reference_blueprint=json.dumps(reference_blueprint, ensure_ascii=False)
                               if reference_blueprint else "(none provided)",
                               transcript=transcript)
    failures = []
    for api_key in keys:
        try:
            if api_key.startswith(("AIza", "AQ.")):
                for model in GEMINI_MODELS:
                    try:
                        return _gemini_completion(api_key, model, prompt, target_count)
                    except Exception as exc:
                        failures.append(f"Gemini {model}: {exc}")
            elif api_key.startswith("gsk_"):
                for model in GROQ_MODELS:
                    try:
                        return _openai_compatible_completion(
                            "https://api.groq.com/openai/v1/chat/completions", api_key, model, prompt, target_count)
                    except Exception as exc:
                        failures.append(f"Groq {model}: {exc}")
            elif api_key.startswith("sk-or-"):
                try:
                    return _openai_compatible_completion(
                        "https://openrouter.ai/api/v1/chat/completions", api_key, OPENROUTER_MODEL, prompt,
                        target_count)
                except Exception as exc:
                    failures.append(f"OpenRouter {OPENROUTER_MODEL}: {exc}")
            else:
                failures.append(f"Unrecognized API key prefix: {api_key[:5]}...")
        except Exception as exc:
            failures.append(str(exc))
    details = "; ".join(failures[-6:])
    raise RuntimeError(f"All configured AI providers/models failed. {details}")


def complete_json_with_ai_router(prompt: str, api_keys_str: str | None = None) -> dict:
    """Return one JSON object using the configured Gemini/Groq-compatible failover order."""
    keys_value = (api_keys_str or "").strip() or default_api_keys()
    keys = [key.strip() for key in keys_value.split(",") if key.strip()]
    if "TEST" in keys:
        raise RuntimeError("The TEST key cannot route natural-language commands.")
    if not keys:
        raise RuntimeError("Configure a Gemini (AQ./AIza) or Groq (gsk_) API key.")

    failures = []
    for api_key in keys:
        try:
            if api_key.startswith(("AIza", "AQ.")):
                for model in GEMINI_MODELS:
                    try:
                        return _gemini_json_completion(api_key, model, prompt)
                    except Exception as exc:
                        failures.append(f"Gemini {model}: {exc}")
            elif api_key.startswith("gsk_"):
                for model in GROQ_MODELS:
                    try:
                        return _openai_compatible_json_completion(
                            "https://api.groq.com/openai/v1/chat/completions", api_key, model, prompt)
                    except Exception as exc:
                        failures.append(f"Groq {model}: {exc}")
            elif api_key.startswith("sk-or-"):
                try:
                    return _openai_compatible_json_completion(
                        "https://openrouter.ai/api/v1/chat/completions", api_key, OPENROUTER_MODEL, prompt)
                except Exception as exc:
                    failures.append(f"OpenRouter {OPENROUTER_MODEL}: {exc}")
            else:
                failures.append(f"Unrecognized API key prefix: {api_key[:5]}...")
        except Exception as exc:
            failures.append(str(exc))
    raise RuntimeError("All configured AI providers/models failed. " + "; ".join(failures[-6:]))


def _json_object(text: str) -> dict:
    cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    result = json.loads(cleaned)
    if not isinstance(result, dict):
        raise ValueError("AI response must be a JSON object.")
    return result


def _gemini_json_completion(api_key: str, model: str, prompt: str) -> dict:
    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    body = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"responseMimeType": "application/json", "temperature": 0.1},
    }).encode("utf-8")
    request = urllib.request.Request(endpoint, data=body, headers={
        "x-goog-api-key": api_key, "User-Agent": USER_AGENT, "Content-Type": "application/json",
    })
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = json.loads(response.read().decode("utf-8"))
    text = "".join(part.get("text", "") for part in payload["candidates"][0]["content"]["parts"])
    return _json_object(text)


def _openai_compatible_json_completion(endpoint: str, api_key: str, model: str, prompt: str) -> dict:
    body = json.dumps({
        "model": model, "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1, "response_format": {"type": "json_object"},
    }).encode("utf-8")
    request = urllib.request.Request(endpoint, data=body, headers={
        "Authorization": f"Bearer {api_key}", "User-Agent": USER_AGENT,
        "Content-Type": "application/json",
    })
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return _json_object(payload["choices"][0]["message"]["content"])


def _gemini_completion(api_key: str, model: str, prompt: str, max_count: int = 10) -> list[dict]:
    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    body = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"responseMimeType": "application/json", "temperature": 0.3},
    }).encode("utf-8")
    request = urllib.request.Request(endpoint, data=body, headers={
        "x-goog-api-key": api_key,
        "User-Agent": USER_AGENT,
        "Content-Type": "application/json",
    })
    with urllib.request.urlopen(request, timeout=90) as response:
        payload = json.loads(response.read().decode("utf-8"))
    parts = payload["candidates"][0]["content"]["parts"]
    text = "".join(part.get("text", "") for part in parts if isinstance(part, dict))
    if not text:
        raise ValueError("Gemini response did not contain text.")
    return _parse_clip_response(text, max_count)


def _parse_clip_response(text: str, max_count: int = 10) -> list[dict]:
    cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    parsed = json.loads(cleaned)
    if isinstance(parsed, dict):
        parsed = parsed.get("clips", parsed.get("results", []))
    if not isinstance(parsed, list):
        raise ValueError("AI response must be a JSON array of clips.")
    return parsed[:max_count]


def _openai_compatible_completion(endpoint: str, api_key: str, model: str, prompt: str,
                                  max_count: int = 10) -> list[dict]:
    compatible_prompt = prompt + '\nWrap the result as a JSON object with a "clips" array.'
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": compatible_prompt}],
                       "temperature": 0.3, "response_format": {"type": "json_object"}}).encode("utf-8")
    request = urllib.request.Request(endpoint, data=body, headers={
        "Authorization": f"Bearer {api_key}",
        "User-Agent": USER_AGENT,
        "Content-Type": "application/json",
    })
    with urllib.request.urlopen(request, timeout=90) as response:
        payload = json.loads(response.read().decode("utf-8"))
    content = payload["choices"][0]["message"]["content"]
    result = json.loads(content)
    if isinstance(result, dict):
        result = result.get("clips", result.get("results", []))
    if not isinstance(result, list):
        raise ValueError("Provider response did not contain a clip array.")
    return result[:max_count]


def score_with_gemini(transcript: str, signals: str, trends: str, duration_s: float, n: int = 3) -> list[dict]:
    """Backward-compatible alias for the provider router."""
    return score_with_ai_router(transcript, trends, default_api_keys(), n, signals=signals, duration_s=duration_s)


def finalize(raw: list[dict], duration_s: float, min_score: float, source_url: str) -> list[dict]:
    """Validate Gemini output and compute the weighted index locally (so you can tune WEIGHTS)."""
    clips = []
    for c in raw:
        try:
            s, e = float(c["start_s"]), float(c["end_s"])
            f = {k: max(0.0, min(100.0, float(c["scores"][k]))) for k in WEIGHTS}
        except (KeyError, TypeError, ValueError):
            continue
        if not (25 <= e - s <= 55) or s < 0 or e > duration_s + 1:
            continue
        idx = round(sum(WEIGHTS[k] * f[k] for k in WEIGHTS), 1)
        clips.append({"start_s": s, "end_s": e, "start": hms(s), "end": hms(e), "duration_s": round(e - s),
                      "viral_probability": idx, "factors": f, "why": c.get("why", ""),
                      "hook_text": c.get("hook_text", ""), "titles": c.get("titles", [])[:3],
                      "caption": c.get("caption", ""), "hashtags": c.get("hashtags", [])[:5]})
    clips.sort(key=lambda c: -c["viral_probability"])
    kept: list[dict] = []
    for c in clips:
        if c["viral_probability"] >= min_score and all(c["end_s"] <= k["start_s"] or c["start_s"] >= k["end_s"] for k in kept):
            kept.append(c)
    for i, c in enumerate(kept, 1):
        c["id"] = i
        c["download_cmd"] = (f'yt-dlp "{source_url}" --download-sections "*{c["start"]}-{c["end"]}" '
                             f'--force-keyframes-at-cuts -f "bv*[height<=360]+ba/b[height<=360]" '
                             f'-o "work/clip_{i}.%(ext)s"')
    return kept


def clips_from_signal_peaks(signals: str, duration_s: float, min_score: float,
                            source_url: str, target_count: int) -> list[dict]:
    """Create honest, renderable clips when captions are unavailable, without loading Whisper."""
    if duration_s < 25:
        return []
    peaks = []
    for line in signals.splitlines():
        match = re.search(
            r"(?:loud|chat|mocap)\s+\[(\d{2}):(\d{2}):(\d{2})\].*?z=([+-]?[\d.]+)", line)
        if match:
            hours, minutes, seconds, score = match.groups()
            peaks.append((int(hours) * 3600 + int(minutes) * 60 + int(seconds), float(score)))

    clip_length = min(35.0, duration_s)
    candidates = []
    for peak_s, peak_score in sorted(peaks, key=lambda item: -item[1]):
        start_s = max(0.0, min(peak_s - 5.0, duration_s - clip_length))
        end_s = start_s + clip_length
        if any(start_s < clip["end_s"] and end_s > clip["start_s"] for clip in candidates):
            continue
        strength = max(0.0, peak_score)
        factors = {
            "hook": round(min(92.0, 55.0 + strength * 5.0), 1),
            "pacing": round(min(80.0, 55.0 + strength * 3.0), 1),
            "trend": 55.0,
            "loop": 55.0,
        }
        probability = round(sum(WEIGHTS[key] * factors[key] for key in WEIGHTS), 1)
        if probability < min_score:
            continue
        time_label = hms(peak_s)
        candidates.append({
            "start_s": start_s, "end_s": end_s,
            "start": hms(start_s), "end": hms(end_s),
            "duration_s": round(end_s - start_s),
            "viral_probability": probability, "factors": factors,
            "why": (f"Video-motion/chat/audio peak at {time_label} (z={peak_score:.1f}); "
                    "no transcript was available, so this clip is selected from signal evidence only."),
            "hook_text": "AUDIO PEAK", "hook_line": "AUDIO PEAK",
            "moment_category": "High-Tension Hook",
            "moment_reason": f"A nearby audio/chat peak (z={peak_score:.1f}) is the available hook evidence.",
            "titles": [f"Audio Peak at {time_label}"],
            "caption": "", "hashtags": ["#shorts", "#reels", "#tiktok"],
        })
        if len(candidates) >= max(1, min(10, int(target_count))):
            break

    candidates.sort(key=lambda clip: -clip["viral_probability"])
    for index, clip in enumerate(candidates, 1):
        clip["id"] = index
        clip["download_cmd"] = (
            f'yt-dlp "{source_url}" --download-sections "*{clip["start"]}-{clip["end"]}" '
            f'--force-keyframes-at-cuts -f "bv*[height<=360]+ba/b[height<=360]" '
            f'-o "work/clip_{index}.%(ext)s"')
    return candidates


def run_pipeline(*, source_url: str, work: Path, duration_s: float, segments: list[Segment], chat_offsets: list[float],
                 loudness: list[float], min_score: float, trends_file: str | None,
                 api_keys_str: str | None = None, target_count: int = 10,
                 reference_blueprint: dict | None = None,
                 additional_signals: list[str] | None = None) -> list[dict]:
    from agents import HookScoutAgent, MomentHunterAgent

    reference_blueprint = reference_blueprint or ACTIVE_REFERENCE_BLUEPRINT
    transcript = transcript_to_text(segments)
    (work / "transcript.txt").write_text(transcript, encoding="utf-8")
    (work / "transcript_segments.json").write_text(json.dumps([
        {"start": seg.start, "end": seg.end, "text": seg.text, "words": seg.words}
        for seg in segments], ensure_ascii=False), encoding="utf-8")
    sig_lines = []
    if chat_offsets:
        sp = find_spikes(per_second_counts(chat_offsets, duration_s))
        sig_lines += [f"chat [{hms(t)}] z={z}" for t, z in sp]
    if loudness:
        sig_lines += [f"loud [{hms(t)}] z={z}" for t, z in find_spikes(loudness)]
    sig_lines.extend(additional_signals or [])
    hook_candidates = HookScoutAgent().select(transcript, "\n".join(sig_lines), target_count)
    if hook_candidates:
        sig_lines.append("HOOK SCOUT CANDIDATES (transcript + pace + signal):")
        sig_lines.extend(
            f"hook [{hms(item['start_s'])}] signal={item['signal_score']} pace={item['pace_score']}: {item['text']}"
            for item in hook_candidates
        )
    (work / "signals.txt").write_text("\n".join(sig_lines), encoding="utf-8")
    trends = Path(trends_file).read_text(encoding="utf-8") if trends_file else ""
    print(f"[ingest] {len(transcript.split())} words, {len(sig_lines)} signal peaks")
    if not transcript.strip():
        ram_guard("before signal-only clip scoring")
        clips = clips_from_signal_peaks(
            "\n".join(sig_lines), duration_s, min_score, source_url, target_count)
        if reference_blueprint:
            for clip in clips:
                clip["reference_blueprint"] = reference_blueprint
        (work / "clips.json").write_text(json.dumps(clips, indent=2), encoding="utf-8")
        print(f"\n{len(clips)} signal-only clip(s) at {min_score}+ saved to {work / 'clips.json'}")
        return clips
    ram_guard("before gemini")
    keys = api_keys_str
    if keys is None:
        keys = default_api_keys()
    raw = score_with_ai_router(transcript, trends, keys, target_count,
                               signals="\n".join(sig_lines), duration_s=duration_s,
                               reference_blueprint=reference_blueprint)
    clips = finalize(raw, duration_s, min_score, source_url)
    clips = MomentHunterAgent().tag(clips, transcript, "\n".join(sig_lines))
    if reference_blueprint:
        for clip in clips:
            clip["reference_blueprint"] = reference_blueprint
    (work / "clips.json").write_text(json.dumps(clips, indent=2), encoding="utf-8")
    print(f"\n{len(clips)} clip(s) at {min_score}+ saved to {work / 'clips.json'}")
    for c in clips:
        print(f"  #{c['id']} {c['start']}-{c['end']} ({c['duration_s']}s) {c['viral_probability']}%  {c['hook_text']}")
        print(f"     {c['download_cmd']}")
    return clips
