"""Small, deterministic editing agents used by ClipForge's ingest and render stages."""
from __future__ import annotations

import json
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any
from urllib.parse import urlparse


POWER_WORDS = {
    "insane", "crazy", "unbelievable", "never", "finally", "secret", "win", "won", "fail",
    "impossible", "best", "worst", "shocking", "destroyed", "amazing", "huge", "love", "hate",
    "unbelievably", "incredible", "furious", "terrifying", "hilarious",
}
FILLER_WORDS = {"um", "uh", "erm", "ah", "like"}


class HookScoutAgent:
    """Find timestamped transcript moments close to unusually loud audio or chat spikes."""

    def select(self, transcript: str, signals: str, limit: int = 10) -> list[dict[str, Any]]:
        peaks = []
        for line in signals.splitlines():
            match = re.search(r"\[(\d{2}):(\d{2}):(\d{2})\].*?z=([\d.]+)", line)
            if match:
                hours, minutes, seconds, score = match.groups()
                peaks.append((int(hours) * 3600 + int(minutes) * 60 + int(seconds), float(score)))
        moments = []
        for line in transcript.splitlines():
            match = re.match(r"\[(\d{2}):(\d{2}):(\d{2})\]\s*(.+)", line)
            if not match:
                continue
            hours, minutes, seconds, text = match.groups()
            start = int(hours) * 3600 + int(minutes) * 60 + int(seconds)
            nearby = max((score for time, score in peaks if abs(time - start) <= 25), default=0.0)
            if text.strip():
                pace = min(5.0, len(text.split()) / 12)
                moments.append({"start_s": start, "text": text.strip(), "signal_score": nearby,
                                "pace_score": round(pace, 2)})
        moments.sort(key=lambda item: (-(item["signal_score"] + item["pace_score"]), item["start_s"]))
        return moments[:max(1, min(10, limit))]


class MoCapAgent:
    """Find unusual low-resolution motion and align it with chat activity."""

    FRAME_WIDTH = 160
    FRAME_HEIGHT = 90
    SAMPLE_FPS = 0.5

    def analyze(self, video_path: Path, chat_offsets: list[float], duration_s: float,
                sensitivity: str = "normal") -> list[str]:
        import numpy as np

        from common import find_spikes, per_second_counts, hms

        if sensitivity not in {"low", "normal", "high"}:
            raise ValueError("MoCap sensitivity must be low, normal, or high.")
        frame_bytes = self.FRAME_WIDTH * self.FRAME_HEIGHT
        command = [
            "ffmpeg", "-v", "error", "-i", str(video_path), "-map", "0:v:0", "-an",
            "-vf", "fps=0.5,scale=160:90:force_original_aspect_ratio=decrease,"
                  "pad=160:90:(ow-iw)/2:(oh-ih)/2,format=gray",
            "-f", "rawvideo", "-pix_fmt", "gray", "-",
        ]
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if process.stdout is None or process.stderr is None:
            raise RuntimeError("Could not open FFmpeg motion-analysis pipes.")

        motion_values = []
        previous = None
        try:
            while True:
                frame = process.stdout.read(frame_bytes)
                if not frame:
                    break
                if len(frame) != frame_bytes:
                    raise RuntimeError("FFmpeg returned an incomplete motion-analysis frame.")
                current = np.frombuffer(frame, dtype=np.uint8)
                if previous is not None:
                    motion_values.append(float(np.abs(current.astype(np.int16) -
                                                       previous.astype(np.int16)).mean()))
                previous = current
        except Exception:
            process.kill()
            process.wait()
            raise
        error = process.stderr.read().decode("utf-8", errors="replace")
        return_code = process.wait()
        if return_code:
            raise RuntimeError(f"FFmpeg motion analysis failed: {error.strip()}")

        minimum_gap_s = {"low": 30, "normal": 20, "high": 12}[sensitivity]
        motion_peaks = [
            (round(index / self.SAMPLE_FPS), score)
            for index, score in find_spikes(
                motion_values, top=30, min_gap_s=max(1, round(minimum_gap_s * self.SAMPLE_FPS)),
                win=3)
        ]
        chat_peaks = find_spikes(
            per_second_counts(chat_offsets, duration_s), top=30, min_gap_s=minimum_gap_s, win=5)
        candidate_times = sorted({time for time, _ in motion_peaks} |
                                 {time for time, _ in chat_peaks})
        events = []
        for timestamp in candidate_times:
            motion_score = max((score for time, score in motion_peaks if abs(time - timestamp) <= 8),
                               default=0.0)
            chat_score = max((score for time, score in chat_peaks if abs(time - timestamp) <= 8),
                             default=0.0)
            combined = motion_score + chat_score * 0.5
            events.append((combined, timestamp, motion_score, chat_score))
        events.sort(key=lambda event: (-event[0], event[1]))
        return [
            f"mocap [{hms(timestamp)}] motion_z={motion_score:.1f} "
            f"chat_z={chat_score:.1f} z={combined:.1f}"
            for combined, timestamp, motion_score, chat_score in events[:30]
        ]


class MomentHunterAgent:
    """Tag clips with an explainable category using transcript and audio-peak evidence."""

    CATEGORIES = ("Funny / Comedic", "Clutch Outplay", "Banter / Roast", "High-Tension Hook")
    _RULES = (
        ("Clutch Outplay", re.compile(
            r"\b(clutch|outplay|1v[1-5]|one versus|last second|barely|saved|save|comeback|"
            r"no way|almost died|survived)\b", re.I)),
        ("Banter / Roast", re.compile(
            r"\b(roast|trash talk|talking smack|shut up|you suck|you're bad|"
            r"who asked|skill issue|destroyed you|bot)\b", re.I)),
        ("Funny / Comedic", re.compile(
            r"\b(lol|lmao|haha|hilarious|funny|laugh|joke|comedy|silly|"
            r"what the heck|what the hell|are you serious)\b", re.I)),
    )

    def tag(self, clips: list[dict[str, Any]], transcript: str, signals: str) -> list[dict[str, Any]]:
        transcript_lines = []
        for line in transcript.splitlines():
            match = re.match(r"\[(\d{2}):(\d{2}):(\d{2})\]\s*(.+)", line)
            if match:
                hours, minutes, seconds, text = match.groups()
                timestamp = int(hours) * 3600 + int(minutes) * 60 + int(seconds)
                transcript_lines.append((timestamp, text.strip()))

        peaks = []
        for line in signals.splitlines():
            match = re.search(r"\[(\d{2}):(\d{2}):(\d{2})\].*?z=([+-]?[\d.]+)", line)
            if match:
                hours, minutes, seconds, score = match.groups()
                peaks.append((int(hours) * 3600 + int(minutes) * 60 + int(seconds), float(score)))

        for clip in clips:
            start, end = float(clip.get("start_s", 0)), float(clip.get("end_s", 0))
            nearby_lines = [text for timestamp, text in transcript_lines if start <= timestamp <= end]
            evidence_text = " ".join(nearby_lines)
            nearby_peaks = [score for timestamp, score in peaks if start - 3 <= timestamp <= end]
            category, reason = "High-Tension Hook", "An energetic opening or nearby audio peak supports a tension-led hook."
            for candidate, pattern in self._RULES:
                if pattern.search(evidence_text):
                    category = candidate
                    reason = {
                        "Clutch Outplay": "Transcript contains a clutch, close-call, or comeback cue.",
                        "Banter / Roast": "Transcript contains direct teasing or competitive banter.",
                        "Funny / Comedic": "Transcript contains a clear humor or laughter cue.",
                    }[candidate]
                    break
            else:
                if nearby_peaks:
                    reason = f"A nearby loudness/chat peak (z={max(nearby_peaks):.1f}) supports a high-tension hook."

            hook_line = str(clip.get("hook_text", "")).strip()
            if not hook_line and nearby_lines:
                hook_line = nearby_lines[0][:180]
            clip.update({"moment_category": category, "hook_line": hook_line,
                         "moment_reason": reason})
        return clips


class ViralReverseEngineerAgent:
    """Build a lightweight pacing blueprint from public metadata, captions, and sampled frames."""

    @staticmethod
    def analyze(url: str) -> dict[str, Any]:
        parsed = urlparse(url)
        host = (parsed.hostname or "").lower()
        allowed = ("youtube.com", "youtu.be", "instagram.com", "tiktok.com")
        if parsed.scheme not in {"http", "https"} or not any(
                host == domain or host.endswith("." + domain) for domain in allowed):
            raise ValueError("Reference URL must be a public YouTube, Shorts, Reels, or TikTok URL.")

        with tempfile.TemporaryDirectory(prefix="clipforge_blueprint_") as temp_dir:
            output = str(Path(temp_dir) / "reference.%(ext)s")
            metadata = subprocess.run(
                ["yt-dlp", "--no-warnings", "--no-playlist", "--skip-download",
                 "--print", "%(title)s", "--print", "%(duration)s",
                 "--print", "%(description).500B", url],
                check=True, capture_output=True, text=True, timeout=60,
            ).stdout.splitlines()
            try:
                duration = max(0.0, float(metadata[1]))
            except (IndexError, ValueError):
                duration = 0.0
            if duration <= 180:
                subprocess.run(
                    ["yt-dlp", "--no-warnings", "--no-playlist", "--skip-download",
                     "--write-subs", "--write-auto-subs", "--sub-langs", "en",
                     "--sub-format", "vtt", "-o", output, url],
                    check=True, capture_output=True, text=True, timeout=60,
                )
            captions = ""
            for caption_path in Path(temp_dir).glob("reference*.vtt"):
                captions = caption_path.read_text(encoding="utf-8", errors="replace")[:512_000]
                if captions:
                    break
            visual_cut_times = []
            if duration > 0:
                try:
                    visual_cut_times = ViralReverseEngineerAgent._visual_cut_times(
                        url, min(duration, 30.0), temp_dir)
                except (OSError, subprocess.SubprocessError) as exc:
                    print(f"[reference] Visual scene analysis unavailable: {exc}")

        title = metadata[0][:300] if metadata else ""
        description = "\n".join(metadata[2:])[:500] if len(metadata) > 2 else ""
        cue_lines = []
        for line in captions.splitlines():
            line = re.sub(r"<[^>]*>", "", line).strip()
            if not line or "-->" in line or line.isdigit() or line.upper() == "WEBVTT":
                continue
            cue_lines.append(line)
        text = " ".join(cue_lines)
        sentences = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", text) if sentence.strip()]
        word_counts = [len(sentence.split()) for sentence in sentences]
        first_sentence = sentences[0] if sentences else title
        caption_times = []
        for match in re.finditer(r"(\d{2}):(\d{2}):(\d{2})\.\d{3}\s+-->", captions):
            hours, minutes, seconds = (int(value) for value in match.groups())
            caption_times.append(hours * 3600 + minutes * 60 + seconds)
        intervals = [right - left for left, right in zip(caption_times, caption_times[1:]) if right > left]
        subtitle_rhythm = sum(intervals) / len(intervals) if intervals else 0.0
        visual_intervals = [right - left for left, right in zip(visual_cut_times, visual_cut_times[1:])
                            if right > left]
        visual_rhythm = sum(visual_intervals) / len(visual_intervals) if visual_intervals else 0.0
        average_interval = round(visual_rhythm or subtitle_rhythm, 2)
        energy_words = sum(bool(re.search(r"\b(?:never|insane|crazy|wait|what|no way|win|won|"
                                          r"secret|unbelievable|hilarious)\b|[!?]", sentence, re.I))
                           for sentence in sentences)
        energy = round(100 * energy_words / max(1, len(sentences)))
        return {
            "title": title, "duration_s": duration, "description": description,
            "hook_pattern": first_sentence[:180],
            "average_sentence_words": round(sum(word_counts) / max(1, len(word_counts)), 1),
            "sentences_per_minute": round(len(sentences) * 60 / duration, 1) if duration else 0.0,
            "cut_rhythm_seconds": average_interval,
            "cut_rhythm_source": ("ffmpeg scene detection (first 30 seconds)" if visual_rhythm
                                  else "subtitle cue spacing (fallback; scene cuts unavailable)"),
            "visual_cut_count": len(visual_cut_times),
            "caption_energy": energy,
            "caption_sample": text[:500],
        }

    @staticmethod
    def _visual_cut_times(url: str, duration: float, temp_dir: str) -> list[float]:
        if duration < 1:
            return []
        video_template = str(Path(temp_dir) / "visual_reference.%(ext)s")
        subprocess.run(
            ["yt-dlp", "--no-warnings", "--no-playlist", "--download-sections",
             f"*0-{duration:.3f}", "-f", "bv*[height<=360]/b[height<=360]",
             "-o", video_template, url],
            check=True, capture_output=True, text=True, timeout=120,
        )
        videos = list(Path(temp_dir).glob("visual_reference.*"))
        if not videos:
            return []
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-i", str(videos[0]), "-vf",
             "select='gt(scene,0.3)',showinfo", "-an", "-f", "null", "-"],
            check=True, capture_output=True, text=True, timeout=60,
        )
        return sorted(float(match.group(1)) for match in
                      re.finditer(r"pts_time:([0-9]+(?:\.[0-9]+)?)", result.stderr))


class ZoomDirectorAgent:
    """Create a small set of transcript/peak-aligned smooth zoom pulses."""

    _PUNCHLINE = re.compile(r"\b(lol|lmao|haha|wow|wait|no way|clutch|insane|what|"
                            r"got him|let's go|lets go|oh my god)\b", re.I)

    def keyframes(self, start: float, end: float, segments: list[dict[str, Any]],
                  signals: str, intensity: float, rhythm_s: float | None = None) -> list[dict[str, float]]:
        events = {0.0}
        for segment in segments:
            timestamp = float(segment.get("start", 0))
            text = str(segment.get("text", ""))
            if start <= timestamp <= end and self._PUNCHLINE.search(text):
                events.add(round(timestamp - start, 2))
        for line in signals.splitlines():
            match = re.search(r"\[(\d{2}):(\d{2}):(\d{2})\].*?z=([+-]?[\d.]+)", line)
            if not match:
                continue
            hours, minutes, seconds, score = match.groups()
            timestamp = int(hours) * 3600 + int(minutes) * 60 + int(seconds)
            if start <= timestamp <= end and float(score) >= 1.5:
                events.add(round(timestamp - start, 2))

        pulse_duration = 1.8
        minimum_gap = 2.5
        if rhythm_s is not None and 0 < rhythm_s < float("inf"):
            pulse_duration = max(0.6, min(pulse_duration, rhythm_s))
            minimum_gap = max(1.0, min(minimum_gap, rhythm_s))

        selected = []
        for timestamp in sorted(events):
            if timestamp < 0 or timestamp >= end - start - 0.5:
                continue
            if not selected or timestamp - selected[-1] >= minimum_gap:
                selected.append(timestamp)
        return [{"start_s": timestamp, "peak_s": min(end - start, timestamp + 0.45),
                 "end_s": min(end - start, timestamp + pulse_duration), "intensity": intensity}
                for timestamp in selected[:8]]

    @staticmethod
    def expression(keyframes: list[dict[str, float]]) -> str:
        pulses = []
        for frame in keyframes:
            start, peak, end = (max(0.0, frame[key]) * 30 for key in ("start_s", "peak_s", "end_s"))
            amount = max(0.0, min(0.22, frame["intensity"]))
            up = max(1.0, peak - start)
            down = max(1.0, end - peak)
            pulse = (f"if(between(on,{start:.2f},{peak:.2f}),"
                     f"{amount:.4f}*sin(PI*(on-{start:.2f})/{2 * up:.2f}),"
                     f"if(between(on,{peak:.2f},{end:.2f}),"
                     f"{amount:.4f}*cos(PI*(on-{peak:.2f})/{2 * down:.2f}),0))")
            pulses.append(pulse)
        if not pulses:
            return "1"
        nested = pulses[-1]
        for pulse in reversed(pulses[:-1]):
            nested = f"max({pulse},{nested})"
        return "1+" + nested


class FaceLayoutAgent:
    """Find a stable facecam crop from three low-resolution Haar-cascade samples."""

    SAMPLE_FRACTIONS = (0.15, 0.50, 0.80)

    def detect_facecam(self, video_path: Path, duration_s: float) -> dict[str, Any]:
        import cv2

        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        detector = cv2.CascadeClassifier(str(cascade_path))
        if detector.empty():
            raise RuntimeError(f"OpenCV face cascade was not found: {cascade_path}")
        if duration_s <= 0:
            raise ValueError("Clip duration must be positive for face detection.")

        detections = []
        frame_width = frame_height = 0
        for index, fraction in enumerate(self.SAMPLE_FRACTIONS):
            with tempfile.TemporaryDirectory(prefix="clipforge_face_") as temp_dir:
                frame_path = Path(temp_dir) / f"sample_{index}.png"
                sample_time = duration_s * fraction
                subprocess.run(
                    ["ffmpeg", "-v", "error", "-ss", f"{sample_time:.3f}", "-i", str(video_path),
                     "-frames:v", "1", "-f", "image2", "-y", str(frame_path)],
                    check=True, capture_output=True, text=True, timeout=30,
                )
                try:
                    image = cv2.imread(str(frame_path))
                    if image is None:
                        raise RuntimeError(f"FFmpeg produced an unreadable face sample: {index + 1}")
                    frame_height, frame_width = image.shape[:2]
                    sample_height = max(1, round(frame_height * 640 / frame_width))
                    interpolation = cv2.INTER_AREA if frame_width > 640 else cv2.INTER_CUBIC
                    small = cv2.resize(image, (640, sample_height), interpolation=interpolation)
                    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                                      minSize=(28, 28))
                    if len(faces):
                        x, y, width, height = max(faces, key=lambda face: int(face[2]) * int(face[3]))
                        scale_x, scale_y = frame_width / 640, frame_height / sample_height
                        detections.append((
                            (x + width / 2) * scale_x, (y + height / 2) * scale_y,
                            width * scale_x, height * scale_y,
                        ))
                    del image, small, gray
                finally:
                    if frame_path.exists():
                        frame_path.unlink()

        if not detections:
            return {"detected": False, "crop": None,
                    "frame_width": frame_width, "frame_height": frame_height}

        center_x = median(item[0] for item in detections)
        center_y = median(item[1] for item in detections)
        face_width = median(item[2] for item in detections)
        face_height = median(item[3] for item in detections)
        max_height = min(frame_height, frame_width * 9 / 16)
        crop_height = min(max_height, max(face_height * 3.2, face_width * 2.7))
        crop_width = min(frame_width, crop_height * 16 / 9)
        crop_height = crop_width * 9 / 16
        crop_width = max(2, int(crop_width) // 2 * 2)
        crop_height = max(2, int(crop_height) // 2 * 2)
        crop_x = int(round(center_x - crop_width / 2))
        crop_y = int(round(center_y - crop_height * 0.40))
        crop_x = max(0, min(frame_width - crop_width, crop_x))
        crop_y = max(0, min(frame_height - crop_height, crop_y))
        crop_x -= crop_x % 2
        crop_y -= crop_y % 2
        return {
            "detected": True,
            "crop": {"x": crop_x, "y": crop_y, "width": crop_width, "height": crop_height},
            "center": {"x": round(center_x), "y": round(center_y)},
            "frame_width": frame_width, "frame_height": frame_height,
        }


class TrendPulseAgent:
    """Search a few short-form queries and score clips against the returned live titles."""

    STOP_WORDS = {"the", "and", "for", "with", "this", "that", "you", "your", "from",
                  "shorts", "video", "clip", "viral", "what", "when", "how"}

    def analyze(self, trends: str, clips: list[dict[str, Any]]) -> dict[str, Any]:
        queries = [item.strip()[:80] for item in re.split(r"[,;\n]+", trends) if item.strip()][:3]
        if not queries:
            queries = ["viral gaming moments"]
        found = []
        for query in queries:
            result = subprocess.run(
                ["yt-dlp", "--no-warnings", "--flat-playlist", "--dump-json", "--skip-download",
                 "--playlist-end", "10", f"ytsearch10:{query} shorts"],
                check=True, capture_output=True, text=True, timeout=45,
            )
            for line in result.stdout.splitlines()[:10]:
                try:
                    item = json.loads(line)
                except ValueError:
                    continue
                title = str(item.get("title", "")).strip()
                if title:
                    found.append(title[:200])
        unique_titles = list(dict.fromkeys(found))[:30]
        trend_words = set(re.findall(r"[a-z0-9]{3,}", " ".join(unique_titles).lower())) - self.STOP_WORDS
        clip_results = []
        for clip in clips[:50]:
            source = " ".join(str(value) for value in (
                clip.get("hook_text", ""), clip.get("titles", [""])[0] if clip.get("titles") else "",
                clip.get("moment_category", ""),
            )).lower()
            words = set(re.findall(r"[a-z0-9]{3,}", source)) - self.STOP_WORDS
            overlap = len(words & trend_words)
            fit = min(100, 45 + overlap * 15 + (
                10 if clip.get("moment_category") in MomentHunterAgent.CATEGORIES else 0))
            base_title = str(clip.get("titles", [""])[0] if clip.get("titles") else clip.get("hook_text", ""))
            matched_terms = sorted(words & trend_words)
            optimized_title = base_title or "Wait for the ending"
            if matched_terms and matched_terms[0] not in optimized_title.lower():
                optimized_title = f"{optimized_title} | {matched_terms[0].title()}"
            clip_results.append({
                "clip_id": clip.get("id"), "platform_fit": {
                    "youtube_shorts": min(100, fit + 5), "instagram_reels": fit,
                    "tiktok": min(100, fit + (5 if clip.get("moment_category") == "Funny / Comedic" else 0)),
                },
                "optimized_title": optimized_title[:90],
                "hashtags": ["#shorts", "#reels", "#tiktok"] + [
                    "#" + word for word in matched_terms[:3]
                ],
                "loop_ending_suggestion": "End on the reaction, then cut just before the payoff repeats.",
            })
        return {"queries": queries, "trending_titles": unique_titles[:15], "clips": clip_results}


class StudioCommandAgent:
    """Ask the configured shared AI router for a restricted, structured clip-edit plan."""

    def parse(self, command: str, clips: list[dict[str, Any]], api_keys: str | None = None,
              available_styles: set[str] | None = None) -> dict[str, Any]:
        from common import complete_json_with_ai_router

        styles = available_styles or {
            "hormozi_pop", "mrbeast_impact", "clean_podcast", "cyberpunk_neon",
        }
        compact_clips = [{"id": clip.get("id"), "start_s": clip.get("start_s"),
                          "end_s": clip.get("end_s"), "moment_category": clip.get("moment_category")}
                         for clip in clips[:50]]
        prompt = (
            "Convert the user's ClipForge instruction into JSON only with keys filters, updates, zoom, "
            "caption_style, auto_facecam, rerender. filters is an array of exact moment category labels or empty. "
            "updates is an array of {clip_id,start_delta_s,end_delta_s}; use only supplied clip IDs, "
            "small finite deltas, and do not set timestamps directly. zoom is one of off, subtle, punchy "
            f"or null. caption_style is one of {sorted(styles)} "
            "or null. auto_facecam is true to enable the stacked facecam split, false to disable it, "
            "or null to leave unchanged. rerender is boolean. Never invent IDs. Available clips: "
            f"{compact_clips}. User instruction: {command[:1000]}"
        )
        result = complete_json_with_ai_router(prompt, api_keys)
        if not isinstance(result, dict):
            raise ValueError("Command router must return a JSON object.")

        allowed = set(MomentHunterAgent.CATEGORIES)
        filters = result.get("filters", [])
        if not isinstance(filters, list) or any(not isinstance(item, str) or item not in allowed
                                                for item in filters):
            raise ValueError("Command router returned unsupported category filters.")
        valid_ids = {clip.get("id") for clip in clips[:50]}
        updates = result.get("updates", [])
        if not isinstance(updates, list):
            raise ValueError("Command router returned invalid clip updates.")
        safe_updates = []
        for update in updates[:20]:
            if (not isinstance(update, dict) or not isinstance(update.get("clip_id"), int)
                    or update.get("clip_id") not in valid_ids):
                continue
            try:
                start_delta = float(update.get("start_delta_s", 0))
                end_delta = float(update.get("end_delta_s", 0))
            except (TypeError, ValueError):
                continue
            if not (-30 <= start_delta <= 30 and -30 <= end_delta <= 30):
                continue
            safe_updates.append({"clip_id": update["clip_id"], "start_delta_s": start_delta,
                                 "end_delta_s": end_delta})
        zoom = result.get("zoom")
        if zoom is not None and (not isinstance(zoom, str) or zoom not in {"off", "subtle", "punchy"}):
            raise ValueError("Command router returned an unsupported zoom setting.")
        caption_style = result.get("caption_style")
        if caption_style is not None and (
                not isinstance(caption_style, str) or caption_style not in styles):
            raise ValueError("Command router returned an unsupported caption style.")
        auto_facecam = result.get("auto_facecam")
        if auto_facecam is not None and not isinstance(auto_facecam, bool):
            raise ValueError("Command router returned an invalid facecam setting.")
        return {"filters": filters, "updates": safe_updates, "zoom": zoom,
                "caption_style": caption_style, "auto_facecam": auto_facecam,
                "rerender": bool(result.get("rerender", False))}


@dataclass
class CaptionWord:
    start: float
    end: float
    text: str
    highlight: bool


class CaptionStylistAgent:
    """Build real-timestamp caption words and group them into readable 2-4 word chunks."""

    def words_for_clip(self, segments: list[dict[str, Any]], start: float, end: float,
                       all_caps: bool = True) -> list[CaptionWord]:
        flattened = []
        for index, segment in enumerate(segments):
            seg_start = float(segment.get("start", 0))
            next_start = float(segments[index + 1].get("start", seg_start + 2)) if index + 1 < len(segments) else seg_start + 2
            seg_end = float(segment.get("end") or max(seg_start + 0.1, next_start))
            supplied = segment.get("words") or []
            if supplied:
                flattened.extend((float(word["start"]), float(word["end"]), str(word["text"]))
                                 for word in supplied if word.get("text"))
            else:
                tokens = re.findall(r"\S+", str(segment.get("text", "")))
                duration = max(0.1, seg_end - seg_start)
                flattened.extend((seg_start + i * duration / max(1, len(tokens)),
                                  seg_start + (i + 1) * duration / max(1, len(tokens)), token)
                                 for i, token in enumerate(tokens))

        result = []
        for word_start, word_end, token in flattened:
            clean = token.strip()
            if not clean or re.sub(r"[^\w']", "", clean).lower() in FILLER_WORDS:
                continue
            if word_end <= start or word_start >= end:
                continue
            visible = clean.upper() if all_caps else clean
            normalized = re.sub(r"[^\w']", "", clean).lower()
            result.append(CaptionWord(max(0.0, word_start - start), min(end, word_end) - start,
                                      visible, normalized in POWER_WORDS or "!" in clean or "?" in clean))
        return result

    def chunks(self, words: list[CaptionWord], size: int = 3,
               max_gap_s: float = 0.6) -> list[list[CaptionWord]]:
        size = max(2, min(4, size))
        chunks = []
        current = []
        for word in words:
            if current and (len(current) >= size or word.start - current[-1].end > max_gap_s):
                chunks.append(current)
                current = []
            current.append(word)
        if current:
            chunks.append(current)
        return chunks


class DirectorAgent:
    """Choose render defaults while honoring explicit user-selected framing and style."""

    def direct(self, clip: dict[str, Any], requested_framing: str | None = None,
               caption_preset: str | None = None) -> dict[str, Any]:
        seconds = float(clip.get("duration_s", 35))
        framing = requested_framing or ("blur" if seconds < 40 else "fill")
        framing = {"stack": "stacked", "full": "fill", "h": "horizontal"}.get(framing, framing)
        if framing not in {"stacked", "fill", "blur", "horizontal"}:
            framing = "blur"
        return {
            "framing": framing,
            "zoom_interval": 3.0 if seconds < 40 else 4.0,
            "zoom_intensity": 0.025 if seconds < 40 else 0.018,
            "caption_preset": caption_preset or "hormozi_pop",
        }


class EditorAgent(DirectorAgent):
    """Apply validated framing and caption preset selections for rendering."""


class ShoCapAgent:
    """Attach preview URLs and review decisions to ranked short-form clips."""

    @staticmethod
    def present(clips: list[dict[str, Any]], video_id: str,
                reviews: dict[int, dict[str, str]] | None = None,
                work_root: Path | None = None) -> list[dict[str, Any]]:
        reviews = reviews or {}
        presented = []
        for clip in clips:
            item = dict(clip)
            start_ms = round(float(item["start_s"]) * 1000)
            end_ms = round(float(item["end_s"]) * 1000)
            preview_path = (work_root / video_id /
                            f"clip_{item['id']}_{start_ms}_{end_ms}.mp4") if work_root else None
            item["preview_url"] = (f"/work/{video_id}/{preview_path.name}"
                                   if preview_path and preview_path.is_file() else None)
            item["review"] = reviews.get(item["id"], {"status": "pending", "notes": ""})
            presented.append(item)
        return presented


class UpscalerAgent:
    """Produce a Lanczos 4K-sized output frame for portrait or landscape clips."""

    @staticmethod
    def output_size(framing: str) -> tuple[int, int]:
        return (3840, 2160) if framing == "horizontal" else (2160, 3840)

    @classmethod
    def scale_filter(cls, framing: str) -> str:
        width, height = cls.output_size(framing)
        return f"scale={width}:{height}:flags=lanczos"


class ReferenceClonerAgent:
    """Extract visual caption styling from a supplied image using Gemini vision."""

    @staticmethod
    def analyze_image(image: bytes, mime_type: str, api_key: str) -> dict[str, Any]:
        from google import genai
        from google.genai import types

        prompt = (
            "Analyze the reference short-form video frame and return only JSON with keys "
            "font_weight (one of regular, medium, bold, black), primary_hex, highlight_hex, "
            "safe_zone_margin_percent (0-30), and zoom_intensity (0-0.08). "
            "Infer visible typography and colors; use sensible defaults if not visible."
        )
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=[prompt, types.Part.from_bytes(data=image, mime_type=mime_type)],
            config={"response_mime_type": "application/json", "temperature": 0.1},
        )
        import json
        text = re.sub(r"^```(?:json)?|```$", "", (response.text or "").strip(), flags=re.M).strip()
        style = json.loads(text)
        return ReferenceClonerAgent.validate_style(style)

    @staticmethod
    def validate_style(style: dict[str, Any]) -> dict[str, Any]:
        def hex_color(value: Any, fallback: str) -> str:
            value = str(value or "")
            return value.upper() if re.fullmatch(r"#[0-9A-Fa-f]{6}", value) else fallback

        weight = str(style.get("font_weight", "bold")).lower()
        if weight not in {"regular", "medium", "bold", "black"}:
            weight = "bold"
        return {
            "font_weight": weight,
            "font_name": {"regular": "Arial", "medium": "Arial", "bold": "Arial Black",
                          "black": "Impact"}[weight],
            "primary_hex": hex_color(style.get("primary_hex"), "#FFFFFF"),
            "highlight_hex": hex_color(style.get("highlight_hex"), "#FFE600"),
            "safe_zone_margin_percent": max(0, min(30, int(style.get("safe_zone_margin_percent", 18)))),
            "zoom_intensity": max(0.0, min(0.08, float(style.get("zoom_intensity", 0.02)))),
        }

