from __future__ import annotations

import asyncio
import json
import math
import re
import subprocess
import sys
import tempfile
import threading
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

import common
from agents import (
    CaptionStylistAgent,
    EditorAgent,
    FaceLayoutAgent,
    ReferenceClonerAgent,
    ShoCapAgent,
    StudioCommandAgent,
    TrendPulseAgent,
    UpscalerAgent,
    ViralReverseEngineerAgent,
    ZoomDirectorAgent,
)
from database import AGENT_DEFAULTS, ClipForgeStore, default_store_path

BASE_DIR = Path(__file__).resolve().parent
WORK_DIR = BASE_DIR / "work"
OUTPUT_DIR = BASE_DIR / "output"
STYLES_DIR = BASE_DIR / "styles"
for directory in (WORK_DIR, OUTPUT_DIR, STYLES_DIR):
    directory.mkdir(exist_ok=True)

STORE = ClipForgeStore(default_store_path())
STORE.initialize(STYLES_DIR)


@asynccontextmanager
async def lifespan(_: FastAPI):
    await recover_persistent_tasks()
    yield


app = FastAPI(title="ClipForge API", lifespan=lifespan)
app.mount("/work", StaticFiles(directory=str(WORK_DIR)), name="work")
app.mount("/output", StaticFiles(directory=str(OUTPUT_DIR)), name="output")
INGEST_LOCK = threading.Lock()
INGEST_ASYNC_LOCK = asyncio.Lock()
INGEST_PROGRESS_LOCK = threading.Lock()
INGEST_PROGRESS = {"percent": 0.0, "message": "Idle"}
ACTIVE_INGEST_TASK_ID: str | None = None
RENDER_WORKER_LOCK = threading.Lock()
ACTIVE_WORKERS: set[asyncio.Task] = set()
CAPTION_AGENT = CaptionStylistAgent()
EDITOR_AGENT = EditorAgent()
FACE_LAYOUT_AGENT = FaceLayoutAgent()
ZOOM_AGENT = ZoomDirectorAgent()
COMMAND_AGENT = StudioCommandAgent()
TREND_AGENT = TrendPulseAgent()
SHOCAP_AGENT = ShoCapAgent()
UPSCALER_AGENT = UpscalerAgent()


class IngestRequest(BaseModel):
    url: str
    min_score: float = 55.0
    trends: Optional[str] = None
    api_key: Optional[str] = None
    target_count: int = Field(default=10, ge=1, le=10)
    reference_blueprint: Optional[dict[str, Any]] = None


class RenderRequest(BaseModel):
    clip_id: int = Field(ge=1)
    video_id: str
    url: str
    start: str
    end: str
    duration_s: float
    start_s: Optional[float] = None
    end_s: Optional[float] = None
    framing: str = "stacked"
    caption_style: str = "hormozi_pop"
    all_caps: bool = True
    safe_zone: bool = True
    letterbox: bool = False
    font_size: int = Field(default=1, ge=0, le=2)
    dynamic_zoom: str = "off"
    reference_blueprint: Optional[dict[str, Any]] = None
    auto_facecam: bool = True


class BatchRenderRequest(BaseModel):
    clips: list[RenderRequest]
    count: int = Field(default=10, ge=1, le=10)


class AgentCommandRequest(BaseModel):
    command: str = Field(min_length=1, max_length=1000)
    video_id: Optional[str] = None
    clip_id: Optional[int] = None
    clips: list[dict[str, Any]] = Field(default_factory=list)
    render_requests: list[RenderRequest] = Field(default_factory=list)
    api_key: Optional[str] = None


class RetryRequest(BaseModel):
    api_key: Optional[str] = None


class ReverseEngineerRequest(BaseModel):
    url: str


class TrendPulseRequest(BaseModel):
    trends: str = ""
    clips: list[dict[str, Any]] = Field(default_factory=list)


class CommandRequest(BaseModel):
    prompt: str
    clips: list[dict[str, Any]] = Field(default_factory=list)
    api_key: Optional[str] = None


def _style_presets() -> dict[str, dict[str, Any]]:
    return STORE.get_styles()


@app.get("/", response_class=HTMLResponse)
async def read_index():
    return (BASE_DIR / "index.html").read_text(encoding="utf-8")


def _set_ingest_progress(message: str, percent: float | None = None) -> None:
    with INGEST_PROGRESS_LOCK:
        INGEST_PROGRESS["message"] = message
        if percent is not None:
            INGEST_PROGRESS["percent"] = max(0.0, min(100.0, percent))
        task_id = ACTIVE_INGEST_TASK_ID
        current_percent = INGEST_PROGRESS["percent"]
    if task_id:
        STORE.update_task(task_id, status="running", progress=current_percent,
                          stage=message, message=message)


def _run_ingest_script(main_func, argv: list[str], progress_callback=None,
                       reference_blueprint: dict[str, Any] | None = None) -> dict[str, Any]:
    with INGEST_LOCK:
        old_argv = sys.argv
        previous_blueprint = common.ACTIVE_REFERENCE_BLUEPRINT
        try:
            common.ACTIVE_REFERENCE_BLUEPRINT = reference_blueprint
            sys.argv = argv
            if progress_callback:
                result = main_func(progress_callback=progress_callback)
            else:
                result = main_func()
            if not isinstance(result, dict):
                raise RuntimeError("The ingest pipeline did not return a result object.")
            return result
        finally:
            common.ACTIVE_REFERENCE_BLUEPRINT = previous_blueprint
            sys.argv = old_argv


def _normalize_ingest_result(result: Any) -> dict[str, Any]:
    if not isinstance(result, dict):
        raise ValueError("The ingest pipeline returned an invalid response.")
    video_id = result.get("video_id")
    if not isinstance(video_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", video_id):
        raise ValueError("The ingest pipeline returned an invalid video ID.")
    raw_clips = result.get("clips")
    if not isinstance(raw_clips, list):
        raise ValueError("The ingest pipeline did not return a clips list.")

    clips = []
    for index, raw_clip in enumerate(raw_clips, 1):
        if not isinstance(raw_clip, dict):
            raise ValueError(f"Clip {index} is not a JSON object.")
        clip = dict(raw_clip)
        try:
            score = float(clip.get("viral_probability", 0))
        except (TypeError, ValueError):
            score = 0.0
        clip["viral_probability"] = score if math.isfinite(score) else 0.0
        clip["id"] = (clip["id"] if isinstance(clip.get("id"), int)
                      and not isinstance(clip["id"], bool) and clip["id"] > 0 else index)

        try:
            start_value = clip.get("start_s")
            if start_value is None:
                start_value = _seconds(str(clip.get("start", "00:00:00")))
            start_s = max(0.0, float(start_value))
        except (TypeError, ValueError):
            start_s = 0.0
        try:
            end_value = clip.get("end_s")
            if end_value is None:
                end_value = _seconds(str(clip.get("end", "")))
            end_s = float(end_value)
        except (TypeError, ValueError):
            end_s = start_s + 30.0
        if not math.isfinite(end_s) or end_s <= start_s:
            end_s = start_s + 30.0
        clip.update({
            "start_s": start_s, "end_s": end_s,
            "start": common.hms(start_s), "end": common.hms(end_s),
            "duration_s": max(1, round(end_s - start_s)),
        })
        clip["moment_category"] = str(clip.get("moment_category") or "High-Tension Hook")
        clip["hook_text"] = str(clip.get("hook_text") or "")
        clip["hook_line"] = str(clip.get("hook_line") or clip["hook_text"])
        clip["why"] = str(clip.get("why") or "")
        clip["moment_reason"] = str(clip.get("moment_reason") or clip["why"] or
                                    "Moment analysis did not provide a reason.")
        clip["titles"] = clip.get("titles") if isinstance(clip.get("titles"), list) else []
        clip["hashtags"] = clip.get("hashtags") if isinstance(clip.get("hashtags"), list) else []
        clip["factors"] = clip.get("factors") if isinstance(clip.get("factors"), dict) else {}
        if not isinstance(clip.get("reference_blueprint"), dict):
            clip["reference_blueprint"] = None
        clips.append(clip)
    reviews = STORE.clip_reviews(video_id)
    presented = SHOCAP_AGENT.present(clips, video_id, reviews, WORK_DIR)
    return {"video_id": video_id, "clips": presented}


@app.get("/api/ingest/progress")
async def ingest_progress():
    with INGEST_PROGRESS_LOCK:
        return dict(INGEST_PROGRESS)


@app.post("/api/ingest")
async def ingest_video(req: IngestRequest):
    api_keys = (req.api_key or "").strip()
    test_mode = "TEST" in [key.strip() for key in api_keys.split(",")]
    if test_mode:
        script = ""
        main_func = None
    elif "youtube.com" in req.url or "youtu.be" in req.url:
        import youtube_ingest
        script, main_func = "youtube_ingest.py", youtube_ingest.main
    elif "twitch.tv" in req.url:
        import twitch_ingest
        script, main_func = "twitch_ingest.py", twitch_ingest.main
    else:
        raise HTTPException(status_code=400, detail="Unsupported URL")

    task_id = uuid.uuid4().hex
    STORE.create_task(task_id, "ingest", req.model_dump(exclude={"api_key"}),
                      status="running", message="Starting ingest")
    STORE.record_agent_event("Supervisor", "running", "Ingest workflow started", task_id=task_id)
    if req.trends:
        STORE.save_trends(req.trends, {"topics": [
            topic.strip() for topic in re.split(r"[,;\n]+", req.trends) if topic.strip()
        ], "source": "ingest"})
    global ACTIVE_INGEST_TASK_ID
    ACTIVE_INGEST_TASK_ID = task_id
    trends_path = None
    try:
        async with INGEST_ASYNC_LOCK:
            _set_ingest_progress("Starting ingest", 0.0)
            if test_mode:
                video_id = "test_video_" + "".join(filter(str.isalnum, req.url))[-8:]
                work = WORK_DIR / video_id
                work.mkdir(parents=True, exist_ok=True)
                clips = common.run_pipeline(
                    source_url=req.url, work=work, duration_s=600,
                    segments=[common.Segment(0, "Test transcript with a real caption sample.")],
                    chat_offsets=[], loudness=[], min_score=req.min_score, trends_file=None,
                    api_keys_str="TEST", target_count=req.target_count,
                    reference_blueprint=req.reference_blueprint,
                )
                result = {"video_id": video_id, "clips": clips}
            else:
                if not api_keys:
                    api_keys = common.default_api_keys()
                argv = [script, req.url, "--min-score", str(req.min_score), "--workdir", str(WORK_DIR),
                        "--target-count", str(req.target_count), "--task-id", task_id]
                if api_keys:
                    argv.extend(["--api-keys", api_keys])
                if req.trends:
                    trends_path = WORK_DIR / f"trends_{task_id}.txt"
                    trends_path.write_text(req.trends, encoding="utf-8")
                    argv.extend(["--trends", str(trends_path)])
                progress_callback = _set_ingest_progress
                result = await run_in_threadpool(_run_ingest_script, main_func, argv, progress_callback,
                                                 req.reference_blueprint)
            response = _normalize_ingest_result(result)
            response["task_id"] = task_id
            _set_ingest_progress("Ingest complete", 100.0)
            STORE.update_task(task_id, status="completed", progress=100, stage="Complete",
                              message="Ingest completed", result_json=response)
            STORE.record_agent_event("Supervisor", "completed", "Ingest workflow completed",
                                     task_id=task_id, progress=100, result={"video_id": response["video_id"]})
            return response
    except HTTPException as exc:
        STORE.update_task(task_id, status="failed", stage="Failed", message="Ingest failed", error=str(exc.detail))
        STORE.record_agent_event("Supervisor", "failed", "Ingest workflow failed",
                                 task_id=task_id, result={"error": str(exc.detail)})
        raise
    except Exception as exc:
        _set_ingest_progress(f"Ingest failed: {exc}")
        STORE.update_task(task_id, status="failed", stage="Failed", message="Ingest failed", error=str(exc))
        STORE.record_agent_event("Supervisor", "failed", "Ingest workflow failed",
                                 task_id=task_id, result={"error": str(exc)})
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        ACTIVE_INGEST_TASK_ID = None
        if trends_path and trends_path.exists():
            trends_path.unlink()


@app.post("/api/reverse-engineer")
async def reverse_engineer(req: ReverseEngineerRequest):
    try:
        blueprint = await run_in_threadpool(ViralReverseEngineerAgent.analyze, req.url)
        return {"blueprint": blueprint}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Reference analysis failed: {exc}") from exc


@app.post("/api/trend-pulse")
async def trend_pulse(req: TrendPulseRequest):
    try:
        result = await run_in_threadpool(TREND_AGENT.analyze, req.trends, req.clips)
        STORE.save_trends(req.trends, result)
        return result
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Trend search failed: {exc}") from exc


@app.post("/api/command")
async def command_agent(req: CommandRequest):
    if not req.prompt.strip():
        raise HTTPException(status_code=400, detail="Enter a command.")
    try:
        return await run_in_threadpool(
            COMMAND_AGENT.parse, req.prompt, req.clips, req.api_key or common.default_api_keys(),
            set(_style_presets()))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Command processing failed: {exc}") from exc


def _seconds(value: str) -> float:
    parts = value.split(":")
    if len(parts) != 3:
        raise ValueError(f"Invalid timestamp: {value}")
    return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])


def _ass_time(seconds: float) -> str:
    centiseconds = max(0, round(seconds * 100))
    hours, remainder = divmod(centiseconds, 360000)
    minutes, remainder = divmod(remainder, 6000)
    secs, cs = divmod(remainder, 100)
    return f"{hours}:{minutes:02d}:{secs:02d}.{cs:02d}"


def _ass_color(hex_color: str) -> str:
    color = hex_color.lstrip("#")
    red, green, blue = color[0:2], color[2:4], color[4:6]
    return f"&H00{blue}{green}{red}&"


def _zoom_intensity(mode: str, style: dict[str, Any]) -> float:
    zoom_modes = {"off": 0.0, "subtle": 0.12, "punchy": 0.22}
    if mode not in zoom_modes:
        raise ValueError("Dynamic zoom must be off, subtle, or punchy.")
    try:
        style_zoom = float(style.get("zoom_intensity", 0.025))
    except (TypeError, ValueError) as exc:
        raise ValueError("Style preset zoom_intensity must be between 0 and 0.08.") from exc
    if not math.isfinite(style_zoom) or not 0 <= style_zoom <= 0.08:
        raise ValueError("Style preset zoom_intensity must be between 0 and 0.08.")
    return min(0.22, zoom_modes[mode] * style_zoom / 0.025)


def generate_ass(path: Path, req: RenderRequest, transcript_segments: list[dict[str, Any]] | None = None,
                 style: dict[str, Any] | None = None,
                 reference_blueprint: dict[str, Any] | None = None,
                 split_seam: bool = False, horizontal: bool = False) -> None:
    style = style or {}
    start = req.start_s if req.start_s is not None else _seconds(req.start)
    end = req.end_s if req.end_s is not None else _seconds(req.end)
    font_sizes = (48, 64, 78)
    font_size = font_sizes[req.font_size]
    font = style.get("font_name", "Impact")
    primary = _ass_color(style.get("primary_hex", "#FFFFFF"))
    highlight = _ass_color(style.get("highlight_hex", "#FFE600"))
    canvas_width, canvas_height = (1920, 1080) if horizontal else (1080, 1920)
    margin_v = (672 if split_seam else
                int(canvas_height * float(style.get("safe_zone_margin_percent", 18)) / 100)
                if req.safe_zone else 80)
    alignment = 8 if split_seam else 2
    outline = max(4, int(font_size * 0.09))
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {canvas_width}
PlayResY: {canvas_height}

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font},{font_size},{highlight},{primary},&H00000000,&H64000000,1,0,0,0,100,100,0,0,1,{outline},3,{alignment},70,70,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    word_items = CAPTION_AGENT.words_for_clip(transcript_segments or [], start, end, req.all_caps)
    lines = []
    caption_energy = float((reference_blueprint or {}).get("caption_energy", 40))
    chunk_size = 2 if caption_energy >= 65 else 4 if caption_energy <= 20 else 3
    for chunk in CAPTION_AGENT.chunks(word_items, chunk_size):
        if not chunk:
            continue
        event_start, event_end = chunk[0].start, chunk[-1].end
        parts = [r"{\t(0,120,\fscx116\fscy116)\t(120,240,\fscx100\fscy100)}"]
        for index, word in enumerate(chunk):
            next_start = chunk[index + 1].start if index + 1 < len(chunk) else word.end
            duration_cs = max(1, round((max(word.end, next_start) - word.start) * 100))
            color_tag = rf"{{\1c{highlight}}}" if word.highlight else ""
            escaped = word.text.replace("{", r"\{").replace("}", r"\}")
            parts.append(f"{color_tag}{{\\kf{duration_cs}}}{escaped}")
        lines.append(f"Dialogue: 0,{_ass_time(event_start)},{_ass_time(event_end)},Default,,0,0,0,,{' '.join(parts)}\n")
    path.write_text(header + "".join(lines), encoding="utf-8")


def _safe_video_dir(video_id: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", video_id):
        raise ValueError("Invalid video_id")
    return WORK_DIR / video_id


def _render_clip(req: RenderRequest, progress=None) -> dict[str, Any]:
    video_dir = _safe_video_dir(req.video_id)
    video_dir.mkdir(parents=True, exist_ok=True)
    start_s = req.start_s if req.start_s is not None else _seconds(req.start)
    end_s = req.end_s if req.end_s is not None else _seconds(req.end)
    clip_path = video_dir / f"clip_{req.clip_id}_{round(start_s * 1000)}_{round(end_s * 1000)}.mp4"
    if progress:
        progress("Downloading slice...")
    if not clip_path.exists():
        cmd = [
            "yt-dlp", "--newline", "--no-playlist",
            "--extractor-args", "youtube:player_client=android,web",
            "-f", "bv*[height<=360]+ba/b[height<=360]",
            req.url, "--download-sections", f"*{req.start}-{req.end}",
            "--force-keyframes-at-cuts",
            "--merge-output-format", "mp4", "-o", str(clip_path),
        ]
        subprocess.run(cmd, check=True, timeout=600)

    if progress:
        progress("Syncing captions...")
    segments_path = video_dir / "transcript_segments.json"
    segments = json.loads(segments_path.read_text(encoding="utf-8")) if segments_path.exists() else []
    face_layout = None
    requested_framing = req.framing
    if req.framing == "stacked" and req.auto_facecam:
        if progress:
            progress("Detecting facecam from three sample frames...")
        face_layout = FACE_LAYOUT_AGENT.detect_facecam(clip_path, end_s - start_s)
        if not face_layout["detected"]:
            requested_framing = "fill"

    presets = _style_presets()
    director = EDITOR_AGENT.direct(req.model_dump(), requested_framing, req.caption_style)
    preset = presets.get(director["caption_preset"], presets.get("hormozi_pop", {}))
    ass_path = video_dir / f"clip_{req.clip_id}.ass"
    framing = director["framing"]
    generate_ass(ass_path, req, segments, preset, req.reference_blueprint,
                 split_seam=framing == "stacked", horizontal=framing == "horizontal")

    if progress:
        progress(f"Mastering {'16:9' if framing == 'horizontal' else '9:16'} HD...")
    output_filename = f"render_{req.clip_id}_{director['framing']}_4k.mp4"
    output_path = OUTPUT_DIR / output_filename
    zoom_intensity = _zoom_intensity(req.dynamic_zoom, preset)
    signals_path = video_dir / "signals.txt"
    signals = signals_path.read_text(encoding="utf-8") if signals_path.exists() else ""
    try:
        reference_rhythm = float((req.reference_blueprint or {}).get("cut_rhythm_seconds", 0))
    except (TypeError, ValueError):
        reference_rhythm = 0.0
    if not math.isfinite(reference_rhythm) or reference_rhythm <= 0:
        reference_rhythm = None
    keyframes = ZOOM_AGENT.keyframes(start_s, end_s, segments, signals, zoom_intensity, reference_rhythm)
    zoom_expression = ZOOM_AGENT.expression(keyframes)

    if framing == "stacked":
        if face_layout and face_layout["detected"]:
            crop = face_layout["crop"]
            face_crop = f"crop={crop['width']}:{crop['height']}:{crop['x']}:{crop['y']}"
        else:
            face_crop = "crop=iw*0.22:ih*0.3:iw*0.78:ih*0.7"
        face_zoom = (f",zoompan=z='{zoom_expression}':x='iw/2-(iw/zoom/2)':"
                     "y='ih/2-(ih/zoom/2)':d=1:s=1080x672:fps=30") if req.dynamic_zoom != "off" else ""
        game_zoom = (f",zoompan=z='{zoom_expression}':x='iw/2-(iw/zoom/2)':"
                     "y='ih/2-(ih/zoom/2)':d=1:s=1080x1248:fps=30") if req.dynamic_zoom != "off" else ""
        game_crop = ("crop=w='min(iw,ih*1080/1248)':h='min(ih,iw*1248/1080)':"
                     "x='(iw-ow)/2':y='(ih-oh)/2',scale=1080:1248")
        v_filter = (
            f"[0:v]split=2[face_src][game_src];"
            f"[face_src]{face_crop},scale=1080:672:force_original_aspect_ratio=increase,"
            f"crop=1080:672{face_zoom}[face];"
            f"[game_src]{game_crop}{game_zoom}[game];"
            "[face][game]vstack=inputs=2[v0]"
        )
    elif framing == "fill":
        v_filter = "[0:v]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920[v0]"
    elif framing == "horizontal":
        letterbox = ",drawbox=x=0:y=0:w=iw:h=ih*0.12:color=black:t=fill,drawbox=x=0:y=ih*0.88:w=iw:h=ih*0.12:color=black:t=fill" if req.letterbox else ""
        v_filter = f"[0:v]scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2{letterbox}[v0]"
    else:
        v_filter = ("[0:v]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,boxblur=40:5[bg];"
                    "[0:v]scale=1080:-2[fg];[bg][fg]overlay=(W-w)/2:(H-h)/2[v0]")

    frame_width, frame_height = UPSCALER_AGENT.output_size(framing)
    frame_size = f"{frame_width}x{frame_height}"
    zoom_filter = (f"zoompan=z='{zoom_expression}':"
                   f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=1:s={frame_size}:fps=30")
    ass_filter_path = str(ass_path.resolve()).replace("\\", "/").replace(":", r"\:").replace("'", r"\'")
    if progress:
        progress("Upscaling to 4K...")
    video_chain = (f"[v0]{zoom_filter}," if req.dynamic_zoom != "off" and framing != "stacked"
                   else f"[v0]{UPSCALER_AGENT.scale_filter(framing)},")
    filter_complex = (f"{v_filter};{video_chain}"
                      f"eq=contrast=1.07:saturation=1.16,unsharp=5:5:0.8:5:5:0.0,"
                      f"subtitles=filename='{ass_filter_path}'[out]")
    ffmpeg_cmd = [
        "ffmpeg", "-y", "-i", str(clip_path), "-filter_complex", filter_complex,
        "-map", "[out]", "-map", "0:a?", "-c:v", "libx264", "-threads", "2",
        "-preset", "veryfast", "-crf", "19", "-c:a", "aac", "-b:a", "160k",
        "-af", "loudnorm=I=-14:TP=-1.5:LRA=11", str(output_path),
    ]
    subprocess.run(ffmpeg_cmd, check=True, timeout=1800)
    return {"output": f"/output/{output_filename}", "framing": framing,
            "face_detected": face_layout["detected"] if face_layout else None}


@app.post("/api/render")
async def render_clip(req: RenderRequest):
    try:
        return _render_clip(req)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def _run_render_batch(job_id: str, clips: list[RenderRequest]) -> None:
    with RENDER_WORKER_LOCK:
        task = STORE.get_task(job_id)
        if not task:
            return
        result_state = task.get("result") or {}
        result_state.setdefault("results", [])
        result_state.setdefault("completed", 0)
        result_state.setdefault("total", len(clips))
        completed_count = int(result_state["completed"])
        STORE.update_task(job_id, status="running", stage="Rendering",
                          message="Render batch running")
        STORE.record_agent_event("Supervisor", "running", "Render batch started",
                                 task_id=job_id)
        try:
            for index, req in enumerate(clips[completed_count:], start=completed_count):
                def update_step(step: str, current=index):
                    agent = "Upscaler" if step.startswith("Upscaling") else "Editor"
                    message = f"Clip {current + 1}/{len(clips)}: {step}"
                    STORE.update_task(
                        job_id, status="running", progress=round(current * 100 / len(clips), 1),
                        stage=step, message=message)
                    STORE.record_agent_event(agent, "running", message, task_id=job_id,
                                             progress=round(current * 100 / len(clips), 1))

                STORE.record_agent_event("Editor", "running", f"Editing clip {index + 1}",
                                         task_id=job_id)
                rendered = _render_clip(req, update_step)
                result_state["results"].append({"clip_id": req.clip_id, **rendered})
                result_state["completed"] = index + 1
                STORE.update_task(
                    job_id, status="running", progress=round((index + 1) * 100 / len(clips), 1),
                    stage="Clip complete", message=f"Completed clip {index + 1}/{len(clips)}",
                    result_json=result_state)
                STORE.record_agent_event("Editor", "completed", f"Clip {req.clip_id} rendered",
                                         task_id=job_id, progress=round((index + 1) * 100 / len(clips), 1),
                                         result=rendered)
                STORE.record_agent_event("Upscaler", "completed", "4K output ready",
                                         task_id=job_id, progress=round((index + 1) * 100 / len(clips), 1))
            STORE.update_task(job_id, status="completed", progress=100, stage="Complete",
                              message="Render batch completed", result_json=result_state)
            STORE.record_agent_event("Supervisor", "completed", "Render batch completed",
                                     task_id=job_id, progress=100)
        except Exception as exc:
            STORE.update_task(job_id, status="failed", stage="Failed", message="Render batch failed",
                              error=str(exc), result_json=result_state)
            STORE.record_agent_event("Supervisor", "failed", "Render batch failed",
                                     task_id=job_id, result={"error": str(exc)})


def _schedule_render_worker(job_id: str, clips: list[RenderRequest]) -> None:
    async def run_worker():
        await asyncio.to_thread(_run_render_batch, job_id, clips)

    worker = asyncio.create_task(run_worker())
    ACTIVE_WORKERS.add(worker)
    worker.add_done_callback(ACTIVE_WORKERS.discard)


def _create_render_task(clips: list[RenderRequest]) -> str:
    if not clips:
        raise ValueError("No clips were supplied.")
    if len(clips) > 10:
        raise ValueError("A batch can contain at most 10 clips.")
    job_id = uuid.uuid4().hex
    STORE.create_task(job_id, "render_batch",
                      {"clips": [clip.model_dump() for clip in clips]},
                      status="queued", message="Render batch queued")
    STORE.record_agent_event("Supervisor", "queued", "Render batch queued", task_id=job_id)
    return job_id


@app.post("/api/render-batch", status_code=202)
async def render_batch(req: BatchRenderRequest, background_tasks: BackgroundTasks):
    clips = req.clips[:req.count]
    if not clips:
        raise HTTPException(status_code=400, detail="No clips were supplied.")
    if len(clips) > 10:
        raise HTTPException(status_code=400, detail="A batch can contain at most 10 clips.")
    job_id = _create_render_task(clips)
    background_tasks.add_task(_run_render_batch, job_id, clips)
    return {"job_id": job_id}


@app.get("/api/render-batch/{job_id}")
async def render_batch_status(job_id: str):
    task = STORE.get_task(job_id)
    if not task or task["kind"] != "render_batch":
        raise HTTPException(status_code=404, detail="Render job not found.")
    result = task.get("result") or {}
    return {
        "job_id": job_id,
        "status": task["status"],
        "current": result.get("completed", 0) + (1 if task["status"] == "running" else 0),
        "completed": result.get("completed", 0),
        "total": result.get("total", len(task["request"].get("clips", []))),
        "step": task["stage"],
        "message": task["message"],
        "results": result.get("results", []),
        "error": task["error"],
        "progress": task["progress"],
    }


async def recover_persistent_tasks():
    STORE.mark_interrupted_ingests()
    max_attempts = int(STORE.get_agent_configs()["Supervisor"].get("max_attempts", 2))
    for task in STORE.recover_render_tasks():
        if task["attempts"] > max_attempts:
            STORE.update_task(task["id"], status="failed", stage="Restart recovery",
                              message="Render recovery retry limit reached.",
                              error="The task exceeded the automatic restart retry limit.")
            STORE.record_agent_event("Supervisor", "failed", "Render recovery retry limit reached",
                                     task_id=task["id"])
            continue
        clips = [RenderRequest.model_validate(item) for item in task["request"].get("clips", [])]
        if clips:
            _schedule_render_worker(task["id"], clips)


@app.get("/api/tasks")
async def task_history():
    return {"tasks": STORE.list_tasks(50)}


@app.get("/api/tasks/{task_id}")
async def get_task(task_id: str):
    task = STORE.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found.")
    return task


@app.post("/api/tasks/{task_id}/retry")
async def retry_task(task_id: str, req: RetryRequest | None = None):
    task = STORE.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found.")
    if task["status"] not in {"failed", "interrupted"}:
        raise HTTPException(status_code=409, detail="Only failed or interrupted tasks can be retried.")
    max_attempts = int(STORE.get_agent_configs()["Supervisor"].get("max_attempts", 2))
    if task["attempts"] >= max_attempts:
        raise HTTPException(status_code=409, detail="This task reached the retry limit.")
    if task["kind"] == "ingest":
        api_keys = (req.api_key or "").strip() if req else ""
        api_keys = api_keys or common.default_api_keys()
        if not api_keys:
            raise HTTPException(
                status_code=400,
                detail="Enter an API key in the dashboard to retry this ingest.",
            )
        STORE.update_task(task_id, attempts=task["attempts"] + 1)
        return await ingest_video(IngestRequest(**task["request"], api_key=api_keys))
    if task["kind"] != "render_batch":
        raise HTTPException(status_code=400, detail="This task type cannot be retried automatically.")
    clips = [RenderRequest.model_validate(item) for item in task["request"].get("clips", [])]
    if not clips:
        raise HTTPException(status_code=400, detail="The saved task has no render requests.")
    STORE.update_task(task_id, status="queued", progress=0, stage="Retry queued",
                      message="Retry queued", error=None,
                      attempts=task["attempts"] + 1, result_json={"results": [], "completed": 0})
    _schedule_render_worker(task_id, clips)
    return {"task_id": task_id, "status": "queued"}


@app.get("/api/dashboard")
async def dashboard_state():
    return {
        "agents": STORE.get_agent_configs(),
        "events": STORE.list_agent_events(50),
        "tasks": STORE.list_tasks(20),
        "trends": STORE.list_trends(10),
    }


@app.post("/api/agents/{agent_name}/command")
async def agent_command(agent_name: str, req: AgentCommandRequest):
    if agent_name not in AGENT_DEFAULTS:
        raise HTTPException(status_code=404, detail="Unknown agent.")
    command = req.command.strip()
    STORE.record_agent_event(agent_name, "received", f"{agent_name} command received",
                             command=command)
    try:
        if agent_name == "MoCap":
            match = re.search(r"\bsensitivity\s*[:=]?\s*(low|normal|high)\b", command, re.I)
            if not match:
                raise ValueError("Use 'sensitivity low', 'sensitivity normal', or 'sensitivity high'.")
            config = STORE.update_agent("MoCap", {"motion_sensitivity": match.group(1).lower()})
            result = {"agent": agent_name, "config": config}
        elif agent_name == "ShoCap":
            match = re.fullmatch(r"\s*(approve|reject)\s+(?:clip\s*)?#?(\d+)(?:\s+(.+))?\s*",
                                 command, re.I)
            if not match or not req.video_id:
                raise ValueError("Use 'approve 2' or 'reject 2' for the current video.")
            action, clip_id, notes = match.groups()
            known_ids = {clip.get("id") for clip in req.clips}
            if known_ids and int(clip_id) not in known_ids:
                raise ValueError(f"Clip {clip_id} is not in the current preview list.")
            status = "approved" if action.lower() == "approve" else "rejected"
            STORE.save_clip_review(req.video_id, int(clip_id), status, notes or "")
            result = {"agent": agent_name, "clip_id": int(clip_id), "status": status}
        elif agent_name == "Editor":
            result = COMMAND_AGENT.parse(command, req.clips,
                                         req.api_key or common.default_api_keys(),
                                         set(_style_presets()))
            config_updates = {}
            if result.get("caption_style"):
                config_updates["caption_style"] = result["caption_style"]
                STORE.update_agent("Editor", config_updates)
            result = {"agent": agent_name, **result}
        elif agent_name == "Upscaler":
            if not re.search(r"\b(upscale|render|4k)\b", command, re.I):
                raise ValueError("Use 'upscale' or 'render 4k' to start 4K rendering.")
            if not req.render_requests:
                raise ValueError("No clips were supplied for upscaling.")
            job_id = _create_render_task(req.render_requests[:10])
            _schedule_render_worker(job_id, req.render_requests[:10])
            result = {"agent": agent_name, "task_id": job_id, "status": "queued", "target": "4k"}
        else:
            match = re.fullmatch(r"\s*retry\s+([a-f0-9]{32})\s*", command, re.I)
            if command.lower() == "status":
                return {"agent": agent_name, "tasks": STORE.list_tasks(20)}
            if not match:
                raise ValueError("Use 'retry <task-id>' to resume a failed render.")
            result = await retry_task(match.group(1), RetryRequest(api_key=req.api_key))
            result["agent"] = agent_name
        STORE.record_agent_event(agent_name, "completed", "Command applied", command=command,
                                 result=result, progress=100)
        return result
    except ValueError as exc:
        STORE.record_agent_event(agent_name, "failed", str(exc), command=command)
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/styles")
async def get_styles():
    return {"styles": _style_presets()}


@app.post("/api/agent-edit")
async def agent_edit(request: Request):
    data = await request.json()
    return {"message": f"Agent processing: '{data.get('prompt')}' for clip {data.get('clip_id')}."}


def _reference_frame_from_url(url: str) -> tuple[bytes, str]:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    allowed = ("youtube.com", "youtu.be", "tiktok.com", "instagram.com", "twitch.tv")
    if parsed.scheme not in {"http", "https"} or not any(host == domain or host.endswith("." + domain)
                                                          for domain in allowed):
        raise ValueError("Reference URL must be a public YouTube, TikTok, Instagram, or Twitch URL.")
    with tempfile.TemporaryDirectory(prefix="clipforge_ref_") as temp_dir:
        source = Path(temp_dir) / "reference.%(ext)s"
        subprocess.run(["yt-dlp", "--quiet", "--no-warnings", "--download-sections", "*0-5",
                        "--force-keyframes-at-cuts", "-f", "bv*[height<=720]/b",
                        "-o", str(source), url], check=True, timeout=90)
        videos = list(Path(temp_dir).glob("reference.*"))
        if not videos:
            raise RuntimeError("Could not download a reference frame.")
        frame_path = Path(temp_dir) / "frame.jpg"
        subprocess.run(["ffmpeg", "-v", "error", "-ss", "1", "-i", str(videos[0]),
                        "-frames:v", "1", "-vf", "scale=640:-2", str(frame_path)],
                       check=True, timeout=30)
        return frame_path.read_bytes(), "image/jpeg"


@app.post("/api/analyze-reference")
async def analyze_reference(request: Request):
    content_type = request.headers.get("content-type", "")
    upload = None
    url = ""
    api_key = ""
    preset_name = ""
    if "multipart/form-data" in content_type:
        form = await request.form()
        upload = form.get("file")
        url = str(form.get("url", "")).strip()
        api_key = str(form.get("api_key", "")).strip()
        preset_name = str(form.get("name", "")).strip()
    else:
        data = await request.json()
        url = str(data.get("url", "")).strip()
        api_key = str(data.get("api_key", "")).strip()
        preset_name = str(data.get("name", "")).strip()
    if not api_key:
        api_key = common.default_api_keys()
    gemini_key = next((key.strip() for key in api_key.split(",")
                       if key.strip().startswith(("AIza", "AQ."))), "")
    if not gemini_key:
        raise HTTPException(status_code=400, detail="Reference cloning requires a Google Gemini API key.")
    try:
        if upload is not None and getattr(upload, "filename", ""):
            payload = await upload.read(30 * 1024 * 1024 + 1)
            if len(payload) > 30 * 1024 * 1024:
                raise ValueError("Reference uploads must be 30 MB or smaller.")
            content_type = getattr(upload, "content_type", "") or ""
            if content_type.startswith("image/"):
                image, mime_type = payload, content_type
            elif content_type.startswith("video/"):
                suffix = Path(upload.filename).suffix or ".mp4"
                with tempfile.TemporaryDirectory(prefix="clipforge_upload_") as temp_dir:
                    video_path = Path(temp_dir) / f"reference{suffix}"
                    frame_path = Path(temp_dir) / "frame.jpg"
                    video_path.write_bytes(payload)
                    subprocess.run(["ffmpeg", "-v", "error", "-ss", "1", "-i", str(video_path),
                                    "-frames:v", "1", "-vf", "scale=640:-2", str(frame_path)],
                                   check=True, timeout=30)
                    image, mime_type = frame_path.read_bytes(), "image/jpeg"
            else:
                raise ValueError("Upload an image or video file.")
        elif url:
            image, mime_type = _reference_frame_from_url(url)
        else:
            raise ValueError("Upload a reference image/video or provide a supported URL.")

        style = ReferenceClonerAgent.analyze_image(image, mime_type, gemini_key)
        stem = re.sub(r"[^a-z0-9_-]+", "_", preset_name.lower()).strip("_")[:48]
        if not stem:
            stem = f"reference_{uuid.uuid4().hex[:8]}"
        STORE.save_style(stem, style)
        return {"preset": stem, "style": style}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


PLUGINS_DIR = BASE_DIR / "plugins"
PLUGINS_DIR.mkdir(exist_ok=True)


def load_plugins():
    for plugin in PLUGINS_DIR.glob("*.py"):
        print(f"[plugins] Loading {plugin.name}...")


load_plugins()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
