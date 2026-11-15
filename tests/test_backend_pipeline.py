import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import common
import app as app_module
from agents import (
    CaptionStylistAgent,
    CaptionWord,
    ShoCapAgent,
    StudioCommandAgent,
    UpscalerAgent,
    ViralReverseEngineerAgent,
    ZoomDirectorAgent,
)
from app import RenderRequest, _zoom_intensity, generate_ass, _run_render_batch
from database import ClipForgeStore


class CaptionStylistTests(unittest.TestCase):
    def test_chunks_split_at_long_silences(self):
        words = [
            CaptionWord(0.0, 0.2, "first", False),
            CaptionWord(0.3, 0.5, "phrase", False),
            CaptionWord(1.5, 1.8, "second", False),
        ]

        chunks = CaptionStylistAgent().chunks(words, size=3)

        self.assertEqual([[word.text for word in chunk] for chunk in chunks],
                         [["first", "phrase"], ["second"]])

    def test_ass_uses_horizontal_canvas_and_clip_relative_word_times(self):
        request = RenderRequest(
            clip_id=1, video_id="test", url="https://example.com/video",
            start="00:00:10", end="00:00:12", start_s=10, end_s=12,
            duration_s=2, framing="horizontal", font_size=0,
        )
        segments = [{
            "start": 10.2, "end": 11.0, "text": "first phrase",
            "words": [
                {"start": 10.2, "end": 10.4, "text": "first"},
                {"start": 10.7, "end": 11.0, "text": "phrase"},
            ],
        }]

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "clip.ass"
            generate_ass(path, request, segments, {"font_name": "Arial", "zoom_intensity": 0.025},
                         horizontal=True)
            output = path.read_text(encoding="utf-8")

        self.assertIn("PlayResX: 1920\nPlayResY: 1080", output)
        self.assertIn("Style: Default,Arial,48,", output)
        self.assertIn("Dialogue: 0,0:00:00.20,0:00:01.00", output)
        self.assertIn(r"{\kf50}FIRST {\kf30}PHRASE", output)


class ReferencePacingTests(unittest.TestCase):
    def test_selected_reference_style_controls_zoom_strength(self):
        self.assertAlmostEqual(_zoom_intensity("subtle", {"zoom_intensity": 0.035}), 0.168)
        self.assertEqual(_zoom_intensity("off", {"zoom_intensity": 0.08}), 0.0)

    def test_reference_rhythm_changes_zoom_pulse_spacing(self):
        agent = ZoomDirectorAgent()
        segments = [
            {"start": 1.5, "text": "wow"},
            {"start": 3.2, "text": "no way"},
        ]

        default = agent.keyframes(0, 8, segments, "", 0.12)
        reference = agent.keyframes(0, 8, segments, "", 0.12, rhythm_s=1.2)

        self.assertEqual(len(default), 2)
        self.assertEqual(len(reference), 3)
        self.assertLessEqual(reference[0]["end_s"] - reference[0]["start_s"], 1.2)

    def test_visual_cut_timestamps_are_read_from_ffmpeg_output(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            def fake_run(command, **kwargs):
                if command[0] == "yt-dlp":
                    Path(temp_dir, "visual_reference.mp4").write_bytes(b"video")
                    return None
                return type("Result", (), {"stderr": "pts_time:1.250\npts_time:4.500\n"})()

            with patch("agents.subprocess.run", side_effect=fake_run):
                times = ViralReverseEngineerAgent._visual_cut_times(
                    "https://example.com/reference", 10, temp_dir)

        self.assertEqual(times, [1.25, 4.5])


class PersistentPipelineTests(unittest.TestCase):
    def test_sqlite_imports_styles_and_recovers_render_tasks(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            styles = root / "styles"
            styles.mkdir()
            (styles / "custom.json").write_text(json.dumps({"font_name": "Arial"}), encoding="utf-8")
            store = ClipForgeStore(root / "clipforge.sqlite3")

            store.initialize(styles)
            store.save_style("custom", {"font_name": "Impact"})
            store.save_trends("game", {"topics": ["game"]})
            store.create_task("render-1", "render_batch", {"clips": []}, status="running")
            store.record_agent_event("Editor", "completed", "Rendered", task_id="render-1",
                                     result={"clip_id": 1})

            recovered = store.recover_render_tasks()

            self.assertEqual(store.get_styles()["custom"]["font_name"], "Impact")
            self.assertEqual(store.list_trends()[0]["data"]["topics"], ["game"])
            self.assertEqual(recovered[0]["status"], "queued")
            self.assertEqual(recovered[0]["attempts"], 1)
            self.assertEqual(store.list_agent_events()[0]["agent"], "Editor")
            self.assertEqual(store.list_agent_events()[0]["result"], {"clip_id": 1})

    def test_download_source_requests_360p_and_returns_downloaded_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            captured = {}
            downloaded_urls = []

            class FakeDownloader:
                def __init__(self, options):
                    captured.update(options)

                def __enter__(self):
                    return self

                def __exit__(self, *_):
                    return False

                def download(self, urls):
                    downloaded_urls.extend(urls)
                    Path(captured["outtmpl"].replace("%(ext)s", "mp4")).write_bytes(b"source")

            with patch.dict(sys.modules, {"yt_dlp": types.SimpleNamespace(YoutubeDL=FakeDownloader)}):
                result = common.download_source_video("https://example.com/vod", root)

            self.assertEqual(captured["format"], "bv*[height<=360]+ba/b[height<=360]")
            self.assertEqual(downloaded_urls, ["https://example.com/vod"])
            self.assertTrue(result.is_file())

    def test_source_is_deleted_only_after_successful_extraction(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.mp4"
            clip = root / "clip.mp4"
            source.write_bytes(b"source")
            clip.write_bytes(b"clip")

            with patch("common.extract_clip_slices", return_value=[clip]):
                result = common.extract_clip_slices_and_cleanup_source(source, root, [])

            self.assertEqual(result, [clip])
            self.assertFalse(source.exists())
            self.assertTrue(clip.exists())

            source.write_bytes(b"source")
            with patch("common.extract_clip_slices", side_effect=RuntimeError("ffmpeg failed")):
                with self.assertRaisesRegex(RuntimeError, "ffmpeg failed"):
                    common.extract_clip_slices_and_cleanup_source(source, root, [])
            self.assertTrue(source.exists())

    def test_video_chunks_are_saved_and_deleted_before_next_chunk(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.mp4"
            source.write_bytes(b"source")
            analyzed = []
            progress = []

            def fake_ffmpeg(command, **_kwargs):
                Path(command[-1]).write_bytes(b"chunk")

            def analyze_chunk(video_path, chunk_work, index, start_s, end_s):
                self.assertTrue(video_path.is_file())
                if analyzed:
                    self.assertFalse(analyzed[-1].is_file())
                    self.assertTrue((chunk_work.parent.parent / "chunk_results.json").is_file())
                analyzed.append(video_path)
                return [{"id": index + 1, "start_s": start_s, "end_s": end_s}]

            with patch("common.subprocess.run", side_effect=fake_ffmpeg):
                results = common.analyze_video_in_chunks(
                    source, root, 100, analyze_chunk,
                    lambda message, percent: progress.append((message, percent)))

            self.assertEqual(len(results), 5)
            self.assertEqual([(clip["start_s"], clip["end_s"]) for clip in results],
                             [(0, 20), (20, 40), (40, 60), (60, 80), (80, 100)])
            self.assertTrue(all(not path.exists() for path in analyzed))
            self.assertEqual(json.loads((root / "chunk_results.json").read_text()), results)
            self.assertEqual([percent for _, percent in progress],
                             [70, 74, 74, 78, 78, 82, 82, 86, 86, 90])

    def test_video_chunk_is_deleted_when_analysis_fails(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.mp4"
            source.write_bytes(b"source")

            def fake_ffmpeg(command, **_kwargs):
                Path(command[-1]).write_bytes(b"chunk")

            def fail_analysis(video_path, *_args):
                self.assertTrue(video_path.is_file())
                raise RuntimeError("analysis failed")

            with patch("common.subprocess.run", side_effect=fake_ffmpeg):
                with self.assertRaisesRegex(RuntimeError, "analysis failed"):
                    common.analyze_video_in_chunks(source, root, 100, fail_analysis)

            self.assertFalse((root / "analysis_chunks" / "chunk_01" /
                              "analysis_chunk.mp4").exists())

    def test_video_url_chunks_download_lowest_quality_and_delete_after_analysis(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            downloads = []

            def fake_download(source_url, chunk_work, start_s, end_s, output_stem,
                              format_selector, recode_video=False):
                if downloads:
                    self.assertFalse(downloads[-1].exists())
                path = chunk_work / f"{output_stem}.webm"
                path.write_bytes(b"chunk")
                downloads.append(path)
                self.assertEqual(source_url, "https://example.com/vod")
                self.assertEqual(format_selector, "worstvideo/worst")
                self.assertFalse(recode_video)
                return path

            def analyze_chunk(video_path, _chunk_work, index, start_s, end_s):
                self.assertTrue(video_path.is_file())
                return [{"id": index + 1, "start_s": start_s, "end_s": end_s}]

            with patch("common.download_video_range", side_effect=fake_download):
                results = common.analyze_video_url_in_chunks(
                    "https://example.com/vod", root, 100, analyze_chunk)

            self.assertEqual([(clip["start_s"], clip["end_s"]) for clip in results],
                             [(0, 20), (20, 40), (40, 60), (60, 80), (80, 100)])
            self.assertEqual(len(downloads), 5)
            self.assertTrue(all(not path.exists() for path in downloads))

    def test_video_url_chunk_is_deleted_when_analysis_fails(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            downloaded = []

            def fake_download(_url, chunk_work, _start_s, _end_s, output_stem,
                              _format_selector, _recode_video=False):
                path = chunk_work / f"{output_stem}.webm"
                path.write_bytes(b"chunk")
                downloaded.append(path)
                return path

            def fail_analysis(video_path, *_args):
                self.assertTrue(video_path.is_file())
                raise RuntimeError("analysis failed")

            with patch("common.download_video_range", side_effect=fake_download):
                with self.assertRaisesRegex(RuntimeError, "analysis failed"):
                    common.analyze_video_url_in_chunks(
                        "https://example.com/vod", root, 100, fail_analysis)

            self.assertEqual(len(downloaded), 1)
            self.assertFalse(downloaded[0].exists())

    def test_download_video_range_requests_only_the_lowest_quality_section(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            captured = {}

            class FakeDownloader:
                def __init__(self, options):
                    captured["options"] = options

                def __enter__(self):
                    return self

                def __exit__(self, *_args):
                    return False

                def download(self, urls):
                    captured["urls"] = urls
                    captured["ranges"] = list(
                        captured["options"]["download_ranges"]({"duration": 100}, self))
                    Path(captured["options"]["outtmpl"].replace("%(ext)s", "webm")).write_bytes(
                        b"section")

            with patch("yt_dlp.YoutubeDL", FakeDownloader):
                downloaded = common.download_video_range(
                    "https://example.com/vod", root, 20, 40, "analysis_chunk",
                    "worstvideo/worst")

            self.assertEqual(downloaded.name, "analysis_chunk.webm")
            self.assertEqual(captured["urls"], ["https://example.com/vod"])
            self.assertEqual(captured["options"]["format"], "worstvideo/worst")
            self.assertEqual(captured["ranges"], [{"start_time": 20, "end_time": 40}])

    def test_video_range_removes_partial_output_when_download_fails(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)

            class FailingDownloader:
                def __init__(self, options):
                    self.options = options

                def __enter__(self):
                    return self

                def __exit__(self, *_args):
                    return False

                def download(self, _urls):
                    Path(self.options["outtmpl"].replace("%(ext)s", "webm")).write_bytes(
                        b"partial")
                    raise RuntimeError("download failed")

            with patch("yt_dlp.YoutubeDL", FailingDownloader):
                with self.assertRaisesRegex(RuntimeError, "download failed"):
                    common.download_video_range(
                        "https://example.com/vod", root, 20, 40, "analysis_chunk",
                        "worstvideo/worst")

            self.assertEqual(list(root.glob("analysis_chunk.*")), [])

    def test_showcap_downloads_previews_from_candidate_timestamps(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            clips = [{"id": 2, "start_s": 12.5, "end_s": 42.0}]
            preview = root / "clip_2_12500_42000.mp4"

            with patch("common.download_video_range", return_value=preview) as download:
                previews = ShoCapAgent.prepare_previews(
                    "https://example.com/vod", root, clips)

            self.assertEqual(previews, [preview])
            download.assert_called_once_with(
                "https://example.com/vod", root, 12.5, 42.0, "clip_2_12500_42000",
                "bv*[height<=360]+ba/b[height<=360]", recode_video=True)

    def test_chunk_clip_merge_ranks_and_removes_overlapping_results(self):
        clips = [
            {"start_s": 0, "end_s": 30, "viral_probability": 70},
            {"start_s": 20, "end_s": 50, "viral_probability": 90},
            {"start_s": 50, "end_s": 80, "viral_probability": 80},
        ]

        merged = common.merge_chunk_clips(clips, "https://example.com/vod", 2)

        self.assertEqual([clip["viral_probability"] for clip in merged], [90, 80])
        self.assertEqual([clip["id"] for clip in merged], [1, 2])
        self.assertEqual(merged[0]["start"], "00:00:20")
        self.assertIn("--download-sections", merged[0]["download_cmd"])

    def test_transcript_chunk_slices_localize_segment_and_word_times(self):
        segment = common.Segment(9.5, "boundary phrase", 10.5, [
            {"start": 9.5, "end": 10.0, "text": "boundary"},
            {"start": 10.0, "end": 10.5, "text": "phrase"},
        ])

        sliced = common.slice_segments([segment], 10, 20)

        self.assertEqual(len(sliced), 1)
        self.assertEqual((sliced[0].start, sliced[0].end), (0, 0.5))
        self.assertEqual(sliced[0].words, [{"start": 0, "end": 0.5, "text": "phrase"}])

    def test_chunk_signal_timestamps_return_to_video_timeline(self):
        shifted = common.shift_signal_timestamps(
            ["mocap [00:00:12] motion_z=1.0 chat_z=0.0 z=1.0"], 108)

        self.assertEqual(shifted, ["mocap [00:02:00] motion_z=1.0 chat_z=0.0 z=1.0"])

    def test_upscaler_targets_4k_portrait_and_landscape_dimensions(self):
        self.assertEqual(UpscalerAgent.output_size("stacked"), (2160, 3840))
        self.assertEqual(UpscalerAgent.output_size("horizontal"), (3840, 2160))
        self.assertEqual(UpscalerAgent.scale_filter("horizontal"),
                         "scale=3840:2160:flags=lanczos")

    def test_editor_accepts_sqlite_backed_custom_caption_styles(self):
        response = {
            "filters": [], "updates": [], "zoom": None,
            "caption_style": "reference_custom", "auto_facecam": None, "rerender": False,
        }
        with patch("common.complete_json_with_ai_router", return_value=response):
            actions = StudioCommandAgent().parse(
                "apply reference custom style", [], available_styles={"reference_custom"})

        self.assertEqual(actions["caption_style"], "reference_custom")

    def test_recovered_render_batch_skips_persisted_completed_clips(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            styles = root / "styles"
            styles.mkdir()
            store = ClipForgeStore(root / "clipforge.sqlite3")
            store.initialize(styles)
            store.create_task("batch-1", "render_batch", {"clips": []}, status="running")
            store.update_task("batch-1", result_json={
                "results": [{"clip_id": 1, "output": "/output/clip-1.mp4"}],
                "completed": 1,
                "total": 2,
            })
            clips = [
                RenderRequest(clip_id=1, video_id="video", url="https://example.com/vod",
                              start="00:00:00", end="00:00:30", duration_s=30),
                RenderRequest(clip_id=2, video_id="video", url="https://example.com/vod",
                              start="00:00:30", end="00:01:00", duration_s=30),
            ]

            with patch.object(app_module, "STORE", store), \
                    patch.object(app_module, "_render_clip",
                                 return_value={"output": "/output/clip-2.mp4"}) as render:
                _run_render_batch("batch-1", clips)

            render.assert_called_once()
            self.assertEqual(render.call_args.args[0].clip_id, 2)
            task = store.get_task("batch-1")
            self.assertEqual(task["status"], "completed")
            self.assertEqual([item["clip_id"] for item in task["result"]["results"]], [1, 2])


if __name__ == "__main__":
    unittest.main()
