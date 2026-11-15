# ClipForge

ClipForge analyzes a YouTube or Twitch VOD, ranks up to 10 non-overlapping short clips, and renders 4K MP4s with timestamped captions.

## Setup (Windows, PowerShell)

1. Install Python 3.12 and FFmpeg.
2. Create and activate a virtual environment, then install dependencies:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```
3. Optional: configure comma-separated provider keys in `$env:CLIPFORGE_API_KEYS` (Gemini `AIza...` or `AQ....`, Groq `gsk_...`, OpenRouter `sk-or-...`). Alternatively, put `CLIPFORGE_API_KEYS=...` in a project `.env` file; it is ignored by Git and used when the dashboard key field is blank. Keys may also be entered in the dashboard. Provider availability, model support, and free quotas can change; failover is best-effort, not a quota guarantee.
4. Optional: put trending topics in a text file and pass it with `--trends`.

## Run the dashboard

```powershell
python app.py
```

Open `http://127.0.0.1:8000`. Enter one or more comma-separated API keys, choose the clip count, and analyze a supported VOD. `TEST` runs sample scoring immediately without contacting an AI provider.

The dashboard offers:

- Gemini, Groq, and OpenRouter model/key failover for clip scoring.
- Local Whisper word timestamps when captions are unavailable.
- Sequential batch rendering (up to 10 clips), with status polling.
- Style presets persisted in SQLite (the initial presets are imported from `styles/`), including reference-style analysis from an image/video upload or supported video URL.
  Reference URL analysis samples scene cuts from the first 30 seconds; if that is unavailable, it falls back to
  subtitle-cue spacing. Rendering applies the selected preset's typography, colors, zoom intensity, and reference
  pacing to the generated captions and zoom pulses.
- A persistent five-agent control room: MoCap detects motion/chat candidates and accepts sensitivity commands;
  ShoCap presents extracted previews and saves approve/reject decisions; Editor handles style and recut commands;
  Upscaler queues 4K renders; Supervisor reports task status and retries failed work.

## Persistent processing and media lifecycle

- SQLite stores style presets, trend data, agent configuration, review decisions, agent events, and task history;
  video/audio files are never stored in the database. The default database is `clipforge.sqlite3` beside `app.py`.
  Set `CLIPFORGE_DB_PATH` to choose another path. SQLite WAL/SHM files are ignored by Git.
- Ingest analyzes five sequential 20% URL ranges using the lowest available video quality. MoCap analyzes each
  downloaded range, persists its results, then deletes the temporary video before downloading the next range.
  Selected timestamps are handed to ShoCap, which downloads only the 360p-or-lower preview ranges. Previews and
  per-chunk analysis results remain under `work/` for review and rerendering; generated 4K outputs are stored
  under `output/`.
- Queued/running render batches are recovered after an application restart. Interrupted ingests remain in task history
  and can be retried from the dashboard with an API key; API keys are not persisted.
- 4K output uses FFmpeg Lanczos scaling (2160×3840 portrait or 3840×2160 landscape); this increases output
  dimensions but does not reconstruct detail absent from the low-resolution source.

For command-line ingestion:

```powershell
python youtube_ingest.py "https://www.youtube.com/watch?v=VIDEO_ID" --target-count 10
python twitch_ingest.py "https://www.twitch.tv/videos/1234567890" --target-count 10
```

Useful options include `--min-score 55`, `--no-chat`, YouTube-only `--no-audio`, `--workdir work`, and `--api-keys "KEY1,KEY2"`.

## Notes

- Rendering uses FFmpeg with two encoding threads and the `veryfast` preset to limit memory use; render time and achievable daily volume depend on source resolution, clip duration, CPU, and provider limits.
- Clip scores are model estimates, not guaranteed engagement. The local weighted score uses the factors in `common.py`.
- Reference cloning and live AI scoring require valid provider credentials. TEST mode is local and does not render a source video.
