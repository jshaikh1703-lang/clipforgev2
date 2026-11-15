from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


AGENT_DEFAULTS = {
    "MoCap": {"enabled": True, "motion_sensitivity": "normal"},
    "ShoCap": {"enabled": True, "preview_enabled": True},
    "Editor": {"enabled": True, "caption_style": "hormozi_pop"},
    "Upscaler": {"enabled": True, "target": "4k", "algorithm": "lanczos"},
    "Supervisor": {"enabled": True, "max_attempts": 2},
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ClipForgeStore:
    def __init__(self, path: Path):
        self.path = path

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    @contextmanager
    def _connection(self):
        connection = self._connect()
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def initialize(self, styles_dir: Path) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS style_presets (
                    name TEXT PRIMARY KEY,
                    data_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS trend_data (
                    query TEXT PRIMARY KEY,
                    data_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS agent_configs (
                    name TEXT PRIMARY KEY,
                    config_json TEXT NOT NULL,
                    instructions TEXT NOT NULL DEFAULT '',
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS task_history (
                    id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    status TEXT NOT NULL,
                    progress REAL NOT NULL DEFAULT 0,
                    stage TEXT NOT NULL DEFAULT '',
                    message TEXT NOT NULL DEFAULT '',
                    request_json TEXT NOT NULL DEFAULT '{}',
                    result_json TEXT,
                    error TEXT,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS agent_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT REFERENCES task_history(id) ON DELETE CASCADE,
                    agent TEXT NOT NULL,
                    status TEXT NOT NULL,
                    progress REAL NOT NULL DEFAULT 0,
                    message TEXT NOT NULL DEFAULT '',
                    command TEXT,
                    result_json TEXT,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS agent_events_task_idx ON agent_events(task_id, id);
                CREATE TABLE IF NOT EXISTS clip_reviews (
                    video_id TEXT NOT NULL,
                    clip_id INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    notes TEXT NOT NULL DEFAULT '',
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (video_id, clip_id)
                );
            """)

            for name, config in AGENT_DEFAULTS.items():
                connection.execute(
                    "INSERT OR IGNORE INTO agent_configs(name, config_json, updated_at) VALUES (?, ?, ?)",
                    (name, _json(config), _now()),
                )
            for path in sorted(styles_dir.glob("*.json")):
                try:
                    style = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as exc:
                    print(f"[db] Could not import style {path.name}: {exc}")
                    continue
                if isinstance(style, dict):
                    connection.execute(
                        "INSERT OR IGNORE INTO style_presets(name, data_json, updated_at) VALUES (?, ?, ?)",
                        (path.stem, _json(style), _now()),
                    )

    def get_styles(self) -> dict[str, dict[str, Any]]:
        with self._connection() as connection:
            rows = connection.execute("SELECT name, data_json FROM style_presets ORDER BY name").fetchall()
        return {row["name"]: json.loads(row["data_json"]) for row in rows}

    def save_style(self, name: str, data: dict[str, Any]) -> None:
        with self._connection() as connection:
            connection.execute(
                """INSERT INTO style_presets(name, data_json, updated_at) VALUES (?, ?, ?)
                   ON CONFLICT(name) DO UPDATE SET data_json=excluded.data_json,
                       updated_at=excluded.updated_at""",
                (name, _json(data), _now()),
            )

    def get_agent_configs(self) -> dict[str, dict[str, Any]]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT name, config_json, instructions FROM agent_configs ORDER BY name"
            ).fetchall()
        return {row["name"]: {
            **json.loads(row["config_json"]), "instructions": row["instructions"],
        } for row in rows}

    def update_agent(self, name: str, config: dict[str, Any] | None = None,
                     instructions: str | None = None) -> dict[str, Any]:
        if name not in AGENT_DEFAULTS:
            raise ValueError(f"Unknown agent: {name}")
        with self._connection() as connection:
            row = connection.execute(
                "SELECT config_json, instructions FROM agent_configs WHERE name=?", (name,)
            ).fetchone()
            current = json.loads(row["config_json"]) if row else dict(AGENT_DEFAULTS[name])
            current.update(config or {})
            saved_instructions = instructions if instructions is not None else (
                row["instructions"] if row else "")
            connection.execute(
                """INSERT INTO agent_configs(name, config_json, instructions, updated_at)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(name) DO UPDATE SET config_json=excluded.config_json,
                       instructions=excluded.instructions, updated_at=excluded.updated_at""",
                (name, _json(current), saved_instructions, _now()),
            )
        return {**current, "instructions": saved_instructions}

    def save_trends(self, query: str, data: dict[str, Any]) -> None:
        if not query.strip():
            return
        with self._connection() as connection:
            connection.execute(
                """INSERT INTO trend_data(query, data_json, updated_at) VALUES (?, ?, ?)
                   ON CONFLICT(query) DO UPDATE SET data_json=excluded.data_json,
                       updated_at=excluded.updated_at""",
                (query.strip(), _json(data), _now()),
            )

    def list_trends(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT query, data_json, updated_at FROM trend_data ORDER BY updated_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [{"query": row["query"], "data": json.loads(row["data_json"]),
                 "updated_at": row["updated_at"]} for row in rows]

    def create_task(self, task_id: str, kind: str, request: dict[str, Any],
                    status: str = "queued", message: str = "Queued") -> None:
        now = _now()
        with self._connection() as connection:
            connection.execute(
                """INSERT INTO task_history
                   (id, kind, status, message, request_json, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (task_id, kind, status, message, _json(request), now, now),
            )

    def update_task(self, task_id: str, **fields: Any) -> None:
        allowed = {"status", "progress", "stage", "message", "result_json", "error", "attempts"}
        updates = {key: value for key, value in fields.items() if key in allowed}
        if not updates:
            return
        if "result_json" in updates and not isinstance(updates["result_json"], str):
            updates["result_json"] = _json(updates["result_json"])
        updates["updated_at"] = _now()
        assignments = ", ".join(f"{key}=?" for key in updates)
        with self._connection() as connection:
            cursor = connection.execute(
                f"UPDATE task_history SET {assignments} WHERE id=?",
                (*updates.values(), task_id),
            )
            if cursor.rowcount == 0:
                raise KeyError(f"Task not found: {task_id}")

    @staticmethod
    def _task(row: sqlite3.Row) -> dict[str, Any]:
        task = dict(row)
        task["request"] = json.loads(task.pop("request_json"))
        raw_result = task.pop("result_json")
        task["result"] = json.loads(raw_result) if raw_result else None
        return task

    def get_task(self, task_id: str) -> dict[str, Any] | None:
        with self._connection() as connection:
            row = connection.execute("SELECT * FROM task_history WHERE id=?", (task_id,)).fetchone()
        return self._task(row) if row else None

    def list_tasks(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT * FROM task_history ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [self._task(row) for row in rows]

    def recover_render_tasks(self) -> list[dict[str, Any]]:
        with self._connection() as connection:
            rows = connection.execute(
                """SELECT * FROM task_history WHERE kind='render_batch'
                   AND status IN ('queued', 'running') ORDER BY created_at"""
            ).fetchall()
            for row in rows:
                connection.execute(
                    """UPDATE task_history SET status='queued', stage='Restart recovery',
                       message='Resuming queued render after restart', attempts=attempts+1,
                       updated_at=? WHERE id=?""",
                    (_now(), row["id"]),
                )
            recovered = [
                connection.execute("SELECT * FROM task_history WHERE id=?", (row["id"],)).fetchone()
                for row in rows
            ]
        return [self._task(row) for row in recovered if row]

    def mark_interrupted_ingests(self) -> None:
        with self._connection() as connection:
            connection.execute(
                """UPDATE task_history SET status='interrupted', stage='Restart recovery',
                   message='Ingest was interrupted by a restart; submit it again to continue.',
                   updated_at=? WHERE kind='ingest' AND status='running'""",
                (_now(),),
            )

    def record_agent_event(self, agent: str, status: str, message: str,
                           task_id: str | None = None, progress: float = 0,
                           command: str | None = None, result: Any = None) -> None:
        with self._connection() as connection:
            connection.execute(
                """INSERT INTO agent_events
                   (task_id, agent, status, progress, message, command, result_json, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (task_id, agent, status, progress, message, command,
                 _json(result) if result is not None else None, _now()),
            )

    def list_agent_events(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT * FROM agent_events ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        events = [dict(row) for row in rows]
        for event in events:
            result_json = event.pop("result_json")
            event["result"] = json.loads(result_json) if result_json else None
        return events

    def save_clip_review(self, video_id: str, clip_id: int, status: str, notes: str = "") -> None:
        with self._connection() as connection:
            connection.execute(
                """INSERT INTO clip_reviews(video_id, clip_id, status, notes, updated_at)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(video_id, clip_id) DO UPDATE SET status=excluded.status,
                       notes=excluded.notes, updated_at=excluded.updated_at""",
                (video_id, clip_id, status, notes, _now()),
            )

    def clip_reviews(self, video_id: str) -> dict[int, dict[str, str]]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT clip_id, status, notes FROM clip_reviews WHERE video_id=?", (video_id,)
            ).fetchall()
        return {row["clip_id"]: {"status": row["status"], "notes": row["notes"]} for row in rows}


def default_store_path() -> Path:
    configured = os.environ.get("CLIPFORGE_DB_PATH")
    return Path(configured).expanduser() if configured else Path(__file__).with_name("clipforge.sqlite3")


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)
