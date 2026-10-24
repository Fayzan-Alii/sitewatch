import json
import sqlite3
import time
import uuid
from pathlib import Path


class Store:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "sitewatch.db"
        with self.connect() as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS records (id TEXT PRIMARY KEY, kind TEXT NOT NULL, created REAL NOT NULL, data TEXT NOT NULL)"
            )
            db.execute(
                "CREATE INDEX IF NOT EXISTS records_kind ON records(kind, created)"
            )

    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.execute("PRAGMA journal_mode=WAL")
        return db

    def put(self, kind, data, id=None):
        id = id or uuid.uuid4().hex
        with self.connect() as db:
            db.execute(
                "INSERT INTO records VALUES (?,?,?,?) ON CONFLICT(id) DO UPDATE SET data=excluded.data",
                (id, kind, time.time(), json.dumps(data)),
            )
        return {**data, "id": id}

    def get(self, id):
        with self.connect() as db:
            row = db.execute("SELECT data FROM records WHERE id=?", (id,)).fetchone()
        return {**json.loads(row[0]), "id": id} if row else None

    def list(self, kind, limit=500):
        with self.connect() as db:
            rows = db.execute(
                "SELECT id,data FROM records WHERE kind=? ORDER BY created DESC LIMIT ?",
                (kind, limit),
            ).fetchall()
        return [{**json.loads(data), "id": id} for id, data in rows]
