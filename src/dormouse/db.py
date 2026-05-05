"""Робота з SQLite базою даних dormouse."""

import sqlite3
from pathlib import Path

from dormouse.assets import _cache_dir

DB_DIR = _cache_dir()
DB_PATH = DB_DIR / "dormouse.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS raw_texts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,
    source_id TEXT,
    source_name TEXT,
    text TEXT NOT NULL,
    author_hash TEXT,
    collected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    corpus TEXT,
    word_count INTEGER,
    processed BOOLEAN DEFAULT FALSE
);

CREATE TABLE IF NOT EXISTS lexicon (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    word TEXT NOT NULL UNIQUE,
    type TEXT NOT NULL,
    meaning TEXT,
    example TEXT,
    normalized_form TEXT,
    first_seen DATE,
    last_seen DATE,
    frequency_30d INTEGER DEFAULT 0,
    status TEXT DEFAULT 'active',
    tags TEXT
);

CREATE TABLE IF NOT EXISTS pairs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_text TEXT NOT NULL,
    optimized_text TEXT NOT NULL,
    style TEXT NOT NULL,
    method TEXT NOT NULL,
    tokens_source_claude INTEGER,
    tokens_optimized_claude INTEGER,
    tokens_source_gpt4 INTEGER,
    tokens_optimized_gpt4 INTEGER,
    tokens_source_llama INTEGER,
    tokens_optimized_llama INTEGER,
    compression_ratio REAL,
    validated BOOLEAN DEFAULT FALSE,
    validator TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS normal_texts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,
    text TEXT NOT NULL,
    category TEXT,
    word_count INTEGER,
    tokens_claude INTEGER,
    tokens_gpt4 INTEGER,
    imported_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

# Категорії каналів → corpus
CORPUS_MAP = {
    "суржик": "розмовний",
    "коментарі": "розмовний",
    "особисті": "розмовний",
    "новини": "еталон",
    "поп-наука": "еталон",
}


def get_connection(db_path: Path | None = None) -> sqlite3.Connection:
    """Повертає з'єднання з БД, створює таблиці якщо потрібно."""
    path = db_path or DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(SCHEMA)
    _migrate(conn)
    return conn


def _migrate(conn: sqlite3.Connection):
    """Міграції для існуючих БД."""
    columns = {
        row[1] for row in conn.execute("PRAGMA table_info(raw_texts)").fetchall()
    }
    if "corpus" not in columns:
        conn.execute("ALTER TABLE raw_texts ADD COLUMN corpus TEXT")
        # Заповнюємо corpus з source для існуючих записів
        for category, corpus in CORPUS_MAP.items():
            conn.execute(
                "UPDATE raw_texts SET corpus = ? WHERE source = ?",
                (corpus, f"telegram:{category}"),
            )
        conn.commit()
