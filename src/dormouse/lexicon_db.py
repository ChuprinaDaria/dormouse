"""SQLite лексикон — єдина база для нормалізації, компресії і маппінгу.

Одна таблиця, один lookup на слово: суржик → нормалізоване → EN.
B-Tree індекс, <1ms на запит. Масштабується до 50K+ записів.
"""

import json
import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS lexicon (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    word TEXT NOT NULL,
    normalized TEXT,
    en_compressed TEXT,
    type TEXT,
    ngram INTEGER DEFAULT 1,
    is_filler BOOLEAN DEFAULT 0,
    source TEXT
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_lexicon_word ON lexicon(word);
CREATE INDEX IF NOT EXISTS idx_lexicon_ngram ON lexicon(ngram);
"""


def get_lexicon(db_path: Path | None = None) -> sqlite3.Connection:
    """Повертає з'єднання з лексиконом."""
    if db_path is None:
        from dormouse.assets import get_asset
        path = get_asset("lexicon.db")
    else:
        path = db_path
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(SCHEMA)
    return conn


def lookup(conn: sqlite3.Connection, word: str) -> sqlite3.Row | None:
    """Шукає слово в лексиконі. O(log n) через B-Tree."""
    return conn.execute(
        "SELECT * FROM lexicon WHERE word = ?", (word.lower(),)
    ).fetchone()


def lookup_batch(conn: sqlite3.Connection, words: list[str]) -> dict[str, sqlite3.Row]:
    """Batch lookup — один запит на всі слова."""
    if not words:
        return {}
    placeholders = ",".join("?" * len(words))
    rows = conn.execute(
        f"SELECT * FROM lexicon WHERE word IN ({placeholders})",
        [w.lower() for w in words],
    ).fetchall()
    return {row["word"]: row for row in rows}


def get_expressions(conn: sqlite3.Connection, ngram: int) -> list[sqlite3.Row]:
    """Повертає всі вирази певної довжини (для multi-word matching)."""
    return conn.execute(
        "SELECT * FROM lexicon WHERE ngram = ? ORDER BY length(word) DESC",
        (ngram,),
    ).fetchall()


def build_from_sources(
    replacements_path: Path,
    en_compressed: dict[str, str],
    db_path: Path | None = None,
):
    """Будує лексикон з replacements.json + EN словника.

    Все зливається в одну таблицю: word → normalized + en_compressed.
    Зберігає записи з інших джерел (opus, manual, ua-slang-mcp, corpus).
    """
    conn = get_lexicon(db_path)

    # Видаляємо ТІЛЬКИ записи з джерел що перебудовуються
    conn.execute("DELETE FROM lexicon WHERE source IN ('replacements', 'mapper')")

    # 1. Replacements.json → normalized
    with open(replacements_path, encoding="utf-8") as f:
        data = json.load(f)

    for r in data["replacements"]:
        word = r["from"].lower()
        normalized = r["to"]  # може бути None (filler)
        is_filler = 1 if normalized is None else 0
        ngram = r.get("ngram", len(word.split()))

        # Шукаємо EN еквівалент для нормалізованої форми
        en_c = en_compressed.get(word) or (en_compressed.get(normalized) if normalized else None)

        conn.execute(
            """INSERT OR REPLACE INTO lexicon
               (word, normalized, en_compressed, type, ngram, is_filler, source)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (word, normalized, en_c, r.get("type", ""), ngram, is_filler, "replacements"),
        )

    # 2. EN-only записи (слова яких нема в replacements, але є EN маппінг)
    all_words = {r["from"].lower() for r in data["replacements"]}

    for word in set(en_compressed) - all_words:
        en_c = en_compressed.get(word)
        ngram = len(word.split())
        # Перевіряємо чи запис вже є (наприклад з OPUS)
        existing = conn.execute(
            "SELECT id, en_compressed FROM lexicon WHERE word = ?",
            (word,),
        ).fetchone()
        if existing:
            # Оновлюємо EN маппінг якщо він NULL
            if en_c and not existing["en_compressed"]:
                conn.execute(
                    "UPDATE lexicon SET en_compressed = ? WHERE id = ?",
                    (en_c, existing["id"]),
                )
        else:
            conn.execute(
                """INSERT INTO lexicon
                   (word, normalized, en_compressed, type, ngram, is_filler, source)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (word, None, en_c, "en_only", ngram, 0, "mapper"),
            )

    # 3. Regex patterns зберігаємо окремо (JSON blob)
    # Їх мало, не варто в SQL
    conn.execute(
        "CREATE TABLE IF NOT EXISTS patterns "
        "(id INTEGER PRIMARY KEY, regex TEXT, replacement TEXT, type TEXT)"
    )
    conn.execute("DELETE FROM patterns")
    for p in data.get("patterns", []):
        conn.execute(
            "INSERT INTO patterns (regex, replacement, type) VALUES (?, ?, ?)",
            (p["regex"], p["to"], p.get("type", "")),
        )

    conn.commit()

    total = conn.execute("SELECT COUNT(*) FROM lexicon").fetchone()[0]
    patterns = conn.execute("SELECT COUNT(*) FROM patterns").fetchone()[0]
    conn.close()

    return total, patterns
