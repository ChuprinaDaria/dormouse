"""Full-text search по нормалізованому + лематизованому тексту через SQLite FTS5.

Пошук "помилка" знаходить тексти з "баг", "баґ", "баги", "помилці", "помилок" —
бо при індексації: rule_engine нормалізує + pymorphy3 лематизує.
"""

import sqlite3
from pathlib import Path

from dormouse.assets import _cache_dir
from dormouse.morphology import lemmatize_text
from dormouse.rule_engine import crack

SEARCH_DB = _cache_dir() / "search.db"

SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS search_index USING fts5(
    normalized_text,
    original_text UNINDEXED,
    source UNINDEXED,
    source_id UNINDEXED,
    tokenize='unicode61'
);
"""


def get_search_db(db_path: Path | None = None) -> sqlite3.Connection:
    """Повертає з'єднання з пошуковою БД."""
    path = db_path or SEARCH_DB
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(SCHEMA)
    return conn


def _normalize_for_index(text: str) -> str:
    """Нормалізація + лематизація для індексу."""
    return lemmatize_text(crack(text))


def _normalize_query(query: str) -> str:
    """Нормалізація + лематизація пошукового запиту."""
    return lemmatize_text(crack(query))


def index_text(
    conn: sqlite3.Connection,
    text: str,
    source: str = "",
    source_id: str = "",
) -> str:
    """Індексує текст: нормалізує і зберігає в FTS5.

    Returns:
        Нормалізований текст.
    """
    normalized = _normalize_for_index(text)
    conn.execute(
        "INSERT INTO search_index (normalized_text, original_text, source, source_id) "
        "VALUES (?, ?, ?, ?)",
        (normalized, text, source, source_id),
    )
    return normalized


def index_batch(
    conn: sqlite3.Connection,
    texts: list[dict],
    commit: bool = True,
) -> int:
    """Батч-індексація через executemany.

    Args:
        texts: [{"text": "...", "source": "...", "source_id": "..."}, ...]
        commit: Автоматичний commit після батчу.

    Returns:
        Кількість проіндексованих текстів.
    """
    rows = []
    for item in texts:
        text = item.get("text", "")
        if not text or not text.strip():
            continue
        normalized = _normalize_for_index(text)
        rows.append((normalized, text, item.get("source", ""), item.get("source_id", "")))

    if rows:
        conn.executemany(
            "INSERT INTO search_index (normalized_text, original_text, source, source_id) "
            "VALUES (?, ?, ?, ?)",
            rows,
        )

    if commit:
        conn.commit()
    return len(rows)


def search(
    conn: sqlite3.Connection,
    query: str,
    limit: int = 20,
) -> list[dict]:
    """Шукає тексти по нормалізованому змісту.

    Запит нормалізується перед пошуком: "баг" → "помилка",
    тому знайде всі тексти де була будь-яка форма "баг/баги/баґ".

    Returns:
        [{"original": "...", "normalized": "...", "source": "...", "rank": float}, ...]
    """
    normalized_query = _normalize_query(query)

    # FTS5 MATCH query — кожне слово як окремий терм
    words = normalized_query.split()
    if not words:
        return []

    # Шукаємо кожне слово (OR для більшого охоплення)
    fts_query = " OR ".join(f'"{w}"' for w in words if len(w) > 1)
    if not fts_query:
        return []

    rows = conn.execute(
        "SELECT original_text, normalized_text, source, source_id, rank "
        "FROM search_index WHERE search_index MATCH ? "
        "ORDER BY rank LIMIT ?",
        (fts_query, limit),
    ).fetchall()

    return [
        {
            "original": row["original_text"],
            "normalized": row["normalized_text"],
            "source": row["source"],
            "source_id": row["source_id"],
            "rank": row["rank"],
        }
        for row in rows
    ]


def search_exact(
    conn: sqlite3.Connection,
    query: str,
    limit: int = 20,
) -> list[dict]:
    """Точний пошук — всі слова мають бути присутні (AND)."""
    normalized_query = _normalize_query(query)
    words = normalized_query.split()
    if not words:
        return []

    fts_query = " AND ".join(f'"{w}"' for w in words if len(w) > 1)
    if not fts_query:
        return []

    rows = conn.execute(
        "SELECT original_text, normalized_text, source, source_id, rank "
        "FROM search_index WHERE search_index MATCH ? "
        "ORDER BY rank LIMIT ?",
        (fts_query, limit),
    ).fetchall()

    return [
        {
            "original": row["original_text"],
            "normalized": row["normalized_text"],
            "source": row["source"],
            "source_id": row["source_id"],
            "rank": row["rank"],
        }
        for row in rows
    ]


def count(conn: sqlite3.Connection) -> int:
    """Кількість проіндексованих текстів."""
    return conn.execute(
        "SELECT COUNT(*) FROM search_index"
    ).fetchone()[0]
