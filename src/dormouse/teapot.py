"""Teapot — ядро семантичного пошуку dormouse.

stir()   — індексація текстів (FTS5 + embeddings)
mumble() — гібридний пошук (embeddings + FTS5 fallback)
sip()    — класифікація текстів по темам через embeddings
"""

import logging
import struct

from dormouse.parsers import chunk_texts, parse_file
from dormouse.search import get_search_db, index_text
from dormouse.search import search as fts_search

logger = logging.getLogger("dormouse")

EMBEDDINGS_SCHEMA = """
CREATE TABLE IF NOT EXISTS embeddings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chunk_text TEXT NOT NULL,
    original_text TEXT NOT NULL,
    source TEXT DEFAULT '',
    embedding BLOB NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_embeddings_source ON embeddings(source);
"""


def _serialize_embedding(vector: list[float]) -> bytes:
    """Серіалізація вектора в BLOB."""
    return struct.pack(f"{len(vector)}f", *vector)


def _deserialize_embedding(data: bytes) -> list[float]:
    """Десеріалізація BLOB у вектор."""
    n = len(data) // 4
    return list(struct.unpack(f"{n}f", data))


def _normalize_text(text: str) -> str:
    """crack() + seq2seq (optional)."""
    from dormouse.rule_engine import crack

    normalized = crack(text)
    try:
        from dormouse.seq2seq import normalize as seq2seq_normalize

        normalized = seq2seq_normalize(normalized)
    except (ImportError, Exception):
        pass
    return normalized


def _resolve_source(source) -> list[str]:
    """Розпізнає джерело: list[str] або шлях до файлу."""
    if isinstance(source, list):
        return source
    # str або Path — спробуємо як файл
    return parse_file(source)


class Teapot:
    """Чайник семантичного пошуку.

    Args:
        db_path: Шлях до SQLite БД (None = default).
        embedder: Embedder інстанс (None = FTS5-only режим).
    """

    def __init__(self, db_path=None, embedder=None):
        self._db_path = db_path
        self._embedder = embedder
        self._conn = get_search_db(db_path)
        self._conn.executescript(EMBEDDINGS_SCHEMA)

    def stir(self, source, **kwargs) -> "Teapot":
        """Індексація текстів: парсинг -> chunking -> нормалізація -> FTS5 + embeddings.

        Args:
            source: list[str] або шлях до файлу (str/Path).

        Returns:
            self (для chaining).
        """
        texts = _resolve_source(source)
        if not texts:
            return self

        chunks = chunk_texts(texts, **kwargs)
        if not chunks:
            return self

        src = str(source) if not isinstance(source, list) else ""

        # Видаляємо старі дані для цього source (запобігає дублям при re-index)
        if src:
            self._conn.execute("DELETE FROM embeddings WHERE source = ?", (src,))
            self._conn.execute(
                "DELETE FROM search_index WHERE rowid IN "
                "(SELECT id FROM search_index_content WHERE c3 = ?)",
                (src,),
            )

        normalized_chunks = []
        for chunk in chunks:
            norm = _normalize_text(chunk)
            normalized_chunks.append(norm)
            # FTS5 індексація
            index_text(self._conn, chunk, source=src)

        # Embeddings (якщо є ембедер)
        if self._embedder is not None and normalized_chunks:
            vectors = self._embedder.embed(normalized_chunks)
            for chunk, norm, vec in zip(chunks, normalized_chunks, vectors):
                blob = _serialize_embedding(vec)
                self._conn.execute(
                    "INSERT INTO embeddings (chunk_text, original_text, source, embedding) "
                    "VALUES (?, ?, ?, ?)",
                    (norm, chunk, src, blob),
                )

        self._conn.commit()
        return self

    def count(self) -> int:
        """Кількість проіндексованих ембедінгів."""
        return self._conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]

    def mumble(self, query: str, top_k: int = 10) -> list[dict]:
        """Гібридний пошук: embeddings (якщо є) або FTS5 fallback.

        Args:
            query: Пошуковий запит.
            top_k: Максимальна кількість результатів.

        Returns:
            [{"text": str, "score": float, "chunk_id": int}, ...]
        """
        if self._embedder is None:
            return self._mumble_fts5(query, top_k)

        return self._mumble_embeddings(query, top_k)

    def _mumble_embeddings(self, query: str, top_k: int) -> list[dict]:
        """Пошук через cosine similarity ембедінгів (з дедуплікацією)."""
        query_vec = self._embedder.embed_one(query)

        rows = self._conn.execute(
            "SELECT id, chunk_text, original_text, embedding FROM embeddings"
        ).fetchall()

        if not rows:
            return []

        results = []
        for row in rows:
            chunk_id = row[0]
            original_text = row[2]
            stored_vec = _deserialize_embedding(row[3])
            score = self._embedder.cosine_similarity(query_vec, stored_vec)
            results.append({
                "text": original_text,
                "score": score,
                "chunk_id": chunk_id,
            })

        results.sort(key=lambda x: x["score"], reverse=True)

        # Дедуплікація: залишаємо найкращий score для кожного унікального тексту
        seen: set[str] = set()
        deduped = []
        for r in results:
            if r["text"] not in seen:
                seen.add(r["text"])
                deduped.append(r)
                if len(deduped) >= top_k:
                    break

        return deduped

    def _mumble_fts5(self, query: str, top_k: int) -> list[dict]:
        """Fallback пошук через FTS5 (з дедуплікацією)."""
        fts_results = fts_search(self._conn, query, limit=top_k * 3)
        seen: set[str] = set()
        deduped = []
        for r in fts_results:
            text = r["original"]
            if text not in seen:
                seen.add(text)
                deduped.append({
                    "text": text,
                    "score": abs(r["rank"]),
                    "chunk_id": 0,
                })
                if len(deduped) >= top_k:
                    break
        return deduped

    def sip(
        self,
        source,
        topics: list[str],
        threshold: float = 0.3,
    ) -> dict[str, list[dict]]:
        """Класифікація текстів по темам через cosine similarity.

        Args:
            source: list[str] або шлях до файлу (str/Path).
            topics: Список тем для класифікації.
            threshold: Мінімальний score для включення в результат.

        Returns:
            {"topic": [{"text": str, "score": float}, ...], ...}

        Raises:
            ValueError: Якщо embedder не задано.
        """
        if self._embedder is None:
            raise ValueError("sip() потребує embedder — передайте його в Teapot(embedder=...)")

        texts = _resolve_source(source)

        # Ембедінги текстів і тем
        text_vecs = self._embedder.embed(texts) if texts else []
        topic_vecs = self._embedder.embed(topics)

        result: dict[str, list[dict]] = {topic: [] for topic in topics}

        for text, text_vec in zip(texts, text_vecs):
            for topic, topic_vec in zip(topics, topic_vecs):
                score = self._embedder.cosine_similarity(text_vec, topic_vec)
                if score >= threshold:
                    result[topic].append({"text": text, "score": score})

        # Сортуємо кожну тему за score desc
        for topic in result:
            result[topic].sort(key=lambda x: x["score"], reverse=True)

        return result

    def brew(
        self,
        query: str,
        *,
        llm,
        extract: dict[str, type] | None = None,
        strategy: str = "hierarchical",
        pre_filter_k: int | None = None,
        limit: int = 50,
    ) -> list[dict]:
        """LLM-powered пошук і витягування.

        Args:
            query: Пошуковий запит природною мовою (UA).
            llm: LocalLLM інстанс (або будь-який об'єкт з classify/extract).
            extract: Схема витягування. None = повернути чанки як є.
            strategy: "hierarchical", "filter", або "full_scan".
            pre_filter_k: Кількість чанків для pre-filter. None = adaptive.
            limit: Максимум результатів.

        Returns:
            Якщо extract задана: [{"name": "...", ...}, ...]
            Якщо extract=None: [{"text": str, "score": float, "chunk_id": int}, ...]
        """
        total = self.count()
        if total == 0:
            return []

        # Етап 1: Pre-filter (embeddings)
        if strategy in ("hierarchical", "filter"):
            k = pre_filter_k or min(500, max(50, int(total * 0.1)))
            k = min(k, total)
            candidates = self.mumble(query, top_k=k)
        else:  # full_scan
            candidates = self._all_chunks()

        # Етап 2: LLM classify
        if strategy in ("hierarchical", "full_scan"):
            candidates = [c for c in candidates if llm.classify(c["text"], query)]

        # Етап 3: Extract або повернути як є
        if extract is None:
            return candidates[:limit]

        results = []
        for chunk in candidates:
            extracted = llm.extract(chunk["text"], extract)
            if extracted is not None:
                results.append(extracted)

        results = _deduplicate(results)
        return results[:limit]

    def _all_chunks(self) -> list[dict]:
        """Повертає всі проіндексовані чанки."""
        rows = self._conn.execute(
            "SELECT id, original_text FROM embeddings"
        ).fetchall()
        return [
            {"text": row[1], "score": 0.0, "chunk_id": row[0]}
            for row in rows
        ]


def _deduplicate(results: list[dict]) -> list[dict]:
    """Дедуплікація результатів extract за вмістом."""
    seen = set()
    deduped = []
    for item in results:
        key = tuple(sorted(item.items()))
        if key not in seen:
            seen.add(key)
            deduped.append(item)
    return deduped
