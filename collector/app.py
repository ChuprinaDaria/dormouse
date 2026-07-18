"""Pair collector for dormouse deployments — private infrastructure.

Minimal FastAPI service accepting (dirty, dormouse_output) pairs from
deployments and appending them to a JSONL sink after PII scrubbing.
NOT part of the dormouse-ua pip package (setuptools only packages src/).

Env:
    COLLECTOR_TOKEN  — bearer token, required
    COLLECTOR_SINK   — target JSONL path (default: ./collected_pairs.jsonl)

Run:
    pip install -r collector/requirements.txt
    COLLECTOR_TOKEN=secret uvicorn collector.app:app --host 0.0.0.0 --port 8080
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from dormouse.pii import scrub_pair

app = FastAPI(title="dormouse pair collector", docs_url=None, redoc_url=None)
_write_lock = threading.Lock()


class PairIn(BaseModel):
    dirty: str = Field(min_length=1, max_length=4096)
    dormouse_output: str = Field(min_length=1, max_length=4096)
    meta: dict = Field(default_factory=dict)


def _sink_path() -> Path:
    return Path(os.environ.get("COLLECTOR_SINK", "collected_pairs.jsonl"))


def _check_token(authorization: str | None) -> None:
    expected = os.environ.get("COLLECTOR_TOKEN")
    if not expected:
        raise HTTPException(status_code=503, detail="collector not configured")
    if authorization != f"Bearer {expected}":
        raise HTTPException(status_code=401, detail="invalid token")


@app.post("/pairs")
def collect_pair(pair: PairIn, authorization: str | None = Header(default=None)) -> dict:
    _check_token(authorization)

    scrubbed = scrub_pair(pair.dirty, pair.dormouse_output)
    if scrubbed is None:
        return {"status": "dropped", "reason": "pii_dominated"}

    record = {
        "dirty": scrubbed[0],
        "dormouse_output": scrubbed[1],
        "meta": {k: v for k, v in pair.meta.items() if isinstance(v, (str, int, float, bool))},
        "collected_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    with _write_lock, open(_sink_path(), "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return {"status": "ok"}


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
