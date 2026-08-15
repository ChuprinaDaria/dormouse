"""ONNX-Runtime MT translator for the fine-tuned MarianMT models (Track B).

Loaded lazily so importing ``dormouse`` stays cheap: the first ``translate``
call downloads the model via ``assets.ensure_mt_model`` and instantiates
``optimum.onnxruntime.ORTModelForSeq2SeqLM``.

Design notes:
- ONNX Runtime handles CPU SIMD dispatch (SSE/AVX/AVX2/AVX-512) at runtime, so
  the same wheel works on Phenom II as well as modern hardware.
- Beam search comes from HF ``generate()`` — one API, two directions.
- Failure to load falls back to the caller: `translate` returns ``None``, and
  ``optimizer.py`` / ``unsqueeze.py`` keep the existing rule-based path.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

from dormouse.assets import ensure_mt_model

logger = logging.getLogger("dormouse")

_INSTANCES: dict[str, "MTTranslator"] = {}
_INSTANCES_LOCK = threading.Lock()

# Threshold at which we treat the input as shouted and lowercase it.
_CAPS_RATIO = 0.7


def _normalize_caps(text: str) -> tuple[str, bool]:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return text, False
    upper_ratio = sum(1 for c in letters if c.isupper()) / len(letters)
    if upper_ratio > _CAPS_RATIO:
        return text.lower(), True
    return text, False


class MTTranslator:
    """One instance per direction. Not thread-safe internally — HF generate is."""

    def __init__(self, direction: str, model_dir: Path):
        self.direction = direction
        self.model_dir = model_dir
        self._model = None
        self._tokenizer = None

    def _lazy_load(self) -> None:
        if self._model is not None:
            return
        # v0.6+ ships MarianMT PyTorch weights; ONNX INT8 export is a follow-up.
        from transformers import MarianMTModel, MarianTokenizer
        self._tokenizer = MarianTokenizer.from_pretrained(self.model_dir)
        self._model = MarianMTModel.from_pretrained(self.model_dir).eval()

    def translate(
        self, text: str, *, num_beams: int = 4, max_new_tokens: int = 256,
    ) -> str | None:
        try:
            self._lazy_load()
        except Exception:
            logger.warning(
                "dormouse: MTTranslator(%s) load failed, falling back", self.direction,
                exc_info=True,
            )
            return None

        # Lowercase inputs that are mostly SHOUTED — the model was trained on
        # mixed-case chat and produces garbage on all-caps ("I'M INVOLVED TO
        # PESTROY IN THE SERVANT" on the v06 smoke test). Restore caps after.
        src, was_shouted = _normalize_caps(text)
        try:
            inputs = self._tokenizer(src, return_tensors="pt", truncation=True, max_length=256)
            outputs = self._model.generate(
                **inputs, num_beams=num_beams, max_new_tokens=max_new_tokens,
            )
            out = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
            return out.upper() if was_shouted else out
        except Exception:
            logger.warning("dormouse: MTTranslator(%s) inference failed", self.direction,
                           exc_info=True)
            return None


def get_translator(direction: str) -> MTTranslator | None:
    """Return the cached MTTranslator for a direction, downloading if needed.

    Returns None if the model isn't available offline or the manifest for that
    direction isn't pinned yet — callers should treat None as "use the
    rule-based fallback".
    """
    with _INSTANCES_LOCK:
        if direction in _INSTANCES:
            return _INSTANCES[direction]
    try:
        model_dir = ensure_mt_model(direction)
    except (FileNotFoundError, NotImplementedError, ValueError) as e:
        logger.info("dormouse: MT model unavailable for %s (%s)", direction, e)
        return None
    inst = MTTranslator(direction, model_dir)
    with _INSTANCES_LOCK:
        _INSTANCES[direction] = inst
    return inst
