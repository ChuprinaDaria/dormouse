"""Бенчмарк токенізації: скільки токенів жере текст у різних LLM."""

from dataclasses import dataclass

from dormouse.optimizer import squeeze


@dataclass
class NibbleStats:
    text: str
    gpt4: int = 0
    llama: int = 0
    squeezed_text: str = ""
    gpt4_squeezed: int = 0
    llama_squeezed: int = 0

    @property
    def savings_gpt4_pct(self) -> float:
        if self.gpt4 == 0:
            return 0.0
        return round((1 - self.gpt4_squeezed / self.gpt4) * 100, 1)

    @property
    def savings_llama_pct(self) -> float:
        if self.llama == 0:
            return 0.0
        return round((1 - self.llama_squeezed / self.llama) * 100, 1)

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "gpt4": self.gpt4,
            "llama": self.llama,
            "squeezed_text": self.squeezed_text,
            "gpt4_squeezed": self.gpt4_squeezed,
            "llama_squeezed": self.llama_squeezed,
            "savings_gpt4_pct": self.savings_gpt4_pct,
            "savings_llama_pct": self.savings_llama_pct,
        }


def _count_tokens_tiktoken(text: str, encoding: str = "cl100k_base") -> int:
    """GPT-4 токени через tiktoken."""
    import tiktoken

    enc = tiktoken.get_encoding(encoding)
    return len(enc.encode(text))


def _count_tokens_llama(text: str) -> int:
    """Llama токени — fallback на tiktoken o200k_base."""
    import tiktoken

    enc = tiktoken.get_encoding("o200k_base")
    return len(enc.encode(text))


def nibble(text: str) -> NibbleStats:
    """Гризе текст по токенах — скільки жере у різних LLM.

    Потребує: pip install dormouse[tokens]
    """
    squeezed = squeeze(text)

    stats = NibbleStats(
        text=text,
        squeezed_text=squeezed,
    )

    try:
        stats.gpt4 = _count_tokens_tiktoken(text)
        stats.gpt4_squeezed = _count_tokens_tiktoken(squeezed)
        stats.llama = _count_tokens_llama(text)
        stats.llama_squeezed = _count_tokens_llama(squeezed)
    except ImportError:
        raise ImportError(
            "tiktoken потрібен для nibble(). "
            "Встанови: pip install dormouse[tokens]"
        )

    return stats


def nibble_batch(texts: list[str]) -> list[NibbleStats]:
    """Батч-аналіз токенізації."""
    return [nibble(t) for t in texts]
