"""Оптимізація українських текстів для LLM.

Pipeline: нормалізація → компресія → UA→EN маппінг.
"""

from dataclasses import dataclass, field

from dormouse.compressor import compress
from dormouse.rule_engine import crack_open


@dataclass
class SqueezedText:
    text: str
    original: str
    method: str = "rule"
    replacements_made: int = 0
    tokens_saved: dict[str, int] = field(default_factory=dict)
    compression_ratio: float | None = None


def squeeze(
    text: str,
    target: str | None = None,
    verbose: bool = False,
    use_seq2seq: bool = False,
) -> str | SqueezedText:
    """Оптимізує текст: менше токенів, краще розуміння LLM.

    Args:
        text: Вхідний текст (розмовна, суржик, сленг).
        target: None (тільки нормалізація + компресія UA),
                "cloud" (+ EN стиснення для cloud моделей).
        verbose: Повертати SqueezedText замість рядка.
        use_seq2seq: Увімкнути seq2seq v2 для перекладу виразів (experimental).
    """
    if not text or not text.strip():
        if verbose:
            return SqueezedText(text=text, original=text)
        return text

    # Шар 1: нормалізація (суржик/сленг → стандартна)
    result = crack_open(text)
    output = result.text

    # Шар 2: компресія (видалення надлишковості)
    output = compress(output)

    # Шар 3: UA→EN маппінг (якщо target вказано)
    if target == "cloud":
        from dormouse.mapper import map_to_en
        output = map_to_en(output, use_seq2seq=use_seq2seq)

    if not verbose:
        return output

    opt = SqueezedText(
        text=output,
        original=result.original,
        method="rule",
        replacements_made=result.replacements_made,
    )

    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        src_tokens = len(enc.encode(result.original))
        opt_tokens = len(enc.encode(output))
        opt.tokens_saved["gpt4"] = src_tokens - opt_tokens
        if src_tokens > 0:
            opt.compression_ratio = round(opt_tokens / src_tokens, 3)
    except ImportError:
        pass

    return opt


def squeeze_batch(
    texts: list[str],
    target: str | None = None,
    verbose: bool = False,
) -> list[str | SqueezedText]:
    """Батч-оптимізація."""
    return [squeeze(t, target=target, verbose=verbose) for t in texts]
