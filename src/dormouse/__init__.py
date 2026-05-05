"""dormouse — оптимізація українських текстів для LLM."""

__version__ = "0.4.1"

from dormouse.optimizer import SqueezedText, squeeze, squeeze_batch

__all__ = [
    "squeeze",
    "squeeze_batch",
    "SqueezedText",
    "__version__",
]


def nibble(text: str):
    """Бенчмарк токенізації (потребує pip install dormouse[tokens])."""
    from dormouse.analyzer import nibble as _nibble

    return _nibble(text)


def nibble_batch(texts: list[str]):
    """Батч-аналіз токенізації."""
    from dormouse.analyzer import nibble_batch as _nibble_batch

    return _nibble_batch(texts)


def sniff(
    items: list[str],
    categories: list[str],
    squeeze_first: bool = False,
):
    """Класифікація через embeddings (потребує pip install dormouse[embeddings])."""
    from dormouse.classifier import sniff as _sniff

    return _sniff(items, categories, squeeze_first=squeeze_first)


def DormouseClient(client, **kwargs):
    """Drop-in SDK wrapper для OpenAI та Anthropic. Автодетекція провайдера."""
    from dormouse.middleware import DormouseClient as _DormouseClient

    return _DormouseClient(client, **kwargs)


def stir(source, **kwargs):
    """Завантажити і проіндексувати файл (потребує pip install dormouse[search])."""
    from dormouse.teapot import Teapot

    try:
        from dormouse.embedder import Embedder

        emb = Embedder()
    except ImportError:
        emb = None
    t = Teapot(embedder=emb)
    return t.stir(source, **kwargs)


def mumble(query, index=None, top_k=10):
    """Семантичний пошук по проіндексованих даних."""
    from dormouse.teapot import Teapot

    if index is not None:
        return index.mumble(query, top_k=top_k)
    t = Teapot()
    return t.mumble(query, top_k=top_k)


def sip(source, topics, threshold=0.5, **kwargs):
    """Класифікація тексту по заданих темах."""
    from dormouse.embedder import Embedder
    from dormouse.teapot import Teapot

    emb = Embedder()
    t = Teapot(embedder=emb)
    return t.sip(source, topics, threshold=threshold)


def brew(query, *, llm, index=None, extract=None, strategy="hierarchical", limit=50):
    """LLM-powered пошук і витягування (потребує pip install dormouse[search])."""
    from dormouse.teapot import Teapot

    if index is not None:
        return index.brew(query, llm=llm, extract=extract, strategy=strategy, limit=limit)
    t = Teapot()
    return t.brew(query, llm=llm, extract=extract, strategy=strategy, limit=limit)


def LocalLLM(model="MamayLM-4B", **kwargs):
    """Уніфікований інтерфейс до локальної LLM."""
    from dormouse.local_model import LocalLLM as _LocalLLM

    return _LocalLLM(model, **kwargs)
