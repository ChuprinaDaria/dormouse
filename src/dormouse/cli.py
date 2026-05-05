"""CLI інтерфейс dormouse."""

import sys

import click


@click.group()
@click.version_option()
def main():
    """dormouse — стискання українських текстів для LLM."""


@main.command()
@click.argument("text", required=False)
@click.option("--input", "-i", "input_file", type=click.Path(exists=True))
@click.option("--output", "-o", "output_file", type=click.Path())
@click.option("--target", "-t", type=click.Choice(["cloud"]), default=None,
              help="Цільова платформа: cloud (EN стиснення для cloud моделей)")
@click.option("--verbose", "-v", is_flag=True, help="Детальний вивід")
def squeeze(text, input_file, output_file, target, verbose):
    """Стискає текст: менше токенів, краще розуміння LLM."""
    from dormouse.optimizer import squeeze as sq

    texts = _read_input(text, input_file)

    results = []
    for t in texts:
        r = sq(t, target=target, verbose=verbose)
        if verbose:
            click.echo(f"{r.original}")
            click.echo(f"  → {r.text}")
            click.echo(f"  замін: {r.replacements_made}, метод: {r.method}")
            if r.tokens_saved:
                for k, v in r.tokens_saved.items():
                    click.echo(f"  {k}: збережено {v} токенів")
            if r.compression_ratio is not None:
                click.echo(f"  compression: {r.compression_ratio:.1%}")
            click.echo()
            results.append(r.text)
        else:
            results.append(r)

    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("\n".join(results))
        click.echo(f"Записано {len(results)} рядків → {output_file}")
    elif not verbose:
        for r in results:
            click.echo(r)


@main.command()
@click.argument("text")
def nibble(text):
    """Бенчмарк токенізації тексту для різних LLM."""
    from dormouse.analyzer import nibble as _nibble

    try:
        stats = _nibble(text)
    except ImportError as e:
        click.echo(str(e), err=True)
        raise SystemExit(1)

    click.echo(f"Текст: {stats.text}")
    click.echo(f"Стиснутий: {stats.squeezed_text}")
    click.echo()
    click.echo(f"GPT-4:  {stats.gpt4} → {stats.gpt4_squeezed} "
               f"(збережено {stats.savings_gpt4_pct}%)")
    click.echo(f"Llama:  {stats.llama} → {stats.llama_squeezed} "
               f"(збережено {stats.savings_llama_pct}%)")


@main.command()
@click.option("--categories", "-c", required=True, help="Категорії через кому")
@click.option("--input", "-i", "input_file", type=click.Path(exists=True))
@click.option("--squeeze-first", is_flag=True, help="Стиснути перед класифікацією")
@click.argument("texts", nargs=-1)
def sniff(categories, input_file, squeeze_first, texts):
    """Класифікує тексти по категоріях (embeddings, локально)."""
    from dormouse.classifier import sniff as _sniff

    cats = [c.strip() for c in categories.split(",")]

    if input_file:
        with open(input_file, encoding="utf-8") as f:
            items = [line.strip() for line in f if line.strip()]
    elif texts:
        items = list(texts)
    else:
        click.echo("Потрібні тексти: dormouse sniff -c 'кат1,кат2' 'текст'",
                    err=True)
        raise SystemExit(1)

    try:
        results = _sniff(items, cats, squeeze_first=squeeze_first)
    except ImportError as e:
        click.echo(str(e), err=True)
        raise SystemExit(1)

    for r in results:
        click.echo(f"[{r.confidence:.2f}] {r.category}: {r.text}")


@main.command()
@click.argument("file", type=click.Path(exists=True))
@click.option("--sheet", help="Sheet name для Excel")
def stir(file, sheet):
    """Завантажує і індексує файл для пошуку."""
    from dormouse.teapot import Teapot

    try:
        from dormouse.embedder import Embedder

        emb = Embedder()
    except ImportError:
        emb = None
        click.echo("Warning: sentence-transformers не встановлений, тільки FTS5", err=True)
    t = Teapot(embedder=emb)
    kwargs = {}
    if sheet:
        kwargs["sheet"] = sheet
    t.stir(file, **kwargs)
    click.echo(f"Проіндексовано {t.count()} фрагментів з {file}")


@main.command()
@click.argument("query")
@click.option("--top", "-n", default=10, help="Кількість результатів")
@click.option("--file", "-f", "input_file", type=click.Path(exists=True))
def mumble(query, top, input_file):
    """Семантичний пошук по тексту."""
    from dormouse.teapot import Teapot

    try:
        from dormouse.embedder import Embedder

        emb = Embedder()
    except ImportError:
        emb = None
    t = Teapot(embedder=emb)
    if input_file:
        t.stir(input_file)
    results = t.mumble(query, top_k=top)
    if not results:
        click.echo("Нічого не знайдено.")
        return
    for r in results:
        click.echo(f"[{r['score']:.3f}] {r['text'][:100]}")


@main.command()
@click.argument("file", type=click.Path(exists=True))
@click.option("--topics", "-t", required=True, help="Теми через кому")
@click.option("--threshold", default=0.5, help="Мінімальний score")
@click.option("--output", "-o", "output_file", type=click.Path())
def sip(file, topics, threshold, output_file):
    """Сортує текст по темах."""
    from dormouse.embedder import Embedder
    from dormouse.teapot import Teapot

    topic_list = [t.strip() for t in topics.split(",")]
    emb = Embedder()
    tp = Teapot(embedder=emb)
    result = tp.sip(file, topic_list, threshold=threshold)
    for topic, items in result.items():
        click.echo(f"\n--- {topic} ({len(items)}) ---")
        for item in items:
            click.echo(f"  [{item['score']:.3f}] {item['text'][:80]}")
    if output_file:
        import json

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        click.echo(f"\nЗаписано → {output_file}")


@main.command()
@click.argument("query")
@click.option("--file", "-f", "input_file", type=click.Path(exists=True), required=True,
              help="Файл для пошуку")
@click.option("--model", "-m", default="MamayLM-4B", help="Модель LLM")
@click.option("--backend", "-b", default="ollama", type=click.Choice(["ollama", "hf"]),
              help="Backend: ollama або hf")
@click.option("--extract", "-e", "extract_fields", default=None,
              help="Поля для витягування через кому (name,price,description)")
@click.option("--strategy", "-s", default="hierarchical",
              type=click.Choice(["hierarchical", "filter", "full_scan"]),
              help="Стратегія пошуку")
@click.option("--limit", "-n", default=50, help="Максимум результатів")
def brew(query, input_file, model, backend, extract_fields, strategy, limit):
    """LLM-powered пошук і витягування з файлу."""
    from dormouse.local_model import LocalLLM
    from dormouse.teapot import Teapot

    try:
        from dormouse.embedder import Embedder
        emb = Embedder()
    except ImportError:
        emb = None
        click.echo("Warning: sentence-transformers не встановлений, тільки FTS5", err=True)

    llm = LocalLLM(model, backend=backend)
    t = Teapot(embedder=emb)
    t.stir(input_file)

    extract_schema = None
    if extract_fields:
        extract_schema = {f.strip(): str for f in extract_fields.split(",")}

    click.echo(f"Шукаю: {query}")
    click.echo(f"Модель: {model} ({backend}), стратегія: {strategy}")
    click.echo()

    results = t.brew(
        query,
        llm=llm,
        extract=extract_schema,
        strategy=strategy,
        limit=limit,
    )

    if not results:
        click.echo("Нічого не знайдено.")
        return

    click.echo(f"Знайдено: {len(results)}")
    click.echo()
    for i, r in enumerate(results, 1):
        if extract_schema:
            fields = ", ".join(f"{k}: {v}" for k, v in r.items())
            click.echo(f"  {i}. {fields}")
        else:
            score = r.get("score", 0)
            text = r["text"][:100]
            click.echo(f"  {i}. [{score:.3f}] {text}")


def _read_input(text, input_file):
    """Читає тексти з аргументу, файлу або stdin."""
    if text:
        return [text]
    if input_file:
        with open(input_file, encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
    if not sys.stdin.isatty():
        return [line.strip() for line in sys.stdin if line.strip()]
    click.echo("Потрібен текст: dormouse squeeze 'текст' або --input файл",
               err=True)
    raise SystemExit(1)
