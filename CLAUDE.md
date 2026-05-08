# dormouse

Оптимізація українських текстів для LLM: менше токенів, краще розуміння.
Курсова робота AI&ML. Python бібліотека + CLI.

**PyPI:** `pip install dormouse-ua` (package name `dormouse-ua`, import name `dormouse`)
**GitHub:** https://github.com/ChuprinaDaria/dormouse (публічне, чисте репо без історії)
**HuggingFace:** https://huggingface.co/Dariachup/dormouse (model card + assets)
**Версія:** 0.4.2

## Публікація і deployment

### Assets — бандлені в пакет (з v0.4.2)
Все бандлиться в pip пакет (29MB wheel), нічого не треба качати:
- `src/dormouse/data/replacements.json` (41KB) — 360 правил
- `src/dormouse/data/lexicon.db` (12MB) — лексикон 47K
- `src/dormouse/data/expr_seq2seq.pt` (28MB) — GRU модель
- `src/dormouse/data/expr_vocab_src.json` (393KB) — vocab src
- `src/dormouse/data/expr_vocab_tgt.json` (162KB) — vocab tgt
- `src/dormouse/data/expr_config.json` — config моделі

Логіка в `src/dormouse/assets.py` (порядок пошуку):
1. `DORMOUSE_DATA_DIR` env var → dev mode (локальна `data/`)
2. `importlib.resources` → бандлені файли в пакеті
3. `~/.cache/dormouse/v{version}/` → cache
4. Download: GitHub Releases → HuggingFace fallback
5. `DORMOUSE_OFFLINE=1` → заборона download

### Default dependencies (~15MB)
`pip install dormouse-ua` ставить: click, pymorphy3 + dicts-uk, tiktoken, openai, anthropic, openpyxl.
Для torch/embeddings/search: `pip install dormouse-ua[ml]`

### DormouseClient — автодетекція провайдера
```python
from dormouse import DormouseClient
# Один клас для OpenAI і Anthropic — визначає по class name
client = DormouseClient(OpenAI())      # → OpenAI proxy
client = DormouseClient(Anthropic())   # → Anthropic proxy
```

### Публікація нової версії
```bash
# 1. Bump version в pyproject.toml + src/dormouse/__init__.py
# 2. Скопіювати зміни в /tmp/dormouse-clean (чисте репо)
# 3. Скопіювати model+lexicon в src/dormouse/data/ (бандлиться в пакет)
# 4. Build
cd /tmp/dormouse-clean && python -m build
# 5. Upload
TWINE_USERNAME=__token__ TWINE_PASSWORD=pypi-xxx twine upload dist/*
# 6. Push
git push origin main
```

### CI
- `.github/workflows/ci.yml` — lint + tests (Python 3.11, 3.12)
- `.github/workflows/publish.yml` — PyPI publish on release (trusted publisher)
- Тести без assets: `DORMOUSE_OFFLINE=1`, skip test_unsqueeze, test_mapper, test_import

### Два репо
- `/home/dchuprina/dormouse` — робоче репо з повною історією, data/, scripts/, exports
- `/tmp/dormouse-clean` — чисте репо для PyPI/GitHub (модель+лексикон бандляться в src/dormouse/data/)
- GitHub `ChuprinaDaria/dormouse` = чисте репо

### LLMLingua benchmark (реальні дані, 20 промптів)
| Method | Tokens | Savings | Quality |
|---|---|---|---|
| Original UA | 1,312 | — | 4.65/5 |
| dormouse | 620 | 53% | 4.50/5 |
| LLMLingua (on UA) | 1,182 | 10% | 4.60/5 |
| dormouse+LLMLingua | 595 | 55% | 4.60/5 |
Дані: `data/exports/eval_benchmark_comparison.json`

## Що працює (не ламати)

### Cloud squeeze — 73% економія токенів ✅
```
Вхідний текст
│
├→ crack_open()          ← src/dormouse/data/replacements.json (360 правил)
│   нормалізація + апострофи (U+2019, U+02BC → U+0027)
│
├→ compress()            ← вбудовані правила
│   видалення fillers
│
├→ [Cloud path]
│   ├→ seq2seq v2        ← expr_seq2seq.pt (28K pairs, 98.2% exact match)
│   │   вирази цілком (ExpressionTranslator, GRU 7.3M params)
│   └→ map_to_en(cloud)  ← lexicon.db (47K entries)
│       пословний + lemma fallback
│
└→ [Teapot + LLM path]
    ├→ stir/mumble       ← embeddings (MiniLM-L12-v2)
    ├→ sip               ← topic classification (topic phrases, top-1)
    └→ brew()            ← LocalLLM (MamayLM-4B / Qwen3-4B)
```

### Eval результати

#### Corpus eval (2026-04-29, 4 дні)
| Метрика | Значення |
|---------|---------|
| Token savings (cloud + seq2seq) | **73%** |
| Token savings (cloud without seq2seq) | **49%** |
| Lexicon coverage | **88.23%** (53K texts) |
| Seq2seq exact match | **98.2%** (500 val pairs) |
| Normalization speed | 606 texts/sec |
| SIP classification | 99% (8 topics) |
| Tests | 258 passing |

#### Quality preservation (2026-05-05, 100 реальних промптів)
| Model | UA score | EN (squeezed) | Preservation |
|---|---|---|---|
| **GPT-4.1** | 4.79 | **4.86** | **102%** |
| GPT-4.1-mini | 4.71 | 4.68 | 99% |
| GPT-4o-mini | 4.61 | 4.60 | 100% |
| GPT-4.1-nano | 4.58 | 4.56 | 100% |
| GPT-5.5 | 4.00 | 4.00 | 100% |
| Gemini 2.0 Flash | 4.11 | 4.10 | 100% |

Дані: `data/exports/eval_openai_100.json`, `data/exports/eval_gemini_100.json`
Judge: heuristic (length + structure), не LLM-judge. GPT-5.5 score нижчий бо дає коротші відповіді.

#### HF Inference API (малі моделі)
| Model | UA | EN (squeeze) | Δ |
|---|---|---|---|
| Qwen2.5-72B | 4.9/5 | 4.5/5 | -0.4 |
| Qwen2.5-7B | 4.4/5 | 3.6/5 | -0.8 |
| Llama-3.2-1B | 2.7/5 | 2.8/5 | +0.1 |

**Висновок:** squeeze 99-102% preservation для cloud моделей. Для малих (<7B) — brew() з нативною UA.

## Архітектура

```
src/dormouse/
├── __init__.py        — public API (v0.3.2)
├── assets.py          — lazy download + cache (GitHub/HF)
├── optimizer.py       — squeeze(), squeeze_batch(), SqueezedText
├── rule_engine.py     — crack_open(): 360 правил + pymorphy3 + апострофи
├── compressor.py      — compress(): fillers, інтенсифікатори
├── mapper.py          — map_to_en(): lexicon + lemma + transliteration
├── seq2seq.py         — ExpressionTranslator (GRU encoder-decoder, 7.3M)
├── teapot.py          — stir/mumble/sip/brew (dedup в FTS5 і embeddings)
├── embedder.py        — sentence-transformers (MiniLM-L12-v2)
├── search.py          — FTS5 + лематизація
├── middleware.py      — DormouseClient: OpenAI SDK proxy
├── anthropic_proxy.py — Anthropic SDK proxy
├── unsqueeze.py       — EN→UA (для middleware)
├── local_model.py     — LocalLLM: Ollama/HF backends
├── cli.py             — Click CLI
├── data/
│   └── replacements.json  ← бандлиться через importlib.resources
├── analyzer.py        — nibble (tiktoken)
├── classifier.py      — sniff (embeddings)
├── language_guard.py  — детекція мови
├── stream_buffer.py   — буферизація стрімів
├── morphology.py      — pymorphy3 обгортка
├── style_classifier.py — TF-IDF стиль
├── parsers.py         — PDF, Excel, CSV, TXT
└── db.py              — SQLite corpus
```

## Видалено (v0.3.0 cleanup)
- `mcp_tool.py` — мертвий код, не використовувався
- `DreamWeaver` (seq2seq v1) — UA→UA нормалізація, замінений ExpressionTranslator v2
- `wake_up()`, `dream()` — старі функції DreamWeaver
- `scripts/train_seq2seq.py` — старий training script
- `src/dormouse/models/` — видалено в v0.4.2, файли перенесені в `src/dormouse/data/`

## Дані (робоче репо)

### В data/
- `data/db/lexicon.db` — лексикон 47K записів
- `data/db/search.db` — FTS5 пошуковий індекс
- `data/db/dormouse.db` — corpus (53K texts)
- `data/lexicon/replacements.json` — 360 правил (копія в src/dormouse/data/)
- `data/channels.json` — Telegram канали
- `data/exports/` — eval результати (full_eval_2026-04-29.json)

### Gaps (top проблеми лексикону)
- Артефакти апострофів — **ВИПРАВЛЕНО** (нормалізація в crack_open)
- Словоформи (були, може, можуть) — потребують додавання
- Власні назви — ігноруються

## Скрипти

| Скрипт | Призначення |
|--------|-------------|
| `scripts/build_lexicon.py` | Побудова лексикону |
| `scripts/collect_telegram.py` | Збір корпусу |
| `scripts/sync_slang.py` | Синк сленгу з ua-slang MCP |
| `scripts/generate_expression_pairs.py` | Генерація expression pairs |
| `scripts/train_expressions.py` | Тренування ExpressionTranslator |
| `scripts/full_eval.py` | Повна евалуація (4 дні) |
| `scripts/eval_cloud.py` | Евалуація з cloud API |
| `scripts/eval_hf.py` | Евалуація з HF Inference API |
| `scripts/fill_gaps.py` | Автозаповнення прогалин лексикону |

## Тестування

```bash
# Локально (з даними)
DORMOUSE_DATA_DIR=./data pytest tests/ -v --ignore=tests/test_middleware_e2e.py

# Без assets (як в CI)
DORMOUSE_OFFLINE=1 pytest tests/ --ignore=tests/test_middleware_e2e.py \
  --ignore=tests/test_unsqueeze.py --ignore=tests/test_mapper.py \
  --ignore=tests/test_import.py -k "not cloud and not version" -q
```

## Публічний API

```python
from dormouse import squeeze, squeeze_batch, SqueezedText
from dormouse import DormouseClient  # OpenAI/Anthropic middleware
from dormouse import nibble          # токен-аналіз
from dormouse import sniff           # класифікація
from dormouse import stir, mumble, sip  # пошук + класифікація (локально)
from dormouse import brew, LocalLLM    # LLM-powered витягування
```

## Правила розробки

- Мова коммітів: англійська (conventional commits)
- Лінтер: ruff (line-length 99, target py311)
- Тести обов'язкові для нової функціональності
- `.env` та API ключі — ніколи в git
- PyPI package name: `dormouse-ua`, import name: `dormouse`
- Шляхи до даних: через `assets.py`, НЕ hardcoded
- replacements.json: через `importlib.resources`, бандлиться в пакет
