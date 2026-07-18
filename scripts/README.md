# Датасет + трен пайплайн

Розширення трейн-даних для ExpressionTranslator (брудна UA → стислий EN),
дотрен і цикл релізів. Кожен крок закінчується перевірюваним артефактом.

## Два шари даних

- **Layer A** — `data/pairs/*.jsonl`: канонічні пари `{"dirty","clean","source","license"}` (UA↔UA).
- **Layer B** — `data/train/*.jsonl`: скомпоновані трейн-пари `{"src","tgt","clean","source","license"}`,
  де `tgt` — стислий EN, отриманий детерміністично з чистої сторони
  (`crack_open → compress → строгий лексикон-маппінг`, без транслітерації).
  Заморожений eval (`data/eval/frozen_v1.jsonl`) — теж Layer B: це задача,
  яку реально виконує деплойнута модель.

## Порядок запуску

```bash
pip install -e ".[dev,ml]" -r scripts/requirements.txt

# 0. Асети поточної моделі (модель+лексикон з wheel 0.4.2 на PyPI)
python scripts/fetch_assets.py                     # → data/assets/

# 1. Датасети
python scripts/download_ua_gec.py                  # UA-GEC (бандлений у pip ua-gec)
python scripts/download_omnigec.py                 # OmniGEC з HF hub      [потрібен доступ до huggingface.co]
python scripts/download_brown_uk.py                # Brown-UK з GitHub     [потрібен доступ до github.com]
python scripts/download_ubertext.py --url <...>    # UberText 2.0 social   [URL з lang.org.ua]

# 2. Уніфікація в Layer A (нормалізація, фільтри, PII-скраб, дедуп)
python scripts/unify_pairs.py --source ua_gec  --license "CC BY 4.0"
python scripts/unify_pairs.py --source omnigec --license "see MANIFEST"

# 3. Layer B: композиція EN-таргетів + регенерація лексиконних пар
DORMOUSE_DATA_DIR=data/assets python scripts/generate_expression_pairs.py --expand-unigrams
DORMOUSE_DATA_DIR=data/assets python scripts/compose_train.py --source ua_gec
DORMOUSE_DATA_DIR=data/assets python scripts/compose_train.py --source omnigec

# 4. Заморожений eval (ОДИН раз, ДО тренування) + базлайн
python scripts/make_frozen_eval.py
python scripts/eval_frozen.py --model-dir data/assets \
    --out data/eval/baseline_v0.4.2.json --model-label assets-v0.4.2
# після цього ПЕРЕгенерувати трейн-файли (кроки 3) — вони виключать frozen-хеші

# 5. Синтетика (клеан-речення → брудні пари)
python scripts/synth_corrupt.py                    # конфіг: scripts/configs/synth_corrupt.yaml
DORMOUSE_DATA_DIR=data/assets python scripts/compose_train.py --source synth

# 6. Трен (конфіг міксу: scripts/configs/train_mix.yaml)
DORMOUSE_DATA_DIR=data/assets python scripts/train_expressions.py --run-name run_v1
# → data/checkpoints/run_v1/{expr_seq2seq.pt, expr_config.json, expr_vocab_*.json,
#                            training_run.json, frozen_metrics.json}
```

CPU-трен на ~100K пар — десятки хвилин; GPU підхоплюється автоматично.
Smoke-перевірка: `--epochs 2 --max-pairs 3000 --run-name run_smoke`.

## Гейт релізу

Порівняти `data/checkpoints/<run>/frozen_metrics.json` з
`data/eval/baseline_v0.4.2.json`:

1. загальний `exact_match` ≥ базлайну;
2. на lexicon-страті просадка ≤ 1 п.п. (захист поточної поведінки);
3. `latin_ok_rate` і `none_rate` не гірші.

Просіло — не реліз: розбір, не «підкрутити eval». `frozen_v1.jsonl`
незмінний назавжди; наступна версія — тільки новий `frozen_v2.jsonl`
(TODO: ~200 ручних пар з доменів магазини/LMS/сапорт).

## Реліз

1. Чекпоінт → `src/dormouse/data/` чистого репо (бандлиться у wheel), bump
   версії в `pyproject.toml` + `__init__.py` + `assets.VERSION`.
2. Асети → GitHub Release tag + HF `Dariachup/dormouse`.
3. Датасет → приватний HF datasets репо (`huggingface_hub.upload_folder`),
   ревізія фіксується.
4. Рядок у `DATA_REVISIONS.md`: pip-версія ↔ ревізія даних ↔ sha256
   чекпоінта ↔ frozen-метрики.

## Приватні пари

- Оригінальні 28K пар: покласти як `data/train/existing28k.jsonl`
  (`{"src","tgt"}`) — mix-конфіг підхопить автоматично (`optional: true`).
- Пари з деплойментів (collector/): `unify_pairs.py --source harvest` →
  `compose_train.py --source harvest` → додати джерело в `train_mix.yaml`.

## PII

`src/dormouse/pii.py` — обов'язковий скраб ДО запису будь-якої пари на диск
(unify_pairs і collector роблять це самі). Опційний вищий recall для імен:
spaCy `uk_core_news_sm` (не залежність пакета; можна пропустити ручним
переглядом `data/pairs/` перед комітом у датасет-репо).
