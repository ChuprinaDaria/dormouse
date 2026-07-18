# Інструкція: дотрен ExpressionTranslator на RunPod (GPU)

Ця інструкція — для Claude Code на локальній машині Даші (і на RunPod-поді).
Контекст: у гілці `claude/dormouse-dataset-seq2seq-xbh661` побудований повний
датасет+трен пайплайн (див. `scripts/README.md`). У хмарній сесії, де його
робили, huggingface.co і сирий github.com були заблоковані, тому OmniGEC,
Brown-UK і UberText треба скачати локально. UA-GEC, лексикон і frozen eval
уже готові й закомічені/відтворювані.

## 1. Забрати гілку локально (PR НЕ створювався)

Це саме гілка в основному репо, не PR і не форк. Реліз-репо вона не чіпає,
поки її явно не змерджити в main:

```bash
git clone https://github.com/ChuprinaDaria/dormouse.git dormouse-train
cd dormouse-train
git checkout claude/dormouse-dataset-seq2seq-xbh661
# або в наявному клоні:
# git fetch origin claude/dormouse-dataset-seq2seq-xbh661
# git checkout claude/dormouse-dataset-seq2seq-xbh661

pip install -e ".[dev,ml]" -r scripts/requirements.txt
DORMOUSE_OFFLINE=1 pytest tests/test_dataset_pipeline.py tests/test_pii.py -q  # має бути зелено
```

PR з цієї гілки відкривається тільки коли буде готовий реліз (після гейта).

## 2. Скачати всі дані (звідки що)

```bash
# 2.1. Асети поточної моделі (модель 0.4.2 + лексикон 47K) — з PyPI wheel:
python scripts/fetch_assets.py                       # → data/assets/

# 2.2. UA-GEC 2.x — бандлений у pip-пакет ua-gec (CC BY 4.0, Grammarly):
python scripts/download_ua_gec.py                    # → data/raw/ua_gec/ + clean_sentences/

# 2.3. OmniGEC (UNLP 2025) — Hugging Face, організація lang-uk:
#      UberText-GEC (Telegram, автовиправлення GPT-4o-mini) + укр. зрізи
#      Reddit-MultiGEC і WikiEdits-MultiGEC. Скрипт сам резолвить точні id
#      через пошук по хабу (не вгадуємо шляхи) і пише їх у MANIFEST:
python scripts/download_omnigec.py --max-records 200000

# 2.4. Brown-UK (БрУК, 1M слів чистої укр., CC BY-NC-SA) — GitHub brown-uk/corpus:
python scripts/download_brown_uk.py                  # → clean_sentences/brown_uk.txt

# 2.5. UberText 2.0 social (lang.org.ua; URL взяти на сторінці корпусу
#      після прийняття умов — сирі Telegram-тексти для синтетики):
python scripts/download_ubertext.py --url <URL_соц_підкорпусу> --max-sentences 200000
```

Кожен крок дописує джерело/версію/ліцензію/sha256 у `data/raw/MANIFEST.md`.

Далі — уніфікація і композиція (UA-GEC уже пройдений у хмарній сесії, але
файли data/ не в git, тож локально прогнати все):

```bash
python scripts/unify_pairs.py --source ua_gec  --license "CC BY 4.0"
python scripts/unify_pairs.py --source omnigec --license "see MANIFEST"
DORMOUSE_DATA_DIR=data/assets python scripts/generate_expression_pairs.py --expand-unigrams
DORMOUSE_DATA_DIR=data/assets python scripts/compose_train.py --source ua_gec
DORMOUSE_DATA_DIR=data/assets python scripts/compose_train.py --source omnigec
python scripts/synth_corrupt.py
DORMOUSE_DATA_DIR=data/assets python scripts/compose_train.py --source synth
```

**ВАЖЛИВО — frozen eval уже існує і закомічений** (`data/eval/frozen_v1.jsonl`,
1499 пар, sha256 у `data/eval/FROZEN.md`). `make_frozen_eval.py` НЕ запускати —
він і сам відмовиться перезаписувати. Усі trainwriting-скрипти автоматично
виключають frozen-хеші. Якщо є оригінальні 28K пар зі старого робочого репо —
покласти як `data/train/existing28k.jsonl` (`{"src","tgt"}`), mix-конфіг
підхопить сам (`optional: true`).

## 3. Тренування на RunPod

Для GRU на 7.3M параметрів вистачає найдешевшого GPU (RTX 3090/4090/A5000 —
будь-який; VRAM потрібно < 2GB). Темплейт RunPod PyTorch підходить.

```bash
# на поді:
git clone https://github.com/ChuprinaDaria/dormouse.git && cd dormouse
git checkout claude/dormouse-dataset-seq2seq-xbh661
pip install -e ".[dev,ml]" -r scripts/requirements.txt

# перенести на под підготовлені data/pairs/ + data/train/ + data/assets/
# (runpodctl send/receive або scp), АБО прогнати кроки 2.1–2.5 прямо на поді

# smoke (1-2 хв) — перевірка що все живе:
DORMOUSE_DATA_DIR=data/assets python scripts/train_expressions.py \
    --epochs 2 --max-pairs 3000 --run-name run_smoke

# повний трен (device=cuda підхопиться сам; 30 епох max, early stop patience 5):
DORMOUSE_DATA_DIR=data/assets python scripts/train_expressions.py --run-name run_v1
```

Вихід: `data/checkpoints/run_v1/` — `expr_seq2seq.pt` + `expr_config.json` +
`expr_vocab_{src,tgt}.json` (деплойний формат, чекпоінт self-verify через
`wake_up_expr`), `training_run.json` (гіперпараметри, хеші даних, криві),
`frozen_metrics.json` (автоматичний прогін на frozen_v1 у кінці).

**Гейт релізу** (порівняти з `data/eval/baseline_v0.4.2.json`):
- загальний exact_match ≥ 24.7%;
- lexicon-страта ≥ 32.3% (просадка ≤ 1 п.п. — захист поточної поведінки);
- `latin_ok_rate` / `none_rate` не гірші за 100% / 0%.
Просіло — не реліз; чекпоінт забрати з пода (`runpodctl send` / scp) в
будь-якому разі, для розбору.

Після успішного гейта: рядок у `DATA_REVISIONS.md` (версія ↔ ревізія даних ↔
sha256 чекпоінта ↔ метрики), далі реліз за `scripts/README.md` («Реліз»).

## 4. Чи додавати книги / ще українських даних, і скільки

Коротко: **більше реальних GEC-пар — так; книги — тільки як сировина для
синтетики, і небагато.**

- Модель маленька (7.3M, word-level GRU, вирази ≤ 16 токенів). Стеля корисного
  об'єму — приблизно **200–400K пар** у фінальному міксі; далі приріст з'їдає
  ростучий словник (min_freq=2), а не якість.
- Пріоритет №1 — **OmniGEC/UberText-GEC**: це реальні брудні Telegram-тексти,
  найближчі до цільового домену. 100–200K сирих записів дадуть ~50–150K
  композитних пар — цього досить, щоб домінували реальні дані.
- **Книги (Brown-UK, УберТекст fiction) — тільки в синтетику**: вони чисті,
  парами не є, і стиль далекий від чатів. Синтетика обрізається на 40% міксу
  конфігом, тому більш ніж ~60–80K синт-пар сенсу не мають. Достатньо:
  Brown-UK повністю (~50K речень) + 100–200K речень UberText social. Окремо
  качати ще книжкові корпуси — НЕ треба.
- Якщо хочеться максимального результату — замість книг додати **доменні
  дані**: реальні діалоги з магазинів/LMS/сапорту (через collector/ або
  ручний експорт). 5–10K реальних доменних пар дадуть більше, ніж +100K книг.

## 5. Ручна робота з парами (так, є сенс — 2-3 години максимум)

1. **Спот-чек композиції** (найважливіше): подивитись по ~100 випадкових пар
   з `data/train/ua_gec.jsonl` і `data/train/omnigec.jsonl`. Червоні прапорці:
   таргет не відповідає змісту, обрізані вікна без сенсу, англійська каша.
   Якщо >10-15% сміття — підняти строгість: у `compose_train.py` можна
   викинути вікна без жодного кириличного слова і пари, де tgt коротший за
   1/3 src. (Дрібне сміття ок — детермінований «вчитель» має свої причуди,
   модель вчиться саме його функції.)
2. **Топ-частотні src**: `sort | uniq -c | sort -rn` по src у трейні — верхні
   50 глянути очима, вони важать найбільше.
3. **~200 ручних пар з доменів магазини/LMS/сапорт** — це заплановане
   frozen_v2 (TODO у `data/eval/FROZEN.md`). Писати у форматі Layer B
   (`{"src","tgt"}`), НЕ додавати в трейн — це майбутній eval. Ще 300–500
   таких же ручних пар можна додати в трейн окремим джерелом
   `data/train/manual.jsonl` з вагою 2.0 у `train_mix.yaml` — ручні доменні
   пари найцінніші в усьому міксі.
4. **PII-вибірка**: перед пушем датасету на HF глянути 50 випадкових пар з
   `data/pairs/` на пропущені імена/адреси (евристика імен — не ідеальна).

## Що вже зроблено (не переробляти)

- Гілка: 9 комітів — PII-модуль, download-скрипти, уніфікація+композиція,
  frozen eval + базлайн 0.4.2, синтетика, трен-скрипт з деплой-сумісними
  чекпоінтами, runbook, колектор.
- `data/eval/frozen_v1.jsonl` — незмінний, у git. Базлайн:
  `data/eval/baseline_v0.4.2.json` (EM 24.7%; lexicon 33.3%; ua_gec 0.7%).
- Тести: 283 passed локально; `tests/test_dataset_pipeline.py` містить
  roundtrip-тест деплойного формату чекпоінта.
- Хмарний smoke-трен пройшов увесь цикл (мікс → трен → self-verify → frozen
  eval); повний CPU-ран зупинено навмисно — тренувати на RunPod.
