# dormouse pair collector

Приватний сервіс збору пар з деплойментів. **Не входить у pip-пакет** —
`setuptools` пакує тільки `src/`. Телеметрії в `dormouse-ua` немає і не буде;
деплоймент надсилає пари сюди явно, зі свого коду.

## Запуск

```bash
pip install -r collector/requirements.txt
COLLECTOR_TOKEN=<секрет> COLLECTOR_SINK=/var/data/pairs.jsonl \
    uvicorn collector.app:app --host 0.0.0.0 --port 8080
```

## API

`POST /pairs` з `Authorization: Bearer <токен>`:

```json
{"dirty": "шо там по замовленню", "dormouse_output": "order status?", "meta": {"deployment": "shop-bot"}}
```

PII-скраб (`dormouse.pii`) виконується **до** запису на диск: телефони, email,
IBAN, картки, URL з токенами, хендли, адреси, імена → плейсхолдери. Пари, що
складаються переважно з PII, відкидаються (`{"status": "dropped"}`).

Зібрані пари проходять той самий пайплайн, що й датасети:
`scripts/unify_pairs.py --source harvest` → `scripts/compose_train.py`.
