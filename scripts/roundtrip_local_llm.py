#!/usr/bin/env python3
"""Round-trip benchmark: is a small local LLM better driven in English?

Two arms over the same customer messages:

  CHAIN   UA -> dormouse(uk-en) -> local LLM (English prompt) -> dormouse(en-uk) -> UA
  DIRECT  UA -> local LLM (Ukrainian prompt, told to answer in Ukrainian)

Needs an ollama daemon on 127.0.0.1:11434 and the dormouse MT models
(downloaded automatically, or pointed at with DORMOUSE_MT_DIR_UK_EN /
DORMOUSE_MT_DIR_EN_UK).

    python scripts/roundtrip_local_llm.py [model]   # default qwen2.5:3b
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request

from dormouse.mt_translator import get_translator

OLLAMA = "http://127.0.0.1:11434/api/generate"
MODEL = sys.argv[1] if len(sys.argv) > 1 else "qwen2.5:3b"

SHOP = (
    "You are a support agent for a small handmade jewellery shop (brass and silver, "
    "ships from Poland). Answer the customer politely and concretely, max 45 words."
)
SHOP_UA = (
    "Ти агент підтримки невеликого магазину hand-made прикрас (латунь і срібло, "
    "відправка з Польщі). Відповідай клієнту ввічливо і конкретно, максимум 45 слів. "
    "Відповідай українською."
)

PROMPTS = [
    "Доброго дня, а скільки коштує доставка в Україну і за скільки днів прийде?",
    "хочу замовити два браслети, знижка якась є на два?",
    "а можна оплатити при отриманні? бо карткою не хочу",
    "замовляла ще тиждень тому, де моє замовлення, номер 1042",
    "а якщо розмір не підійде, можна поміняти або повернути?",
    "скажіть будь ласка цей перстень є в сріблі чи тільки латунь?",
    "шо по гарантії якщо застібка зламається?",
    "можете зробити на замовлення з гравіюванням імені?",
    "скиньте реквізити куди платити і я сьогодні оплачу",
    "чи є у вас щось до 200 злотих в подарунок дівчині",
]


def ask(prompt: str, system: str) -> str:
    body = json.dumps({
        "model": MODEL, "prompt": prompt, "stream": False, "system": system,
        "options": {"temperature": 0.3, "num_predict": 140},
    }).encode()
    req = urllib.request.Request(OLLAMA, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)["response"].strip()


def main() -> None:
    uk_en = get_translator("uk-en")
    en_uk = get_translator("en-uk")
    if uk_en is None or en_uk is None:
        sys.exit("dormouse MT models unavailable")

    # Warm the lazy weight load so it does not land in the first timing.
    uk_en.translate("привіт")
    en_uk.translate("hello")
    print(f"[llm={MODEL}]\n")

    for i, ua in enumerate(PROMPTS, 1):
        t = time.time()
        en_q = uk_en.translate(ua)
        en_a = ask(en_q, SHOP)
        # Translate line by line: the model likes numbered lists, and MarianMT
        # handles one sentence at a time far better than a whole block.
        ua_a = "\n".join(
            en_uk.translate(ln) or ln for ln in en_a.split("\n") if ln.strip()
        )
        t_chain = time.time() - t

        t = time.time()
        direct = ask(ua, SHOP_UA)
        t_direct = time.time() - t

        print(f"===== {i} =====")
        print(f"[UA in   ]  {ua}")
        print(f"[EN in   ]  {en_q}")
        print(f"[EN out  ]  {en_a}")
        print(f"[CHAIN UA]  {ua_a}")
        print(f"[DIRECT  ]  {direct}")
        print(f"[time    ]  chain={t_chain:.1f}s direct={t_direct:.1f}s")
        print()


if __name__ == "__main__":
    main()
