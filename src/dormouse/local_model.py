"""LocalLLM — уніфікований інтерфейс до локальних LLM.

Backends:
  - "ollama": Ollama API (localhost:11434)
  - "hf": HuggingFace Inference API (потребує HF_TOKEN)
"""

import json
import os
from urllib.error import URLError
from urllib.request import Request, urlopen

MODEL_ALIASES = {
    "MamayLM-4B": {
        "ollama": "hf.co/INSAIT-Institute/MamayLM-Gemma-3-4B-IT-v1.0-GGUF:Q4_K_M",
        "hf": "INSAIT-Institute/MamayLM-Gemma-3-4B-IT-v1.0",
    },
    "Qwen3-4B": {
        "ollama": "qwen3:4b",
        "hf": "Qwen/Qwen3-4B",
    },
    "Qwen2.5-7B": {
        "ollama": "qwen2.5:7b",
        "hf": "Qwen/Qwen2.5-7B-Instruct",
    },
}

OLLAMA_URL = "http://localhost:11434"


class LocalLLM:
    """Уніфікований інтерфейс до локальної LLM."""

    def __init__(
        self,
        model: str = "MamayLM-4B",
        backend: str = "ollama",
        temperature: float = 0.1,
        max_tokens: int = 200,
        timeout: int = 600,
    ):
        if backend not in ("ollama", "hf"):
            raise ValueError(f"Невідомий backend: {backend!r}. Доступні: 'ollama', 'hf'")
        self._backend = backend
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._timeout = timeout
        alias = MODEL_ALIASES.get(model, {}).get(backend, "")
        self._is_qwen3 = "qwen3" in model.lower() or "qwen3" in alias.lower()
        if model in MODEL_ALIASES and backend in MODEL_ALIASES[model]:
            self._resolved_model = MODEL_ALIASES[model][backend]
        else:
            self._resolved_model = model

    def ask(self, prompt: str, system: str | None = None) -> str:
        """Вільна генерація."""
        if self._backend == "ollama":
            return self._ask_ollama(prompt, system)
        return self._ask_hf(prompt, system)

    def classify(self, text: str, query: str) -> bool:
        """Бінарна класифікація: текст відповідає запиту?"""
        prompt = (
            f"Does this text contain information about: {query}?\n\n"
            f"Text: {text}\n\n"
            f"Answer ONLY one word: yes or no"
        )
        response = self.ask(prompt)
        return _parse_yes_no(response)

    def extract(self, text: str, schema: dict[str, type]) -> dict | None:
        """Структурне витягування з тексту."""
        fields_desc = ", ".join(
            f"{name} ({_type_label(tp)})" for name, tp in schema.items()
        )
        prompt = (
            f"Extract information from this text as JSON.\n"
            f"Fields: {fields_desc}\n"
            f"If no information found — return null.\n\n"
            f"Text: {text}\n\n"
            f"JSON:"
        )
        response = self.ask(prompt)
        return _parse_json(response)

    def _ask_ollama(self, prompt: str, system: str | None = None) -> str:
        payload = {
            "model": self._resolved_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self._temperature,
                "num_predict": self._max_tokens,
            },
        }
        if self._is_qwen3:
            payload["think"] = False
        if system:
            payload["system"] = system

        data = json.dumps(payload).encode("utf-8")
        req = Request(
            f"{OLLAMA_URL}/api/generate",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urlopen(req, timeout=self._timeout) as resp:
                result = json.loads(resp.read())
                return result.get("response", "").strip()
        except (URLError, OSError) as e:
            raise ConnectionError(
                f"Ollama недоступний ({OLLAMA_URL}). "
                f"Встанови: https://ollama.com — {e}"
            )

    def _ask_hf(self, prompt: str, system: str | None = None) -> str:
        from huggingface_hub import InferenceClient

        token = os.environ.get("HF_TOKEN")
        client = InferenceClient(token=token)
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        resp = client.chat_completion(
            messages=messages,
            model=self._resolved_model,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
        )
        return resp.choices[0].message.content.strip()


# --- Standalone helpers (backward compat) ---


def is_ollama_available() -> bool:
    """Перевіряє чи доступний Ollama."""
    try:
        req = Request(f"{OLLAMA_URL}/api/tags")
        with urlopen(req, timeout=2) as resp:
            return resp.status == 200
    except (URLError, OSError):
        return False


def list_models() -> list[str]:
    """Повертає список доступних моделей Ollama."""
    try:
        req = Request(f"{OLLAMA_URL}/api/tags")
        with urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
            return [m["name"] for m in data.get("models", [])]
    except (URLError, OSError):
        return []


def generate(
    prompt: str,
    model: str = "qwen2.5:1.5b",
    system: str | None = None,
    temperature: float = 0.3,
    max_tokens: int = 512,
) -> str:
    """Legacy generate через Ollama."""
    llm = LocalLLM(model, backend="ollama", temperature=temperature, max_tokens=max_tokens)
    return llm.ask(prompt, system=system)


def run(
    prompt: str,
    model: str = "qwen2.5:1.5b",
    optimize_first: bool = True,
    system: str | None = None,
) -> str:
    """Legacy run: оптимізує промпт → модель виконує."""
    if optimize_first:
        from dormouse.optimizer import squeeze
        prompt = squeeze(prompt)
    return generate(prompt, model=model, system=system)


# --- Parsing helpers ---


def _parse_yes_no(response: str) -> bool:
    """Парсить yes/no/так/ні з відповіді LLM."""
    token = response.strip().lower().rstrip(".!,")
    return token in ("yes", "так", "true", "1")


def _parse_json(response: str) -> dict | None:
    """Парсить JSON з відповіді LLM. Повертає None якщо не вдалося."""
    text = response.strip()
    if text.lower() in ("null", "none", ""):
        return None
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


def _type_label(tp: type) -> str:
    """Людський опис типу для промпта."""
    labels = {str: "text", int: "integer", float: "number", bool: "yes/no"}
    return labels.get(tp, tp.__name__)
