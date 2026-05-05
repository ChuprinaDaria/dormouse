"""E2E тести middleware з реальним OpenAI / Anthropic API.

Запускати: pytest tests/test_middleware_e2e.py -m e2e -v
Потребує: OPENAI_API_KEY та/або ANTHROPIC_API_KEY в env.
"""

import os

import pytest

from dormouse.middleware import DormouseClient

pytestmark = pytest.mark.e2e

# --- OpenAI E2E ---

openai_available = False
try:
    if os.environ.get("OPENAI_API_KEY"):
        from openai import OpenAI
        openai_available = True
except ImportError:
    pass


@pytest.fixture
def client():
    if not openai_available:
        pytest.skip("OPENAI_API_KEY not set or openai not installed")
    return DormouseClient(OpenAI(), target="cloud")


class TestE2ESync:
    def test_basic_roundtrip(self, client):
        """Повний roundtrip: укр → squeeze → GPT → unsqueeze → укр."""
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": "Поясни що таке git rebase одним реченням"},
            ],
        )
        content = response.choices[0].message.content
        assert content is not None
        assert len(content) > 10

    def test_surzhyk_roundtrip(self, client):
        """Суржик нормалізується перед відправкою."""
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": "шо таке docker, поясни коротко"},
            ],
        )
        content = response.choices[0].message.content
        assert content is not None
        assert len(content) > 10


class TestE2EStreaming:
    def test_streaming_roundtrip(self, client):
        """Streaming roundtrip — збираємо всі чанки."""
        chunks = []
        for chunk in client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": "Назви три мови програмування"},
            ],
            stream=True,
        ):
            content = chunk.choices[0].delta.content
            if content:
                chunks.append(content)

        full_response = "".join(chunks)
        assert len(full_response) > 10


# --- Anthropic E2E ---

anthropic_available = False
try:
    if os.environ.get("ANTHROPIC_API_KEY"):
        from anthropic import Anthropic
        anthropic_available = True
except ImportError:
    pass


@pytest.fixture
def anthropic_client():
    if not anthropic_available:
        pytest.skip("ANTHROPIC_API_KEY not set or anthropic not installed")
    return DormouseClient(Anthropic(), target="cloud")


class TestE2EAnthropic:
    def test_basic_roundtrip(self, anthropic_client):
        """Повний roundtrip: укр → squeeze → Claude → unsqueeze → укр."""
        response = anthropic_client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=256,
            messages=[
                {"role": "user", "content": "Поясни що таке git rebase одним реченням"},
            ],
        )
        content = response.content[0].text
        assert content is not None
        assert len(content) > 10

    def test_streaming_roundtrip(self, anthropic_client):
        """Streaming roundtrip — збираємо всі чанки."""
        chunks = []
        for event in anthropic_client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=256,
            messages=[
                {"role": "user", "content": "Назви три мови програмування"},
            ],
            stream=True,
        ):
            if (
                event.type == "content_block_delta"
                and hasattr(event.delta, "text")
            ):
                chunks.append(event.delta.text)

        full_response = "".join(chunks)
        assert len(full_response) > 10

    def test_system_prompt(self, anthropic_client):
        """System prompt передається правильно."""
        response = anthropic_client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=128,
            system="Відповідай одним словом.",
            messages=[
                {"role": "user", "content": "Яка столиця Франції?"},
            ],
        )
        content = response.content[0].text
        assert content is not None
