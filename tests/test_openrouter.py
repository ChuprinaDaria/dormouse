"""Тести для OpenRouter helper."""

from dormouse.middleware import openrouter_headers


class TestOpenRouterHeaders:
    def test_both_params(self):
        headers = openrouter_headers(referer="https://myapp.com", title="MyApp")
        assert headers == {"HTTP-Referer": "https://myapp.com", "X-Title": "MyApp"}

    def test_referer_only(self):
        headers = openrouter_headers(referer="https://myapp.com")
        assert headers == {"HTTP-Referer": "https://myapp.com"}

    def test_title_only(self):
        headers = openrouter_headers(title="MyApp")
        assert headers == {"X-Title": "MyApp"}

    def test_empty_returns_empty(self):
        headers = openrouter_headers()
        assert headers == {}
