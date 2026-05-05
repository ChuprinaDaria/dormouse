"""Тести для LanguageGuard — детекція мови/формату."""

from dormouse.language_guard import LanguageGuard


class TestShouldProcess:
    """Тести для should_process() — вирішує чи обробляти message."""

    def test_ukrainian_text_processed(self):
        guard = LanguageGuard()
        assert guard.should_process("шо там по деплою") is True

    def test_english_text_skipped(self):
        guard = LanguageGuard()
        assert guard.should_process("deploy the application to production") is False

    def test_mixed_mostly_latin_skipped(self):
        """Більше 70% латиниці — skip."""
        guard = LanguageGuard()
        assert guard.should_process("deploy application with деплой") is False

    def test_mixed_mostly_cyrillic_processed(self):
        guard = LanguageGuard()
        assert guard.should_process("зроби deploy на сервер") is True

    def test_json_skipped(self):
        guard = LanguageGuard()
        assert guard.should_process('{"key": "value", "count": 1}') is False

    def test_json_array_skipped(self):
        guard = LanguageGuard()
        assert guard.should_process('[{"id": 1}, {"id": 2}]') is False

    def test_code_with_def_skipped(self):
        guard = LanguageGuard()
        code = "def hello():\n    print('world')"
        assert guard.should_process(code) is False

    def test_code_with_import_skipped(self):
        guard = LanguageGuard()
        assert guard.should_process("import os\nimport sys") is False

    def test_code_with_class_skipped(self):
        guard = LanguageGuard()
        assert guard.should_process("class MyApp:\n    pass") is False

    def test_code_with_function_skipped(self):
        guard = LanguageGuard()
        assert guard.should_process("function getData() {\n  return [];\n}") is False

    def test_code_with_const_skipped(self):
        guard = LanguageGuard()
        assert guard.should_process("const x = 42;\nconst y = 'hello';") is False

    def test_empty_string_skipped(self):
        guard = LanguageGuard()
        assert guard.should_process("") is False

    def test_none_skipped(self):
        guard = LanguageGuard()
        assert guard.should_process(None) is False

    def test_whitespace_only_skipped(self):
        guard = LanguageGuard()
        assert guard.should_process("   \n\t  ") is False


class TestSkipPatterns:
    """Тести для кастомних skip_patterns."""

    def test_skip_json_disabled(self):
        """Без json в skip_patterns — JSON з укр текстом обробляється."""
        guard = LanguageGuard(skip_patterns=())
        assert guard.should_process('{"ключ": "значення"}') is True

    def test_skip_code_disabled(self):
        """Без code в skip_patterns — код з укр коментарями обробляється."""
        guard = LanguageGuard(skip_patterns=())
        assert guard.should_process("def привіт():\n    друкувати('світ')") is True

    def test_json_with_default_patterns_skipped(self):
        guard = LanguageGuard()
        assert guard.should_process('{"ключ": "значення"}') is False

    def test_custom_patterns_only_json(self):
        guard = LanguageGuard(skip_patterns=("json",))
        assert guard.should_process('{"ключ": "значення"}') is False
        assert guard.should_process("def привіт():\n    друкувати('світ')") is True


class TestShouldProcessMessage:
    """Тести для should_process_message() — враховує роль."""

    def test_user_role_processed(self):
        guard = LanguageGuard(roles=("user",), squeeze_system=False)
        msg = {"role": "user", "content": "шо там по деплою"}
        assert guard.should_process_message(msg) is True

    def test_system_role_skipped_by_default(self):
        guard = LanguageGuard(roles=("user",), squeeze_system=False)
        msg = {"role": "system", "content": "Ти корисний асистент"}
        assert guard.should_process_message(msg) is False

    def test_system_role_processed_when_enabled(self):
        guard = LanguageGuard(roles=("user",), squeeze_system=True)
        msg = {"role": "system", "content": "Ти корисний асистент"}
        assert guard.should_process_message(msg) is True

    def test_assistant_role_skipped(self):
        guard = LanguageGuard(roles=("user",), squeeze_system=False)
        msg = {"role": "assistant", "content": "відповідь"}
        assert guard.should_process_message(msg) is False

    def test_none_content_skipped(self):
        guard = LanguageGuard(roles=("user",), squeeze_system=False)
        msg = {"role": "user", "content": None}
        assert guard.should_process_message(msg) is False

    def test_multimodal_text_extracted(self):
        """Multimodal message з текстом — обробляється."""
        guard = LanguageGuard(roles=("user",), squeeze_system=False)
        msg = {
            "role": "user",
            "content": [
                {"type": "text", "text": "шо на картинці"},
                {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}},
            ],
        }
        assert guard.should_process_message(msg) is True
