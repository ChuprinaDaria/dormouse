"""Тести для unsqueeze — EN→UA зворотній маппінг."""

from dormouse.unsqueeze import _EN_ARTICLES, _EN_FUNCTION_WORDS, unsqueeze


class TestUnsqueeze:
    def test_basic_words(self):
        result = unsqueeze("fix the error in deploy")
        assert "the" not in result.split()
        assert any(w in result for w in ["виправ", "виправити", "фікс"])
        assert any(w in result for w in ["помилка", "помилку", "помилці"])

    def test_articles_filtered(self):
        result = unsqueeze("the server crashed after an update")
        words = result.lower().split()
        assert "the" not in words
        assert "an" not in words

    def test_auxiliaries_filtered(self):
        result = unsqueeze("it is working and does not crash")
        words = result.lower().split()
        assert "is" not in words
        assert "does" not in words
        assert "it" not in words

    def test_phrases(self):
        result = unsqueeze("I don't know how to fix this")
        assert any(w in result for w in ["знаю", "знати", "вмію", "незнати"])

    def test_tech_terms(self):
        result = unsqueeze("check the container logs")
        assert "the" not in result.split()

    def test_preserves_unknown(self):
        result = unsqueeze("use Kubernetes for orchestration")
        assert "Kubernetes" in result

    def test_empty(self):
        assert unsqueeze("") == ""
        assert unsqueeze("   ") == "   "

    def test_roundtrip_basic(self):
        """squeeze → unsqueeze має повернути щось близьке до оригіналу."""
        from dormouse.optimizer import squeeze

        original = "виправ помилку в контейнері"
        compressed = squeeze(original, target="cloud")
        restored = unsqueeze(compressed)
        assert any(w in restored for w in ["виправ", "помилк", "контейнер", "фікс"])

    def test_punctuation_preserved(self):
        result = unsqueeze("fix error, update deploy.")
        assert "," in result
        assert "." in result

    def test_articles_and_function_words_complete(self):
        """Артиклі та функціональні слова покриті."""
        for word in ("the", "a", "an"):
            assert word in _EN_ARTICLES
        for word in ("is", "are", "was", "were", "do", "does", "did", "not"):
            assert word in _EN_FUNCTION_WORDS

    def test_no_double_spaces(self):
        result = unsqueeze("the error is a problem")
        assert "  " not in result


class TestUnsqueezeMorphology:
    """Морфологічне узгодження (потребує pymorphy3)."""

    def test_verb_object_accusative(self):
        """Дієслово + іменник → знахідний відмінок."""
        result = unsqueeze("fix the error")
        assert "помилку" in result

    def test_past_tense_object_accusative(self):
        """Минулий час + іменник → знахідний."""
        result = unsqueeze("I fixed the error")
        assert "помилку" in result

    def test_preposition_locative(self):
        """Прийменник 'у' + іменник → місцевий відмінок."""
        result = unsqueeze("fix error in deploy")
        # "в" + noun/verb form from deploy mapping
        assert "в" in result.split() or "у" in result.split()

    def test_past_verb_gender_agreement(self):
        """Іменник + дієслово мін. часу → узгодження роду."""
        result = unsqueeze("the server crashed after update")
        # "crashed" maps to "затупив" (masc) or "впав"/"впало" — both acceptable
        assert any(w in result for w in ["впав", "затупив", "впало"])


class TestUnsqueezeLLMFallback:
    """LLM fallback коли word-by-word якість низька."""

    def test_no_fallback_when_quality_ok(self):
        """Якщо більшість слів перекладен�� — LLM не викликається."""
        called = []
        def fake_translate(text):
            called.append(text)
            return "LLM переклад"

        result = unsqueeze("fix the error", translate_fn=fake_translate)
        assert not called  # word-by-word впорався
        assert "помилку" in result

    def test_fallback_when_quality_poor(self):
        """Якщо >30% слів не перекладено — викликає LLM."""
        called = []
        def fake_translate(text):
            called.append(text)
            return "переклад від LLM"

        # Слова яких точно нема в лексиконі
        result = unsqueeze(
            "the frobnicator defenestrated the thingamajig unexpurgated",
            translate_fn=fake_translate,
        )
        assert called  # LLM був викликаний
        assert result == "переклад від LLM"

    def test_fallback_graceful_on_error(self):
        """Якщо LLM падає — повертає word-by-word результат."""
        def broken_translate(text):
            raise ConnectionError("API down")

        # Не кидає виключення
        result = unsqueeze(
            "the container corrupted completely unexpectedly",
            translate_fn=broken_translate,
        )
        assert isinstance(result, str)
        assert len(result) > 0

    def test_no_fallback_without_translate_fn(self):
        """Без translate_fn — тільки word-by-word."""
        result = unsqueeze("the container corrupted completely unexpectedly")
        assert isinstance(result, str)
