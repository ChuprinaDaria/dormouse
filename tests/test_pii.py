"""Тести для pii.py — PII-скраб датасет-пайплайну."""

from dormouse.pii import _luhn_ok, scrub, scrub_pair


class TestPhone:
    def test_intl_format(self):
        assert scrub("телефонуй +380671234567 завтра").text == "телефонуй <PHONE> завтра"

    def test_dashed(self):
        assert "<PHONE>" in scrub("мій номер 067-123-45-67").text

    def test_parens(self):
        assert "<PHONE>" in scrub("(067) 123 45 67").text

    def test_plain_number_not_phone(self):
        assert scrub("замовлення 12345").text == "замовлення 12345"


class TestEmail:
    def test_basic(self):
        assert scrub("пиши на test.user+tag@example.com").text == "пиши на <EMAIL>"


class TestIban:
    def test_ua_iban(self):
        iban = "UA" + "1" * 27
        assert scrub(f"рахунок {iban}").text == "рахунок <IBAN>"

    def test_short_not_iban(self):
        text = "код UA" + "1" * 26
        assert "<IBAN>" not in scrub(text).text


class TestCard:
    def test_luhn_valid_card(self):
        # тестовий Visa номер (Luhn-валідний)
        assert scrub("картка 4111 1111 1111 1111").text == "картка <CARD>"

    def test_luhn_invalid_not_card(self):
        assert "<CARD>" not in scrub("число 4111 1111 1111 1112").text

    def test_luhn_helper(self):
        assert _luhn_ok("4111111111111111")
        assert not _luhn_ok("4111111111111112")


class TestUrl:
    def test_tokened_url_fully_replaced(self):
        res = scrub("дивись https://example.com/reset?token=abc123 швидко")
        assert res.text == "дивись <URL> швидко"

    def test_plain_url_keeps_domain(self):
        res = scrub("дивись https://example.com/some/path/here")
        assert "https://example.com" in res.text
        assert "/some/path" not in res.text

    def test_bare_domain_untouched(self):
        res = scrub("сайт https://example.com живий")
        assert "https://example.com" in res.text


class TestHandle:
    def test_telegram_handle(self):
        assert scrub("пиши @some_user_123 в тг").text == "пиши <HANDLE> в тг"

    def test_email_not_double_scrubbed(self):
        # email обробляється раніше за хендли
        assert scrub("a@b.co").text == "<EMAIL>"


class TestAddr:
    def test_street(self):
        assert "<ADDR>" in scrub("живу на вул. Шевченка, буд. 12").text


class TestNames:
    def test_name_mid_sentence(self):
        res = scrub("подзвони Оксані завтра")
        assert "<NAME>" in res.text
        assert "Оксані" not in res.text

    def test_lowercase_not_name(self):
        assert "<NAME>" not in scrub("подзвони оксані завтра").text

    def test_names_disabled(self):
        assert "Оксані" in scrub("подзвони Оксані", names=False).text


class TestScrubPair:
    def test_pair_scrubbed_both_sides(self):
        pair = scrub_pair("дзвони +380671234567", "телефонуй +380671234567")
        assert pair == ("дзвони <PHONE>", "телефонуй <PHONE>")

    def test_pair_dropped_when_mostly_placeholders(self):
        assert scrub_pair("+380671234567", "a@b.co") is None

    def test_clean_pair_unchanged(self):
        pair = scrub_pair("шо там по багу", "що там з багом")
        assert pair == ("шо там по багу", "що там з багом")


class TestScrubResult:
    def test_counts(self):
        res = scrub("пиши a@b.co або +380671234567")
        assert res.counts == {"<EMAIL>": 1, "<PHONE>": 1}
        assert res.changed

    def test_no_pii(self):
        res = scrub("звичайний текст")
        assert not res.changed
