"""Тести для класифікатора стилю."""

import pytest

sklearn = pytest.importorskip("sklearn")

from dormouse.style_classifier import NoseKit, needs_cracking

TRAIN_TEXTS = [
    "Верховна Рада прийняла закон про бюджет",
    "Президент провів зустріч з міністрами",
    "Метеорологи прогнозують похолодання",
    "Ну шо там по баґу, пофікси плз",
    "Крч зробив але не працює, хз",
    "Блін капець якийсь нічо не зрозуміло",
    "Ваще не понятно шо получається",
    "Канєшно понял, надо ісправити",
    "Єслі чесно ваще не понял",
    "Коли дедлайн був вчора лол",
    "git push --force і молитва",
    "PM: це ж просто кнопка",
]

TRAIN_LABELS = (
    ["норм"] * 3 +
    ["розмовна"] * 3 +
    ["суржик"] * 3 +
    ["мем"] * 3
)


class TestNoseKit:
    def test_train_and_predict(self):
        clf = NoseKit()
        clf.train(TRAIN_TEXTS, TRAIN_LABELS)
        assert clf.is_trained

        result = clf.predict("Верховна Рада затвердила бюджет")
        assert result in ["норм", "розмовна", "суржик", "мем"]

    def test_predict_proba(self):
        clf = NoseKit()
        clf.train(TRAIN_TEXTS, TRAIN_LABELS)

        probs = clf.predict_proba("Ваще не понятно шо получається")
        assert len(probs) == 4
        assert sum(probs.values()) == pytest.approx(1.0, abs=0.01)

    def test_save_load(self, tmp_path):
        clf = NoseKit()
        clf.train(TRAIN_TEXTS, TRAIN_LABELS)
        model_path = tmp_path / "test_model.pkl"
        clf.save(model_path)

        clf2 = NoseKit()
        clf2.load(model_path)
        assert clf2.is_trained
        assert clf2.predict("тест") == clf.predict("тест")

    def test_not_trained_raises(self):
        clf = NoseKit()
        with pytest.raises(RuntimeError):
            clf.predict("тест")


class TestNeedsCracking:
    def test_surzhyk_needs(self):
        assert needs_cracking("шо там получається") is True

    def test_clean_text_no(self):
        assert needs_cracking("Це нормальний текст") is False
