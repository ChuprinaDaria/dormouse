"""Тести для CLI search команд (stir, mumble, sip)."""

from click.testing import CliRunner

from dormouse.cli import main


class TestStirCLI:
    def test_stir_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["stir", "--help"])
        assert result.exit_code == 0
        assert "Завантажує і індексує файл" in result.output

    def test_stir_missing_file(self):
        runner = CliRunner()
        result = runner.invoke(main, ["stir", "nonexistent.txt"])
        assert result.exit_code != 0


class TestMumbleCLI:
    def test_mumble_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["mumble", "--help"])
        assert result.exit_code == 0
        assert "Семантичний пошук" in result.output


class TestSipCLI:
    def test_sip_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["sip", "--help"])
        assert result.exit_code == 0
        assert "Сортує текст по темах" in result.output

    def test_sip_requires_topics(self, tmp_path):
        f = tmp_path / "input.txt"
        f.write_text("тест\n", encoding="utf-8")
        runner = CliRunner()
        result = runner.invoke(main, ["sip", str(f)])
        assert result.exit_code != 0
        assert "topics" in result.output.lower() or "Missing" in result.output
