"""Тести для CLI."""

from click.testing import CliRunner

from dormouse.cli import main


class TestCLI:
    def test_squeeze_basic(self):
        runner = CliRunner()
        result = runner.invoke(main, ["squeeze", "шо там"])
        assert result.exit_code == 0
        assert "що" in result.output

    def test_squeeze_verbose(self):
        runner = CliRunner()
        result = runner.invoke(main, ["squeeze", "-v", "шо там"])
        assert result.exit_code == 0
        assert "замін:" in result.output

    def test_squeeze_pipe(self):
        runner = CliRunner()
        result = runner.invoke(main, ["squeeze"], input="шо там\n")
        assert result.exit_code == 0
        assert "що" in result.output

    def test_squeeze_file(self, tmp_path):
        f = tmp_path / "input.txt"
        f.write_text("шо там\nваще норм\n", encoding="utf-8")
        runner = CliRunner()
        result = runner.invoke(main, ["squeeze", "-i", str(f)])
        assert result.exit_code == 0
        assert "що" in result.output

    def test_squeeze_target_cloud(self):
        runner = CliRunner()
        result = runner.invoke(main, ["squeeze", "-t", "cloud", "привіт"])
        assert result.exit_code == 0

    def test_version(self):
        runner = CliRunner()
        result = runner.invoke(main, ["--version"])
        assert result.exit_code == 0
        assert "0.2.0" in result.output
