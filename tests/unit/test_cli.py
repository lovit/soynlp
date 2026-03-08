"""Unit tests for CLI entry point."""

from unittest.mock import patch

import pytest

from soynlp.cli import main


def test_no_command_prints_usage(capsys):
    """Without subcommand, main() prints usage and exits normally."""
    with patch("sys.argv", ["soynlp"]):
        main()
    captured = capsys.readouterr()
    assert "usage:" in captured.out.lower()


def test_version_flag(capsys):
    """--version flag prints version and exits."""
    with patch("sys.argv", ["soynlp", "--version"]):
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "soynlp" in captured.out


def test_pipeline_requires_config():
    """pipeline subcommand without -c should exit with error."""
    with patch("sys.argv", ["soynlp", "pipeline"]):
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 2


def test_pipeline_calls_run():
    """pipeline -c calls Pipeline.run with the config path."""
    with patch("sys.argv", ["soynlp", "pipeline", "-c", "test.yaml"]):
        with patch("soynlp.cli.Pipeline.run") as mock_run:
            main()
            mock_run.assert_called_once_with("test.yaml")
