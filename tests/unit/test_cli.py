"""Unit tests for CLI entry point."""

import argparse
from unittest.mock import patch

import pytest

from soynlp.cli import _run_task_extract_nouns, _run_task_list, main
from soynlp.pipeline.tasks import TASK_REGISTRY


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


class TestTaskList:
    def test_prints_sorted_task_names(self, capsys):
        _run_task_list(argparse.Namespace())
        output = capsys.readouterr().out.strip().splitlines()
        assert output == sorted(TASK_REGISTRY)

    def test_all_registry_tasks_present(self, capsys):
        _run_task_list(argparse.Namespace())
        output = capsys.readouterr().out.strip().splitlines()
        assert set(output) == set(TASK_REGISTRY)

    def test_via_main(self, capsys):
        with patch("sys.argv", ["soynlp", "task", "list"]):
            main()
        output = capsys.readouterr().out.strip()
        assert "ExtractNoun" in output
        assert "ReadText" in output


class TestTaskExtractNouns:
    def _args(self, tmp_path, **kwargs):
        input_file = tmp_path / "input.txt"
        output_file = tmp_path / "output.txt"
        text = kwargs.pop("text", "자연어 처리는 재미있다\n한국어 자연어 처리 연구\n")
        input_file.write_text(text, encoding="utf-8")
        defaults = dict(
            input=str(input_file),
            output=str(output_file),
            min_noun_score=0.3,
            min_noun_frequency=1,
            min_eojeol_frequency=1,
            extract_compounds=True,
            verbose=False,
        )
        defaults.update(kwargs)
        return argparse.Namespace(**defaults)

    def test_writes_output_file(self, tmp_path):
        args = self._args(tmp_path)
        _run_task_extract_nouns(args)
        assert (tmp_path / "output.txt").exists()

    def test_output_format_is_tsv(self, tmp_path):
        args = self._args(tmp_path, text="자연어 처리는 재미있다\n" * 10)
        _run_task_extract_nouns(args)
        lines = (tmp_path / "output.txt").read_text(encoding="utf-8").strip().splitlines()
        for line in lines:
            parts = line.split("\t")
            assert len(parts) == 3, f"Expected 3 tab-separated fields: {line!r}"
            _, freq, score = parts
            assert int(freq) >= 1
            assert 0.0 <= float(score) <= 1.0

    def test_output_sorted_by_frequency_descending(self, tmp_path):
        args = self._args(tmp_path, text="자연어 처리\n" * 20 + "한국어\n" * 5)
        _run_task_extract_nouns(args)
        lines = (tmp_path / "output.txt").read_text(encoding="utf-8").strip().splitlines()
        freqs = [int(line.split("\t")[1]) for line in lines]
        assert freqs == sorted(freqs, reverse=True)

    def test_min_noun_frequency_filters_results(self, tmp_path):
        text = "자연어 처리\n" * 5 + "한국어\n" * 1

        args_strict = self._args(tmp_path, text=text, min_noun_frequency=3)
        _run_task_extract_nouns(args_strict)
        strict_words = {line.split("\t")[0] for line in (tmp_path / "output.txt").read_text().strip().splitlines() if line}

        args_loose = self._args(tmp_path, text=text, min_noun_frequency=1)
        _run_task_extract_nouns(args_loose)
        loose_words = {line.split("\t")[0] for line in (tmp_path / "output.txt").read_text().strip().splitlines() if line}

        assert strict_words <= loose_words

    def test_via_main(self, tmp_path):
        input_file = tmp_path / "input.txt"
        output_file = tmp_path / "output.txt"
        input_file.write_text("자연어 처리는 재미있다\n" * 5, encoding="utf-8")
        with patch(
            "sys.argv", ["soynlp", "task", "extract-nouns", "-i", str(input_file), "-o", str(output_file), "--no-verbose"]
        ):
            main()
        assert output_file.exists()

    def test_no_extract_compounds_flag(self, tmp_path):
        input_file = tmp_path / "input.txt"
        output_file = tmp_path / "output.txt"
        input_file.write_text("자연어 처리는 재미있다\n" * 5, encoding="utf-8")
        with patch(
            "sys.argv",
            [
                "soynlp",
                "task",
                "extract-nouns",
                "-i",
                str(input_file),
                "-o",
                str(output_file),
                "--no-extract-compounds",
                "--no-verbose",
            ],
        ):
            main()
        assert output_file.exists()
