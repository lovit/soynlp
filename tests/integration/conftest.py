"""Shared fixtures for integration tests."""

import os

import pytest

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
EXAMPLES_DIR = os.path.join(os.path.dirname(__file__), "examples")
NEWS_DATA = os.path.join(ROOT_DIR, "tests/integration/data/news-text/2016-10-20.jsonl")
REVIEW_DATA = os.path.join(ROOT_DIR, "tests/integration/data/movie-review-score/91031.jsonl")


@pytest.fixture
def news_data_path():
    return NEWS_DATA


@pytest.fixture
def review_data_path():
    return REVIEW_DATA


@pytest.fixture
def root_dir():
    return ROOT_DIR


def answer_path(example_name: str, filename: str) -> str:
    return os.path.join(EXAMPLES_DIR, example_name, "answers", filename)


def read_answer(example_name: str, filename: str) -> str:
    path = answer_path(example_name, filename)
    with open(path, encoding="utf-8") as f:
        return f.read()


def read_answer_lines(example_name: str, filename: str) -> list[str]:
    return [line for line in read_answer(example_name, filename).splitlines() if line.strip()]
