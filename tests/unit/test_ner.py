"""Tests for soynlp.ner module (skeleton functions)."""

import pytest

from soynlp.ner import is_date, is_day, is_monetary, is_number, is_period, is_time


class TestNerSkeletonFunctions:
    """All NER functions are currently NotImplementedError stubs."""

    def test_is_date_raises_not_implemented(self):
        with pytest.raises(NotImplementedError):
            is_date("2024년3월1일")

    def test_is_day_raises_not_implemented(self):
        with pytest.raises(NotImplementedError):
            is_day("월요일")

    def test_is_number_raises_not_implemented(self):
        with pytest.raises(NotImplementedError):
            is_number("123")

    def test_is_monetary_raises_not_implemented(self):
        with pytest.raises(NotImplementedError):
            is_monetary("1000원")

    def test_is_period_raises_not_implemented(self):
        with pytest.raises(NotImplementedError):
            is_period("3개월")

    def test_is_time_raises_not_implemented(self):
        with pytest.raises(NotImplementedError):
            is_time("오후3시")
