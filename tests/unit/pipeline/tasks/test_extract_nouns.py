import pytest

from soynlp.pipeline.tasks.extract_nouns import ExtractNounTask, ExtractNounTaskArgs


class TestExtractNounTask:
    def test_missing_key(self):
        task = ExtractNounTask(ExtractNounTaskArgs(verbose=False))
        with pytest.raises(ValueError, match="Not found"):
            task({})

    @pytest.mark.slow
    def test_extract_nouns(self):
        examples = [{"text": "이것은 예문입니다"}] * 100
        task = ExtractNounTask(ExtractNounTaskArgs(verbose=False, extract_compounds=False))
        result = task({"corpus": examples})
        assert "nouns" in result
        assert isinstance(result["nouns"], dict)

    @pytest.mark.slow
    def test_extract_nouns_n_workers(self):
        """n_workers=2로 실행해도 결과가 동일하다."""
        examples = [{"text": "이것은 예문입니다"}] * 200
        task_single = ExtractNounTask(ExtractNounTaskArgs(verbose=False, extract_compounds=False, n_workers=1))
        task_multi = ExtractNounTask(ExtractNounTaskArgs(verbose=False, extract_compounds=False, n_workers=2))
        result_single = task_single({"corpus": examples})
        result_multi = task_multi({"corpus": examples})
        assert set(result_single["nouns"].keys()) == set(result_multi["nouns"].keys())
