import pytest

from soynlp.pipeline.tasks.extract_words import ExtractWordTask, ExtractWordTaskArgs


class TestExtractWordTask:
    def test_missing_key(self):
        task = ExtractWordTask(ExtractWordTaskArgs(verbose=False))
        with pytest.raises(ValueError, match="Not found"):
            task({})

    @pytest.mark.slow
    def test_extract_words_cohesion_only(self):
        examples = [{"text": "이것은 예문입니다 반갑습니다"}] * 50
        task = ExtractWordTask(ExtractWordTaskArgs(verbose=False, extract_cohesion_only=True, min_frequency=1))
        result = task({"corpus": examples})
        assert "word_cohesion" in result
        assert isinstance(result["word_cohesion"], dict)
