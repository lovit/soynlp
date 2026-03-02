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
