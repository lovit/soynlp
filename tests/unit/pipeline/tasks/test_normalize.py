import pytest

from soynlp.pipeline.tasks.normalize import NormalizeTask, NormalizeTaskArgs


class TestNormalizeTask:
    def test_basic_normalize(self):
        examples = [{"text": "안녕하세요!!!! 반갑습니다ㅋㅋㅋㅋㅋ"}]
        task = NormalizeTask(NormalizeTaskArgs())
        result = task({"corpus": examples})
        assert len(result["corpus"]) == 1
        normalized_text = result["corpus"][0]["text"]
        assert "ㅋㅋㅋㅋㅋ" not in normalized_text

    def test_preserves_other_fields(self):
        examples = [{"text": "테스트", "label": True}]
        task = NormalizeTask(NormalizeTaskArgs())
        result = task({"corpus": examples})
        assert result["corpus"][0]["label"] is True

    def test_custom_keys(self):
        examples = [{"content": "테스트!!!"}]
        task = NormalizeTask(NormalizeTaskArgs(in_key="data", text_key="content", out_key="data"))
        result = task({"data": examples})
        assert "content" in result["data"][0]

    def test_missing_key(self):
        task = NormalizeTask(NormalizeTaskArgs())
        with pytest.raises(ValueError, match="Not found"):
            task({})

    def test_repeat_char_removal(self):
        examples = [{"text": "ㅋㅋㅋㅋㅋㅋ 웃겨요"}]
        task = NormalizeTask(NormalizeTaskArgs(remove_repeatchar=2))
        result = task({"corpus": examples})
        assert result["corpus"][0]["text"].count("ㅋ") <= 2
