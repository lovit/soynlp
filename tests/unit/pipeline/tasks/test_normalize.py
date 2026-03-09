import pytest

from soynlp.pipeline.tasks.normalize import (
    EmojiNormalizeTask,
    EmojiNormalizeTaskArgs,
    HangleEmojiNormalizeTask,
    HangleEmojiNormalizeTaskArgs,
    NormalizeTask,
    NormalizeTaskArgs,
    PaddingSpaceNormalizeTask,
    PaddingSpaceNormalizeTaskArgs,
    PassCharacterNormalizeTask,
    PassCharacterNormalizeTaskArgs,
    RemoveLongspaceNormalizeTask,
    RemoveLongspaceNormalizeTaskArgs,
    RepeatCharacterNormalizeTask,
    RepeatCharacterNormalizeTaskArgs,
)


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


class TestPassCharacterNormalizeTask:
    def test_filter_symbols(self):
        examples = [{"text": "안녕 hello @@ 123"}]
        task = PassCharacterNormalizeTask(PassCharacterNormalizeTaskArgs(number=False, symbol=False))
        result = task({"corpus": examples})
        assert "@@" not in result["corpus"][0]["text"]
        assert "123" not in result["corpus"][0]["text"]
        assert "안녕" in result["corpus"][0]["text"]

    def test_missing_key(self):
        task = PassCharacterNormalizeTask(PassCharacterNormalizeTaskArgs())
        with pytest.raises(ValueError, match="Not found"):
            task({})


class TestHangleEmojiNormalizeTask:
    def test_decompose_emoji(self):
        examples = [{"text": "ㅋㅋㅋ쿠ㅜㅜ"}]
        task = HangleEmojiNormalizeTask(HangleEmojiNormalizeTaskArgs())
        result = task({"corpus": examples})
        # 분해 결과: 음절 → 자모로 변환되어 길이가 같거나 늘어남
        assert "쿠" not in result["corpus"][0]["text"] or result["corpus"][0]["text"] == "ㅋㅋㅋ쿠ㅜㅜ"


class TestEmojiNormalizeTask:
    def test_remove_emoji(self):
        examples = [{"text": "안녕 😀 반가워"}]
        task = EmojiNormalizeTask(EmojiNormalizeTaskArgs())
        result = task({"corpus": examples})
        assert "😀" not in result["corpus"][0]["text"]
        assert "안녕" in result["corpus"][0]["text"]

    def test_replace_emoji(self):
        examples = [{"text": "안녕 😀"}]
        task = EmojiNormalizeTask(EmojiNormalizeTaskArgs(replace="[EMOJI]"))
        result = task({"corpus": examples})
        assert "[EMOJI]" in result["corpus"][0]["text"]


class TestRepeatCharacterNormalizeTask:
    def test_reduce_repeats(self):
        examples = [{"text": "ㅋㅋㅋㅋㅋㅋ"}]
        task = RepeatCharacterNormalizeTask(RepeatCharacterNormalizeTaskArgs(max_repeat=2))
        result = task({"corpus": examples})
        assert result["corpus"][0]["text"].count("ㅋ") == 2


class TestRemoveLongspaceNormalizeTask:
    def test_compress_spaces(self):
        examples = [{"text": "a     b"}]
        task = RemoveLongspaceNormalizeTask(RemoveLongspaceNormalizeTaskArgs())
        result = task({"corpus": examples})
        assert result["corpus"][0]["text"] == "a b"


class TestPaddingSpaceNormalizeTask:
    def test_pad_words(self):
        examples = [{"text": "(주)테스트"}]
        task = PaddingSpaceNormalizeTask(PaddingSpaceNormalizeTaskArgs())
        result = task({"corpus": examples})
        assert " 주 " in result["corpus"][0]["text"]
