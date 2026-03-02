import pytest

from soynlp.pipeline.tasks.tokenize import TokenizeTask, TokenizeTaskArgs


class TestTokenizeTask:
    def test_regex_tokenizer(self):
        examples = [{"text": "이것은 test123 입니다"}]
        task = TokenizeTask(TokenizeTaskArgs(tokenizer_type="regex"))
        result = task({"corpus": examples})
        tokens = result["corpus"][0]["tokens"]
        assert isinstance(tokens, list)
        assert len(tokens) > 0

    def test_max_score_tokenizer(self):
        scores = {"이것": 1.0, "예문": 1.0}
        examples = [{"text": "이것은 예문입니다"}]
        task = TokenizeTask(TokenizeTaskArgs(tokenizer_type="max_score", scores_key="scores"))
        result = task({"corpus": examples, "scores": scores})
        assert "tokens" in result["corpus"][0]

    def test_noun_match_tokenizer(self):
        scores = {"이것": 1.0, "예문": 1.0}
        examples = [{"text": "이것은 예문입니다"}]
        task = TokenizeTask(TokenizeTaskArgs(tokenizer_type="noun_match", scores_key="scores"))
        result = task({"corpus": examples, "scores": scores})
        assert "tokens" in result["corpus"][0]

    def test_l_tokenizer(self):
        scores = {"이것": 1.0, "예문": 0.5}
        examples = [{"text": "이것은 예문입니다"}]
        task = TokenizeTask(TokenizeTaskArgs(tokenizer_type="l_tokenizer", scores_key="scores"))
        result = task({"corpus": examples, "scores": scores})
        assert "tokens" in result["corpus"][0]

    def test_namedtuple_score_extraction(self):
        from collections import namedtuple

        NounScore = namedtuple("NounScore", "frequency score")
        scores = {"이것": NounScore(100, 0.8), "예문": NounScore(50, 0.7)}
        examples = [{"text": "이것은 예문입니다"}]
        task = TokenizeTask(TokenizeTaskArgs(tokenizer_type="max_score", scores_key="nouns", score_field="score"))
        result = task({"corpus": examples, "nouns": scores})
        assert "tokens" in result["corpus"][0]

    def test_missing_corpus_key(self):
        task = TokenizeTask(TokenizeTaskArgs())
        with pytest.raises(ValueError, match="Not found"):
            task({})

    def test_missing_scores_key(self):
        examples = [{"text": "이것은 예문입니다"}]
        task = TokenizeTask(TokenizeTaskArgs(tokenizer_type="max_score"))
        with pytest.raises(ValueError, match="Not found"):
            task({"corpus": examples})

    def test_unknown_tokenizer_type(self):
        examples = [{"text": "테스트"}]
        task = TokenizeTask(TokenizeTaskArgs(tokenizer_type="unknown"))
        with pytest.raises(ValueError, match="Unknown tokenizer_type"):
            task({"corpus": examples})
