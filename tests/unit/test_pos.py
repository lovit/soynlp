"""Tests for soynlp.pos module."""

import os
import tempfile

from soynlp.pos import load_default_adverbs, stem_to_adverb


class TestStemToAdverb:
    def test_stem_ending_with_ha(self):
        result = stem_to_adverb({"조용하", "빠르"})
        # Only "조용하" ends with "하", so only "조용히" should be produced
        assert "조용히" in result
        assert len(result) == 1

    def test_single_stem_string(self):
        result = stem_to_adverb("조용하")
        assert "조용히" in result

    def test_no_matching_stems(self):
        result = stem_to_adverb({"먹", "가"})
        assert len(result) == 0

    def test_custom_suffix(self):
        result = stem_to_adverb({"조용하"}, suffixes="히게")
        # stem "조용하" -> "조용히", "조용게"
        assert "조용히" in result
        assert "조용게" in result
        assert len(result) == 2

    def test_empty_stems(self):
        result = stem_to_adverb(set())
        assert len(result) == 0

    def test_multiple_ha_stems(self):
        result = stem_to_adverb({"조용하", "깨끗하", "단단하"})
        assert "조용히" in result
        assert "깨끗히" in result
        assert "단단히" in result
        assert len(result) == 3


class TestLoadDefaultAdverbs:
    def test_load_from_custom_path(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write("빨리 100\n")
            f.write("천천히 50\n")
            f.write("조용히 30\n")
            path = f.name

        try:
            adverbs = load_default_adverbs(path)
            assert isinstance(adverbs, set)
            assert "빨리" in adverbs
            assert "천천히" in adverbs
            assert "조용히" in adverbs
            assert len(adverbs) == 3
        finally:
            os.unlink(path)

    def test_load_default_path(self):
        """Default dictionary file should exist and load successfully."""
        adverbs = load_default_adverbs()
        assert isinstance(adverbs, set)
        assert len(adverbs) > 0


class TestNewsPOSExtractor:
    def test_import(self):
        from soynlp.pos import NewsPOSExtractor

        assert NewsPOSExtractor is not None

    def test_init(self):
        from soynlp.pos import NewsPOSExtractor

        extractor = NewsPOSExtractor(verbose=False)
        assert extractor._verbose is False
        assert extractor._ensure_normalized is True
        assert extractor._extract_eomi is True


class TestChatPOSExtractor:
    def test_import(self):
        from soynlp.pos import ChatPOSExtractor

        assert ChatPOSExtractor is not None

    def test_inherits_news(self):
        from soynlp.pos import ChatPOSExtractor, NewsPOSExtractor

        assert issubclass(ChatPOSExtractor, NewsPOSExtractor)

    def test_init(self):
        from soynlp.pos import ChatPOSExtractor

        extractor = ChatPOSExtractor(verbose=False)
        assert extractor._verbose is False
