import json
import os
from typing import Protocol, runtime_checkable


@runtime_checkable
class DictionaryProtocol(Protocol):
    """커스텀 사전 주입을 위한 인터페이스.

    이 프로토콜을 구현하면 `EojeolTemplateMatcher`, `LRTemplateMatcher` 등에
    도메인 특화 사전을 주입할 수 있다.

    Example:
        >>> class MyMedicalDictionary:
        ...     max_length = 10
        ...
        ...     def get_pos(self, word: str) -> list[str]:
        ...         return ["Noun"] if word in {"암", "세포", "항체"} else []
        ...
        ...     def word_is_tag(self, word: str, tag: str) -> bool:
        ...         return tag == "Noun" and word in {"암", "세포", "항체"}
        ...
        >>> matcher = EojeolTemplateMatcher(MyMedicalDictionary())
    """

    max_length: int

    def get_pos(self, word: str) -> list[str]:
        """단어의 품사 태그 목록을 반환한다."""
        ...

    def word_is_tag(self, word: str, tag: str) -> bool:
        """단어가 주어진 품사 태그에 해당하는지 반환한다."""
        ...


class Dictionary:
    def __init__(self, pos_dict: dict[str, set[str]] | str) -> None:
        if isinstance(pos_dict, dict):
            for key in pos_dict:
                pos_dict[key] = set(pos_dict[key])
            self.pos_dict = pos_dict
            self.max_length = self._check_max_length(self.pos_dict)
        elif isinstance(pos_dict, str):
            if os.path.exists(pos_dict):
                self.load(pos_dict)
            else:
                raise ValueError("dictionary file does not exist")

    def __repr__(self) -> str:
        num_tags = len(self.pos_dict)
        num_words = sum(len(words) for words in self.pos_dict.values())
        tags = list(self.pos_dict.keys())
        return f"Dictionary(num_tags={num_tags}, num_words={num_words}, tags={tags})"

    def _check_max_length(self, pos_dict: dict[str, set[str]]) -> int:
        return max(len(word) for words in pos_dict.values() for word in words)

    def get_pos(self, word: str) -> list[str]:
        tags: list[str] = []
        for pos, words in self.pos_dict.items():
            if word in words:
                tags.append(pos)
        return tags

    def word_is_tag(self, word: str, tag: str) -> bool:
        return word in self.pos_dict.get(tag, set())

    def add_words(self, tag: str, words: set[str] | str, force: bool = False) -> None:
        words = self._type_check(words)

        if not force and tag not in self.pos_dict:
            raise ValueError("Check your tag or use add_words(tag, words, force=True)")

        max_length = max(len(word) for word in words)
        if self.max_length < max_length:
            self.max_length = max_length

        if tag not in self.pos_dict:
            dictionary = words
        else:
            dictionary = self.pos_dict.get(tag, set())
            dictionary.update(words)
        self.pos_dict[tag] = dictionary

    def remove_words(self, tag: str, words: set[str] | str | None = None) -> None:
        if tag not in self.pos_dict:
            raise ValueError(f"tag {tag} does not exist")

        if words is None:
            self.pos_dict.pop(tag)
            return

        words = self._type_check(words)
        dictionary = self.pos_dict[tag]
        dictionary -= words

    def _type_check(self, words: set[str] | str) -> set[str]:
        if isinstance(words, str):
            words = set(words.split())
        return words

    def load(self, filename: str) -> None:
        with open(filename, encoding="utf-8") as fp:
            params = json.load(fp)
            self.max_length = params["max_length"]
            self.pos_dict = {tag: set(words) for tag, words in params["pos_dict"].items()}

    def save(self, filename: str) -> None:
        with open(filename, "w", encoding="utf-8") as fp:
            params = {
                "max_length": self.max_length,
                "pos_dict": {pos: list(words) for pos, words in self.pos_dict.items()},
            }
            json.dump(params, fp, ensure_ascii=False, indent=2)
