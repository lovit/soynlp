import inspect
import json
from collections import Counter
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any

from soynlp import hangle, lemmatizer, normalizer, predicator, tokenizer
from soynlp.artifacts import Artifact, json_value, load_artifact
from soynlp.postagger import Dictionary, KoreanPOSTagger, LRMaxScoreTagger

TEXT_METHODS = {
    "normalize": ("text", "jamo", "hangle-emoji", "emoji", "repeat", "spaces", "characters", "padding"),
    "tokenize": ("regex", "l", "max-score", "noun-match", "max-lr-score", "noun-l-match", "legacy-normalize"),
    "tag": ("korean", "lr-max-score"),
    "lemmatize": ("filtered", "candidates", "chat"),
    "conjugate": ("normal", "chat", "present", "imperative", "pleasure"),
    "hangle": ("decompose", "compose", "jamo", "classify", "distance", "encode", "decode"),
}


NORMALIZER_FACTORIES = {
    "text": normalizer.TextNormalizer.build_normalizer,
    "jamo": normalizer.JamoNormalizer,
    "hangle-emoji": normalizer.HangleEmojiNormalizer,
    "emoji": normalizer.EmojiNormalizer,
    "repeat": normalizer.RepeatCharacterNormalizer,
    "spaces": normalizer.RemoveLongspaceNormalizer,
    "characters": normalizer.PassCharacterNormalizer,
    "padding": normalizer.PaddingSpacetoWordsNormalizer,
}


def construct(factory: Callable, parameters: dict[str, Any], **required: Any) -> Any:
    try:
        inspect.signature(factory).bind(**(required | parameters))
    except TypeError as error:
        raise ValueError(f"Invalid parameters for {factory.__name__}: {error}") from error
    return factory(**(required | parameters))


def resolve_artifact(value: str | Artifact, expected_kind: str | None = None) -> Artifact:
    if isinstance(value, Artifact):
        value.as_dict()
        if expected_kind is not None and value.kind != expected_kind:
            raise ValueError(f"Expected artifact kind {expected_kind!r}, got {value.kind!r}")
        return value
    return load_artifact(value, expected_kind=expected_kind)


def resource_dictionary(path: str | Artifact) -> Dictionary:
    artifact = resolve_artifact(path, expected_kind="dictionary")
    if not isinstance(artifact.data, dict) or not all(
        isinstance(words, list) and all(isinstance(word, str) for word in words) for words in artifact.data.values()
    ):
        raise ValueError("Dictionary data must map POS tags to lists of words")
    return Dictionary({tag: set(words) for tag, words in artifact.data.items()})


def resource_scores(path: str | Artifact | None, score_field: str = "score") -> dict[str, float]:
    if path is None:
        return {}
    artifact = resolve_artifact(path)
    data = artifact.data
    if artifact.kind not in {"scores", "nouns", "words", "dictionary"} or not isinstance(data, dict):
        raise ValueError("Tokenizer resource must contain scores, nouns, words or a dictionary")
    if artifact.kind == "dictionary":
        return {word: 1.0 for words in data.values() for word in words}
    scores = {}
    for word, value in data.items():
        if isinstance(value, dict):
            value = value.get(score_field)
        if not isinstance(value, (int, float)):
            raise ValueError(f"Missing numeric score field {score_field!r} for {word!r}")
        scores[word] = float(value)
    return scores


def validate_text_parameters(command: str, method: str | None, parameters: dict[str, Any]) -> None:
    if command not in TEXT_METHODS or method is not None and method not in TEXT_METHODS[command]:
        raise ValueError(f"Unknown text operation/method: {command!r}/{method!r}")
    method = method or TEXT_METHODS[command][0]
    factories = {
        "normalize": NORMALIZER_FACTORIES,
        "tag": {"korean": KoreanPOSTagger, "lr-max-score": LRMaxScoreTagger},
        "tokenize": {
            "regex": tokenizer.RegexTokenizer,
            "l": tokenizer.LTokenizer,
            "max-score": tokenizer.MaxScoreTokenizer,
            "noun-match": tokenizer.NounMatchTokenizer,
            "max-lr-score": tokenizer.MaxLRScoreTokenizer,
            "noun-l-match": tokenizer.NounLMatchTokenizer,
            "legacy-normalize": tokenizer.normalize,
        },
    }
    if command in factories:
        required = (
            {"noun_scores": {}}
            if method == "noun-match"
            else {"nouns": {}}
            if method == "noun-l-match"
            else {"sentence": ""}
            if method == "legacy-normalize"
            else {}
        )
        try:
            inspect.signature(factories[command][method]).bind(**(required | parameters))
        except TypeError as error:
            raise ValueError(f"Invalid {command} parameters: {error}") from error
    elif command == "hangle" and parameters:
        raise ValueError("Hangle methods have no additional parameters")


def text_processor(
    command: str,
    method: str | None = None,
    *,
    parameters: dict[str, Any] | None = None,
    resource: str | Artifact | None = None,
    dictionary: str | Artifact | None = None,
    offsets: bool = False,
    score_field: str = "score",
    eomi: str | None = None,
    other: str | None = None,
) -> Callable[[str], Any]:
    """Build once, then apply to each example without collecting the corpus."""
    if command not in TEXT_METHODS:
        raise ValueError(f"Unknown text operation: {command!r}")
    method = method or TEXT_METHODS[command][0]
    if method not in TEXT_METHODS[command]:
        raise ValueError(f"Unknown {command} method: {method!r}")
    options = dict(parameters or {})
    if command == "normalize":
        factories = NORMALIZER_FACTORIES
        return construct(factories[method], options)
    if command == "tokenize":
        scores = resource_scores(resource, score_field)
        factories = {
            "regex": (tokenizer.RegexTokenizer, {}),
            "l": (tokenizer.LTokenizer, {"scores": scores}),
            "max-score": (tokenizer.MaxScoreTokenizer, {"scores": scores}),
            "noun-match": (tokenizer.NounMatchTokenizer, {"noun_scores": scores}),
            "max-lr-score": (tokenizer.MaxLRScoreTokenizer, {"Dl": scores, "Dr": {}}),
            "noun-l-match": (tokenizer.NounLMatchTokenizer, {"nouns": scores}),
        }
        if method == "legacy-normalize":
            return lambda text: tokenizer.normalize(text, **options)
        if method in {"noun-match", "noun-l-match"} and resource is None:
            raise ValueError(f"{method} requires a noun or dictionary resource")
        factory, required = factories[method]
        instance = construct(factory, options, **required)
        if method in {"max-lr-score", "noun-l-match"}:
            if offsets:
                raise ValueError("Offsets are available for regex, l, max-score and noun-match")
            return instance.tokenize
        return lambda text: instance.tokenize(text, return_words=not offsets)
    if command == "tag":
        if dictionary:
            options["dictionary"] = resource_dictionary(dictionary)
        if resource:
            if method != "lr-max-score":
                raise ValueError("An LR-Graph resource requires lr-max-score")
            options["lrgraph"] = resolve_artifact(resource, expected_kind="lrgraph").data
        instance = construct(KoreanPOSTagger if method == "korean" else LRMaxScoreTagger, options)
        return instance.tag if method == "korean" else instance.pos
    if command == "lemmatize":
        if method == "filtered":
            if dictionary:
                words = resource_dictionary(dictionary).pos_dict
            else:
                root = Path(__file__).resolve().parents[1] / "lemmatizer" / "dictionary" / "default"
                words = {}
                for tag, relative in [
                    ("Verb", "Stem/Verb.txt"),
                    ("Adjective", "Stem/Adjective.txt"),
                    ("Eomi", "Eomi/Eomi.txt"),
                ]:
                    words[tag] = {line.split()[0] for line in (root / relative).read_text().splitlines() if line.strip()}
            stems = set().union(*(words.get(tag, set()) for tag in ("Stem", "Verb", "Adjective")))
            instance = construct(lemmatizer.Lemmatizer, options, stems=stems, endings=words.get("Eomi", set()))
            return instance.lemmatize
        candidate = lemmatizer.lemma_candidate_chat if method == "chat" else lemmatizer.lemma_candidate
        try:
            inspect.signature(candidate).bind("", "", **options)
        except TypeError as error:
            raise ValueError(f"Invalid lemma parameters: {error}") from error
        return lambda text: set().union(*(candidate(text[:i], text[i:], **options) for i in range(1, len(text) + 1)))
    if command == "conjugate":
        if method in {"normal", "chat"}:
            if eomi is None:
                raise ValueError("Normal and chat conjugation require --eomi")
            function = lemmatizer.conjugate_chat if method == "chat" else lemmatizer.conjugate
            inspect.signature(function).bind("", eomi, **options)
            return lambda text: function(text, eomi, **options) if text else set()
        function = {
            "present": predicator.conjugate_as_present,
            "imperative": predicator.conjugate_as_imperative,
            "pleasure": predicator.conjugate_as_pleasure,
        }[method]
        if options:
            raise ValueError(f"{method} conjugation has no additional parameters")
        return lambda text: function(text) if text else set()
    if options:
        raise ValueError("Hangle methods have no additional parameters")
    if method == "decompose":
        return lambda text: [hangle.decompose(character) for character in text]
    if method == "compose":

        def compose(text):
            parts = json.loads(text) if text.startswith("[") else text.split()
            if not isinstance(parts, list) or len(parts) not in {2, 3}:
                raise ValueError("Compose expects two or three jamo, e.g. ㄱ ㅏ ㄴ")
            return hangle.compose(parts[0], parts[1], parts[2] if len(parts) == 3 else " ")

        return compose
    if method == "jamo":
        return hangle.text_to_jamo
    if method == "classify":
        predicates = {
            "korean": hangle.character_is_korean,
            "complete_korean": hangle.character_is_complete_korean,
            "jaum": hangle.character_is_jaum,
            "moum": hangle.character_is_moum,
            "number": hangle.character_is_number,
            "english": hangle.character_is_english,
            "punctuation": hangle.character_is_punctuation,
        }
        return lambda text: [{"character": c, **{name: predicate(c) for name, predicate in predicates.items()}} for c in text]
    if method == "distance":
        if other is None:
            raise ValueError("Distance requires --other")
        return lambda text: {
            "levenshtein": hangle.levenshtein(text, other),
            "jamo_levenshtein": hangle.jamo_levenshtein(text, other),
            "jaccard": hangle.jaccard_distance(text, other),
            "cosine": hangle.cosine_distance(dict(Counter(text)), dict(Counter(other))),
        }
    encoder = hangle.ConvolutionHangleEncoder()
    if method == "encode":
        return encoder.sent_to_onehot
    return lambda text: encoder.onehot_to_sent(json.loads(text))


def process_documents(
    documents: Iterable[dict[str, Any]], processor: Callable[[str], Any], *, text_key: str = "text", result_key: str = "result"
) -> Iterator[dict[str, Any]]:
    for number, document in enumerate(documents, 1):
        if not isinstance(document.get(text_key), str):
            raise ValueError(f"Document {number}: expected string field {text_key!r}")
        yield {**document, result_key: json_value(processor(document[text_key]))}
