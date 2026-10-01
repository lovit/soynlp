"""Corpus operations; re-iterable input is spooled instead of retained in RAM."""

import inspect
import json
import tempfile
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from soynlp.artifacts import Artifact, json_value
from soynlp.core import LRGraph
from soynlp.noun import LRNounExtractor, LRNounExtractor_v2, NewsNounExtractor, extract_domain_pos_features
from soynlp.pos import ChatPOSExtractor, NewsPOSExtractor
from soynlp.postagger import POSExtractor
from soynlp.predicator import EomiExtractor, PredicatorExtractor, StemExtractor
from soynlp.tokenizer import EojeolPatternTrainer
from soynlp.utils import EojeolCounter
from soynlp.word import BigramExtractor, WordExtractor

from .text import construct, resolve_artifact, resource_dictionary

EXTRACT_METHODS = {
    "nouns": ("lr", "lr-v1", "lr-v2", "news"),
    "words": ("legacy", "corpus"),
    "bigrams": ("frequency", "pmi", "mikolov"),
    "pos": ("news", "chat", "combined"),
    "predicators": ("default",),
    "eomis": ("default",),
    "stems": ("default",),
    "features": ("default",),
}


class ReplayCorpus:
    def __init__(self, documents: Iterable[dict[str, Any]], text_key: str = "text"):
        self._directory = tempfile.TemporaryDirectory(prefix="soynlp-corpus-")
        self._path = Path(self._directory.name) / "corpus.jsonl"
        self._length = 0
        try:
            with self._path.open("w", encoding="utf-8") as stream:
                for number, item in enumerate(documents, 1):
                    text = item.get(text_key)
                    if not isinstance(text, str):
                        raise ValueError(f"Document {number}: expected string field {text_key!r}")
                    stream.write(json.dumps(text, ensure_ascii=False) + "\n")
                    self._length += 1
        except BaseException:
            self._directory.cleanup()
            raise

    def __len__(self) -> int:
        return self._length

    def __iter__(self) -> Iterator[str]:
        with self._path.open(encoding="utf-8") as stream:
            for line in stream:
                yield json.loads(line)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self._directory.cleanup()


def _options(parameters: dict[str, Any] | None) -> tuple[dict, dict, dict]:
    parameters = parameters or {}
    unknown = set(parameters) - {"constructor", "train", "extract"}
    if unknown:
        raise ValueError(f"Corpus parameters use constructor/train/extract objects, unknown: {sorted(unknown)}")
    values = tuple(parameters.get(name, {}) for name in ("constructor", "train", "extract"))
    if not all(isinstance(value, dict) for value in values):
        raise ValueError("constructor/train/extract parameters must be objects")
    return values


def invoke(function, *args, **kwargs):
    try:
        inspect.signature(function).bind(*args, **kwargs)
    except TypeError as error:
        raise ValueError(f"Invalid parameters for {function.__name__}: {error}") from error
    return function(*args, **kwargs)


def extract_corpus(
    corpus: Any,
    kind: str,
    method: str | None = None,
    *,
    parameters: dict[str, Any] | None = None,
    dictionary: str | Artifact | None = None,
    resource: str | Artifact | None = None,
) -> Artifact:
    if kind not in EXTRACT_METHODS:
        raise ValueError(f"Unknown extraction kind: {kind!r}")
    method = method or EXTRACT_METHODS[kind][0]
    if method not in EXTRACT_METHODS[kind]:
        raise ValueError(f"Unknown {kind} method: {method!r}")
    init, train, extract = _options(parameters)
    init = {"verbose": False, **init}
    if kind == "nouns":
        factory = {"lr": LRNounExtractor, "lr-v1": LRNounExtractor, "lr-v2": LRNounExtractor_v2, "news": NewsNounExtractor}[
            method
        ]
        instance = construct(factory, init)
        if method == "lr":
            if train:
                raise ValueError("lr noun method accepts constructor and extract parameters")
            data = instance.extract(corpus, **extract)
        else:
            invoke(instance.train, corpus, **train)
            data = invoke(instance.extract, **extract)
    elif kind == "words":
        instance = construct(WordExtractor, init)
        if method == "legacy":
            invoke(instance.train, corpus, **train)
            scores = invoke(instance.extract, **extract)
            data = {word: {**json_value(score), "score": score.cohesion_forward} for word, score in scores.items()}
        else:
            if train:
                raise ValueError("corpus word method accepts constructor and extract parameters")
            scores = instance.extract(corpus, **extract)
            data = {}
            for group, words in scores.items():
                for word, score in words.items():
                    data.setdefault(word, {})[group] = json_value(score)
                    if group == "cohesion":
                        data[word]["score"] = score.leftside
    elif kind == "bigrams":
        if train:
            raise ValueError("Bigram extraction accepts constructor and extract parameters")
        instance = construct(BigramExtractor, {"score": method, **init})
        data = list(invoke(instance.extract, corpus, **extract).values())
    elif kind == "pos":
        if method == "combined":
            if train or extract:
                raise ValueError("Combined POS extraction accepts constructor parameters")
            instance = construct(POSExtractor, init)
            nouns, predicators = instance.extract(corpus)
            adjectives, verbs = predicators
            data = {"nouns": nouns, "adjectives": adjectives, "verbs": verbs}
        else:
            instance = construct(NewsPOSExtractor if method == "news" else ChatPOSExtractor, init)
            invoke(instance.train, corpus, **train)
            data = invoke(instance.extract, **extract)
    elif kind == "predicators":
        nouns = (
            resolve_artifact(resource, expected_kind="nouns").data
            if resource
            else LRNounExtractor(verbose=False).extract(corpus)
        )
        if dictionary:
            seeds = resource_dictionary(dictionary).pos_dict
            init = {"verbs": seeds.get("Verb", set()), "adjectives": seeds.get("Adjective", set()), **init}
            for tag, option in [("Eomi", "eomis"), ("Josa", "josas")]:
                if tag in seeds:
                    init.setdefault(option, seeds[tag])
        instance = construct(PredicatorExtractor, init, nouns=set(nouns))
        invoke(instance.train, corpus, **train)
        adjectives, verbs = invoke(instance.extract, **extract)
        data = {"verbs": verbs, "adjectives": adjectives}
    else:
        if train:
            raise ValueError(f"{kind} extraction accepts constructor and extract parameters")
        graph = LRGraph.from_sents(corpus)
        words = resource_dictionary(dictionary).pos_dict if dictionary else {}
        stems = set().union(*(words.get(tag, set()) for tag in ("Stem", "Verb", "Adjective")))
        if kind in {"eomis", "stems"}:
            if not dictionary or not stems:
                raise ValueError(f"{kind} extraction requires a dictionary with seed Stem/Verb/Adjective entries")
            if kind == "eomis":
                instance = construct(EomiExtractor, init, lrgraph=graph, stems=stems, nouns=words.get("Noun", set()))
            else:
                if not words.get("Eomi"):
                    raise ValueError("Stem extraction requires seed Eomi entries")
                instance = construct(StemExtractor, init, lrgraph=graph, stems=stems, eomis=words["Eomi"])
            data = invoke(instance.extract, **extract)
            if kind == "stems":
                data = {word: {"frequency": value[0], "score": value[1]} for word, value in data.items()}
        else:
            if set(init) - {"verbose"}:
                raise ValueError("Feature extraction has no constructor options")
            if resource is None:
                raise ValueError("Feature extraction requires a noun score resource")
            nouns = resolve_artifact(resource, expected_kind="nouns").data
            predictions = {word: (value["score"], value["frequency"]) for word, value in nouns.items()}
            scores = invoke(extract_domain_pos_features, predictions, graph, words.get("Josa", set()), **extract)
            data = {word: {"score": value[0], "frequency": value[1]} for word, value in scores.items() if value is not None}
    return Artifact(kind, json_value(data), {"method": method, **(parameters or {})})


def count_eojeols(corpus: Any, parameters: dict[str, Any] | None = None) -> Artifact:
    counter = construct(EojeolCounter, parameters or {}, sents=corpus)
    return Artifact("eojeols", dict(counter.items()), parameters)


def train_patterns(corpus: Any, parameters: dict[str, Any] | None = None) -> Artifact:
    options = dict(parameters or {})
    if "hits" in options:
        if "extract" in options:
            raise ValueError("Use hits or extract for HITS options, not both")
        options["extract"] = options.pop("hits")
    init, train, hits = _options(options)
    if train.get("n_workers", 1) != 1:
        corpus = list(corpus)
    instance = construct(EojeolPatternTrainer, {"verbose": False, **init})
    invoke(instance.train, corpus, **train)
    data = {
        "lrgraph": instance.lrgraph,
        "rlgraph": instance.rlgraph,
        "left_vocabulary": instance.wordset_l,
        "right_vocabulary": instance.wordset_r,
    }
    if hits:
        if not instance.lrgraph or not instance.rlgraph:
            raise ValueError("HITS training requires nonempty left and right vocabularies")
        left, right = invoke(instance.train_hits, **hits)
        data.update({"left_scores": left, "right_scores": right})
    return Artifact("patterns", json_value(data), parameters)


def build_dictionary(*, resource: str | None = None, corpus: Any = None, tag: str = "Noun") -> Artifact:
    if resource:
        artifact = resolve_artifact(resource)
        if artifact.kind == "dictionary":
            data = resource_dictionary(resource).pos_dict
        elif artifact.kind in {"nouns", "words", "stems", "eomis", "features", "eojeols"}:
            if not isinstance(artifact.data, dict):
                raise ValueError("Dictionary source must contain a mapping of words")
            data = {tag: set(artifact.data)}
        elif artifact.kind in {"pos", "predicators"}:
            tag_names = {
                "nouns": "Noun",
                "verbs": "Verb",
                "adjectives": "Adjective",
                "adverbs": "Adverb",
                "eomis": "Eomi",
                "josas": "Josa",
            }
            data = {tag_names.get(name, name): set(words) for name, words in artifact.data.items() if isinstance(words, dict)}
        else:
            raise ValueError(f"Cannot build a dictionary from {artifact.kind!r}")
    elif corpus is not None:
        data = {tag: {word for text in corpus for word in text.split()}}
    else:
        raise ValueError("Dictionary build requires --resource or text input")
    return Artifact("dictionary", json_value(data), {"source": resource, "tag": tag})


def merge_dictionaries(paths: list[str]) -> Artifact:
    merged = {}
    for path in paths:
        for tag, words in resource_dictionary(path).pos_dict.items():
            merged.setdefault(tag, set()).update(words)
    return Artifact("dictionary", json_value(merged), {"sources": paths})
