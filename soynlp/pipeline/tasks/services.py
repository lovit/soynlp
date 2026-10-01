"""Pipeline adapters for the public operations used by the feature CLI."""

from dataclasses import dataclass, field
from typing import Any, ClassVar

from soynlp.artifacts import Artifact, load_artifact, load_matrix, save_artifact, save_matrix
from soynlp.core import LRGraph
from soynlp.documents import read_documents, write_documents
from soynlp.services import TEXT_METHODS, process_documents, text_processor
from soynlp.services.corpus import (
    EXTRACT_METHODS,
    ReplayCorpus,
    build_dictionary,
    count_eojeols,
    extract_corpus,
    invoke,
    merge_dictionaries,
    train_patterns,
)
from soynlp.services.matrix import vectorize
from soynlp.services.text import resource_dictionary, validate_text_parameters
from soynlp.utils import most_similar, pmi, svd

from .task import Task, TaskArgs


@dataclass(slots=True)
class DocumentArgs(TaskArgs):
    path: str
    format: str = "txt"
    text_key: str = "text"
    in_key: str = "corpus"
    out_key: str = "corpus"
    overwrite: bool = False


class ReadDocumentsTask(Task[DocumentArgs]):
    @classmethod
    def required_keys(cls, args):
        return set()

    def __call__(self, parameters):
        return {
            self._args.out_key: list(read_documents(self._args.path, format=self._args.format, text_key=self._args.text_key))
        }


class WriteDocumentsTask(Task[DocumentArgs]):
    @classmethod
    def produced_keys(cls, args):
        return set()

    def __call__(self, parameters):
        write_documents(
            parameters[self._args.in_key],
            self._args.path,
            format=self._args.format,
            text_key=self._args.text_key,
            overwrite=self._args.overwrite,
        )
        return {}


@dataclass(slots=True)
class ArtifactArgs(TaskArgs):
    path: str
    kind: str = "resource"
    in_key: str = "resource"
    out_key: str = "resource"
    overwrite: bool = False


class ReadArtifactTask(Task[ArtifactArgs]):
    @classmethod
    def required_keys(cls, args):
        return set()

    def __call__(self, parameters):
        return {self._args.out_key: load_artifact(self._args.path)}


class WriteArtifactTask(Task[ArtifactArgs]):
    @classmethod
    def produced_keys(cls, args):
        return set()

    def __call__(self, parameters):
        value = parameters[self._args.in_key]
        artifact = value if isinstance(value, Artifact) else Artifact(self._args.kind, value)
        save_artifact(artifact, self._args.path, overwrite=self._args.overwrite)
        return {}


@dataclass(slots=True)
class TextArgs(TaskArgs):
    method: str | None = None
    parameters: dict[str, Any] = field(default_factory=dict)
    in_key: str = "corpus"
    out_key: str = "corpus"
    text_key: str = "text"
    result_key: str = "result"
    resource: str | None = None
    resource_key: str | None = None
    dictionary: str | None = None
    dictionary_key: str | None = None
    score_field: str = "score"
    offsets: bool = False
    eomi: str | None = None
    other: str | None = None


class TextOperationTask(Task[TextArgs]):
    operation: ClassVar[str] = "normalize"

    @classmethod
    def validate_args(cls, args):
        validate_text_parameters(cls.operation, args.method, args.parameters)
        if args.method is not None and args.method not in TEXT_METHODS[cls.operation]:
            raise ValueError(f"Unknown {cls.operation} method: {args.method!r}")
        if cls.operation == "conjugate" and (args.method or "normal") in {"normal", "chat"} and args.eomi is None:
            raise ValueError("Conjugation requires eomi")
        if cls.operation == "hangle" and args.method == "distance" and args.other is None:
            raise ValueError("Distance requires other")

    def __call__(self, parameters):
        args = self._args
        processor = text_processor(
            self.operation,
            args.method,
            parameters=args.parameters,
            resource=parameters[args.resource_key] if args.resource_key else args.resource,
            dictionary=parameters[args.dictionary_key] if args.dictionary_key else args.dictionary,
            score_field=args.score_field,
            offsets=args.offsets,
            eomi=args.eomi,
            other=args.other,
        )
        return {
            args.out_key: list(
                process_documents(parameters[args.in_key], processor, text_key=args.text_key, result_key=args.result_key)
            )
        }


@dataclass(slots=True)
class CorpusArgs(TaskArgs):
    kind: str = "nouns"
    method: str | None = None
    parameters: dict[str, Any] = field(default_factory=dict)
    in_key: str = "corpus"
    text_key: str = "text"
    out_key: str = "resource"
    resource: str | None = None
    resource_key: str | None = None
    dictionary: str | None = None
    dictionary_key: str | None = None


class CorpusOperationTask(Task[CorpusArgs]):
    operation: ClassVar[str] = "extract"

    @classmethod
    def validate_args(cls, args):
        if cls.operation == "extract" and (
            args.kind not in EXTRACT_METHODS or args.method is not None and args.method not in EXTRACT_METHODS[args.kind]
        ):
            raise ValueError(f"Unknown extraction kind/method: {args.kind!r}/{args.method!r}")

    def __call__(self, parameters):
        args = self._args
        with ReplayCorpus(parameters[args.in_key], args.text_key) as corpus:
            if self.operation == "count":
                result = count_eojeols(corpus, args.parameters)
            elif self.operation == "patterns":
                result = train_patterns(corpus, args.parameters)
            elif self.operation == "lrgraph":
                graph = invoke(LRGraph.from_sents, corpus, **args.parameters)
                result = Artifact("lrgraph", dict(graph.iter_original_lr()), args.parameters)
            else:
                result = extract_corpus(
                    corpus,
                    args.kind,
                    args.method,
                    parameters=args.parameters,
                    resource=parameters[args.resource_key] if args.resource_key else args.resource,
                    dictionary=parameters[args.dictionary_key] if args.dictionary_key else args.dictionary,
                )
        return {args.out_key: result}


@dataclass(slots=True)
class DictionaryArgs(TaskArgs):
    in_key: str = "resource"
    out_key: str = "dictionary"
    tag: str = "Noun"
    resources: list[str] = field(default_factory=list)
    resource_keys: list[str] = field(default_factory=list)
    path: str = "-"
    overwrite: bool = False


class DictionaryOperationTask(Task[DictionaryArgs]):
    operation: ClassVar[str] = "build"

    @classmethod
    def required_keys(cls, args):
        return set(args.resource_keys) if cls.operation == "merge" else {args.in_key}

    def __call__(self, parameters):
        args = self._args
        if self.operation == "merge":
            result = merge_dictionaries([*args.resources, *(parameters[key] for key in args.resource_keys)])
        elif self.operation == "build":
            result = build_dictionary(resource=parameters[args.in_key], tag=args.tag)
        else:
            words = resource_dictionary(parameters[args.in_key]).pos_dict
            if self.operation == "export":
                examples = (
                    {"text": f"{tag}\t{word}"} for tag, vocabulary in sorted(words.items()) for word in sorted(vocabulary)
                )
                write_documents(examples, args.path, format="txt", overwrite=args.overwrite)
                return {}
            result = Artifact(
                "dictionary-summary",
                {
                    "tags": {tag: len(vocabulary) for tag, vocabulary in words.items()},
                    "entries": sum(map(len, words.values())),
                },
            )
        return {args.out_key: result}


@dataclass(slots=True)
class NeighborArgs(TaskArgs):
    word: str
    in_key: str = "resource"
    out_key: str = "neighbors"
    direction: str = "right"
    topk: int = 10


class LRGraphNeighborsTask(Task[NeighborArgs]):
    @classmethod
    def validate_args(cls, args):
        if args.direction not in {"left", "right"}:
            raise ValueError("direction must be left or right")

    def __call__(self, parameters):
        args = self._args
        graph = LRGraph(parameters[args.in_key].data)
        neighbors = graph.get_r(args.word, args.topk) if args.direction == "right" else graph.get_l(args.word, args.topk)
        return {args.out_key: Artifact("lrgraph-neighbors", neighbors)}


@dataclass(slots=True)
class VectorArgs(TaskArgs):
    method: str = "bow"
    in_key: str = "corpus"
    text_key: str = "text"
    out_key: str = "matrix"
    metadata_key: str = "matrix_metadata"
    vocabulary_key: str | None = None
    tokenizer_method: str = "whitespace"
    tokenizer_resource_key: str | None = None
    parameters: dict[str, Any] = field(default_factory=dict)


class VectorizeTask(Task[VectorArgs]):
    @classmethod
    def required_keys(cls, args):
        return {args.in_key, *(key for key in (args.vocabulary_key, args.tokenizer_resource_key) if key)}

    @classmethod
    def produced_keys(cls, args):
        return {args.out_key, args.metadata_key}

    @classmethod
    def validate_args(cls, args):
        if args.method not in {"bow", "contexts"}:
            raise ValueError("Vectorize method must be bow or contexts")

    def __call__(self, parameters):
        args = self._args
        vocabulary = parameters[args.vocabulary_key].data if args.vocabulary_key else None
        with ReplayCorpus(parameters[args.in_key], args.text_key) as corpus:
            matrix, vocabulary, rows = vectorize(
                corpus,
                args.method,
                parameters=args.parameters,
                vocabulary=vocabulary,
                tokenizer_method=args.tokenizer_method,
                tokenizer_resource=parameters[args.tokenizer_resource_key] if args.tokenizer_resource_key else None,
            )
        metadata = Artifact(args.method, {"vocabulary": vocabulary, "row_vocabulary": rows}, args.parameters)
        return {args.out_key: matrix, args.metadata_key: metadata}


@dataclass(slots=True)
class MatrixArgs(TaskArgs):
    in_key: str = "matrix"
    metadata_key: str = "matrix_metadata"
    out_key: str = "matrix"
    out_metadata_key: str = "matrix_metadata"
    path: str | None = None
    parameters: dict[str, Any] = field(default_factory=dict)
    query: str | None = None
    topk: int = 10
    overwrite: bool = False


class MatrixOperationTask(Task[MatrixArgs]):
    operation: ClassVar[str] = "pmi"

    @classmethod
    def required_keys(cls, args):
        return set() if cls.operation == "read" else {args.in_key, args.metadata_key}

    @classmethod
    def produced_keys(cls, args):
        if cls.operation == "write":
            return set()
        if cls.operation == "similar":
            return {args.out_key}
        keys = {args.out_key, args.out_metadata_key}
        if cls.operation == "svd":
            keys |= {args.out_key + "_columns", args.out_metadata_key + "_columns"}
        return keys

    @classmethod
    def validate_args(cls, args):
        if cls.operation in {"read", "write"} and args.path is None:
            raise ValueError("Matrix IO requires path")
        if cls.operation == "similar" and args.query is None:
            raise ValueError("Matrix similarity requires query")

    def __call__(self, parameters):
        args = self._args
        if self.operation == "read":
            assert args.path is not None
            matrix, metadata = load_matrix(args.path)
            return {args.out_key: matrix, args.out_metadata_key: metadata}
        matrix, metadata = parameters[args.in_key], parameters[args.metadata_key]
        vocabulary = metadata.data.get("vocabulary")
        rows = metadata.data.get("row_vocabulary")
        if self.operation == "write":
            assert args.path is not None
            save_matrix(
                args.path,
                matrix,
                vocabulary=vocabulary,
                row_vocabulary=rows,
                parameters=metadata.parameters,
                kind=metadata.kind,
                overwrite=args.overwrite,
            )
            return {}
        if self.operation == "similar":
            assert args.query is not None
            if rows is None or args.query not in rows:
                raise ValueError(f"Unknown matrix row: {args.query!r}")
            scores = most_similar(args.query, matrix, {word: i for i, word in enumerate(rows)}, rows, args.topk)
            return {
                args.out_key: Artifact("similarity", [{"word": word, "similarity": float(score)} for word, score in scores])
            }
        if self.operation == "pmi":
            result, px, py = invoke(pmi, matrix, **args.parameters)
            data = {"vocabulary": vocabulary, "row_vocabulary": rows}
            return {
                args.out_key: result,
                args.out_metadata_key: Artifact("pmi", data, {"px": px.tolist(), "py": py.tolist(), **args.parameters}),
            }
        options = {"n_components": 2, "random_state": 0, **args.parameters}
        u, sigma, vt = invoke(svd, matrix, **options)
        dimensions = [f"component:{i}" for i in range(len(sigma))]
        options["singular_values"] = sigma.tolist()
        return {
            args.out_key: u,
            args.out_key + "_columns": vt.T,
            args.out_metadata_key: Artifact("svd-rows", {"vocabulary": dimensions, "row_vocabulary": rows}, options),
            args.out_metadata_key + "_columns": Artifact(
                "svd-columns", {"vocabulary": dimensions, "row_vocabulary": vocabulary}, options
            ),
        }


def _variant(name, base, operation):
    return type(name + "Task", (base,), {"operation": operation, "__module__": __name__})


SERVICE_TASKS = {
    "ReadDocuments": ReadDocumentsTask,
    "WriteDocuments": WriteDocumentsTask,
    "ReadArtifact": ReadArtifactTask,
    "WriteArtifact": WriteArtifactTask,
    "Extract": CorpusOperationTask,
    "Vectorize": VectorizeTask,
    "LRGraphNeighbors": LRGraphNeighborsTask,
}
SERVICE_TASKS.update(
    {
        name: _variant(name, TextOperationTask, operation)
        for name, operation in {
            "TextNormalize": "normalize",
            "TextTokenize": "tokenize",
            "Tag": "tag",
            "Lemmatize": "lemmatize",
            "Conjugate": "conjugate",
            "Hangle": "hangle",
        }.items()
    }
)
SERVICE_TASKS.update(
    {
        name: _variant(name, CorpusOperationTask, operation)
        for name, operation in {"CountEojeols": "count", "TrainPatterns": "patterns", "BuildLRGraph": "lrgraph"}.items()
    }
)
SERVICE_TASKS.update(
    {
        name: _variant(name, DictionaryOperationTask, operation)
        for name, operation in {
            "BuildDictionary": "build",
            "MergeDictionary": "merge",
            "InspectDictionary": "inspect",
            "ExportDictionary": "export",
        }.items()
    }
)
SERVICE_TASKS.update(
    {
        name: _variant(name, MatrixOperationTask, operation)
        for name, operation in {
            "ReadMatrix": "read",
            "WriteMatrix": "write",
            "PMI": "pmi",
            "SVD": "svd",
            "Similar": "similar",
        }.items()
    }
)
