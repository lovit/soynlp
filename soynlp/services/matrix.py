"""Reusable vectorization and matrix transformations."""

from pathlib import Path
from typing import Any

from soynlp.artifacts import Artifact, load_artifact, load_matrix, save_matrix
from soynlp.utils import most_similar, pmi, svd
from soynlp.vectorizer import BaseVectorizer, sent_to_word_contexts_matrix

from .corpus import invoke
from .text import construct, text_processor


def read_vocabulary(path: str | None) -> list[str] | None:
    if path is None:
        return None
    target = Path(path)
    artifact = load_artifact(target.with_suffix(".json") if target.suffix == ".npz" else target)
    vocabulary = artifact.data if artifact.kind == "vocabulary" else artifact.data.get("vocabulary")
    if (
        not isinstance(vocabulary, list)
        or not all(isinstance(word, str) for word in vocabulary)
        or len(set(vocabulary)) != len(vocabulary)
    ):
        raise ValueError("Vocabulary resource must contain a list of distinct strings")
    return vocabulary


def vectorize(
    corpus: Any,
    method: str = "bow",
    *,
    parameters: dict[str, Any] | None = None,
    vocabulary: list[str] | None = None,
    tokenizer_method: str = "whitespace",
    tokenizer_resource: str | None = None,
) -> tuple[Any, list[str], list[str]]:
    options = {"verbose": False, **(parameters or {})}
    tokenize = (
        str.split
        if tokenizer_method == "whitespace"
        else text_processor("tokenize", tokenizer_method, resource=tokenizer_resource)
    )
    if options.get("n_workers", 1) != 1:
        corpus = list(corpus)
    if method == "bow":
        workers = options.pop("n_workers", 1)
        instance = construct(BaseVectorizer, options, tokenizer=tokenize)
        if vocabulary is None:
            instance.fit(corpus, n_workers=workers)
        else:
            instance.idx2vocab = list(vocabulary)
            instance.vocabulary_ = {word: index for index, word in enumerate(vocabulary)}
            instance.n_vocabs = len(vocabulary)
        matrix = instance.transform(corpus)
        return matrix, instance.vocabs(), [f"document:{i}" for i in range(len(corpus))]
    if method != "contexts":
        raise ValueError(f"Unknown vectorization method: {method!r}")
    matrix, words = invoke(sent_to_word_contexts_matrix, corpus, tokenizer=tokenize, vocabulary=vocabulary, **options)
    return matrix, words, words


def matrix_pmi(path: str, parameters: dict[str, Any] | None = None):
    matrix, metadata = load_matrix(path)
    transformed, px, py = invoke(pmi, matrix, **(parameters or {}))
    return transformed, metadata, {"px": px.tolist(), "py": py.tolist(), **(parameters or {})}


def matrix_svd(path: str, output: str, parameters: dict[str, Any] | None = None, *, overwrite: bool = False) -> None:
    options = {"n_components": 2, "random_state": 0, **(parameters or {})}
    matrix, metadata = load_matrix(path)
    components = options["n_components"]
    if type(components) is not int or components < 1 or components > matrix.shape[0]:
        raise ValueError("n_components must be positive and no greater than the row count")
    target = Path(output)
    if target.suffix != ".npz":
        raise ValueError("SVD output must have an .npz extension")
    columns = target.with_name(target.stem + "-columns.npz")
    if not overwrite:
        for candidate in (target, target.with_suffix(".json"), columns, columns.with_suffix(".json")):
            if candidate.exists():
                raise FileExistsError(f"Output already exists: {candidate}")
    u, singular_values, vt = invoke(svd, matrix, **options)
    dimensions = [f"component:{i}" for i in range(components)]
    options["singular_values"] = singular_values.tolist()
    save_matrix(
        target,
        u,
        vocabulary=dimensions,
        row_vocabulary=metadata.data.get("row_vocabulary"),
        parameters=options,
        kind="svd-rows",
        overwrite=overwrite,
    )
    save_matrix(
        columns,
        vt.T,
        vocabulary=dimensions,
        row_vocabulary=metadata.data.get("vocabulary"),
        parameters=options,
        kind="svd-columns",
        overwrite=overwrite,
    )


def matrix_similar(path: str, query: str, topk: int = 10) -> Artifact:
    matrix, metadata = load_matrix(path)
    vocabulary = metadata.data.get("row_vocabulary")
    if vocabulary is None:
        raise ValueError("Similarity requires row_vocabulary in matrix metadata")
    if query not in vocabulary:
        raise ValueError(f"Unknown matrix row: {query!r}")
    scores = most_similar(query, matrix, {word: i for i, word in enumerate(vocabulary)}, vocabulary, topk)
    return Artifact(
        "similarity", [{"word": word, "similarity": float(score)} for word, score in scores], {"query": query, "topk": topk}
    )
