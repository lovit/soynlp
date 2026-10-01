"""Feature based command line interface; algorithms live in public services."""

import argparse
import json
import logging
import sys
from typing import Any

from soynlp import __version__
from soynlp.artifacts import Artifact, atomic_output, load_artifact, save_artifact, save_matrix
from soynlp.core import LRGraph
from soynlp.documents import read_documents, write_documents
from soynlp.pipeline import Pipeline
from soynlp.services import TEXT_METHODS, process_documents, text_processor
from soynlp.services.corpus import (
    EXTRACT_METHODS,
    ReplayCorpus,
    build_dictionary,
    count_eojeols,
    extract_corpus,
    merge_dictionaries,
    train_patterns,
)
from soynlp.services.matrix import matrix_pmi, matrix_similar, matrix_svd, read_vocabulary, vectorize
from soynlp.services.text import resource_dictionary


def json_parameters(value: str) -> dict[str, Any]:
    try:
        result = json.loads(value)
    except json.JSONDecodeError as error:
        raise argparse.ArgumentTypeError(f"Invalid JSON parameters: {error.msg}") from error
    if not isinstance(result, dict):
        raise argparse.ArgumentTypeError("Parameters must be a JSON object")
    return result


def add_input(parser: argparse.ArgumentParser, required: bool = True) -> None:
    inputs = parser.add_mutually_exclusive_group(required=required)
    inputs.add_argument("--text", help="Process one example")
    inputs.add_argument("--input", help="TXT or JSONL path; - reads stdin")
    parser.add_argument("--input-format", choices=("txt", "jsonl"), default="txt")
    parser.add_argument("--text-key", default="text", help="Text field in JSONL records")


def add_output(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-o", "--output", default="-", help="Output path; default: stdout")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output")


def input_documents(args):
    if args.text is not None:
        return iter([{args.text_key: args.text}])
    return read_documents(args.input, format=args.input_format, text_key=args.text_key)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unsupervised Korean natural language processing")
    parser.add_argument("-v", "--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--verbose", action="store_true", help="Write diagnostic logs to stderr")
    commands = parser.add_subparsers(dest="command")
    for name, methods in TEXT_METHODS.items():
        command = commands.add_parser(name, help=f"{name.capitalize()} individual text examples")
        add_input(command)
        add_output(command)
        command.add_argument("--method", choices=methods, default=methods[0])
        command.add_argument("--parameters", type=json_parameters, default={}, help="Algorithm options as a JSON object")
        default_key = {
            "normalize": "normalized",
            "tokenize": "tokens",
            "tag": "tags",
            "lemmatize": "lemmas",
            "conjugate": "forms",
        }.get(name, "result")
        command.add_argument("--result-key", default=default_key)
        if name in {"tokenize", "tag"}:
            command.add_argument("--resource", help="Versioned scores, nouns, words, dictionary or LR-Graph resource")
        if name == "tokenize":
            command.add_argument("--offsets", action="store_true", help="Include token positions")
            command.add_argument("--score-field", default="score")
        if name in {"tag", "lemmatize"}:
            command.add_argument("--dictionary", help="Versioned POS dictionary resource")
        if name == "conjugate":
            command.add_argument("--eomi", help="Ending used by normal/chat methods")
        if name == "hangle":
            command.add_argument("--other", help="Second text for distance calculations")
    extraction = commands.add_parser("extract", help="Extract corpus resources")
    extractors = extraction.add_subparsers(dest="kind", required=True)
    for kind, methods in EXTRACT_METHODS.items():
        command = extractors.add_parser(kind)
        add_input(command)
        add_output(command)
        command.add_argument("--method", choices=methods, default=methods[0])
        command.add_argument("--parameters", type=json_parameters, default={}, help="constructor/train/extract option objects")
        command.add_argument("--dictionary")
        command.add_argument("--resource")
    corpus = commands.add_parser("corpus", help="Convert or count a corpus")
    corpora = corpus.add_subparsers(dest="action", required=True)
    for action in ("convert", "count-eojeols"):
        command = corpora.add_parser(action)
        add_input(command)
        add_output(command)
        command.add_argument("--parameters", type=json_parameters, default={})
        if action == "convert":
            command.add_argument("--output-format", choices=("txt", "jsonl"), default="jsonl")
    training = commands.add_parser("train", help="Train corpus statistics")
    trainer = training.add_subparsers(dest="action", required=True).add_parser("patterns")
    add_input(trainer)
    add_output(trainer)
    trainer.add_argument("--parameters", type=json_parameters, default={})
    dictionary = commands.add_parser("dictionary", help="Build, merge, inspect or export POS dictionaries")
    dictionaries = dictionary.add_subparsers(dest="action", required=True)
    builder = dictionaries.add_parser("build")
    add_input(builder, required=False)
    builder.add_argument("--resource")
    builder.add_argument("--tag", default="Noun")
    add_output(builder)
    merger = dictionaries.add_parser("merge")
    merger.add_argument("resources", nargs="+")
    add_output(merger)
    for action in ("inspect", "export"):
        command = dictionaries.add_parser(action)
        command.add_argument("--resource", required=True)
        add_output(command)
    graph = commands.add_parser("lrgraph", help="Build LR-Graphs and query neighbors")
    graphs = graph.add_subparsers(dest="action", required=True)
    builder = graphs.add_parser("build")
    add_input(builder)
    add_output(builder)
    builder.add_argument("--parameters", type=json_parameters, default={})
    neighbors = graphs.add_parser("neighbors")
    neighbors.add_argument("--resource", required=True)
    neighbors.add_argument("--word", required=True)
    neighbors.add_argument("--direction", choices=("left", "right"), default="right")
    neighbors.add_argument("--topk", type=int, default=10)
    add_output(neighbors)
    vectorizer = commands.add_parser("vectorize", help="Create term frequency and context matrices")
    vectorizers = vectorizer.add_subparsers(dest="method", required=True)
    for method in ("bow", "contexts"):
        command = vectorizers.add_parser(method)
        add_input(command)
        add_output(command)
        command.add_argument("--vocabulary", help="Vocabulary JSON or existing matrix NPZ")
        command.add_argument(
            "--tokenizer-method", choices=("whitespace", "regex", "l", "max-score", "noun-match"), default="whitespace"
        )
        command.add_argument("--tokenizer-resource")
        command.add_argument("--parameters", type=json_parameters, default={})
    matrix = commands.add_parser("matrix", help="Transform saved matrices or find similar rows")
    matrices = matrix.add_subparsers(dest="action", required=True)
    for action in ("pmi", "svd", "similar"):
        command = matrices.add_parser(action)
        command.add_argument("--input", required=True, help="Matrix NPZ with matching JSON metadata")
        add_output(command)
        if action == "similar":
            command.add_argument("--query", required=True)
            command.add_argument("--topk", type=int, default=10)
        else:
            command.add_argument("--parameters", type=json_parameters, default={})
    pipeline = commands.add_parser("pipeline", help="Run a YAML pipeline")
    pipeline.add_argument("-c", "--config_file", "--config-file")
    pipelines = pipeline.add_subparsers(dest="action")
    for action in ("run", "validate"):
        command = pipelines.add_parser(action)
        command.add_argument("-c", "--config_file", "--config-file", required=True)
    pipelines.add_parser("tasks")
    return parser


def write_resource(artifact: Artifact, args) -> None:
    if args.output == "-":
        print(json.dumps(artifact.as_dict(), ensure_ascii=False, allow_nan=False))
    else:
        save_artifact(artifact, args.output, overwrite=args.overwrite)


def execute_corpus(args) -> None:
    if args.command == "dictionary":
        if args.action == "merge":
            artifact = merge_dictionaries(args.resources)
        elif args.action == "build":
            if bool(args.resource) + bool(args.input is not None or args.text is not None) != 1:
                raise ValueError("Dictionary build requires exactly one resource or text input")
            if args.resource:
                artifact = build_dictionary(resource=args.resource, tag=args.tag)
            else:
                with ReplayCorpus(input_documents(args), args.text_key) as corpus:
                    artifact = build_dictionary(corpus=corpus, tag=args.tag)
        else:
            words = resource_dictionary(args.resource).pos_dict
            if args.action == "export":
                from contextlib import nullcontext

                context = (
                    nullcontext(sys.stdout) if args.output == "-" else atomic_output(args.output, overwrite=args.overwrite)
                )
                with context as stream:
                    for tag, vocabulary in sorted(words.items()):
                        for word in sorted(vocabulary):
                            stream.write(f"{tag}\t{word}\n")
                return
            artifact = Artifact(
                "dictionary-summary",
                {
                    "tags": {tag: len(vocabulary) for tag, vocabulary in words.items()},
                    "entries": sum(map(len, words.values())),
                },
            )
        write_resource(artifact, args)
        return
    if args.command == "lrgraph" and args.action == "neighbors":
        resource = load_artifact(args.resource, expected_kind="lrgraph")
        graph = LRGraph(resource.data)
        neighbors = graph.get_r(args.word, args.topk) if args.direction == "right" else graph.get_l(args.word, args.topk)
        write_resource(Artifact("lrgraph-neighbors", neighbors, {"word": args.word, "direction": args.direction}), args)
        return
    if args.command == "corpus" and args.action == "convert":
        write_documents(
            input_documents(args), args.output, format=args.output_format, text_key=args.text_key, overwrite=args.overwrite
        )
        return
    with ReplayCorpus(input_documents(args), args.text_key) as corpus:
        if args.command == "extract":
            artifact = extract_corpus(
                corpus, args.kind, args.method, parameters=args.parameters, dictionary=args.dictionary, resource=args.resource
            )
        elif args.command == "train":
            artifact = train_patterns(corpus, args.parameters)
        elif args.command == "lrgraph":
            from soynlp.services.corpus import invoke

            graph = invoke(LRGraph.from_sents, corpus, **args.parameters)
            artifact = Artifact("lrgraph", dict(graph.iter_original_lr()), args.parameters)
        else:
            artifact = count_eojeols(corpus, args.parameters)
    write_resource(artifact, args)


def execute_matrix(args) -> None:
    if args.command == "matrix" and args.action == "similar":
        write_resource(matrix_similar(args.input, args.query, args.topk), args)
        return
    if args.output == "-":
        raise ValueError("Binary matrix output requires -o/--output with an .npz path")
    if args.command == "vectorize":
        vocabulary = read_vocabulary(args.vocabulary)
        with ReplayCorpus(input_documents(args), args.text_key) as corpus:
            matrix, vocabulary, rows = vectorize(
                corpus,
                args.method,
                parameters=args.parameters,
                vocabulary=vocabulary,
                tokenizer_method=args.tokenizer_method,
                tokenizer_resource=args.tokenizer_resource,
            )
        save_matrix(
            args.output,
            matrix,
            vocabulary=vocabulary,
            row_vocabulary=rows,
            parameters={"method": args.method, "tokenizer_method": args.tokenizer_method, **args.parameters},
            kind=args.method,
            overwrite=args.overwrite,
        )
    elif args.action == "svd":
        matrix_svd(args.input, args.output, args.parameters, overwrite=args.overwrite)
    else:
        matrix, metadata, parameters = matrix_pmi(args.input, args.parameters)
        save_matrix(
            args.output,
            matrix,
            vocabulary=metadata.data.get("vocabulary"),
            row_vocabulary=metadata.data.get("row_vocabulary"),
            parameters=parameters,
            kind="pmi",
            overwrite=args.overwrite,
        )


def execute(args) -> None:
    if args.command == "pipeline":
        if args.action == "tasks":
            from soynlp.pipeline.tasks import TASK_REGISTRY

            print(
                json.dumps(
                    {
                        name: [field.name for field in task.args().__dataclass_fields__.values()]
                        for name, task in sorted(TASK_REGISTRY.items())
                    },
                    ensure_ascii=False,
                )
            )
        elif args.action == "validate":
            from soynlp.configs.config import from_yaml

            print(json.dumps(Pipeline.validate(from_yaml(args.config_file)), ensure_ascii=False))
        else:
            if args.config_file is None:
                raise ValueError("Pipeline run requires -c/--config-file")
            Pipeline.run(args.config_file)
        return
    if args.command in {"vectorize", "matrix"}:
        execute_matrix(args)
        return
    if args.command in {"extract", "corpus", "dictionary", "train", "lrgraph"}:
        execute_corpus(args)
        return
    processor = text_processor(
        args.command,
        args.method,
        parameters=args.parameters,
        **{
            name: getattr(args, name)
            for name in ("resource", "dictionary", "offsets", "score_field", "eomi", "other")
            if hasattr(args, name)
        },
    )
    documents = process_documents(input_documents(args), processor, text_key=args.text_key, result_key=args.result_key)
    write_documents(documents, args.output, overwrite=args.overwrite)


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_usage()
        return
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, stream=sys.stderr)
    try:
        execute(args)
    except (ValueError, TypeError) as error:
        parser.error(str(error))
    except (OSError, RuntimeError, ImportError) as error:
        parser.exit(1, f"soynlp: {error}\n")
