"""Feature based command line interface; algorithms live in public services."""

import argparse
import json
import logging
import sys
from typing import Any

from soynlp import __version__
from soynlp.documents import read_documents, write_documents
from soynlp.pipeline import Pipeline
from soynlp.services import TEXT_METHODS, process_documents, text_processor


def json_parameters(value: str) -> dict[str, Any]:
    try:
        result = json.loads(value)
    except json.JSONDecodeError as error:
        raise argparse.ArgumentTypeError(f"Invalid JSON parameters: {error.msg}") from error
    if not isinstance(result, dict):
        raise argparse.ArgumentTypeError("Parameters must be a JSON object")
    return result


def add_input(parser: argparse.ArgumentParser) -> None:
    inputs = parser.add_mutually_exclusive_group(required=True)
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
    pipeline = commands.add_parser("pipeline", help="Run a YAML pipeline")
    pipeline.add_argument("-c", "--config_file", "--config-file", required=True)
    return parser


def execute(args) -> None:
    if args.command == "pipeline":
        Pipeline.run(args.config_file)
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
