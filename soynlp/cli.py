import argparse
import inspect
from importlib.metadata import metadata
from typing import Callable

from soynlp.pipeline.pipeline import Pipeline


def main():
    meta = metadata("soynlp")
    parser = argparse.ArgumentParser(description=meta["Summary"])
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version="%(prog)s {}".format(meta["version"]),
    )
    parser.set_defaults(func=lambda: parser.print_usage())
    subparsers = parser.add_subparsers()

    sp_pipeline = subparsers.add_parser("pipeline", help="Run pipeline with config")
    sp_pipeline.add_argument("-c", "--config_file", type=str, required=True, help="Config file path")
    sp_pipeline.set_defaults(func=Pipeline.run)

    args = parser.parse_args()
    func = args.func
    run(func, args)


def run(func: Callable, args: argparse.Namespace):
    func_signature = inspect.signature(func)
    kwargs = {each_name: getattr(args, each_name) for each_name, each_parameter in func_signature.parameters.items()}
    func(**kwargs)
