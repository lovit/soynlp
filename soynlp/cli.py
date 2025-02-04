import argparse
import inspect
from importlib.metadata import metadata
from typing import Callable

from soynlp.dummy import dummy


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

    sp_dummy = subparsers.add_parser("dummy", help="dummy function")
    sp_dummy.set_defaults(func=dummy)

    args = parser.parse_args()
    func = args.func
    run(func, args)


def run(func: Callable, args: argparse.Namespace):
    func_signature = inspect.signature(func)
    kwargs = {each_name: getattr(args, each_name) for each_name, each_parameter in func_signature.parameters.items()}
    func(**kwargs)
