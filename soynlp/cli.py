import argparse
import inspect
import os
from pprint import pformat
from typing import Callable

from .about import __name__, __version__
from .normalizer.normalizer import task_normalize


def main():
    parser = argparse.ArgumentParser(description=f"{__name__}=={__version__}")
    parser.set_defaults(func=lambda x: print_helps)
    subparsers = parser.add_subparsers(help=f"{__name__} tasks")

    # common parser
    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument("-i", "--input", type=str, nargs="+", required=True, help="input file or directory")
    common_parser.add_argument("-o", "--output", type=str, nargs="+", required=True, help="output file or directory")
    common_parser.add_argument("-v", "--verbose", action="store_true", help="print progress")
    common_parser.add_argument("-f", "--force", action="store_true", help="overwrite `output` even if it already exists")

    # normalize
    normalize_parser = subparsers.add_parser("normalize", help="normalize file[s]", parents=[common_parser])
    normalize_parser.add_argument("--alphabet", action="store_true", help="remain alphabet")
    normalize_parser.add_argument("--hangle", action="store_true", help="remain hangle (Korean characters)")
    normalize_parser.add_argument("--number", action="store_true", help="remain number (0-9)")
    normalize_parser.add_argument("--symbol", action="store_true", help="remain symbols; `().,?!-/[]`")
    normalize_parser.add_argument("--custom", type=str, default=None, help="remain characters such as `@:;`")
    normalize_parser.add_argument("--hangle_emoji", dest="decompose_hangle_emoji", action="store_true",
        help="decompose hangle emoji: ㅋㅋ쿠ㅜㅜ  -> ㅋㅋㅋㅜㅜㅜ")
    normalize_parser.add_argument("--remove_repeatchar", type=int, default=2,
        help="remove repeating chars ㅋㅋㅋㅋㅋㅋㅋㅋ -> ㅋㅋ. (default %(default)s)")
    normalize_parser.add_argument("--remove_longspace", action="store_true", help="`가나다     라` -> `가나다 라`")
    normalize_parser.set_defaults(func=task_normalize)

    # execute
    args = parser.parse_args()
    function = args.func
    execute(function, args)


def execute(function: Callable, args: argparse.Namespace):
    argument_names = inspect.signature(function).parameters
    kwargs = {name: getattr(args, name) for name in argument_names}
    print(f"\n{__name__}=={__version__} CLI\n{function.__name__}")
    for name, value in kwargs.items():
        print(f"  - {name}: {pformat(value)}")
    function(**kwargs)
