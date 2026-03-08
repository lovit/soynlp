import argparse
from importlib.metadata import metadata

from soynlp.pipeline.pipeline import Pipeline


def main() -> None:
    meta = metadata("soynlp")
    parser = argparse.ArgumentParser(description=meta["Summary"])
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"%(prog)s {meta['version']}",
    )
    subparsers = parser.add_subparsers(dest="command")

    sp_pipeline = subparsers.add_parser("pipeline", help="Run pipeline with config")
    sp_pipeline.add_argument("-c", "--config_file", type=str, required=True, help="Config file path")

    args = parser.parse_args()
    if args.command == "pipeline":
        Pipeline.run(args.config_file)
    else:
        parser.print_usage()
