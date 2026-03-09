import argparse
from importlib.metadata import metadata

from soynlp.pipeline.pipeline import Pipeline
from soynlp.pipeline.tasks import TASK_REGISTRY


def _run_pipeline(args: argparse.Namespace) -> None:
    Pipeline.run(args.config_file)


def _run_task_list(args: argparse.Namespace) -> None:
    for name in sorted(TASK_REGISTRY):
        print(name)


def _run_task_extract_nouns(args: argparse.Namespace) -> None:
    from soynlp.noun import LRNounExtractor

    with open(args.input, encoding="utf-8") as f:
        texts = [line.rstrip("\n") for line in f if line.strip()]

    extractor = LRNounExtractor(verbose=args.verbose)
    nouns = extractor.extract(
        train_data=texts,
        min_noun_score=args.min_noun_score,
        min_noun_frequency=args.min_noun_frequency,
        min_eojeol_frequency=args.min_eojeol_frequency,
        extract_compounds=args.extract_compounds,
    )

    sorted_nouns = sorted(nouns.items(), key=lambda x: x[1].frequency, reverse=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for word, score in sorted_nouns:
            f.write(f"{word}\t{score.frequency}\t{score.score:.4f}\n")


def main() -> None:
    meta = metadata("soynlp")
    parser = argparse.ArgumentParser(description=meta["Summary"])
    parser.add_argument("-v", "--version", action="version", version=f"%(prog)s {meta['version']}")
    subparsers = parser.add_subparsers(dest="command")

    # soynlp pipeline -c config.yaml
    sp_pipeline = subparsers.add_parser(
        "pipeline",
        help="YAML 설정 파일로 파이프라인 실행 (예: soynlp pipeline -c pipeline.yaml)",
    )
    sp_pipeline.add_argument("-c", "--config_file", type=str, required=True, help="YAML 파이프라인 설정 파일 경로")

    # soynlp task <subcommand>
    sp_task = subparsers.add_parser("task", help="개별 태스크 실행")
    task_subparsers = sp_task.add_subparsers(dest="task_command")

    # soynlp task list
    task_subparsers.add_parser("list", help="사용 가능한 태스크 목록 출력")

    # soynlp task extract-nouns
    sp_extract_nouns = task_subparsers.add_parser(
        "extract-nouns",
        help="텍스트 파일에서 명사 추출 (결과: word TAB freq TAB score)",
    )
    sp_extract_nouns.add_argument("-i", "--input", type=str, required=True, help="입력 텍스트 파일 경로 (줄당 한 문장)")
    sp_extract_nouns.add_argument(
        "-o", "--output", type=str, required=True, help="출력 파일 경로 (word TAB freq TAB score 형식)"
    )
    sp_extract_nouns.add_argument("--min-noun-score", type=float, default=0.3, help="최소 명사 점수 (기본값: 0.3)")
    sp_extract_nouns.add_argument("--min-noun-frequency", type=int, default=1, help="최소 명사 빈도 (기본값: 1)")
    sp_extract_nouns.add_argument("--min-eojeol-frequency", type=int, default=1, help="최소 어절 빈도 (기본값: 1)")
    sp_extract_nouns.add_argument(
        "--no-extract-compounds", dest="extract_compounds", action="store_false", help="복합명사 추출 비활성화"
    )
    sp_extract_nouns.add_argument("--verbose", dest="verbose", action="store_true", default=True, help="로그 출력 (기본값)")
    sp_extract_nouns.add_argument("--no-verbose", dest="verbose", action="store_false", help="로그 출력 비활성화")

    args = parser.parse_args()

    if args.command == "pipeline":
        _run_pipeline(args)
    elif args.command == "task":
        if args.task_command == "list":
            _run_task_list(args)
        elif args.task_command == "extract-nouns":
            _run_task_extract_nouns(args)
        else:
            sp_task.print_usage()
    else:
        parser.print_usage()
