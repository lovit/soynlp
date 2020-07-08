import argparse


def extract_noun(args):
    print(args)


def main():
    parser = argparse.ArgumentParser(description="soynlp Command Line Interface")
    parser.add_argument('--silent', dest='silent', action='store_true', help="Off verbose mode")
    subparsers = parser.add_subparsers()

    # noun
    subparser_noun = subparsers.add_parser('extract_noun', help="Extract noun list, tokenize/vectorize documents only with extracted nouns")
    subparser_noun.add_argument('--corpus_path', type=str, required=True, help="Corpus file path")
    subparser_noun.add_argument('--output_dir', type=str, required=True, help='Output directory')
    subparser_noun.add_argument('--experiment_name', type=str, default='', help='Prefix of output files')
    subparser_noun.add_argument('--method', type=str, default='v2', choices=['v1', 'v2'])
    subparser_noun.set_defaults(func=extract_noun)

    # execute
    args = parser.parse_args()
    task_function = args.func
    task_function(args)


if __name__ == '__main__':
    main()
