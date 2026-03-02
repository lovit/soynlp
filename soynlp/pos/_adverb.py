from soynlp.utils.utils import installpath


def load_default_adverbs(path: str | None = None) -> set[str]:
    if path is None:
        path = f"{installpath}/postagger/dictionary/default/Adverb/adverb.txt"
    with open(path, encoding="utf-8") as f:
        return {word.strip().split()[0] for word in f}


def stem_to_adverb(stems: set[str] | str, suffixes: str | None = None) -> set[str]:
    if isinstance(stems, str):
        stems = [stems]  # type: ignore[assignment]
    if suffixes is None:
        suffixes = "히"
    return {stem[:-1] + suffix for stem in stems for suffix in suffixes if stem[-1] == "하"}
