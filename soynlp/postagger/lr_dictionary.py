"""Dictionary adapter for the L/R tagger's historical interface."""

from pathlib import Path

from .dictionary import Dictionary


class LRDictionary:
    def __init__(self, folders=None, use_base=True, min_frequency=3, dictionary: Dictionary | None = None):
        self.dictionary = Dictionary({}) if dictionary is None else dictionary
        if use_base and dictionary is None:
            self._load_folder(Path(__file__).parent / "dictionary" / "pos", min_frequency)
        if isinstance(folders, str):
            folders = [folders]
        for folder in folders or []:
            self._load_folder(Path(folder), min_frequency)
        self._refresh()

    def _load_folder(self, folder: Path, min_frequency: int) -> None:
        if not folder.is_dir():
            raise FileNotFoundError(folder)
        for path in sorted(folder.rglob("*.txt")):
            tag = path.parent.name if path.parent != folder else path.stem.split("_")[0]
            words = set()
            for line in path.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if not parts:
                    continue
                try:
                    frequency = int(parts[1]) if len(parts) > 1 else min_frequency
                except ValueError:
                    frequency = min_frequency
                if frequency >= min_frequency:
                    words.add(parts[0])
            self.dictionary.add_words(tag, words, force=True)

    def _refresh(self):
        self._pos = set(self.dictionary.pos_dict)
        self._lmax = max(
            (len(w) for tag, words in self.dictionary.pos_dict.items() if tag != "Josa" for w in words), default=0
        )
        self._rmax = max(
            (
                len(w)
                for tag, words in self.dictionary.pos_dict.items()
                if tag in {"Josa", "Verb", "Adjective", "Eomi"}
                for w in words
            ),
            default=0,
        )

    def pos_L(self, word: str) -> str | None:
        return next((tag for tag in self.dictionary.get_pos(word) if tag not in {"Josa", "Eomi"}), None)

    def pos_R(self, word: str) -> str | None:
        return next((tag for tag in self.dictionary.get_pos(word) if tag in {"Josa", "Eomi", "Verb", "Adjective"}), None)

    def add_words(self, words, tag):
        self.dictionary.add_words(tag, set(words.split()) if isinstance(words, str) else set(words), force=True)
        self._refresh()

    def remove_words(self, words, tag):
        self.dictionary.remove_words(tag, set(words.split()) if isinstance(words, str) else set(words))
        self._refresh()

    def save_domain_dictionary(self, folder: str, head: str | None = None):
        directory = Path(folder)
        directory.mkdir(parents=True, exist_ok=True)
        for tag, words in self.dictionary.pos_dict.items():
            (directory / f"{tag}_{head or ''}.txt").write_text("".join(f"{w}\n" for w in sorted(words)), encoding="utf-8")
