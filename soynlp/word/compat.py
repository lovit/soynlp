"""Keep master train/extract and pickle files alongside the new corpus API."""

from typing import Any

from soynlp._legacy.word import WordExtractor as LegacyWordExtractor

from .word import WordExtractor as CorpusWordExtractor


class WordExtractor(CorpusWordExtractor):
    def __init__(self, *args, **kwargs):
        modern_keys = {"max_l_length", "max_r_length", "verbose", "R_suffix"}
        self._legacy_options = {key: value for key, value in kwargs.items() if key not in modern_keys}
        self._legacy_mode = bool(self._legacy_options) or bool(args and (args[0] is None or not isinstance(args[0], int)))
        for old, new in [("max_left_length", "max_l_length"), ("max_right_length", "max_r_length")]:
            if new in kwargs:
                self._legacy_options[old] = kwargs[new]
        if kwargs.get("verbose") is False:
            self._legacy_options["verbose_points"] = 0
        self._legacy = None
        if self._legacy_mode:
            self._legacy = LegacyWordExtractor(*args, **self._legacy_options)
            super().__init__(verbose=kwargs.get("verbose", True))
        else:
            super().__init__(*args, **kwargs)
        if kwargs.get("verbose") is False:
            self._legacy_options["verbose_points"] = 0

    def _legacy_extractor(self):
        if self._legacy is None:
            self._legacy = LegacyWordExtractor(**self._legacy_options)
        return self._legacy

    def __getattr__(self, name):
        return getattr(self._legacy_extractor(), name)

    def train(self, sents, num_for_pruning=0, cumulate=True):
        self._legacy_mode = True
        self._legacy_extractor().train(sents, num_for_pruning, cumulate)
        self.L, self.R = self._legacy_extractor().L, self._legacy_extractor().R

    def extract(self, train_data=None, *args, **kwargs) -> Any:
        if self._legacy_mode or (train_data is None and self._legacy is not None):
            if train_data is not None:
                kwargs["scores"] = train_data
            return self._legacy_extractor().extract(*args, **kwargs)
        return super().extract(train_data, *args, **kwargs)

    def save(self, fname):
        if not self._legacy_mode:
            raise ValueError("Use train() for legacy model files; save_artifact() stores corpus extraction results")
        return self._legacy_extractor().save(fname)

    def load(self, fname):
        self._legacy_extractor().load(fname)
        self._legacy_mode = True
        self.L, self.R = self._legacy_extractor().L, self._legacy_extractor().R
