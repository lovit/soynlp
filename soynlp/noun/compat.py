"""Adapters preserving the master training API alongside corpus extraction."""

from typing import Any, cast

from soynlp._legacy.noun import LRNounExtractor as LegacyNounExtractor

from .lr import LRNounExtractor as CorpusNounExtractor


class LRNounExtractor(CorpusNounExtractor):
    def __init__(self, *args, **kwargs):
        modern_keys = {"max_l_length", "max_r_length", "pos_features", "neg_features", "verbose"}
        legacy_keys = set(kwargs) - modern_keys
        self._legacy_mode = bool(legacy_keys or args)
        self._legacy = None
        self._legacy_options = {key: value for key, value in kwargs.items() if key not in modern_keys or key == "verbose"}
        if self._legacy_mode:
            self._legacy = LegacyNounExtractor(*args, **self._legacy_options)
            super().__init__(verbose=kwargs.get("verbose", True))
        else:
            super().__init__(*args, **kwargs)
            for old, new in [("max_left_length", "max_l_length"), ("max_right_length", "max_r_length")]:
                if new in kwargs:
                    self._legacy_options[old] = kwargs[new]

    def _legacy_extractor(self):
        if self._legacy is None:
            self._legacy = LegacyNounExtractor(**self._legacy_options)
        return self._legacy

    def __getattr__(self, name):
        return getattr(self._legacy_extractor(), name)

    @property
    def is_trained(self):
        if self._legacy_mode:
            return self._legacy_extractor().lrgraph is not None
        return super().is_trained

    def train(self, sents, min_noun_frequency=5):
        self._legacy_mode = True
        self._legacy_extractor().train(sents, min_noun_frequency)
        self.lrgraph = cast(Any, self._legacy_extractor().lrgraph)

    def train_extract(self, sents, min_noun_score=0.5, min_noun_frequency=5, noun_candidates=None):
        self.train(sents, min_noun_frequency)
        return self.extract(
            min_noun_score=min_noun_score, min_noun_frequency=min_noun_frequency, noun_candidates=noun_candidates
        )

    def extract(self, train_data=None, *args, **kwargs):
        if self._legacy_mode or isinstance(train_data, (int, float)):
            self._legacy_mode = True
            if train_data is None:
                return self._legacy_extractor().extract(*args, **kwargs)
            return self._legacy_extractor().extract(train_data, *args, **kwargs)
        return super().extract(train_data, *args, **kwargs)

    def predict(self, *args, **kwargs):
        if self._legacy_mode:
            return self._legacy_extractor().predict(*args, **kwargs)
        return super().predict(*args, **kwargs)
