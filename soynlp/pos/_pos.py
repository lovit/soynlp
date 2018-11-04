from soynlp.noun import LRNounExtractor_v2
from soynlp.predicator import PredicatorExtractor
from soynlp.utils import LRGraph


class POSExtractor:

    def __init__(self, verbose=True, extract_pos_feature=True,
        extract_determiner=True, ensure_normalized=True, extract_eomi=True,
        extract_stem=True):

        self._verbose = verbose
        # noun extraction
        self._extract_pos_feature = extract_pos_feature
        self._extract_determiner = extract_determiner
        self._ensure_normalized = ensure_normalized
        # predicator extraction
        self._extract_eomi = extract_eomi
        self._extract_stem = extract_stem

    def extract(self, sents):
        # noun extraction
        noun_extractor = LRNounExtractor_v2(
            extract_pos_feature = self._extract_pos_feature,
            extract_determiner = self._extract_determiner,
            ensure_normalized = self._ensure_normalized,
            verbose = self._verbose
        )

        nouns = noun_extractor.train_extract(sents, reset_lrgraph=False)

        predicator_lrgraph = LRGraph(noun_extractor.lrgraph._lr)
        predicator_lrgraph.reset_lrgraph()
        noun_pos_features = {r for r in noun_extractor._pos_features}
        noun_pos_features.update({r for r in noun_extractor._common_features})

        # predicator extraction
        predicator_extractor = PredicatorExtractor(
            nouns,
            noun_pos_features,
            extract_eomi = self._extract_eomi,
            extract_stem = self._extract_stem,
            verbose = self._verbose
        )

        predicator_extractor.train(sents)
        predicators = predicator_extractor.extract() # 782 개

        self.noun_extractor = noun_extractor
        self.predicator_extractor = predicator_extractor

        nouns_ = {}
        removals = {}

        for noun, score in nouns.items():
            if self._is_noun_predicator_compound(noun, nouns, predicators):
                removals[noun] = score
            else:
                nouns_[noun] = score
        return nouns_, removals, predicators

    def _is_noun_predicator_compound(self, noun, nouns, predicators):
        def is_noun_josa(prefix):
            if prefix in nouns:
                return True
            for i in range(2, len(prefix)):
                l, r = prefix[:i], prefix[i:]
                if (l in nouns) and is_pos_feature(r):
                    return True
            return False

        def is_pos_feature(r):
            return ((r in predicators) or
                    (r in self.noun_extractor._pos_features) or
                    (r in self.noun_extractor._common_features))

        if noun in predicators:
            return True

        n = len(noun)
        for i in range(-1, -n + 1, -1):
            prefix, suffix = noun[:i], noun[i:]
            if (prefix in predicators) and (suffix in predicators):
                return True
            if is_noun_josa(prefix) and (suffix in predicators):
                return True
        return False