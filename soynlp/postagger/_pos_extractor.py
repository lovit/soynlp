from soynlp.noun import LRNounExtractor_v2
from soynlp.predicator import PredicatorExtractor
from soynlp.utils import LRGraph

class POSExtractor:

    def __init__(self, l_max_length=10, r_max_length=8, verbose=True, logpath=None):
        self.l_max_length = l_max_length
        self.r_max_length = r_max_length
        self.verbose = verbose
        self.logpath = logpath

    def _print(self, message, replace=False, newline=True):
        header = '[POS Extractor]'
        if replace:
            print('\r{} {}'.format(header, message),
                  end='\n' if newline else '', flush=True)
        else:
            print('{} {}'.format(header, message),
                  end='\n' if newline else '', flush=True)

    def extract(self, sentences):

        self._num_of_eojeols = 0
        self._num_of_covered_eojeols = 0

        nouns = self._extract_nouns(sentences)
        predicators = self._extract_predicators(self._lrgraph, nouns)

        del self._lrgraph

        if self.verbose:
            message = '{} nouns, {} predicators were extracted'.format(
                len(nouns), len(predicators))
            self._print(message, replace=True, newline=True)

        return nouns, predicators

    def _extract_nouns(self, sentences):

        noun_extractor = LRNounExtractor_v2(
            l_max_length = self.l_max_length,
            r_max_length = self.r_max_length,
            min_eojeol_count = 2,
            min_num_of_features = 2,
            max_count_when_noun_is_eojeol = 15,
            extract_compound = False,
            logpath = self.logpath,
            extract_pos_feature = True,
            verbose = self.verbose
        )

        noun_extractor.train(sentences)

        nouns = noun_extractor.extract(
            reset_lrgraph = False,
            min_count = 10,
            minimum_noun_score = 0.4,
        )

        self._lrgraph = LRGraph(
            {l:{r:v for r,v in rdict.items()}
             for l, rdict in noun_extractor.lrgraph._lr.items()}
        )
        self._num_of_eojeols = noun_extractor._num_of_eojeols
        self._num_of_covered_eojeols = noun_extractor._num_of_covered_eojeols

        self.noun_extractor = noun_extractor

        if self.verbose:
            message = 'noun extraction was done. {} % eojeols are covered'.format(
                '%.2f' % (100 * self._num_of_covered_eojeols / self._num_of_eojeols))
            self._print(message, replace=True, newline=True)

        return nouns

    def _extract_predicators(self, sentences_or_lrgraph, nouns=None):

        predicator_extractor = PredicatorExtractor(
            nouns,
            extract_eomi=True,
            extract_stem=True,
            verbose = self.verbose
        )

        predicator_extractor.train(sentences_or_lrgraph, min_eojeol_count=2)

        predicators = predicator_extractor.extract(
            candidates=None, min_count=10, reset_lrgraph=True,
            # Eomi extractor
            min_num_of_features=5, min_eomi_score=0.3, min_eomi_frequency=1,
            # Stem extractor
            min_num_of_unique_R_char=10, min_entropy_of_R_char=0.5,
            min_entropy_of_R=1.5, min_stem_score=0.7, min_stem_frequency=100
        )

        self._num_of_covered_eojeols += predicator_extractor._num_of_covered_eojeols

        self.predicator_extractor = predicator_extractor

        if self.verbose:
            message = 'predicator extraction was done. {} % eojeols are covered (cum)'.format(
                '%.2f' % (100 * self._num_of_covered_eojeols / self._num_of_eojeols))
            self._print(message, replace=True, newline=True)

        return predicators