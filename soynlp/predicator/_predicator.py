""" TERM DEFINITION
(l, r) : L and R position subwords
stem : stem of Adjective and Verb
ending : suffix, canonical form of ending

stems : set of stem including Adjectives and Verbs
composable_stems : stems that can be compounded with other prefix
    - [] + 하다 : 덕질+하다, 냐옹+하다, 냐옹+하냥
endings : set of ending
pos_l_features : canonical form set of stems (L subwords)
pos_composable_l_features : canonical form set of composable stems (L subwords)
lrgraph : L-R graph including [stem + Ending], Adverbs, 
          and maybe some Noun + Josa
"""

from collections import defaultdict
from collections import namedtuple
from soynlp.hangle import character_is_complete_korean
from soynlp.utils import EojeolCounter
from soynlp.utils import get_process_memory
from soynlp.utils import LRGraph
from soynlp.utils.utils import installpath
from soynlp.lemmatizer import conjugate
from soynlp.lemmatizer import lemma_candidate
from soynlp.lemmatizer import _conjugate_stem
from soynlp.normalizer import normalize_sent_for_lrgraph
from ._eomi import EomiExtractor
from ._stem import StemExtractor
from ._adjective_vs_verb import conjugate_as_present
from ._adjective_vs_verb import conjugate_as_imperative
from ._adjective_vs_verb import conjugate_as_pleasure
from ._adjective_vs_verb import rule_classify

Predicator = namedtuple('Predicator', 'frequency lemma')

class PredicatorExtractor:

    def __init__(self, nouns, josas=None, adjectives=None,
        verbs=None, eomis=None, extract_eomi=False, extract_stem=False,
        verbose=True, ensure_normalized=False):

        if not josas:
            josas = self._load_default_josa()

        if (adjectives is None) or (verbs is None):
            adjectives, verbs = self._load_default_stems()

        if eomis is None:
            eomis = self._load_default_eomis()

        self._josas = josas
        self._adjective_stems = adjectives
        self._verb_stems = verbs
        self._stems = {stem for stem in adjectives}
        self._stems = self._stems.union(verbs)
        self._eomis = eomis
        self.verbose = verbose
        self.extract_eomi = extract_eomi
        self.extract_stem = extract_stem
        self.ensure_normalized = ensure_normalized

        self._stem_surfaces = self._transform_stem_as_surfaces()
        self.eojeol_counter = None

        nouns = self._remove_stem_prefix(nouns)
        self._nouns = nouns
        self._eomis_ = {eomi for eomi in self._eomis} # original eomis

    def _load_default_josa(self):
        path = '%s/postagger/dictionary/default/Josa/josa_chat.txt' % installpath
        with open(path, encoding='utf-8') as f:
            josas = {word.strip() for word in f}
        return josas

    def _load_default_stems(self, min_frequency=2):
        def load(path):
            stems = set()
            with open(path, encoding='utf-8') as f:
                for line in f:
                    word, frequency = line.split()
                    if int(frequency) < min_frequency:
                        continue
                    stems.add(word)
            return stems

        dirs = '%s/lemmatizer/dictionary/default/Stem' % installpath
        adjectives = load('%s/Adjective.txt' % dirs)
        verbs = load('%s/Verb.txt' % dirs)
        return adjectives, verbs

    def _load_default_eomis(self, min_frequency=20):
        path = '%s/lemmatizer/dictionary/default/Eomi/Eomi.txt' % installpath
        eomis = set()
        with open(path, encoding='utf-8') as f:
            for line in f:
                word, frequency = line.split()
                if int(frequency) < min_frequency:
                    continue
                eomis.add(word)
        return eomis

    def _remove_stem_prefix(self, nouns):
        def parse_noun(stem):
            if stem[:-1] in nouns:
                return stem[:-1]
            if stem[:-2] in nouns:
                return stem[:-2]
            return None

        removals = {parse_noun(stem) for stem in self._stems
                    if parse_noun(stem) is not None}
        return {noun for noun in nouns if not (noun in removals)}

    def _transform_stem_as_surfaces(self):
        surfaces = set()
        for stem in self._stems:
            try:
                for l in _conjugate_stem(stem):
                    surfaces.add(l)
            except Exception as e:
                print('Exception stem = {}, {}'.format(stem, e))
                continue
        return surfaces

    def _print(self, message, replace=False, newline=True):
        header = '[Predicator Extractor]'
        if replace:
            print('\r{} {}'.format(header, message),
                  end='\n' if newline else '', flush=True)
        else:
            print('{} {}'.format(header, message),
                  end='\n' if newline else '', flush=True)

    @property
    def is_trained(self):
        return self.lrgraph

    def train_extract(self, inputs, min_eojeol_frequency=2,
        filtering_checkpoint=100000, candidates=None,
        min_predicator_frequency=1, reset_lrgraph=True,
        # Eomi extractor
        min_num_of_features=5, min_eomi_score=0.3, min_eomi_frequency=1,
        # Stem extractor
        min_num_of_unique_R_char=10, min_entropy_of_R_char=0.5,
        min_entropy_of_R=1.5, min_stem_score=0.7, min_stem_frequency=100):

        self.train(inputs, min_eojeol_frequency, filtering_checkpoint,
            min_num_of_features, min_eomi_score, min_eomi_frequency,
            min_num_of_unique_R_char, min_entropy_of_R_char,
            min_entropy_of_R, min_stem_score, min_stem_frequency)

        predicators = self.extract(
            candidates, min_predicator_frequency)
        return predicators

    def train(self, inputs, min_eojeol_frequency=2, filtering_checkpoint=100000,
        # Eomi extractor
        min_num_of_features=5, min_eomi_score=0.3, min_eomi_frequency=1,
        # Stem extractor
        min_num_of_unique_R_char=10, min_entropy_of_R_char=0.5,
        min_entropy_of_R=1.5, min_stem_score=0.7, min_stem_frequency=100):

        # handle inputs
        if isinstance(inputs, LRGraph):
            self._train_with_eojeol_counter(
                inputs.to_EojeolCounter(), min_eojeol_frequency)
        elif isinstance(inputs, EojeolCounter):
            self._train_with_eojeol_counter(
                inputs, min_eojeol_frequency)
        else:
            self._train_with_sentences(inputs,
                min_eojeol_frequency, filtering_checkpoint)

        # prepare predicator lrgraph
        if self.extract_eomi or self.extract_stem:
            lrgraph = self._prepare_predicator_lrgraph()

        # extract eomi & stem
        if self.extract_eomi:
            self._extract_eomi(lrgraph, min_num_of_features,
                min_eomi_score, min_eomi_frequency)

        if self.extract_stem:
            if self.extract_eomi:
                lrgraph.reset_lrgraph()
            self._extract_stem(lrgraph, min_num_of_unique_R_char,
                min_entropy_of_R_char, min_entropy_of_R,
                min_stem_score, min_stem_frequency)

        if self.verbose:
            message = 'has been trained'
            self._print(message, replace=False, newline=True)

    def _train_with_sentences(self, sentences, min_eojeol_frequency=2,
        filtering_checkpoint=100000):

        check = filtering_checkpoint > 0

        if self.verbose:
            message = 'counting eojeols ... '
            self._print(message, replace=False, newline=False)

        if self.ensure_normalized:
            preprocess = lambda x:x
        else:
            preprocess = normalize_sent_for_lrgraph

        # Eojeol counting
        eojeol_counter = EojeolCounter(
            sentences,
            min_count = min_eojeol_frequency,
            verbose = self.verbose,
            preprocess = preprocess
        )

        self._train_with_eojeol_counter(eojeol_counter)

    def _train_with_eojeol_counter(self, eojeol_counter, min_eojeol_frequency=2):
        eojeol_counter._counter = {
            eojeol:count for eojeol, count in eojeol_counter._counter.items()
            if count >= min_eojeol_frequency}

        eojeol_counter._set_count_sum()
        self._num_of_eojeols = len(eojeol_counter)
        self._num_of_covered_eojeols = 0
        self._count_of_eojeols = eojeol_counter._count_sum
        self._count_of_covered_eojeols = 0

        self.eojeol_counter = eojeol_counter

        if self.verbose:
            mem = '%.3f' % get_process_memory()
            message = '#eojeols={}, mem={} Gb'.format(self._num_of_eojeols, mem)
            self._print(message, replace=True, newline=True)

    def extract(self, candidates=None, min_predicator_frequency=1):
        """candidates is EojeolCounter or dict format"""

        # reset covered eojeol count
        self._num_of_covered_eojeols = 0

        predicators = self._extract_predicator(
            candidates, min_predicator_frequency)

        adjectives, verbs = self._separate_adjective_verb(predicators)

        return adjectives, verbs

    def _prepare_predicator_lrgraph(self):

        def contains_noun(eojeol):
            n = len(eojeol)
            for e in range(2, n + 1):
                if eojeol[:e] in self._nouns:
                    return True
            return False

        eojeols = {k:v for k, v in self.eojeol_counter._counter.items()}
        eojeols = {eojeol:count for eojeol, count in eojeols.items()
                   if len(eojeol) > 1 and not contains_noun(eojeol)}
        lrgraph = EojeolCounter()._to_lrgraph(eojeols)
        return lrgraph

    def _extract_eomi(self, lrgraph, min_num_of_features=5,
        min_eomi_score=0.3, min_eomi_frequency=1):

        eomi_extractor = EomiExtractor(
            lrgraph = lrgraph,
            stems = self._stems,
            nouns = self._nouns,
            min_num_of_features = min_num_of_features,
            verbose = self.verbose,
            logpath = None
        )

        extracted_eomis = eomi_extractor.extract(
            condition=None,
            min_eomi_score = min_eomi_score,
            min_eomi_frequency = min_eomi_frequency,
            reset_lrgraph=True
        )

        extracted_eomis = {eomi for eomi in extracted_eomis if not (eomi in self._eomis)}

        n_before = len(self._eomis)
        self._eomis.update(extracted_eomis)
        n_after = len(self._eomis)

        if self.verbose:
            message = 'eomis: {} -> {}'.format(n_before, n_after)
            self._print(message, replace=False, newline=True)

    def _extract_stem(self, lrgraph, min_num_of_unique_R_char=10, min_entropy_of_R_char=0.5,
        min_entropy_of_R=1.5, min_stem_score=0.7, min_stem_frequency=100):

        stem_extractor = StemExtractor(
            lrgraph = lrgraph,
            stems = self._stems,
            eomis = self._eomis,
            min_num_of_unique_R_char = min_num_of_unique_R_char,
            min_entropy_of_R_char = min_entropy_of_R_char,
            min_entropy_of_R = min_entropy_of_R
        )

        extracted_stems = stem_extractor.extract(
            L_ignore=None,
            min_stem_score = min_stem_score,
            min_stem_frequency = min_stem_frequency
        )

        extracted_stems = {stem for stem in extracted_stems if not (stem in self._stems)}

        n_before = len(self._stems)
        self._stems.update(extracted_stems)
        n_after = len(self._stems)

        if self.verbose:
            message = 'stems: {} -> {}'.format(n_before, n_after)
            self._print(message, replace=False, newline=True)

    def _extract_predicator(self, eojeol_counter=None, min_frequency=1):

        def all_character_are_complete_korean(s):
            for c in s:
                if not character_is_complete_korean(c):
                    return False
            return True

        if (eojeol_counter is None) or (not eojeol_counter):
            eojeol_counter = {
                eojeol:count for eojeol, count in self.eojeol_counter.items()
                if (count > min_frequency) and all_character_are_complete_korean(eojeol)
            }

        lemmas = self._as_lemma_candidates(eojeol_counter)

        # TODO
        # evaluation

        if self.verbose:
            message = '{} predicators are extracted'.format(len(lemmas))
            self._print(message, replace=True, newline=True)

        return lemmas

    def _as_lemma_candidates(self, eojeol_counter=None):

        def is_noun_josa(eojeol):
            for i in range(1, len(eojeol)):
                if ((eojeol[:i] in self._nouns) and
                    (eojeol[i:] in self._josas)):
                    return True
            return False

        self._num_of_covered_eojeols = 0
        self._count_of_covered_eojeols = 0

        lemmas = {}
        eomi_to_word_count = defaultdict(lambda: [])
        num_eojeol = len(eojeol_counter)

        for i, (eojeol, count) in enumerate(eojeol_counter.items()):
            if self.verbose and i % 5000 == 4999:
                message = 'lemmatizing {} / {} words'.format(i+1, num_eojeol)
                self._print(message, replace=True, newline=False)
            if is_noun_josa(eojeol):
                continue

            n = len(eojeol)
            lemma_candidates = set()

            for i in range(1, n+1):
                l, r = eojeol[:i], eojeol[i:]
                for stem, eomi in lemma_candidate(l, r):
                    if (stem in self._stems) and (eomi in self._eomis):
                        lemma_candidates.add((stem, eomi))

            lemma_candidates_ = set()
            for stem, eomi in lemma_candidates:
                if eojeol in conjugate(stem, eomi):
                    lemma_candidates_.add((stem, eomi))

            if lemma_candidates_:
                lemmas[eojeol] = Predicator(count, lemma_candidates_)
                for stem, eomi in lemma_candidates_:
                    eomi_to_word_count[eomi].append((eojeol, count))
                self._num_of_covered_eojeols += 1
                self._count_of_covered_eojeols += count

        lemmas = self._remove_wrong_eomis(lemmas, eomi_to_word_count)
        if self.verbose:
            perc = '%.3f' % (100 * self._count_of_covered_eojeols / self._count_of_eojeols)
            message = 'lemma candidating was done'
            self._print(message, replace=True, newline=True)

        return lemmas

    def _remove_wrong_eomis(self, lemmas, eomi_to_word_count):
        def noun_proportion(word_count):
            sum_ = sum(1 for w, v in word_count if len(w) == 2)
            prop = sum(1 for w, v in word_count if (w in self._nouns) and (len(w) == 2))
            prop_len2 = 0
            if sum_ > 0:
                prop /= sum_
                prop_len2 = sum_ / sum(1 for w, v in word_count)
            return prop, prop_len2

        #checks = {
        #    '안서', '어든', '아기', 'ㄴ라', 'ㄴ대', 'ㄴ기', '아두',
        #    'ㄴ단', 'ㄴ지', '어자', 'ㄴ물', 'ㄴ장', 'ㄴ타', 'ㄴ어',
        #    'ㄴ물', 'ㄴ고', 'ㄴ터', '물', '애', 'ㄴ가', '앙위'
        #}
        remove_eomis = set()
        remove_morphs = {}

        # find
        for eomi, word_count in eomi_to_word_count.items():
            if len(eomi) >= 3:
                continue

            prop, prop_len2 = noun_proportion(word_count)
            if prop < 0.5:
                continue

            if prop_len2 == 1:
                remove_eomis.add(eomi)
                remove_words = tuple(word for word, _ in word_count if len(word) == 2)
            else:
                remove_words = tuple(
                    word for word, _ in word_count
                    if ((len(word) == 2 and word in self._nouns) or
                        (len(word) == 3 and len(eomi) == 2))
                )
            remove_morphs[eomi] = remove_words

            # debug code
            #if eomi in checks:
            #    print(eomi, prop, prop_len2)
            #    for word in remove_words:
            #        print('  {}, {}'.format(word, word in self._nouns))

        # remove
        self._eomis = {eomi for eomi in self._eomis if not (eomi in remove_eomis)}
        if self.verbose:
            words = {word for words in remove_morphs.values() for word in words}
            message = '{} eomis are removed, {} words are modified.'.format(
                len(remove_eomis), len(words))
            self._print(message, replace=True, newline=True)

        for eomi, words in remove_morphs.items():
            for word in words:
                if not (word in lemmas):
                    continue
                predicator = lemmas[word]
                if len(predicator.lemma) == 1:
                    lemmas.pop(word)
                else:
                    lemmas[word] = Predicator(
                        predicator.frequency,
                        {lemma for lemma in predicator.lemma if lemma[1] != eomi})

        return lemmas

    def _separate_adjective_verb(self, predicators, num_threshold=3):
        adjectives = {}
        verbs = {}

        # proportion
        for word, predicator in predicators.items():
            frequency = predicator.frequency
            lemmas = predicator.lemma
            adj = set()
            v = set()
            for lemma in lemmas:
                # dictionary first, verb preference
                if lemma[0] in self._verb_stems:
                    v.add(lemma)
                    continue
                if lemma[0] in self._adjective_stems:
                    adj.add(lemma)
                    continue

                # others are extracted stems
                # rule based classifier, verb preference
                answer = rule_classify(lemma[0])
                if answer is 'Verb':
                    v.add(lemma)
                    continue
                if answer is 'Adjective':
                    adj.add(lemma)
                    continue

                # surfacial form test
                surfaces = conjugate_as_present(lemma[0])
                surfaces.update(conjugate_as_imperative(lemma[0]))
                surfaces.update(conjugate_as_pleasure(lemma[0]))
                surfaces = {surface for surface in surfaces
                            if surface in predicators}

                if len(surfaces) <= 1:
                    adj.add(lemma)
                else:
                    v.add(lemma)

            if adj:
                adjectives[word] = Predicator(frequency, adj)
            if v:
                verbs[word] = Predicator(frequency, v)

        return adjectives, verbs
