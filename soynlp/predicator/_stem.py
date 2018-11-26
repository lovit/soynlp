import math
from soynlp.lemmatizer import conjugate
from soynlp.lemmatizer import lemma_candidate

class StemExtractor:

    def __init__(self, lrgraph, stems, eomis, min_num_of_unique_R_char=10,
        min_entropy_of_R_char=0.5, min_entropy_of_R=1.5, verbose=True):

        self.lrgraph = lrgraph
        self.stems = stems
        self.eomis = eomis
        self.min_num_of_unique_R_char = min_num_of_unique_R_char
        self.min_entropy_of_R_char = min_entropy_of_R_char
        self.min_entropy_of_R = min_entropy_of_R
        self.verbose = verbose

        # L : stem surfaces
        # R : eomi surfaces
        self.L, self.R = self._conjugate_stem_and_eomi(lrgraph, stems, eomis)
        self._josa = {
            '거나', '게', '게는', '게도', '고',
            '고도', '고만', '는', '다', '다가',
            '서는', '아', '은'
        }

    def _print(self, message, replace=False, newline=True):
        header = '[Stem Extractor]'
        if replace:
            print('\r{} {}'.format(header, message),
                  end='\n' if newline else '', flush=True)
        else:
            print('{} {}'.format(header, message),
                  end='\n' if newline else '', flush=True)

    def _conjugate_stem_and_eomi(self, lrgraph, stems, eomis):
        eojeol_counter = lrgraph.to_EojeolCounter()

        stem_surfaces = set()
        eomi_surfaces = set()

        n_stems = len(stems)
        n_eomis = len(eomis)
        for i, stem in enumerate(stems):

            if self.verbose and i % 100 == 0:
                message = 'Checking combination of {} / {} stems + {} eomis'.format(
                    i, n_stems, n_eomis)
                self._print(message, replace=True, newline=False)

            stem_len = len(stem)
            for eomi in eomis:
                try:
                    for word in conjugate(stem, eomi):
                        if (eojeol_counter[word] == 0) or (len(word) <= stem_len):
                            continue
                        l, r = word[:stem_len], word[stem_len:]
                        stem_surfaces.add(l)
                        eomi_surfaces.add(r)
                except:
                    continue

        if self.verbose:
            message = 'Initializing was done with {} stems and {} eomis{}'.format(
                len(stems), len(eomis), ' ' * 10)
            self._print(message, replace=True, newline=True)

        del eojeol_counter
        return stem_surfaces, eomi_surfaces

    def extract(self, L_ignore=None, min_stem_score=0.7,
        min_stem_frequency=100):

        if L_ignore is None:
            L_ignore = {}

        candidates = {}
        for r in self.R:
            for l, count in self.lrgraph.get_l(r, -1):
                if (l in self.L) or (l in L_ignore):
                    continue
                candidates[l] = candidates.get(l, 0) + count

        # 1st. frequency filtering
        candidates = {l:count for l, count in candidates.items()
            if count >= min_stem_frequency}

        if self.verbose:
            message = 'batch prediction for {} candidates'.format(len(candidates))
            self._print(message, replace=False, newline=True)

        stem_surfaces = self._batch_prediction(
            candidates, min_stem_score, min_stem_frequency)

        self.stem_surfaces, self.removals = self._post_processing(
            stem_surfaces)

        self.stems = self._to_stem(self.stem_surfaces)
        # TODO extracted 된 canonical form 이 다른 어미들과 조합되어 다르게 활용될 수 있는지 확인

        if self.verbose:
            message = '{} stems, {} surfacial stems, {} removals'.format(
                len(self.stems), len(self.stem_surfaces), len(self.removals))
            self._print(message, replace=False, newline=True)

        return self.stems

    def _batch_prediction(self, candidates,
        min_stem_score, min_frequency):

        # add known L for unknown L prediction
        extracted = {l:None for l in self.L}

        # from longer to shorter
        for l in sorted(candidates, key=lambda x:-len(x)):

            if ((l in self.L) or
                (l in self.R) or
                (len(l) == 1) or
                (l[-1] == '다') or # Hard coding rule
                (l in extracted)):
                continue

            score, freq = self.predict(l,
                min_stem_score, min_frequency)

            # no use entropy of R ?
            # entropy_of_R = _entropy([v for _, v in features])

            if (score < min_stem_score) or (freq < min_frequency):
                continue

            extracted[l] = (score, freq)

        # remove known L
        extracted = {l:score for l, score in extracted.items() if not (l in self.L)}

        return extracted

    def predict(self, l, min_stem_score=0.7, min_frequency=1, debug=False):

        features = self.lrgraph.get_r(l, -1)
        char_count = self._count_first_chars(features)

        unique_of_char = len(char_count)
        #entropy_of_char = self._entropy(tuple(char_count.values()))
        entropy_of_char = self._entropy(
            tuple(self._select_pos_features(l, features).values())
        )

        pos, neg, unk = self._predict(l, features)
        score = (pos - neg) / (pos + neg) if (pos + neg) > 0 else 0
        freq = pos if score >= min_stem_score else neg + unk

        if debug:
            print('pos={}, neg={}, unk={}, n_features_={}, n_char={}, entropy_r={}'.format(
                pos, neg, unk, len(features), unique_of_char, entropy_of_char))

        if ((unique_of_char < self.min_num_of_unique_R_char) or
            (entropy_of_char < self.min_entropy_of_R_char)):
            return (0, 0)

        if freq < min_frequency:
            return (0, freq)
        else:
            return (score, freq)

    def _predict(self, l, features):
        pos, neg, unk = 0, 0, 0
        for r, freq in features:
            if r in self._josa: # 조사로도 이용되는 어미는 skip
                continue
            if not r:
                neg += freq
            elif r in self.R:
                pos += freq
            elif self._r_is_predicator(r):
                neg += freq
            elif self._exist_longer_eomi(l, r):
                neg += freq
            else:
                unk += freq
        return pos, neg, unk

    def _count_first_chars(self, features):
        char_count = [(r[0], count) for r, count in features if r]
        counter = {}
        for char, count in char_count:
            counter[char] = counter.get(char, 0) + count
        return counter

    def _entropy(self, counts):
        if len(counts) <= 1:
            return 0
        sum_ = sum(counts)
        entropy = [v/sum_ for v in counts]
        entropy = -1 * sum((p * math.log(p) for p in entropy))
        return entropy

    def _select_pos_features(self, l, features):
        def is_pos(r):
            if (r in self._josa) or self._r_is_predicator(r) or self._exist_longer_eomi(l, r):
                return False
            return r in self.R
        return {r:count for r, count in features if is_pos(r)}

    def _r_is_predicator(self, r):
        n = len(r)
        for i in range(1, n):
            if (r[:i] in self.L) and (r[i:] in self.R):
                return True
        return False

    def _exist_longer_eomi(self, l, r):
        for i in range(1, len(l)+1):
            if (l[-i:] + r) in self.R:
                return True
        return False

    # TODO check postprocessing rule
    def _post_processing(self, extracted):
        def is_stem_and_eomi(l):
            n = len(l)
            for i in range(1, n):
                if not ((l[:i] in self.L) or (l[:i] in extracted)):
                    continue
                for j in range(i+1, n+1):
                    if l[i:j] in self.R:
                        return True
            return False

        def exist_subword(l):
            for i in range(2, len(l)):
                if l[:i] in extracted:
                    return True
            return False

        removals = set()
        for l in sorted(extracted, key=lambda x:len(x)):
            if is_stem_and_eomi(l) or exist_subword(l):
                removals.add(l)
        extracted = {l:score for l, score in extracted.items()
                     if not (l in removals)}

        return extracted, removals

    def _to_stem(self, surfaces):

        def merge_score(freq0, score0, freq1, score1):
            return (freq0 + freq1, (score0 * freq0 + score1 * freq1) / (freq0 + freq1))

        stems = {}
        for l, (freq0, score0) in surfaces.items():
            for r, count in self.lrgraph.get_r(l, -1):
                try:
                    for stem, eomi in lemma_candidate(l, r):
                        if eomi in self.eomis:
                            continue
                        stems[stem] = merge_score(
                            freq0, score0, *stems.get(stem, (0, 0)))
                except:
                    continue

        return stems