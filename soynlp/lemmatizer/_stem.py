import math

class StemExtractor:

    def __init__(self, lrgraph, L, R, min_num_of_unique_R_char=10,
        min_entropy_of_R_char=0.5, min_entropy_of_R=1.5):

        self.lrgraph = lrgraph
        self.L = L
        self.R = R
        self.min_num_of_unique_R_char = min_num_of_unique_R_char
        self.min_entropy_of_R_char = min_entropy_of_R_char
        self.min_entropy_of_R = min_entropy_of_R

    def extract(self, L_ignore=None, minimum_stem_score=0.7,
        minimum_frequency=100):

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
            if count >= minimum_frequency}

        extracted = self._batch_prediction(
            candidates, minimum_stem_score, minimum_frequency)

        # extracted = _post_processing(extract, self.L, self.R)

        stems = _to_stem(extracted)
        return stems, extracted

    def _batch_prediction(self, candidates,
        minimum_stem_score, minimum_frequency):

        # add known L for unknown L prediction
        extracted = {l:None for l in self.L}

        # from longer to shorter
        for l in sorted(candidates, key=lambda x:-len(x)):

            if ((l in self.L) or
                (l in self.R) or
                (len(l) == 1) or
                (l[-1] == '다') or
                (l in extracted)):
                continue

            features = _get_R_features(l, self.lrgraph)
            score, freq = predict(l, features, extracted, self.R, minimum_stem_score,
                minimum_frequency, self.min_num_of_unique_R_char, self.min_entropy_of_R_char)

            # no use entropy of R ?
            # entropy_of_R = _entropy([v for _, v in features])

            if (score < minimum_stem_score) or (freq < minimum_frequency):
                continue

            extracted[l] = (score, freq)

        # remove known L
        extracted = {l:score for l, score in extracted.items() if not (l in self.L)}

        return extracted

def _get_R_features(l, lrgraph):
    features = lrgraph.get_r(l, -1)
    return [feature for feature in features if feature[0]]

def predict(l, features, L_extracted, R, minimum_stem_score,
    minimum_frequency, min_num_of_unique_R_char, min_entropy_of_R_char):

    R_char = _count_first_chars(features)
    unique_R_char = len(R_char)
    entropy_of_R_char = _entropy(tuple(R_char.values()))
    if ((unique_R_char < min_num_of_unique_R_char) or
        (entropy_of_R_char < min_entropy_of_R_char)):
        return (0, 0)

    pos, neg, unk = _predict(l, features, L_extracted, R)
    score = (pos - neg) / (pos + neg) if (pos + neg) > 0 else 0
    freq = pos if score >= minimum_stem_score else neg + unk

    if freq < minimum_frequency:
        return (0, freq)
    else:
        return (score, freq)

def _predict(l, features, L_extracted, R):
    pos, neg, unk = 0, 0, 0
    for r, freq in features:
        if r in R:
            pos += freq
        elif _r_is_PredicateEomi(r, L_extracted, R):
            neg += freq
        elif _exist_longer_eomi(l, r, R):
            neg += freq
        else:
            unk += freq
    return pos, neg, unk

def _r_is_PredicateEomi(r, L, R):
    n = len(r)
    for i in range(1, n):
        if (r[:i] in L) and (r[i:] in R):
            return True
    return False

def _exist_longer_eomi(l, r, R):
    for i in range(1, len(l)+1):
        if (l[-i:] + r) in R:
            return True
    return False

def _count_first_chars(rcount):
    counter = {}
    for r, count in rcount:
        if not r:
            continue
        char = r[0]
        counter[char] = counter.get(char, 0) + count
    return counter

def _entropy(counts):
    if not counts or len(counts) == 1:
        return 0
    sum_ = sum(counts)
    entropy = [v/sum_ for v in counts]
    entropy = -1 * sum((p * math.log(p) for p in entropy))
    return entropy

def _post_processing(L_extracted, L, R):
    def is_stem_and_eomi(l):
        n = len(l)
        for i in range(1, n):
            if not ((l[:i] in L) or (l[:i] in L_extracted)):
                continue
            for j in range(i+1, n+1):
                if l[i:j] in R:
                    return True
        return False

    def exist_subword(l):
        for i in range(2, len(l)):
            if l[:i] in L_extracted:
                return True
        return False

    removals = set()
    for l in sorted(L_extracted, key=lambda x:len(x)):
        if is_stem_and_eomi(l) or exist_subword(l):
            removals.add(l)
    extracteds = {l:score for l, score in L_extracted.items()
                  if not (l in removals)}

    return extracteds, removals

def _to_stem(L_extracted):
    # TODO
    return L_extracted