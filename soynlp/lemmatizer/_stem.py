import math

def extract_domain_stem(lrgraph, L, R, L_ignore=None,
    min_score_of_L=0.7, min_frequency_of_L=100, min_num_of_unique_R_char=10,
    min_entropy_of_R_char=0.5, min_entropy_of_R=1.5):

    if L_ignore is None:
        L_ignore = {}

    L_candidates = {}
    for r in R:
        for l, count in lrgraph.get_l(r, -1):
            if (l in L) or (l in L_ignore):
                continue
            L_candidates[l] = L_candidates.get(l, 0) + count

    # 1st. frequency filtering
    L_candidates = {l:count for l, count in L_candidates.items()
        if count >= min_frequency_of_L}

    L_extracted = _batch_predicting_L(lrgraph, L, R, L_candidates,
        min_score_of_L, min_frequency_of_L, min_num_of_unique_R_char,
        min_entropy_of_R_char, min_entropy_of_R)

    # L_extracted = _post_processing(L_extract, L, R)

    stems = _to_stem(L_extracted)
    return stems, L_extracted

def _batch_predicting_L(lrgraph, L, R, L_candidates, min_score_of_L,
    min_frequency_of_L, min_num_of_unique_R_char,
    min_entropy_of_R_char, min_entropy_of_R):

    # add known L for unknown L prediction
    L_extracted = {l:None for l in L}

    # from longer to shorter
    for l in sorted(L_candidates, key=lambda x:-len(x)):

        if (l in L) or (l in R) or (len(l) == 1) or (l[-1] == '다'):
            continue

        features = _get_R_features(l, lrgraph)
        score, freq = predict(l, features, L_extracted, R, min_score_of_L,
            min_frequency_of_L, min_num_of_unique_R_char, min_entropy_of_R_char)

        entropy_of_R = _entropy([v for _, v in features])

        if (score < min_score_of_L) or (freq < min_frequency_of_L):
            continue

        L_extracted[l] = (score, freq)

    # remove known L
    L_extracted = {l:score for l, score in L_extracted.items()
                   if not (l in L)}

    return L_extracted

def _get_R_features(l, lrgraph):
    features = lrgraph.get_r(l, -1)
    return [feature for feature in features if feature[0]]

def predict(l, features, L_extracted, R, min_score_of_L,
    min_frequency_of_L, min_num_of_unique_R_char, min_entropy_of_R_char):

    R_char = _count_first_chars(features)
    unique_R_char = len(R_char)
    entropy_of_R_char = _entropy(tuple(R_char.values()))
    if ((unique_R_char < min_num_of_unique_R_char) or
        (entropy_of_R_char < min_entropy_of_R_char)):
        return (0, 0)

    pos, neg, unk = _predict(l, features, L_extracted, R)
    score = (pos - neg) / (pos + neg) if (pos + neg) > 0 else 0
    freq = pos if score >= min_score_of_L else neg + unk

    if freq < min_frequency_of_L:
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