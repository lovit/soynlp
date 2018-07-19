import math

def extract_domain_stem(lrgraph, L, R, L_ignore=None,
    min_L_score=0.7, min_L_frequency=100, min_num_of_unique_firstchar=10,
    min_entropy_of_firstchar=0.5, min_stem_entropy=1.5):

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
        if count >= min_L_frequency}

    L_extracted = _batch_predicting_L(
        lrgraph, L, R, L_candidates,
        min_L_score, min_L_frequency,
        min_num_of_unique_firstchar, min_stem_entropy)

    stems = _to_stem(L_extracted)
    return stems, L_extracted

def _batch_predicting_L(lrgraph, L, R, L_candidates, min_L_score,
    min_L_frequency, min_num_of_unique_firstchar, min_stem_entropy):

    # add known L for unknown L prediction
    L_extracted = {l:None for l in L}

    # from longer to shorter
    for l in sorted(L_candidates, key=lambda x:-len(x)):

        if (l in L) or (l in R) or (len(l) == 1) or (l[-1] == '다'):
            continue

        features = _get_R_features(l, lrgraph)
        score, freq = predict(l, features, L_extracted, R, min_L_score,
            min_L_frequency, min_num_of_unique_firstchar, min_stem_entropy)

        # Stem 에 맞게 변형
#         # noun entropy
#         noun_sum = sum((c for l, c in features if l in nouns))
#         noun_entropy = [c/noun_sum for l, c in features if l in nouns]
#         noun_entropy = sum([-math.log(p) * p for p in noun_entropy])

        if (score < min_L_score) or (freq < min_L_frequency):
            continue

        L_extracted[l] = (score, freq)

    # remove known L
    L_extracted = {l:score for l, score in L_extracted.items()
                   if not (l in L)}

    return L_extracted

def _get_R_features(l, lrgraph):
    features = lrgraph.get_r(l, -1)
    return [feature for feature in features if feature[0]]

def predict(l, features, L_extracted, R, min_L_score,
    min_L_frequency, min_num_of_unique_firstchar, min_L_entropy):

    n_unique, entropy = _first_character_criterions(features)
    if ((n_unique < min_num_of_unique_firstchar) or
        (entropy < min_L_entropy)):
        return (0, 0)

    pos, neg, unk = _predict(l, features, L_extracted, R)
    score = (pos - neg) / (pos + neg) if (pos + neg) > 0 else 0
    freq = pos if score >= min_L_score else neg + unk

    if freq < min_L_frequency:
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

def _first_character_criterions(rcount):
    counter = {}
    for r, count in rcount:
        first = r[-1]
        counter[first] = counter.get(first, 0) + count

    n_unique = len(counter)
    n_sum = sum(counter.values())
    entropy = [freq/n_sum for freq in counter.values()]
    entropy = -1 * sum((p * math.log(p) for p in entropy))
    return n_unique, entropy

def _to_stem(L_extracted):
    # TODO
    return L_extracted