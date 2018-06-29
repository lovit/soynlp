import math

def extract_domain_pos_features(prediction_scores, lrgraph,
    known_pos_features, ignore_features=None,
    min_noun_score=0.3, min_noun_frequency=100,
    min_pos_score=0.3, min_pos_feature_frequency=1000,
    min_num_of_unique_lastchar=4, min_entropy_of_lastchar=0.5,
    min_noun_entropy=1.5):

    nouns = {noun for noun, score in prediction_scores.items()
             if ((score[0] >= min_noun_score) 
                 and (score[1] >= min_noun_frequency))}

    if ignore_features is None:
        ignore_features = {}

    pos_candidates = {}
    for noun in nouns:
        for r, count in lrgraph.get_r(noun, -1):
            if (r in known_pos_features) or (r in ignore_features):
                continue
            pos_candidates[r] = pos_candidates.get(r, 0) + count

    # 1st. frequency filtering (ignoring L is noun)
    pos_candidates = {r:count for r, count in pos_candidates.items() 
                      if count >= min_pos_feature_frequency}

    # add known pos features for unknown feature prediction
    domain_pos_features = {r:None for r in known_pos_features}

    # from shorter to longer
    for r in sorted(pos_candidates, key=lambda x:len(x)):

        if (r in known_pos_features) or (r in nouns):
            continue

        features = _get_noun_feature(r, lrgraph)
        score, freq = predict(r, features, nouns, domain_pos_features,
            min_pos_score, min_pos_feature_frequency,
            min_num_of_unique_lastchar, min_entropy_of_lastchar)

        # noun entropy
        noun_sum = sum((c for l, c in features if l in nouns))
        noun_entropy = [c/noun_sum for l, c in features if l in nouns]
        noun_entropy = sum([-math.log(p) * p for p in noun_entropy])

        if ((score >= min_pos_score) and
            (freq >= min_pos_feature_frequency) and
            (noun_entropy >= min_noun_entropy)):
            domain_pos_features[r] = (score, freq)

    # remove known features
    domain_pos_features = {r:score for r, score in domain_pos_features.items()
                           if not (r in known_pos_features)}

    return domain_pos_features

def _get_noun_feature(r, lrgraph):
    return [(l,c) for l, c in lrgraph.get_l(r, -1) if len(l) > 1]

def predict(r, features, nouns, pos_r,
    min_pos_score=0.3, min_pos_feature_frequency=30,
    min_num_of_unique_lastchar=4, min_entropy_of_lastchar=0.5):

    n_unique, n_sum, entropy = _last_character_criterions(features)
    if ((n_unique < min_num_of_unique_lastchar) or 
        (entropy < min_entropy_of_lastchar)):
        return (0, 0)

    pos, neg, unk = _predict(r, features, nouns, pos_r)
    score = (pos - neg) / (pos + neg) if (pos + neg) > 0 else 0
    freq = pos if score >= min_pos_score else neg + unk

    if freq < min_pos_feature_frequency :
        return (0, freq)
    else:
        return (score, freq)

def _predict(r, features, nouns, pos_r):

    pos, neg, unk = 0, 0, 0

    for l, freq in features:
        if len(l) <= 1:
            continue
        if _is_NJ(r, nouns, pos_r):
            neg += freq
        elif _exist_longer_noun(l, r, nouns):
            neg += freq
        elif l in nouns:
            pos += freq
        else:
            unk += freq

    return pos, neg, unk

def _exist_longer_noun(l, r, nouns):

    for i in range(1, len(r)+1):
        if (l + r[:i]) in nouns:
            return True
    return False

def _is_NJ(r, nouns, pos_r):

    n = len(r)
    for i in range(1, n):
        if r[:i] in nouns:
            for j in range(i, n):
                return r[j:] in pos_r
    return False

def _last_character_criterions(lcount):

    counter = {}
    for l, count in lcount:
        last = l[-1]
        counter[last] = counter.get(last, 0) + count

    n_unique = len(counter)
    n_sum = sum(counter.values())
    entropy = [freq/n_sum for freq in counter.values()]
    entropy = -1 * sum((p * math.log(p) for p in entropy))

    return n_unique, n_sum, entropy