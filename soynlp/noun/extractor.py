def _load_text(fname):
    with open(fname, encoding='utf-8') as f:
        lines = [line.strip() for line in f]
    return lines


def _load_predictor_coefficients(fname):
    tuples = _load_text(fname)
    tuples = [t.split() for t in tuples]
    tuples = [(t[0], float(t[1])) for t in tuples if len(t) == 2]
    pos, neg, common = {}, {}, {}
    for r, score in tuples:
        if score > 0:
            pos[r] = max(pos.get(r, 0), score)
        elif score < 0:
            neg[r] = min(neg.get(r, 0), score)
        else:
            common[r] = 0
    return pos, neg, common


def _load_predictor_lists(pos_fname, neg_fname):
    pos = {r: 1.0 for r in _load_text(pos_fname)}
    neg = {r: -1.0 for r in _load_text(neg_fname)}
    common = {r: 0 for r in pos if r in neg}
    pos = {r: coef for r, coef in pos.items() if r in common}
    neg = {r: coef for r, coef in neg.items() if r in common}
    return pos, neg, common