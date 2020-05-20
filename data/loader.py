import os
from glob import glob


installpath = os.path.abspath(os.path.dirname(__file__))


def load(idx='134963', mode='norm', max_samples=-1):
    """
    Args:
	idx: str
            movie idx
        mode: str
            `mode` = 'norm' or not
        max_samplse: int
            Number of maximal samples

    Returns:
        texts: list of str
            Movie comments
        scores: list of int
            Annotated scores
    """
    suffix = '' if mode != 'norm' else '_norm'
    paths = glob(f'{installpath}/{idx}{suffix}.txt')
    if not paths:
        raise ValueError(f'Not found file. Check idx {idx}')
    with open(paths[0], encoding='utf-8') as f:
        docs = [line.strip() for line in f]
    docs = [line.rsplit('\t', 1) for line in docs]
    docs = [row for row in docs if len(row) == 2]
    if max_samples > 0:
        docs = docs[:max_samples]
    texts, scores = zip(*docs)
    scores = [int(s) for s in scores]
    return texts, scores

