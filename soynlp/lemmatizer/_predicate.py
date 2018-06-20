""" TERM DEFINITION
(l, r) : L and R position subwords
root : root of Adjective and Verb
ending : suffix, canonical form of ending

roots : set of root including Adjectives and Verbs
composable_roots : roots that can be compounded with other prefix
    - [] + 하다 : 덕질+하다, 냐옹+하다, 냐옹+하냥
endings : set of ending
pos_l_features : canonical form set of roots (L subwords)
pos_composable_l_features : canonical form set of composable roots (L subwords)
lrgraph : L-R graph including [Root + Ending], Adverbs, 
          and maybe some Noun + Josa
"""

from soynlp.utils import LRGraph

def predict_r(r, minimum_r_score=0.3, debug=False):
    raise NotImplemented

def _predict_r(features, r):
    raise NotImplemented

def _exist_longer_l(l, r):
    raise NotImplemented

def _has_composable_l(l, r):
    raise NotImplemented

def _refine_features(features, r):
    return [(l, count) for l, count in features if
        (l in pos_l_features and not _exist_longer_l(l, r))]