from collections import Counter
import numpy as np
from ._hangle import split_jamo as _jamo

def levenshtein(s1, s2, cost={}):
    # based on Wikipedia/Levenshtein_distance#Python
    if len(s1) < len(s2):
        return levenshtein(s2, s1)

    if len(s2) == 0:
        return len(s1)
    
    def get_cost(c1, c2, cost):
        return 0 if (c1 == c2) else cost.get((c1, c2), 1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1 # j+1 instead of j since previous_row and current_row are one character longer
            deletions = current_row[j] + 1       # than s2
            substitutions = previous_row[j] + get_cost(c1, c2, cost)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]

def jamo_levenshtein(s1, s2):
    if len(s1) < len(s2):
        return jamo_levenshtein(s2, s1)

    if len(s2) == 0:
        return len(s1)
    
    def get_jamo_cost(c1, c2):
        return 0 if (c1 == c2) else levenshtein(_jamo(c1), _jamo(c2))/3

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1 # j+1 instead of j since previous_row and current_row are one character longer
            deletions = current_row[j] + 1       # than s2
            substitutions = previous_row[j] + get_jamo_cost(c1, c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]

def cosine_distance(s1, s2, unitfy=lambda x:Counter(x)):
    '''distance = 1 - cosine similarity; [0, 2] '''
    if (not s1) or (not s2):
        return 2
    
    d1 = unitfy(s1)
    d2 = unitfy(s2)
    prod = 0
    for c1, f in d1.items():
        prod += (f * d2.get(c1, 0))
    return 1 - ( prod / np.sqrt( (sum([f**2 for f in d1.values()]) * sum([f**2 for f in d2.values()])) ) )

def jaccard_distance(s1, s2, unitfy=lambda x:set(x)):
    if (not s1) or (not s2):
        return 1
    
    s1_set = unitfy(s1)
    s2_set = unitfy(s2)
    return 1 - len(s1_set.intersection(s2_set)) / len(s1_set.union(s2_set))
