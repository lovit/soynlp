# -*- encoding:utf8 -*-
import sys
from collections import namedtuple

# b:begin, m: middle, e: end
LR = namedtuple('LR', 'l l_tag r r_tag b m e')

class BaseTemplateMatcher:
    def generate(self, token):
        raise NotImplementedError

class EojeolTemplateMatcher(BaseTemplateMatcher):
    def __init__(self, dictionary, single_tags=None, lr_templates=None):
        if not single_tags:
            single_tags = ['Noun', 'Verb', 'Adjective', 'Adverb', 'Exclamation']
        if not lr_templates:
            lr_templates = [('Noun', 'Verb'), ('Noun', 'Adjective'), ('Noun', 'Josa')]
        self.dictionary = dictionary
        self.single_tags = single_tags
        self.lr_templates = lr_templates

    def generate(self, eojeol):
        n = len(eojeol)
        candidates = []
        for tag in self.dictionary.get_pos(eojeol):
            if tag in self.single_tags:
                candidates.append([LR(eojeol, tag, '', None, 0, n, n)])
        if not candidates:
            candidates.append([LR(eojeol, None, '', None, 0, n, n)])

        for b in range(1, n):
            l, r = eojeol[:b], eojeol[b:]
            for l_tag, r_tag in self.lr_templates:
                if self.dictionary.word_is_tag(l, l_tag) and self.dictionary.word_is_tag(r, r_tag):
                    candidates.append([LR(l, l_tag, r, r_tag, 0, b, n)])
        
        compound_noun = self._decompose_compound(eojeol, 'Noun')
        if compound_noun:
            candidates.append(compound_noun)
            
        compound_adverb = self._decompose_compound(eojeol, 'Adverb')
        if compound_adverb:
            candidates.append(compound_adverb)

        return candidates

    def _decompose_compound(self, eojeol, tag):
        n, b = len(eojeol), 0
        words = []
        while b < n:
            e = b + 1
            next_round = False
            for i in range(e, n+1):
                if next_round or (b == 0 and i == n):
                    break
                subword = eojeol[b:i]
                if self.dictionary.word_is_tag(subword, tag):
                    words.append(LR(subword, tag, '', None, b, i, i))
                    b = i
                    next_round = True
                    break
            if not next_round:
                return []
        return words

class LRTemplateMatcher(BaseTemplateMatcher):
    def __init__(self, dictionary, ltags=None, templates=None):
        if not ltags:
            ltags = {'Noun', 'Adjective', 'Verb', 'Adverb', 'Exclamation'}
        if not templates:
            templates = {'Noun': ('Josa', 'Verb', 'Adjective')}
        
        self.dictionary = dictionary
        self.ltags = ltags
        self.rtags = {tag for tags in templates.values() for tag in tags}
        self.templates = templates
            
    def generate(self, token):
        if sys.version_info.major == 2:
            token = unicode(token)
        candidates = self._initialize_L(token)
        candidates = self._expand_R(token, candidates)
        return candidates
    
    def _pos_L(self, word):
        poses = self.dictionary.get_pos(word)
        poses = {pos for pos in poses if pos in self.ltags}
        return poses

    def _initialize_L(self, t):
        n = len(t)
        candidates = []
        
        for b in range(n):
            for e in range(b+2, min(n, b + self.dictionary.max_length) + 1):
                l = t[b:e]
                l_tags = self._pos_L(l)
                
                if not l_tags:
                    continue
                    
                for l_tag in l_tags:
                    candidates.append([l, l_tag, b, e])
                    
        # candidates = self._remove_subset_l(candidates)
        return sorted(candidates, key=lambda x:x[2])

    def _remove_subset_l(self, candidates):
        candidates_ = []
        for l_tag in self.ltags:
            
            # Sort by length of L
            sorted_ = sorted(
                filter(lambda x:x[1] == l_tag, candidates), 
                key=lambda x:-(x[3] - x[2])
            )
            
            while sorted_:
                candidates_.append(sorted_.pop(0))
                (b, e) = (candidates_[-1][2], candidates_[-1][3])
                
                 # Find overlapped
                removals = [i for i, c in enumerate(sorted_) if b <= c[2] and e >= c[3]]
                
                for idx in reversed(removals):
                    del sorted_[idx]
                    
        return candidates_

    def _expand_R(self, t, candidates):
        n = len(t)
        expanded = []

        for (l, l_tag, b, e1) in candidates:
            last = min(self.dictionary.max_length + e1, n)
            
            for e2 in range(e1, last + 1):
                r = t[e1:e2]
                
                if not r:
                    expanded.append(LR(l, l_tag, r, None, b, e1, e2))
                else:
                    for r_tag in self.templates.get(l_tag, []):
                        if not self.dictionary.word_is_tag(r, r_tag):
                            continue
                        expanded.append(LR(l, l_tag, r, r_tag, b, e1, e2))
                        
       #  expanded = self._remove_subset_r(expanded)
        return sorted(expanded, key=lambda x:x.b)

    def _remove_subset_r(self, expanded):
        expanded_ = []
        rtags = [rtag for rtag in self.rtags] + [None]
        for r_tag in rtags:
            
            # Sory by length of R
            sorted_ = sorted(
                filter(lambda x:x.r_tag == r_tag, expanded),
                key=lambda x:-(x.e - x.m)
            )
            
            while sorted_:
                expanded_.append(sorted_.pop(0))
                
                l_tag = expanded_[-1].l_tag
                (b, m, e) = expanded_[-1][-3:]
                
                 # Find subset
                removals = [i for i, c in enumerate(sorted_) 
                            if ((c.l_tag == l_tag) 
                                and (b == c.b and m == c.m)
                                and (m <= c.m and e >= c.e))
                           ]

                for idx in reversed(removals):
                    del sorted_[idx]

        return expanded_