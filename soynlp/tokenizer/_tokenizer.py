# -*- encoding:utf8 -*-
import sys
if sys.version_info <= (2,7):
    reload(sys)
    sys.setdefaultencoding('utf-8')
from pprint import pprint
import re
import numpy as np


class RegexTokenizer:
    
    def __init__(self):
        self._patterns = [
            ('number', re.compile(u'[-+]?\d*[\.]?[\d]+|[-+]?\d+', re.UNICODE)),
            ('korean', re.compile(u'[가-힣]+', re.UNICODE)),
            ('jaum', re.compile(u'[ㄱ-ㅎ]+', re.UNICODE)),
            ('moum', re.compile(u'[ㅏ-ㅣ]+', re.UNICODE)),
            ('english & latin', re.compile(u"[a-zA-ZÀ-ÿ]+[[`']?s]*|[a-zA-ZÀ-ÿ]+", re.UNICODE))
        ]
        
        self.doublewhite_pattern = re.compile('\s+')

    def __call__(self, s, debug=True, flatten=True):
        return self.tokenize(s, debug, flatten)

    def tokenize(self, s, debug=False, flatten=True):
        '''
        Usage
        
        s = "이거에서+3.12같은34숫자나-1.2like float해해 같은aÀÿfafAis`s-1찾아서3.1.2.1해ㅋㅋㅜㅠ봐 Bob`s job.1"
        tokenizer = RegularTokenizer()
        tokenizer.tokenize(s)

        [['이거에서', '+3.12', '같은', '34', '숫자나', '-1.2', 'like'],
         ['float', '해해'],
         ['같은', 'aÀÿfafAis`s', '-1', '찾아서', '3.1', '.2', '.1', '해', 'ㅋㅋ', 'ㅜㅠ', '봐'],
         ['Bob`s'],
         ['job', '.1']]
        '''
        tokens = [self._tokenize(t, debug) for t in s.split()]
        if flatten:
            tokens = [subtoken for token in tokens for subtoken in token if subtoken]
        return tokens
    
    def _tokenize(self, s, debug=False):
        for name, pattern in self._patterns:
            
            founds = pattern.findall(s)
            if not founds: 
                continue

            if debug:
                print('\n%s' % name)
                print(founds)
            
            found = founds.pop(0)
            len_found = len(found)
            
            s_ = ''
            b = 0
            for i, c in enumerate(s):
                
                if b > i: 
                    continue
                    
                if s[i:i+len_found] == found:
                    s_ += ' %s ' % s[i:i+len_found]
                    b = i + len_found
                    
                    if not founds:
                        s_ += s[b:]
                        break
                    else:
                        found = founds.pop(0)
                        len_found = len(found)
                    
                    continue
                s_ += c
            s = s_
            
        s = self.doublewhite_pattern.sub(' ', s).strip().split()
        # TODO: handle 3.1.2.1
        return s


class LTokenizer:
    
    def __init__(self, scores=None, default_score=0.0):
        self._scores = scores if scores else {}
        self._ds = default_score

    def __call__(self, sentence, tolerance=0.0, flatten=True, remove_r=False):
        return self.tokenize(sentence, tolerance, flatten, remove_r)

    def tokenize(self, sentence, tolerance=0.0, flatten=True, remove_r=False):
        
        def token_to_lr(token, tolerance=0.0):
            length = len(token)
            if length <= 2: return (token, '')
            candidates = [(token[:e], token[e:]) for e in range(2, length + 1)]
            candidates = [(self._scores.get(t[0], self._ds), t[0], t[1]) for t in candidates]
            if tolerance > 0:
                max_score = max([c[0] for c in candidates])
                candidates = [c for c in candidates if (max_score - c[0]) <= tolerance]
                best = sorted(candidates, key=lambda x:len(x[1]), reverse=True)[0]
            else:
                best = sorted(candidates, key=lambda x:(x[0], len(x[1])), reverse=True)[0]
            return (best[1], best[2])

        tokens = [token_to_lr(token, tolerance) for token in sentence.split()]
        
        if remove_r:
            tokens = [token[0] for token in tokens]
        
        if (flatten) and (remove_r == False):
            tokens = [subtoken for token in tokens for subtoken in token if subtoken]
        
        return tokens
    

class MaxScoreTokenizer:
    
    def __init__(self, scores=None, max_length=10, default_score=0.0):
        self._scores = scores if scores else {}
        self._max_length = max_length
        self._ds = default_score

    def __call__(self, sentence, flatten=True):
        return self.tokenize(sentence, flatten)

    def tokenize(self, sentence, flatten=True):
        tokens = [self._recursive_tokenize(token) for token in sentence.split()]
        if flatten:
            tokens = [subtoken[0] for token in tokens for subtoken in token]
        return tokens

    def _recursive_tokenize(self, token, range_l=0, debug=False):
        
        length = len(token)
        if length <= 2:
            return [(token, 0, length, self._ds, length)]

        if range_l == 0:
            range_l = min(self._max_length, length)

        scores = self._initialize(token, range_l, length)
        if debug:
            pprint(scores)
        
        result = self._find(scores)
        
        adds = self._add_inter_subtokens(token, result)
        
        if result[-1][2] != length:
            adds += self._add_last_subtoken(token, result)
            
        if result[0][1] != 0:
            adds += self._add_first_subtoken(token, result)
            
        return sorted(result + adds, key=lambda x:x[1])

    def _initialize(self, token, range_l, length):
        scores = []
        for b in range(0, length - 1):
            for r in range(2, range_l + 1):
                e = b + r
                
                if e > length: 
                    continue
                
                subtoken = token[b:e]
                score = self._scores.get(subtoken, self._ds)
                scores.append((subtoken, b, e, score, r))
                
        return sorted(scores, key=lambda x:(-x[3], -x[4], x[1]))

    def _find(self, scores):
        result = []
        num_iter = 0
        
        while scores:
            word, b, e, score, r = scores.pop(0)
            result.append((word, b, e, score, r))

            if not scores:
                break

            removals = []
            for i, (_1, b_, e_, _2, _3) in enumerate(scores):
                if (b_ < e and b < e_) or (b_ < e and e_ > b):
                    removals.append(i)

            for i in reversed(removals):
                del scores[i]

            num_iter += 1
            if num_iter > 100: break

        return sorted(result, key=lambda x:x[1])
    
    def _add_inter_subtokens(self, token, result):
        adds = []        
        for i, base in enumerate(result[:-1]):
            if base[2] == result[i+1][1]:
                continue
            
            b = base[2]
            e = result[i+1][1]
            subtoken = token[b:e]
            adds.append((subtoken, b, e, self._ds, e - b))
        
        return adds

    def _add_first_subtoken(self, token, result):
        e = result[0][1]
        subtoken = token[0:e]
        score = self._scores.get(subtoken, self._ds)
        return [(subtoken, 0, e, score, e)]
    
    def _add_last_subtoken(self, token, result):
        b = result[-1][2]
        subtoken = token[b:]
        score = self._scores.get(subtoken, self._ds)
        return [(subtoken, b, len(token), score, len(subtoken))]

class MaxLRScoreTokenizer:
    def __init__(self, Dl=None, Dr=None,
                 preference_l=None, preference_r=None,
                 lrgraph=None, tokenizer_builder=None,
                 max_lscore_difference=0.3, max_lscore_diffratio=0.5, # Expansion L
                 ensurable_score_l=0.5, ensurable_score_lr_diff=0.3   # R overlap L
                ):

        # Normalize L-R graph to prob graph
        def norm(rdict):
            sum_ = sum(rdict.values())
            return {r:c/sum_ for r,c in rdict.items()}
        self.lrgraph = lrgraph if lrgraph else {}
        self.lrgraph_norm = {l:norm(rdict) for l,rdict in self.lrgraph.items()}

        # Expanding dictionary from lrgraph
        #self.Dl, self.Dr = tokenizer_builder(self.lrgraph) if tokenizer_builder else LRTokenizerBuilder()(self.lrgraph)
        self.Dl, self.Dr = tokenizer_builder(self.lrgraph) if tokenizer_builder else ({}, {})
        
        # Dictionary type check
        if not Dl: Dl = {}
        if not Dr: Dr = {}

        if not type(Dl) == dict:
            Dl = {l:1.0 for l in Dl}
        self.Dl.update(Dl)
        if not type(Dr) == dict:
            Dr = {r:1.0 for r in Dr}
        self.Dr.update(Dr)

        # Add preference words into dictionary
        self.Pl = preference_l if preference_l else {}
        self.Pr = preference_r if preference_r else {}

        for l in self.Pl:
            if not (l in self.Dl):
                self.Dl[l] = 1.0
        for r in self.Pr:
            if not (r in self.Dr):
                self.Dr[r] = 1.0

        self.lmax = max((len(w) for w in self.Dl)) if self.Dl else 0
        self.rmax = max((len(w) for w in self.Dr)) if self.Dr else 0
        self.base_tokenizer = MaxScoreTokenizer(scores=self.Dr)
        
        self.max_lscore_difference = max_lscore_difference
        self.max_lscore_diffratio = max_lscore_diffratio
        self.ensurable_score_l = ensurable_score_l
        self.ensurable_score_lr_diff = ensurable_score_lr_diff

    def __call__(self, sent, debug=True, flatten=True):
        return self.tokenize(sent, debug, flatten)

    def tokenize(self, sent, debug=False, flatten=True):
        sent_ = [self._tokenize(t, debug) for t in sent.split() if t]
        if flatten:
            sent_ = [word for words in sent_ for word in words]
        return sent_

    def _tokenize(self, t, debug=False):
        candidates = self._initialize(t)
        candidates_ = self._remove_l_subset(candidates)
        scores = self._score(candidates_)
        best = self._find_best(scores)
        
        if best:
            post = self._postprocessing(t, best)
        else:
            post = self._base_tokenizing_subword(t, 0)

        if not debug:
            post = [[(p[0], 'L'), (p[1], 'R')] for p in post]
            post = [w for p in post for w in p if w[0]]
        return post
    
    def _initialize(self, t):
        candidates = self._initialize_L(t)
        candidates = self._initialize_LR(t, candidates)
        return candidates

    def _initialize_L(self, t):
        n = len(t)
        candidates = []
        for b in range(n):
            for e in range(b+1, min(n, b+self.lmax)+1):
                l = t[b:e]
                if not (l in self.Dl):
                    continue
                candidates.append([l,     # 0
                                   b,     # 1                  
                                   e,     # 2
                                   e-b    # 3
                                  ])
        return candidates

    def _initialize_LR(self, t, candidates):
        n = len(t)
        expanded = []
        for (l, b, e, len_l) in candidates:
            for len_r in range(min(self.rmax, n-e)+1):
                if len_l == 1 and len_r == 0:
                    continue
                r = t[e:e+len_r]
                if r and not (r in self.Dr):
                    continue
                expanded.append([l,
                                 r,
                                 b,
                                 e,
                                 e + len_r,
                                 len_l,
                                 len_r,
                                 len_l + len_r,
                                ])
        return sorted(expanded, key=lambda x:x[4])
    
    def _remove_l_subset(self, candidates):    
        for c in candidates:
            c.append(self.Dl.get(c[0], 0))
            c.append(self.Dr.get(c[1], 0))
        candidates = sorted(candidates, key=lambda x:-x[-2])

        candidates_ = []
        while candidates:
            best = candidates.pop(0)
            b, e, lscore = best[2], best[3], best[-2]

            exist_longer = False
            for c in candidates:
                if c[2] > b or c[3] < e or not (c[2] < b or c[3] > e):
                    continue
                if ((lscore - c[-2]) < self.max_lscore_difference) or \
                    ((self.ensurable_score_l * 0.5 < lscore) and \
                        ((lscore+1e-5) / (c[-2]+1e-5) < self.max_lscore_diffratio)):
                    exist_longer = True
                    break

            if not exist_longer:
                candidates_.append(best)

        return candidates_

    def _score(self, candidates):
        from collections import defaultdict
        # With checking R is overlapped next L
        begin_to_words = defaultdict(lambda: [])
        for c in candidates:
            begin_to_words[c[2]].append(c)
        begin_to_words = dict(begin_to_words)

        scored = []

        candidates = sorted(candidates, key=lambda x:(-x[-2], -x[-1], x[2], -x[5]))
        while candidates:
            c = candidates.pop(0)
            l, r, p0, p1, p2, len_l, len_r, len_lr, score_l, score_r = c

            # Check whether R is overlapped next L
            if len_r:
                overlappped = False
                for b in range(p1, p2):
                    if overlappped:
                        break
                    for word in begin_to_words.get(b, []):
                        score_diff = word[-2] + self.Pl.get(word[0], 0) - score_r
                        if (self.ensurable_score_l <= word[-2]) or (score_diff > self.ensurable_score_lr_diff):
                            overlappped = True
                            break
                if overlappped:
                    continue

            total_score = (score_l * 2 if not r else score_l + score_r) + self.Pl.get(l, 0) + self.Pr.get(r, 0)
            c.append(total_score)
            scored.append(c)
        return scored

    def _find_best(self, scores):
        best = []
        sorted_ = sorted(scores, key=lambda x:-x[-1])
        while sorted_:
            best.append(sorted_.pop(0))
            (b, e) = (best[-1][2], best[-1][4])
            removals = [i for i, c in enumerate(sorted_) if b < c[4] and e > c[2]] # Overlap
            for idx in reversed(removals):
                del sorted_[idx]
        return sorted(best, key=lambda x:x[2])

    def _postprocessing(self, t, words):
        n = len(t)
        adds = []
        if words and words[0][2] > 0:
            adds += self._add_first_subword(t, words)
        if words and words[-1][3] < n:
            adds += self._add_last_subword(t, words, n)
        adds += self._add_inter_subwords(t, words)
        post = [w for w in words] + adds
        return sorted(post, key=lambda x:x[2])

    def _add_inter_subwords(self, t, words):
        adds = []        
        for i, base in enumerate(words[:-1]):
            if base[4] == words[i+1][2]:
                continue
            b = base[4]
            e = words[i+1][2]
            subword = t[b:e]
            adds += self._base_tokenizing_subword(subword, b)
        return adds

    def _add_last_subword(self, t, words, n):
        b = words[-1][3]
        subword = t[b:]
        return self._base_tokenizing_subword(subword, b)

    def _add_first_subword(self, t, words):    
        e = words[0][2]
        subword = t[0:e]
        return self._base_tokenizing_subword(subword, 0)

    def _base_tokenizing_subword(self, t, b):
        words = self.base_tokenizer.tokenize(t)
        words_ = []
        b_ = 0
        for w in words:
            n = len(w)
            # TODO: 여기를 바꿔야해
            if w in self.Dr:
                words_.append(['', w, b+b_, b+b_, b+n, 0, n, n, 0, self.base_tokenizer.scores[w]])
            else:
                words_.append([w, '', b+b_, b+n, b+n, n, 0, n, 0, 0])
        return words_