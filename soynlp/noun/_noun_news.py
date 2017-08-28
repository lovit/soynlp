class NewsNounExtractor:
    
    def __init__(self, l_max_length=10, r_max_length=7, predictor_fnames=None, base_noun_dictionary=None, verbose=True):
        self.l_max_length = l_max_length
        self.r_max_length = r_max_length
        self.verbose = verbose
        self.r_scores = {}
        self.base_noun_dictionary = base_noun_dictionary if base_noun_dictionary else {}
        self.noun_dictionary = {}
        
        if not predictor_fnames:
            import os
            directory = '/'.join(os.path.abspath(__file__).replace('\\', '/').split('/')[:-2])
            predictor_fnames = ['%s/trained_models/noun_predictor_sejong' % directory]
            if verbose:
                print('used default noun predictor; Sejong corpus based logistic predictor')
        for fname in predictor_fnames:
            self._load_predictor(fname)
        self.josa_dictionary = {r for r, s in self.r_scores.items() if s > 0.1}
        self.josa_dictionary.update({'는'})

    def _load_predictor(self, fname):
        try:
            with open(fname, encoding='utf-8') as f:
                for num_line, line in enumerate(f):
                    r, score = line.split('\t')
                    score = float(score)
                    self.r_scores[r] = max(self.r_scores.get(r, 0), score)
        except FileNotFoundError:
            print('predictor file was not found')
        except Exception as e:
            print(' ... %s parsing error line (%d) = %s' % (e, num_line, line))
            
    def train_extract(self, sents, min_count=3, minimum_noun_score=0.4, minimum_feature_proportion=0.6, wordset=None):
        self.train(sents)        
        return self.extract(wordset, min_count, minimum_noun_score, minimum_feature_proportion)
    
    def train(self, sents):
        if self.verbose:
            print('scan vocabulary ... ', end='')
        self.lrgraph, self.rlgraph, self.eojeols = self._build_graph(sents)
        self.lcount = {k:sum(d.values()) for k,d in self.lrgraph.items()}
        self.rcount = {k:sum(d.values()) for k,d in self.rlgraph.items()}
        if self.verbose:
            print('done (Lset, Rset, Eojeol) = ({}, {}, {})'.format(len(self.lcount), len(self.rcount), len(self.eojeols)))

    def _build_graph(self, sents):
        from collections import defaultdict
        from collections import Counter
        dictdictize = lambda dd: {k:dict(d) for k,d in dd.items()}
        
        eojeol_max_length = self.l_max_length + self.r_max_length
        eojeols = Counter((eojeol for sent in sents for eojeol in sent.split() if len(eojeol) <= eojeol_max_length))

        lrgraph = defaultdict(lambda: defaultdict(lambda: 0))
        rlgraph = defaultdict(lambda: defaultdict(lambda: 0))
        for eojeol, count in eojeols.items():
            n = len(eojeol)
            for e in range(1, min(self.l_max_length, n) + 1):
                if (n - e) > self.r_max_length:
                    continue
                (l, r) = (eojeol[:e], eojeol[e:])
                lrgraph[l][r] += count
                if r:
                    rlgraph[r][l] += count
        return dictdictize(lrgraph), dictdictize(rlgraph), eojeols

    def extract(self, wordset=None, min_count=3, minimum_noun_score=0.4, minimum_feature_proportion=0.6):
        if not wordset:
            wordset = {l:c for l, c in self.lcount.items() if len(l) >= 2}
        wordset = {l for l,c in wordset.items() if c > min_count and not (l in self.r_scores)}
        
        noun_scores = {}
        for i, l in enumerate(wordset):
            noun_scores[l] = self.predict(l)
            if self.verbose and (i+1) % 1000 == 0:
                print('\rpredicting noun score ... {} / {}'.format(i+1, len(wordset)), end='', flush=True)
        if self.verbose:
            print('\rpredicting noun score ... done', flush=True)
        
        noun_scores = self._postprocessing(noun_scores, minimum_noun_score, minimum_feature_proportion)
        return noun_scores
        
    def predict(self, l):
        from collections import namedtuple
        NounScore = namedtuple('NounScore', 'score count feature_proportion eojeol_proportion n_positive_feature unique_positive_feature_proportion')
        
        (norm, score, _total, n_positive_feature, n_feature) = (0, 0, 0, 0, 0)
        for r, frequency in self.lrgraph.get(l, {}).items():
            _total += frequency
            if not r in self.r_scores:
                continue
            norm += frequency
            score += frequency * self.r_scores[r]
            n_feature += 1
            n_positive_feature += 1 if self.r_scores[r] > 0 else 0

        score = score / norm if norm else 0
        n_eojeol = self.eojeols.get(l, 0)
        feature_proportion = norm / (_total - n_eojeol) if (_total - n_eojeol) > 0 else 0
        eojeol_proportion = n_eojeol / _total if _total > 0 else 0
        unique_positive_feature_proportion = 0 if n_feature <= 0 else n_positive_feature / n_feature
        return NounScore(score, _total, feature_proportion, eojeol_proportion, n_positive_feature, unique_positive_feature_proportion)
#         return (score, _total, feature_proportion, eojeol_proportion, n_positive_feature, unique_positive_feature_proportion)
    
    def _postprocessing(self, noun_scores, minimum_noun_score=0.4, minimum_feature_proportion=0.6):
        import sys
        self._noun_scores_ = dict(filter(lambda x:x[1][0] > minimum_noun_score and x[1][1] > minimum_feature_proportion and len(x[0]) > 1, noun_scores.items()))

        if self.verbose:
            print('finding NJsubJ (대학생(으)+로), NsubJ (떡볶+(이)), NVsubE (사기(당)+했다) ... ', end='')
        njsunjs = {l for l in self._noun_scores_ if self._is_NJsubJ(l)}
        nsubs = {l0 for l in self._noun_scores_ for l0 in self._find_NsubJ(l) if not (l in njsunjs)}
        nvsubes = {l for l in self._noun_scores_ if self._is_NVsubE(l) and self._is_NWsub(l) and not self._is_compound(l)}
        if self.verbose:
            print('done')
        
        #     unijosa = {}
        self._noun_scores_postprocessed = {}
        for i, (noun, score) in enumerate(self._noun_scores_.items()):
            if (i+1) % 1000 == 0:
                print('\rchecking hardrules ... {} / {}'.format(i+1, len(self._noun_scores_)), flush=True, end='')
            if(noun in njsunjs) or (noun in nsubs) or (noun in nvsubes):
                continue
            if not self._hardrule_unijosa_filter(noun) and not self._is_compound(noun):
    #             unijosa[noun] = score
                continue
            if not self._hardrule_suffix_filter(noun):
                continue
            if not self._hardrule_dang_hada_filter(noun):
                continue
            self._noun_scores_postprocessed[noun] = score
            
        if self.verbose:
            print('\rchecking hardrules ... done')
        return self._noun_scores_postprocessed
    
    def _is_NJsubJ(self, l, candidate_noun_threshold=0.4, njsub_proportion_threshold=0.8, min_frequency_droprate=0.7):
        """### NJsub + J: 대학생으 + 로"""
        def match_NJsubJ(token):
            for e in l0_candidates:
                if token[e:] in self.josa_dictionary:
                    return True
            return False

        l0_candidates = {l[:e] for e in range(2, len(l))}
        l0_candidates = {len(l0) for l0 in l0_candidates if l0 in self.base_noun_dictionary or self.predict(l0)[0] > candidate_noun_threshold}
        if not l0_candidates:
            return False
        base = self.l_frequency(l[:max(l0_candidates)])
        if base == 0:
            return False
        elif self.l_frequency(l) / base  > min_frequency_droprate:
            return False
        tokens = {l+r:c for r, c in self.lrgraph.get(l, {}).items()}
        prop = sum((c for token, c in tokens.items() if match_NJsubJ(token))) / sum(tokens.values())
        return prop > njsub_proportion_threshold

    def _find_NsubJ(self, l, candidate_noun_threshold=0.7, nsubj_proportion_threshold=0.7):
        """### Nsub + J: 떡볶 + 이"""
        proportion = lambda l0, l : self.l_frequency(l0) / self.l_frequency(l)
        l0_candidates = {l[:e] for e in range(2, len(l)) if l[e:] in self.josa_dictionary}
        l0_candidates = {l0 for l0 in l0_candidates if (l0 in self.base_noun_dictionary or self.predict(l0)[0] > candidate_noun_threshold) and proportion(l0, l) > nsubj_proportion_threshold}
        return l0_candidates
    
    def _is_NVsubE(self, l, maximum_eojeol_proportion=0.7):
        """is_NVsubE('성심당') # False
           is_NVsubE('폭행당') # True """
        
        def eojeol_proportion(w):
            sum_ = sum(self.lrgraph.get(w, {}).values())
            return False if not sum_ else (min(1, self.eojeols.get(w, 0) / sum_) > maximum_eojeol_proportion)

        r_extensions = self.lrgraph.get(l, {})
        if not r_extensions:
            return False

        n = len(l)
        for b in range(1, 2 if n <= 3 else 3):
            (l0, r0) = (l[:-b], l[-b:])
            if not (l0 in self._noun_scores_):
                continue
            r_extension_as_eojeol = sum([eojeol_proportion(r0+r)*c for r, c in r_extensions.items()]) / sum(r_extensions.values())
            if r_extension_as_eojeol > maximum_eojeol_proportion:
                return True
        return False
    
    def _is_NWsub(self, l, minimum_frequency_droprate=0.1):
        def frequency_droprate(l0):
            return 0 if self.l_frequency(l0) <= 0 else (self.l_frequency(l) / self.l_frequency(l0))

        for b in range(1, 2 if len(l) <= 3 else 3):
            (l0, r0) = (l[:-b], l[-b:])
            if (l0 in self._noun_scores_) and (frequency_droprate(l0) < minimum_frequency_droprate):
                return True
        return False
    
    def _is_compound(self, l):
        n = len(l)
        if n < 4: return False
        for e in range(2, n):
#             if (e + 1 == n) and (l[:e] in self._noun_scores_):
#                 return True
            if (l[:e] in self._noun_scores_) and (l[e:] in self._noun_scores_):
                return True
        if n >= 6:
            for e1 in range(2, n-3):
                for e2 in range(e1 + 2, n-1):
                    if (l[:e1] in self._noun_scores_) and (l[e1:e2] in self._noun_scores_) and (l[e2:] in self._noun_scores_):
                        return True
        return False
    
    def _is_NJ(self, w):
        for e in range(2, len(w)+1):
            (l, r) = (w[:e], w[e:])
            if (l in self._noun_scores_) and ((not r) or (self.r_scores.get(r, 0) > 0)):
                return True
        return False
    
    def _hardrule_unijosa_filter(self, l, min_count=10):
        def has_passset(r):
            passset = {'과', '는', '되고', '되는', '되다', '된다', 
                       '들', '들에', '들의', '들이', 
                       '로', '로는', '로도', '로서', '를', 
                       '부터', '뿐', '뿐만', '뿐이', '뿐인', 
                       '에게', '에도', '에서', '에와', '와', 
                       '으로', '은', '의', '이', '이나', '이라', '이었', '인', '임', 
                       '처럼', '하다', '한', '할', '했던', '했고', '했다'}        
            return (r in passset)

        if not (l in self._noun_scores_):
            return True
        if self._noun_scores_[l][1] < min_count and self._noun_scores_[l][4] <= 2:
            rdict = self.lrgraph.get(l, {})
            if not rdict:
                return False
            n_passjosa = sum((c for r,c in self.lrgraph[l].items() if (r) and (has_passset(r[:2]) or self._is_NJ(r))))
            n_nonemptyr = sum((c for r, c in self.lrgraph[l].items() if r))
            passset_prop = n_passjosa / n_nonemptyr if n_nonemptyr else 0
            return passset_prop > 0.5
        return True
    
    def _hardrule_dang_hada_filter(self, l, max_h_proportion=0.5):
        from soynlp.hangle import decompose # TODO check import path
        if not (l[-1] == '당') and (l[:-1] in self._noun_scores_):
            return True
        rdict = self.lrgraph.get(l, {})
        n_base = sum((c for r,c in rdict.items() if c))
        n_h = 0
        for r,c in rdict.items():
            if not r: continue
            rdecompose = decompose(r[0])
            if rdecompose and rdecompose[0] == 'ㅎ':
                n_h += c
        return True if n_base <= 0 else (n_h / n_base < max_h_proportion) 
    
    def _hardrule_suffix_filter(self, l):
        def prop_r(l, r):
            base = sum(self.lrgraph.get(l, {}).values())
            return 0 if base == 0 else self.lrgraph.get(l, {}).get(r, 0) / base

        stoplsub = {'갔다', '이는', '겠지', '보는'}
        if l[-2:] in stoplsub: return False
        if (l[-1] == '으') and ((prop_r(l, '로') + prop_r(l, '로써') + prop_r(l, '로만')) > 0.5):
            return False
        if (l[-1] == '지') and (prop_r(l, '만') > 0.5):
            return False
        if (l[-1] == '없') and ((prop_r(l, '는') + prop_r(l, '이') + prop_r(l, '다')) > 0.5):
            return False
        return True
    
    def l_frequency(self, l):
        return self.lcount.get(l, 0)