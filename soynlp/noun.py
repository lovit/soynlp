from collections import defaultdict
import sys

class LRNounExtractor:

    def __init__(self, predictor_fnames=None, verbose=True, 
                 max_l_length=10, max_r_length=6, min_count=5):
        self.coefficient = {}
        self.verbose = verbose
        self.max_l_length = max_l_length
        self.max_r_length = max_r_length
        self.min_count = min_count
        self.lrgraph = None
        
        if not predictor_fnames:
#             predictor_fnames = [__file__[:-9] + 'trained_models/noun_predictor_sejong']
            predictor_fnames = ['/Users/hyunjoongkim/git/soynlp/soynlp/trained_models/noun_predictor_sejong']
            print('used default noun predictor; Sejong corpus predictor')
            
        for fname in predictor_fnames:
            print('used %s' % fname.split('/')[-1])
            self._load_predictor(fname)
        print('%d r features was loaded' % len(self.coefficient))
        
    def _load_predictor(self, fname):
        try:
            with open(fname, encoding='utf-8') as f:
                for num_line, line in enumerate(f):
                    r, score = line.split('\t')
                    score = float(score)
                    self.coefficient[r] = max(self.coefficient.get(r, 0), score)
        except FileNotFoundError:
            print('predictor file was not found')
        except Exception as e:
            print(' ... %s parsing error line (%d) = %s' % (e, num_line, line))
    
    def train_extract(self, sents, wordset_l=None, wordset_r=None):
        self.train(sents, wordset_l, wordset_r)
        return self.extract()
    
    def train(self, sents, wordset_l=None, wordset_r=None):
        if (not wordset_l) or (not wordset_r):
            wordset_l, wordset_r = self._scan_vocabulary(sents)
        self.lrgraph = self._build_lrgraph(sents, wordset_l, wordset_r)
    
    def _scan_vocabulary(self, sents):
        """
        Parameters
        ----------
            sents: list-like iterable object which has string
            
        It computes subtoken frequency first. 
        After then, it builds lr-graph with sub-tokens appeared at least min count
        """
        
        _ckpt = int(len(sents) / 40)
        
        wordset_l = defaultdict(lambda: 0)
        wordset_r = defaultdict(lambda: 0)
        
        for i, sent in enumerate(sents):
            for token in sent.split(' '):
                if not token:
                    continue
                token_len = len(token)
                for i in range(1, min(self.max_l_length, token_len)+1):
                    wordset_l[token[:i]] += 1
                for i in range(1, min(self.max_r_length, token_len)):
                    wordset_r[token[-i:]] += 1
            if self.verbose and (i % _ckpt == 0):
                args = ('#' * int(i/_ckpt), '-' * (40 - int(i/_ckpt)), 100.0 * i / len(sent), '%')
                sys.stdout.write('\rscanning: %s%s (%.3f %s)' % args)
            
        wordset_l = {w for w,f in wordset_l.items() if f >= self.min_count}
        wordset_r = {w for w,f in wordset_r.items() if f >= self.min_count}
        if self.verbose:
            print('\rscanning completed')
            print('(L,R) has (%d, %d) tokens' % (len(wordset_l), len(wordset_r)))

        return wordset_l, wordset_r
    
    def _build_lrgraph(self, sents, wordset_l, wordset_r):
        _ckpt = int(len(sents) / 40)
        lrgraph = defaultdict(lambda: defaultdict(lambda: 0))
        
        for i, sent in enumerate(sents):
            for token in sent.split():
                if not token:
                    continue
                token_len = len(token)
                for i in range(1, min(self.max_l_length, token_len)+1):
                    l = token[:i]
                    r = token[i:]
                    if (not l in wordset_l) or (not r in wordset_r):
                        continue
                    lrgraph[l][r] += 1

            if self.verbose and (i % _ckpt == 0):
                args = ('#' * int(i/_ckpt), '-' * (40 - int(i/_ckpt)), 100.0 * i / len(sents), '%')
                sys.stdout.write('\rbuilding lr-graph: %s%s (%.3f %s)' % args)
        if self.verbose:                       
            sys.stdout.write('\rbuilding lr-graph completed')
        lrgraph = {l:{r:f for r,f in rdict.items()} for l,rdict in lrgraph.items()}
        return lrgraph
    
    def extract(self, noun_candidates=None, minimum_noun_score=0.1):
        if not noun_candidates:
            noun_candidates = sorted(self.lrgraph.keys(), key=lambda x:len(x))
            
        nouns = {}
        for word in noun_candidates:
            features = self._get_r_features(word)
            score = self.predict(features) if features else (self._get_subword_score(nouns, word), 0)
            if score[0] < minimum_noun_score:
                continue
            nouns[word] = score
        
        nouns = self._postprocess(nouns, minimum_noun_score)
        return nouns
    
    def _get_r_features(self, word):
        features = self.lrgraph.get(word, {})
        if '' in features:
            del features['']
        return features
    
    def _get_subword_score(self, nouns, word):
        subword_scores = {}
        for e in range(1, len(word)):
            subword = word[:e]
            suffix = word[e:]
            # Add word if compound
            if (subword in nouns) and (suffix in nouns):
                score1 = nouns[subword]
                score2 = nouns[suffix]
                score = score1 if score1[0] > score2[0] else score2
                subword_scores[subword] = score
            elif (subword in nouns) and (self.coefficient.get(suffix,0.0) > minimum_noun_score):
                subword_scores[subword] = self.coefficient.get(suffix,0.0)
        if not subword_scores:
            return None
        return sorted(subword_scores.items(), key=lambda x:x[1][0], reverse=True)[0][1]

    def predict(self, features):
        """Parameters
        ----------
            features: dict
                예시: {을: 35, 는: 22, ...}
        """
        
        score = 0
        norm = 0
        unknown = 0
        
        for r, freq in features.items():
            if r in self.coefficient:
                score += freq * self.coefficient[r]
                norm += freq
            else:
                unknown += freq
        
        return (0 if norm == 0 else score / norm, 
                0 if (norm + unknown == 0) else norm / (norm + unknown))
    
    def _postprocess(self, nouns, minimum_noun_score):
        removals = set()
        for word in sorted(nouns.keys(), key=lambda x:len(x)):
            if len(word) <= 2:
                continue
            if word[-1] == '.':
                removals.add(word)
            for e in range(2, len(word)):
                if (word[:e] in nouns) and (self.coefficient.get(word[e:], 0.0) > minimum_noun_score):
                    removals.add(word)
                    break
        nouns_ = {word:score for word, score in nouns.items() if (word in removals) == False}
        return nouns_