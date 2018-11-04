# -*- encoding:utf8 -*-

from ._template import LR

class BaseTagger:
    def __init__(self, generator, evaluator, postprocessor=None):
        self.evaluator = evaluator
        self.generator = generator
        self.dictionary = generator.dictionary
        self.postprocessor = postprocessor

    def tag(self, sentence, flatten=True, debug=False):
        raise NotImplementedError

class SimpleTagger(BaseTagger):
    def tag(self, sentence, flatten=True, debug=False):
        sent_ = []
        debug_ = []
        eojeols = sentence.split()
        
        for eojeol in eojeols:
            candidates = self.generator.generate(eojeol)
            best = self.evaluator.select_best(candidates)
            
            if self.postprocessor:
                postprocessed = self.postprocessor.postprocess(eojeol, best)
            else:
                postprocessed = best
            
            # Flatten in a token
            postprocessed_ = []
            for word in postprocessed:
                if word.l:
                    postprocessed_.append((word.l, word.l_tag))
                if word.r:
                    postprocessed_.append((word.r, word.r_tag))
            
            sent_.append(postprocessed_)
            
            if debug:
                scored_candidates = [(c, self.evaluator.evaluate(c)) for c in candidates]
                scored_candidates = sorted(scored_candidates, key=lambda x:(x[0].b, x[1]))
                debug_.append(scored_candidates)
        
        # Flatten in a sentence
        if flatten:
            sent_ = [word for words in sent_ for word in words]

        if not debug:
            return sent_
        else:
            return sent_, debug_

class BasePostprocessor:
    def postprocess(self, token, best_wordstream):
        return best_wordstream

class UnknowLRPostprocessor(BasePostprocessor):
    def postprocess(self, token, words):
        n = len(token)
        adds = []
        if words and words[0].b > 0:
            adds.append(self._add_first_subword(token, words))
        if words and words[-1].e < n:
            adds.append(self._add_last_subword(token, words, n))
        adds += self._add_inter_subwords(token, words)
        post = words + adds
        return sorted(post, key=lambda x:x.b)

    def _add_last_subword(self, token, words, n):
        b = words[-1].e
        subword = token[b:]
        return LR(subword, None, '', None, b, n, n)

    def _add_first_subword(self, token, words):    
        e = words[0].b
        subword = token[0:e]
        return LR(subword, None, '', None, 0, e, e)
    
    def _add_inter_subwords(self, token, words):
        adds = []        
        for i, base in enumerate(words[:-1]):
            if base.e == words[i+1].b:
                continue
            b = base.e
            e = words[i+1].b
            subword = token[b:e]
            adds.append(LR(subword, None, '', None, b, e, e))
        return adds