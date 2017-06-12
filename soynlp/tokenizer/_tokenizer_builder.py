from collections import defaultdict
import sys
from soynlp.utils import get_process_memory


class EojeolPatternTrainer:

    def __init__(self, left_max_length=10, right_max_length=6, min_count=10, verbose=True):
        self.left_max_length = left_max_length
        self.right_max_length = right_max_length
        self.min_count = min_count
        self.verbose = verbose
        self.lrgraph = None
        self.rlgraph = None
        self.wordset_l = None
        self.wordset_r = None
    
    def train(self, sents, wordset_l=None, wordset_r=None):
        if (not wordset_l) or (not wordset_r):
            wordset_l, wordset_r = self._scan_vocabulary(sents)
        self.lrgraph, self.rlgraph = self._build_graph(sents, wordset_l, wordset_r)
        # TODO more
        
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
                for i in range(1, min(self.left_max_length, token_len)+1):
                    wordset_l[token[:i]] += 1
                for i in range(1, min(self.right_max_length, token_len)):
                    wordset_r[token[-i:]] += 1
            if self.verbose and (i % _ckpt == 0):
                args = ('#' * int(i/_ckpt), '-' * (40 - int(i/_ckpt)), 100.0 * i / len(sent), '%', get_process_memory())
                sys.stdout.write('\rscanning: %s%s (%.3f %s) %.3f Gb' % args)
            
        wordset_l = {w for w,f in wordset_l.items() if f >= self.min_count}
        wordset_r = {w for w,f in wordset_r.items() if f >= self.min_count}
        if self.verbose:
            print('\rscanning completed')
            print('(L,R) has (%d, %d) tokens. memory = %.3f Gb' % (len(wordset_l), len(wordset_r), get_process_memory()))

        return wordset_l, wordset_r
    
    def _build_graph(self, sents, wordset_l, wordset_r):
        self.wordset_l = wordset_l
        self.wordset_r = wordset_r
        self.wordset_r.add('')
        _ckpt = int(len(sents) / 40)
        lrgraph = defaultdict(lambda: defaultdict(lambda: 0))
        rlgraph = defaultdict(lambda: defaultdict(lambda: 0))
        
        for i, sent in enumerate(sents):
            for token in sent.split():
                if not token:
                    continue
                token_len = len(token)
                for i in range(1, min(self.left_max_length, token_len)+1):
                    l = token[:i]
                    r = token[i:]
                    if (not l in wordset_l) or (not r in wordset_r):
                        continue
                    lrgraph[l][r] += 1
                    rlgraph[r][l] += 1

            if self.verbose and (i % _ckpt == 0):
                args = ('#' * int(i/_ckpt), '-' * (40 - int(i/_ckpt)), 100.0 * i / len(sents), '%', get_process_memory())
                sys.stdout.write('\rbuilding lr-graph: %s%s (%.3f %s), memory = %.3f Gb' % args)
        if self.verbose:                       
            sys.stdout.write('\rbuilding lr-graph completed. memory = %.3f Gb' % get_process_memory())
            
        lrgraph = {l:{r:f for r,f in rdict.items()} for l,rdict in lrgraph.items()}
        rlgraph = {r:{l:f for l,f in ldict.items()} for r, ldict in rlgraph.items()}
        return lrgraph, rlgraph
    
    def save(self, fname):
        with open(fname, 'w', encoding='utf-8') as f:
            f.write('%d %d %d %d\n' % (self.left_max_length, self.right_max_length, self.min_count, 1 if self.verbose else 0))
            f.write('# lrgraph\n')
            for l, rdict in self.lrgraph.items():
                f.write('> %s (%d)\n' % (l, sum(rdict.values())))
                for r, freq in sorted(rdict.items(), key=lambda x:x[1], reverse=True):
                    f.write('  - %s: %d\n' % (r, freq))
            f.write('\n# rlgraph\n')
            for r, ldict in self.rlgraph.items():
                f.write('> %s (%d)\n' % (r, sum(ldict.values())))
                for l, freq in sorted(ldict.items(), key=lambda x:x[1], reverse=True):
                    f.write('  - %s: %d\n' % (l, freq))
            
    def load(self, fname):
        with open(fname, encoding='utf-8') as f:
            param = next(f).strip()
            args = param.split()
            try:
                args = [int(a) for a in args]
                if len(args) != 4:
                    raise ValueError('first line should be parameter info, %s' % str(e))
                self.left_max_length = args[0]
                self.right_max_length = args[1]
                self.min_count = args[2]
                self.verbose == True if args[3] == 1 else False
            except Exception as e:
                raise ValueError('first line should be parameter info, %s' % str(e))
                
            lrgraph = defaultdict(lambda: defaultdict(lambda: 0))
            rlgraph = defaultdict(lambda: defaultdict(lambda: 0))

            load_type = next(f).strip()
            if not load_type == '# lrgraph':
                raise ValueError('Cannot find lrgraph data, %s' % load_type)
                
            key1 = None
            for row in f:
                row = row[:-1]
                if not row: continue
                if row == '# rlgraph':break
                
                if row[:2] == '> ':
                    key1 = row[2:row.rindex('(')].strip()
                    continue
                if row[:4] == '  - ':
                    key2, freq = row[4:].split(': ')
                    freq = int(freq)
                    lrgraph[key1][key2] = freq
            self.wordset_l = set(lrgraph.keys())
                
            for row in f:
                row = row[:-1]
                if not row: continue
                if row[:2] == '> ':
                    key1 = row[2:row.rindex('(')].strip()
                    continue
                if row[:4] == '  - ':
                    key2, freq = row[4:].split(': ')
                    freq = int(freq)
                    rlgraph[key1][key2] = freq
            self.wordset_r = set(rlgraph.keys())
            
            lrgraph = {l:{r:f for r,f in rdict.items()} for l,rdict in lrgraph.items()}
            rlgraph = {r:{l:f for l,f in ldict.items()} for r, ldict in rlgraph.items()}
            self.lrgraph = lrgraph
            self.rlgraph = rlgraph
            
    def train_hits(lrgraph=None, rlgraph=None, sum_of_rank=10000, decaying_factor=0.9, max_iter=10, tolerance=0.0001):
        def normalize(g, sum_of_rank, df):
            factor = df * sum_of_rank / sum(g.values())
            restart = (1 - df) * sum_of_rank / len(g)
            g_ = {word:(factor*rank + restart) for word, rank in g.items() if word != ''}
            return g_

        if lrgraph == None:
            (lrgraph, rlgraph) = (self.lrgraph, self.rlgraph)
        
        rank = sum_of_rank / len(lrgraph)
        rank_l = {l:rank for l in lrgraph.keys()}
        rank = sum_of_rank / len(rlgraph)
        rank_r = {r:rank for r in rlgraph.keys() if r != ''}

        for n_iter in range(max_iter):
            next_rank_l = {}
            for l, rdict in lrgraph.items():
                sum_rrank = sum({freq*rank_r[r] for r, freq in rdict.items() if r != ''})
                next_rank_l[l] = sum_rrank
            next_rank_l = normalize(next_rank_l, sum_of_rank, decaying_factor)

            next_rank_r = {}
            for r, ldict in rlgraph.items():
                if r == '': continue
                sum_lrank = sum({freq*rank_l[l] for l, freq in ldict.items()})
                next_rank_r[r] = sum_lrank
            next_rank_r = normalize(next_rank_r, sum_of_rank, decaying_factor)

            if self.verbose:
                sys.stdout.write('\rtrain hits ... %d in %d' % (n_iter+1, max_iter))

            diff = sum([abs(rank - next_rank_l[w]) for w, rank in rank_l.items()])
            diff += sum([abs(rank - next_rank_r[w]) for w, rank in rank_r.items()])
            rank_l = next_rank_l
            rank_r = next_rank_r
            if diff < (sum_of_rank * tolerance):
                if self.verbose:
                    print('\rgraph was converged at %d iteration' % (n_iter+1))
                break

        if self.verbose:
            print('\rcomputation was done at %d iteration' % (n_iter+1))

        return rank_l, rank_r
