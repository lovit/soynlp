import os
from collections import defaultdict
from glob import glob

class Dictionary:
    def __init__(self, dictionary_root=None, dictionary_word_mincount=3):
        
        if (not dictionary_root) or (not os.path.exists(dictionary_root)):
            directory = '/'.join(os.path.abspath(__file__).replace('\\', '/').split('/')[:-1])
            dictionary_root = '{}/dictionary/'.format(directory)
            print('Use basic dictioanry root')
        
        self._pos = self._load(glob('{}/pos/*/*.txt'.format(dictionary_root)), dictionary_word_mincount, tag_position=-2)        
        self._pos_domain = self._load(glob('{}/extracted/*_*.txt'.format(dictionary_root)), dictionary_word_mincount)
        
        for tag, words in self._pos_domain.items():
            self._pos[tag].update(words)
            
        self._maxlen = {pos:max((len(w) for w in words)) for pos, words in self._pos.items()}
        self._maxlen_ = 0 if not self._maxlen else max(self._maxlen.values())
        self._lmax = max([self._maxlen.get('Adjective', 0),
                          self._maxlen.get('Adverb', 0),
                          self._maxlen.get('Exclamation', 0),
                          self._maxlen.get('Noun', 0),
                          self._maxlen.get('Verb', 0)
            ])
        self._rmax = max([self._maxlen.get('Adjective', 0),
                          self._maxlen.get('Josa', 0),
                          self._maxlen.get('Verb', 0)
            ])
#         self.dictionary_lv = self._load(glob('{}/lv/*.txt'.format(dictionary_root)))
#         self.dictionary_r = self._load(glob('{}/r/*.txt'.format(dictionary_root)))
    
    def _load(self, fnames, min_count=1, tag_position=-1, encoding='utf-8'):
        dictionary = defaultdict(lambda: {})        
        for fname in fnames:
            tag = fname.split('/')[tag_position].split('.')[0].split('_')[0]
            
            with open(fname, encoding=encoding) as f:
                words = [line.split() for line in f]
                words = [col if len(col) == 1 else [col[0], int(col[1])] for col in words]
                max_count = max((col[1] for col in words if len(col) == 2))
                
                for col in words:
                    if len(col) == 1:
                        dictionary[tag][col[0]] = max_count+1
                    elif col[1] >= min_count:
                        dictionary[tag][col[0]] = col[1]
                        
        return dict(dictionary)
    
    def is_L(self, w):
        return (w in self._pos.get('Adjective', {})) \
                or (w in self._pos.get('Adverb', {})) \
                or (w in self._pos.get('Exclamation', {})) \
                or (w in self._pos.get('Noun', {})) \
                or (w in self._pos.get('Verb', {}))
    
    def is_R(self, w):
        return (w in self._pos.get('Adjective', {})) \
                or (w in self._pos.get('Josa', {})) \
                or (w in self._pos.get('Verb', {}))
    
    def pos_L(self, w):
        if w in self._pos.get('Noun', {}): return 'Noun'
        elif w in self._pos.get('Verb', {}): return 'Verb'
        elif w in self._pos.get('Adjective', {}): return 'Adjective'
        elif w in self._pos.get('Adverb', {}): return 'Adverb'
        elif w in self._pos.get('Exclamation', {}): return 'Exclamation'
        return None
    
    def pos_R(self, w):
        if w in self._pos.get('Josa', {}): return 'Josa'
        elif w in self._pos.get('Adjective', {}): return 'Adjective'
        elif w in self._pos.get('Verb', {}): return 'Verb'
        return None