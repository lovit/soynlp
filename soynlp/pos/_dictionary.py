import os
from collections import defaultdict
from glob import glob

class Dictionary:
    def __init__(self, domain_dictionary_folder=None, use_base_dictionary=True, dictionary_word_minscore=3):
        
        self._pos = {}
        self._pos_domain = {}
        
        if use_base_dictionary:
            directory = '/'.join(os.path.abspath(__file__).replace('\\', '/').split('/')[:-1])
            dictionary_root = '{}/dictionary/'.format(directory)
            self._pos = self._load(glob('{}/pos/*/*.txt'.format(dictionary_root)), dictionary_word_minscore, tag_position=-2)
            print('use base dictionary')
        
        if domain_dictionary_folder:
            self._pos_domain = self._load(glob('{}/*_*.txt'.format(domain_dictionary_folder)), dictionary_word_minscore)
            self._pos_domain = self._load(glob('{}/*/*_*.txt'.format(domain_dictionary_folder)), dictionary_word_minscore)
            print('use domain dictionary from {}'.format(domain_dictionary_folder))
            
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
    
    def _load(self, fnames, min_score=1, tag_position=-1, encoding='utf-8'):
        dictionary = defaultdict(lambda: {})        
        for fname in fnames:
            tag = fname.split('/')[tag_position].split('.')[0].split('_')[0]
            
            with open(fname, encoding=encoding) as f:
                words = [line.split() for line in f]
                words = [col if len(col) == 1 else [col[0], float(col[1])] for col in words]
                max_score = max((col[1] for col in words if len(col) == 2))
                
                for col in words:
                    if len(col) == 1:
                        dictionary[tag][col[0]] = max_score+1
                    elif col[1] >= min_score:
                        dictionary[tag][col[0]] = col[1]
                        
        return dict(dictionary)
    
    def add_words(self, words, tag):
        if type(words) == str:
            words = {words}
        words_in_domain_dictionary = self._pos_domain.get(tag, {})
        words_in_domain_dictionary.update({w:0 for w in words if not (w in words_in_domain_dictionary)})
        self._pos_domain[tag] = words_in_domain_dictionary
        words_in_dictionary = self._pos.get(tag, {})
        words_in_dictionary.update({w:0 for w in words if not (w in words_in_dictionary)})
        self._pos[tag] = words_in_dictionary
    
    def remove_words(self, words, tag):
        if type(words) == str:
            words = {words}
        elif type(words) != set:
            words = set(words)
        words_in_domain_dictionary = self._pos_domain.get(tag, {})
        words_in_domain_dictionary = {w:s for w,s in words_in_domain_dictionary.items() if not (w in words)}
        self._pos_domain[tag] = words_in_domain_dictionary
        words_in_dictionary = self._pos.get(tag, {})
        words_in_dictionary = {w:s for w,s in words_in_dictionary.items() if not (w in words)}
        self._pos[tag] = words_in_dictionary
        
    def save_domain_dictionary(self, folder, head=None):
        import os        
        for tag, words in self._pos_domain.items():
            if not os.path.exists(folder):
                os.makedirs('{}/{}/'.format(folder, tag))
            with open('{}/{}/{}_{}.txt'.format(folder, tag, tag, str(head) if head else ''), 'w', encoding='utf-8') as f:
                for word, score in sorted(words.items(), key=lambda x:-x[1]):
                    f.write('{}\t{}\n'.format(word, score))
    
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