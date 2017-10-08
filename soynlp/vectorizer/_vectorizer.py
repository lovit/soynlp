class BaseVectorizer:
    def __init__(self, vocabulary=None,
                 preprocessor=lambda x:x.strip(),
                 tokenizer=lambda x:x.split()
                ):
        
        self._preprocessor = preprocessor
        self._tokenizer = tokenizer
        self._vocab2int = {}
        self._int2vocab = []
        self.n_vocabs = 0
        
        if vocabulary:
            self._set_vocabulary(vocabulary)

    def to_sparse_matrix(self, docs, boolean=False):
        from collections import Counter
        from scipy.sparse import csr_matrix
        
        rows = []
        cols = []
        data = []
        
        for i, doc in enumerate(docs):
            bow_str = Counter(self._tokenizer(self._preprocessor(doc)))
            bow = self.encode_from_bow(bow_str.items())
            if boolean:
                for j, _ in bow:
                    rows.append(i)
                    cols.append(j)
                    data.append(1)
            else:
                for j, v in bow:
                    rows.append(i)
                    cols.append(j)
                    data.append(v)
        
        return csr_matrix((data, (rows, cols)), shape=(i+1, self.n_vocabs))

    def encode_from_bow(self, bow_str):
        return [(self._vocab2int[word], v) for word, v in bow_str if word in self._vocab2int]
        
    def decode_from_bow(self, bow):
        return [(self._int2vocab[i], v) for i,v in bow if 0 <= i < self.n_vocabs]
        
    def save(self, fname):
        with open(fname, 'w', encoding='utf-8') as f:
            for vocab in self._int2vocab:
                f.write('{}\n'.format(vocab))
    
    def load(self, fname, delimiter='\t'):
        with open(fname, encoding='utf-8') as f:
            self._set_vocabulary([doc.strip().split(delimiter)[0] for doc in f])
    
    def _set_vocabulary(self, vocabulary):
        self._int2vocab = sorted(vocabulary)
        self._vocab2int = {v:i for i,v in enumerate(self._int2vocab)}
        self.n_vocabs = len(self._vocab2int)