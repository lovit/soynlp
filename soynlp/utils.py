# -*- encoding:utf8 -*-

import os
import psutil
import sys

def get_available_memory():
    """It returns remained memory as percentage"""

    mem = psutil.virtual_memory()
    return 100 * mem.available / (mem.total)

def get_process_memory():
    """It returns the memory usage of current process"""
    
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 3)

def sort_by_alphabet(filepath):
    if sys.version.split('.')[0] == '2':
        with open(filepath) as f:
            docs = [doc.strip() for doc in f]
            docs = [doc for doc in docs if doc]
    else:
        with open(filepath, encoding= "utf-8") as f:
            docs = [doc.strip() for doc in f]
            docs = [doc for doc in docs if doc]
    if sys.version.split('.')[0] == '2':
        with open(filepath, 'w') as f:
            for doc in sorted(docs):
                f.write('{}\n'.format(doc))
    else:
        with open(filepath, 'w', encoding= "utf-8") as f:
            for doc in sorted(docs):
                f.write('{}\n'.format(doc))

class DoublespaceLineCorpus:    
    def __init__(self, corpus_fname, num_doc = -1, num_sent = -1, iter_sent = False, skip_header = 0):
        self.corpus_fname = corpus_fname
        self.num_doc = 0
        self.num_sent = 0
        self.iter_sent = iter_sent
        self.skip_header = skip_header
        if (num_doc > 0) or (num_sent > 0):
            self.num_doc, self.num_sent = self._check_length(num_doc, num_sent)

    def _check_length(self, num_doc, num_sent):
        num_sent_ = 0
        try:
            try:
                # python version check
                if sys.version.split('.')[0] == '2':
                    f = open(self.corpus_fname)
                else:
                    f = open(self.corpus_fname, encoding= "utf-8")

                # skip headers
                for _ in range(self.skip_header):
                    next(f)
                
                # check length
                for doc_idx, doc in enumerate(f):
                    if (num_doc > 0) and (doc_idx >= num_doc):
                        return doc_idx, num_sent_
                    sents = doc.split('  ')
                    sents = [sent for sent in sents if sent.strip()]
                    num_sent_ += len(sents)
                    if (num_sent > 0) and (num_sent_ > num_sent):
                        return doc_idx+1, min(num_sent, num_sent_)

            finally:
                f.close()
            return doc_idx+1, num_sent_

        except Exception as e:
            print(e)
            return -1, -1
    
    def __iter__(self):
        try:
            try:
                if sys.version.split('.')[0] == '2':
                    f = open(self.corpus_fname)
                else:
                    f = open(self.corpus_fname, encoding='utf-8')
                
                # skip headers
                for _ in range(self.skip_header):
                    next(f)
                    
                # iteration
                num_sent, stop = 0, False
                for doc_idx, doc in enumerate(f):
                    if stop:
                        break

                    # yield doc
                    if not self.iter_sent:
                        yield doc
                        if (self.num_doc > 0) and ((doc_idx + 1) >= self.num_doc):
                            stop = True
                        continue

                    # yield sents
                    for sent in doc.split('  '):
                        if (self.num_sent > 0) and (num_sent >= self.num_sent):
                            stop = True
                            break
                        sent = sent.strip()
                        if sent:
                            yield sent
                            num_sent += 1
            finally:
                f.close()

        except Exception as e:
            print(e)

    def __len__(self):
        if self.num_doc == 0:
            self.num_doc, self.num_sent = self._check_length(-1, -1)
        return self.num_sent if self.iter_sent else self.num_doc