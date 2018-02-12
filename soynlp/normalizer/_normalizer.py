import re
from soynlp.hangle import decompose

doublespace_pattern = re.compile('\s+')
repeatchars_pattern = re.compile('(\w)\\1{3,}')
hangle_filter = re.compile('[^ㄱ-ㅎㅏ-ㅣ가-힣]')
text_filter = re.compile('[^ㄱ-ㅎㅏ-ㅣ가-힣a-zA-Z\.\?\!\"\']')

def repeat_normalize(sent, n_repeats=2):
    if n_repeats > 0:
        sent = repeatchars_pattern.sub('\\1' * n_repeats, sent)
    sent = doublespace_pattern.sub(' ', sent)
    return sent.strip()

def emoticon_normalize(sent, n_repeats=2):
    if not sent:
        return sent
    
    # Pattern matching ㅋ쿠ㅜ
    def pattern(idx):
        # Jaum: 0, Moum: 1, Complete: 2, else -1
        if 12593 <= idx <= 12622:
            return 0
        elif 12623 <= idx <= 12643:
            return 1
        elif 44032 <= idx <= 55203:
            return 2
        else:
            return -1

    idxs = [pattern(ord(c)) for c in sent]
    sent_ = []
    for i, (idx, c) in enumerate(zip(idxs[:-1], sent)):
        if i > 0 and (idxs[i-1] == 0 and idx == 2 and idxs[i+1] == 1):
            cho, jung, jong = decompose(sent[i])
            if (cho == sent[i-1]) and (jung == sent[i+1]) and (jong == ' '):
                sent_.append(cho)
                sent_.append(jung)
            else:
                sent_.append(c)
        else:
            sent_.append(c)
    sent_.append(sent[-1])
    return repeat_normalize(''.join(sent_), n_repeats)

def only_hangle(sent):
    return doublespace_pattern.sub(' ',hangle_filter.sub(' ', sent)).strip()

def only_text(sent):
    return doublespace_pattern.sub(' ',text_filter.sub(' ', sent)).strip()