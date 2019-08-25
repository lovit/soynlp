# -*- encoding:utf8 -*-

import sys
if sys.version_info <= (2,7):
    reload(sys)
    sys.setdefaultencoding('utf-8')
import re
from soynlp.hangle import compose, decompose

doublespace_pattern = re.compile('\s+')
repeatchars_pattern = re.compile('(\w)\\1{3,}')
number_pattern = re.compile('[0-9]')
punctuation_pattern = re.compile('[,\.\?\!]')
symbol_pattern = re.compile('[()\[\]\{\}`]')
hangle_pattern = re.compile('[ㄱ-ㅎㅏ-ㅣ가-힣]')
alphabet_pattern = re.compile('[a-zA-Z]')

hangle_filter = re.compile('[^ㄱ-ㅎㅏ-ㅣ가-힣]')
hangle_number_filter = re.compile('[^ㄱ-ㅎㅏ-ㅣ가-힣0-9]')
text_filter = re.compile('[^ㄱ-ㅎㅏ-ㅣ가-힣a-zA-Z0-9,\.\?\!\"\'-()\[\]\{\}]')

def normalize(doc, alphabet=False, number=False,
    punctuation=False, symbol=False, remove_repeat=0):

    doc = text_filter.sub(' ', doc)
    if not alphabet:
        doc = alphabet_pattern.sub(' ', doc)
    if not number:
        doc = number_pattern.sub(' ', doc)
    if not punctuation:
        doc = punctuation_pattern.sub(' ', doc)
    if not symbol:
        doc = symbol_pattern.sub(' ', doc)
    if remove_repeat > 0:
        doc = repeatchars_pattern.sub('\\1' * remove_repeat, doc)

    return doublespace_pattern.sub(' ', doc).strip()

def remove_doublespace(sent):
    return doublespace_pattern.sub(' ', sent)

def repeat_normalize(sent, num_repeats=2):
    if num_repeats > 0:
        sent = repeatchars_pattern.sub('\\1' * num_repeats, sent)
    sent = doublespace_pattern.sub(' ', sent)
    return sent.strip()

def emoticon_normalize(sent, num_repeats=2):
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
    last_idx = len(idxs) - 1
    for i, (idx, c) in enumerate(zip(idxs, sent)):
        if (i > 0 and i < last_idx) and (idxs[i-1] == 0 and idx == 2 and idxs[i+1] == 1):
            cho, jung, jong = decompose(c)
            if (cho == sent[i-1]) and (jung == sent[i+1]) and (jong == ' '):
                sent_.append(cho)
                sent_.append(jung)
            else:
                sent_.append(c)
        elif (i < last_idx) and (idx == 2) and (idxs[i+1] == 0):
            cho, jung, jong = decompose(c)
            if (jong == sent[i+1]):
                sent_.append(compose(cho, jung, ' '))
                sent_.append(jong)
        elif (i > 0) and (idx == 2 and idxs[i-1] == 0):
            cho, jung, jong = decompose(c)
            if (cho == sent[i-1]):
                sent_.append(cho)
                sent_.append(jung)
        else:
            sent_.append(c)
    return repeat_normalize(''.join(sent_), num_repeats)

def only_hangle(sent):
    return doublespace_pattern.sub(' ',hangle_filter.sub(' ', sent)).strip()

def only_hangle_number(sent):
    return doublespace_pattern.sub(' ',hangle_number_filter.sub(' ', sent)).strip()

def only_text(sent):
    return doublespace_pattern.sub(' ',text_filter.sub(' ', sent)).strip()

def remain_hangle_on_last(eojeol):
    matchs = list(hangle_pattern.finditer(eojeol))
    if not matchs:
        return ''
    last_index = matchs[-1].span()[1]
    return eojeol[:last_index].strip()

def normalize_sent_for_lrgraph(sent):
    sent = text_filter.sub(' ', sent)
    sent = symbol_pattern.sub(' ', sent)
    sent_ = [remain_hangle_on_last(eojeol) for eojeol in sent.split()]
    sent_ = [eojeol for eojeol in sent_ if eojeol]
    if not sent_:
        return ''
    return ' '.join(sent_)