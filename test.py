# -*- encoding:utf8 -*-

import re
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

doublespace_pattern = re.compile('\s+')
repeatchars_pattern = re.compile('(\w)\\1{3,}')
hangle_filter = re.compile('[^ㄱ-ㅎㅏ-ㅣ가-힣]')
hangle_number_filter = re.compile('[^ㄱ-ㅎㅏ-ㅣ가-힣0-9]')
text_filter = re.compile('[^ㄱ-ㅎㅏ-ㅣ가-힣a-zA-Z0-9\.\?\!\"\'-]')



import soynlp
from soynlp.normalizer import *

print(soynlp.__version__)
print emoticon_normalize('ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ쿠ㅜㅜㅜㅜㅜㅜ')

emoticon_normalize('ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ쿠ㅜㅜㅜㅜㅜㅜ', n_repeats=4)
print repeat_normalize('와하하하하하하하하하핫')

def repeat_normalize(sent, n_repeats=2):
    if n_repeats > 0:
        sent = repeatchars_pattern.sub('\\1' * n_repeats, sent)
    sent = doublespace_pattern.sub(' ', sent)
    return sent.strip()

print repeat_normalize('와하하하하하하하하하핫')

test = re.compile(r'(123)\\1')
test_sentence = '1231231123123'
print test.sub('4', test_sentence)
print test.findall(test_sentence)