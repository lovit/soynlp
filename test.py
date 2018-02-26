# -*- encoding:utf8 -*-
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

from soynlp.tokenizer import RegexTokenizer, LTokenizer, MaxScoreTokenizer

tokenizer = RegexTokenizer()

sents = [
    '이렇게연속된문장은잘리지않습니다만',
    '숫자123이영어abc에섞여있으면ㅋㅋ잘리겠죠',
    '띄어쓰기가 포함되어있으면 이정보는10점!꼭띄워야죠'
]

for sent in sents:
    print('   %s\n->%s\n' % (sent, ' '.join(tokenizer.tokenize(sent))))