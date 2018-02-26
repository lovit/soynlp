# -*- encoding:utf8 -*-
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import soynlp
import re
from soynlp.normalizer import *

print(soynlp.__version__)
print emoticon_normalize('ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ쿠ㅜㅜㅜㅜㅜㅜ', 2)
print repeat_normalize('와하하하하하하하하하핫')
print repeat_normalize('와하하하하하하하하하핫', n_repeats=3)
print only_hangle('가나다ㅏㅑㅓㅋㅋ쿠ㅜㅜㅜabcd123!!아핫')
print only_hangle_number('가나다ㅏㅑㅓㅋㅋ쿠ㅜㅜㅜabcd123!!아핫')
print only_text('가나다ㅏㅑㅓㅋㅋ쿠ㅜㅜㅜabcd123!!아핫')
