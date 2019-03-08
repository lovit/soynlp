# -*- encoding:utf8 -*-

import sys
if sys.version_info <= (2,7):
    reload(sys)
    sys.setdefaultencoding('utf-8')
import warnings
import re
import numpy as np

kor_begin     = 44032
kor_end       = 55203
chosung_base  = 588
jungsung_base = 28
jaum_begin = 12593
jaum_end = 12622
moum_begin = 12623
moum_end = 12643

chosung_list = [ 'ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 
        'ㅅ', 'ㅆ', 'ㅇ' , 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']

jungsung_list = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 
        'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 
        'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 
        'ㅡ', 'ㅢ', 'ㅣ']

jongsung_list = [
    ' ', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ',
        'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 
        'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 
        'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']

jaum_list = ['ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄸ', 'ㄹ', 
              'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 
              'ㅃ', 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']

moum_list = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 
              'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ']

doublespace_pattern = re.compile('\s+')
repeatchars_pattern = re.compile('(\w)\\1{3,}')

def normalize(doc, english=False, number=False, punctuation=False, remove_repeat = 0, remains={}):
    message = 'normalize func will be moved soynlp.normalizer at ver 0.1\nargument remains will be removed at ver 0.1'
    warnings.warn(message, DeprecationWarning, stacklevel=2)

    if remove_repeat > 0:
        doc = repeatchars_pattern.sub('\\1' * remove_repeat, doc)

    f = []    
    for c in doc:
        if (c == ' '):
            f.append(c)
            continue
        i = to_base(c)
        if (kor_begin <= i <= kor_end) or (jaum_begin <= i <= jaum_end) or (moum_begin <= i <= moum_end):
            f.append(c)
            continue
        if (english) and ((i >= 97 and i <= 122) or (i >= 65 and i <= 90)):
            f.append(c)
            continue
        if (number) and (i >= 48 and i <= 57):
            f.append(c)
            continue
        if (punctuation) and (i == 33 or i == 34 or i == 39 or i == 44 or i == 46 or i == 63 or i == 96):
            f.append(c)
            continue
        if c in remains:
            f.append(c)
            continue
        else:
            f.append(' ')            
    return doublespace_pattern.sub(' ', ''.join(f)).strip()

def compose(chosung, jungsung, jongsung):
    return chr(kor_begin + chosung_base * chosung_list.index(chosung) + jungsung_base * jungsung_list.index(jungsung) + jongsung_list.index(jongsung))

def decompose(c):
    if not character_is_korean(c):
        return None
    i = to_base(c)
    if (jaum_begin <= i <= jaum_end):
        return (c, ' ', ' ')
    if (moum_begin <= i <= moum_end):
        return (' ', c, ' ')    
    i -= kor_begin
    cho  = i // chosung_base
    jung = ( i - cho * chosung_base ) // jungsung_base 
    jong = ( i - cho * chosung_base - jung * jungsung_base )    
    return (chosung_list[cho], jungsung_list[jung], jongsung_list[jong])

def character_is_korean(c):
    i = to_base(c)
    return (kor_begin <= i <= kor_end) or (jaum_begin <= i <= jaum_end) or (moum_begin <= i <= moum_end)

def character_is_complete_korean(c):
    return (kor_begin <= to_base(c) <= kor_end)

def character_is_jaum(c):
    return (jaum_begin <= to_base(c) <= jaum_end)

def character_is_moum(c):
    return (moum_begin <= to_base(c) <= moum_end)

def to_base(c):
    if sys.version_info.major == 2:
        if type(c) == str or type(c) == unicode:
            return ord(c)
        else:
            raise TypeError
    else:
        if type(c) == str or type(c) == int:
            return ord(c)
        else:
            raise TypeError

def character_is_number(i):
    i = to_base(i)
    return (i >= 48 and i <= 57)

def character_is_english(i):
    i = to_base(i)
    return (i >= 97 and i <= 122) or (i >= 65 and i <= 90)

def character_is_punctuation(i):
    i = to_base(i)
    return (i == 33 or i == 34 or i == 39 or i == 44 or i == 46 or i == 63 or i == 96)

class ConvolutionHangleEncoder:
    """초/중/종성을 구성하는 자음/모음과 띄어쓰기만 인코딩
    one hot vector [ㄱ, ㄴ, ㄷ, ... ㅎ, ㅏ, ㅐ, .. ㅢ, ㅣ,"  ", ㄱ, ㄲ, ... ㅍ, ㅎ,"  ", 0, 1, 2, .. 9]
    """
    def __init__(self):
        self.jung_begin = 19 # len(chosung_list)
        self.jong_begin = 40 # self.jung_begin + len(jungsung_list)
        self.number_begin = 68 # self.jong_begin + len(jongsung_list)
        self.space = 78 # len(chosung_list) + len(jungsung_list) + len(jongsung_list) + 10
        self.unk = 79
        self.dim = 80
        num = [str(i) for i in range(10)]
        space = ' '
        unk = '<unk>'
        idx_to_char = chosung_list + jungsung_list + jongsung_list + num + [space] + [unk]
        self.idx_to_char = np.asarray(idx_to_char)
        self.jamo_to_idx = {
            'ㄱ': 0, 'ㄲ': 1, 'ㄴ': 2, 'ㄷ': 3, 'ㄸ': 4, 'ㄹ': 5, 'ㅁ': 6, 'ㅂ': 7,
            'ㅃ': 8, 'ㅅ': 9, 'ㅆ': 10, 'ㅇ': 11, 'ㅈ': 12, 'ㅉ': 13, 'ㅊ': 14, 'ㅋ': 15,
            'ㅌ': 16, 'ㅍ': 17, 'ㅎ': 18, 'ㅏ': 19, 'ㅐ': 20, 'ㅑ': 21, 'ㅒ': 22, 'ㅓ': 23,
            'ㅔ': 24, 'ㅕ': 25, 'ㅖ': 26, 'ㅗ': 27, 'ㅘ': 28, 'ㅙ': 29, 'ㅚ': 30, 'ㅛ': 31,
            'ㅜ': 32, 'ㅝ': 33, 'ㅞ': 34, 'ㅟ': 35, 'ㅠ': 36, 'ㅡ': 37, 'ㅢ': 38, 'ㅣ': 39,
            ' ': 40, 'ㄳ': 43, 'ㄵ': 45, 'ㄶ': 46, 'ㄺ': 49, 'ㄻ': 50, 'ㄼ': 51, 'ㄽ': 52,
            'ㄾ': 53, 'ㄿ': 54, 'ㅀ': 55, 'ㅄ': 58
        }
    
    def encode(self, sent):
        onehot = self.sent_to_onehot(sent)
        x = np.zeros((len(onehot), self.dim))
        for i, xi in enumerate(onehot):
            for j in xi:
                x[i,j] = 1
        return x
    
    def sent_to_onehot(self, sent):
        chars = self._normalize(sent)
        ords = [ord(c) for c in chars]
        onehot = []
        for char, idx in zip(chars, ords):
            if idx == 32:
                onehot.append((self.space,))
            elif 48 <= idx <= 57:
                onehot.append((idx - 48 + self.number_begin, ))
            else:
                onehot.append(self._decompose(char, idx))
        return onehot
    
    def onehot_to_sent(self, encoded_sent):
        def check_cjj(c):
            cho, jung, jong = c
            if not (0 <= cho < self.jung_begin):
                raise ValueError('Chosung %d is out of index' % cho)
            if not (self.jung_begin <= jung < self.jong_begin):
                raise ValueError('Jungsung %d is out of index' % jung)
            if not (self.jong_begin <= jong < self.number_begin):
                raise ValueError('Jongsung %d is out of index' % jong)

        chars = []
        for c in encoded_sent:
            if len(c) == 1:
                if not 0 <= c[0] < self.dim:
                    raise ValueError('character index %d is out of index [0, %d]' % (c[0], self.dim))
                chars.append(self.idx_to_char[c[0]])
            elif len(c) == 3:
                check_cjj(c)
                cho, jung, jong = tuple(self.idx_to_char[ci] for ci in c)
                chars.append(compose(cho, jung, jong))
            else:
                chars.append(self.idx_to_char[-1])
        return ''.join(chars)
        
    def _normalize(self, sent):
        import re
        regex = re.compile('[^ㄱ-ㅎㅏ-ㅣ가-힣 0-9]')
        sent = regex.sub(' ', sent)
        sent = doublespace_pattern.sub(' ', sent).strip()
        return sent
        
    def _compose(self, cho, jung, jong):
        return chr(kor_begin + chosung_base * cho + jungsung_base * jung + jong)

    def _decompose(self, c, i):
        if kor_begin <= i <= kor_end:
            i -= kor_begin
            cho  = i // chosung_base
            jung = ( i - cho * chosung_base ) // jungsung_base
            jong = ( i - cho * chosung_base - jung * jungsung_base )
            return (cho, self.jung_begin + jung, self.jong_begin + jong)
        else:
            return (self.jamo_to_idx.get(c, self.unk), )