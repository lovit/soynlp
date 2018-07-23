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
    message = 'normalize func will be moved soynlp.normalizer at ver 0.1'
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
        self.dim = 79
    
    def encode(self, sent):
        onehot = self.sent_to_onehot(sent)
        x = np.zeros((len(onehot), self.dim))
        for i, xi in enumerate(onehot):
            for j in xi:
                x[i,j] = 1
        return x
    
    def sent_to_onehot(self, sent):
        sent = self._normalize(sent)
        sent = [ord(c) for c in sent]
        sent_ = []
        for i in sent:
            if i == 32: sent_.append((self.space,))
            elif 48 <= i <= 57: sent_.append((i - 48 + self.number_begin, ))
            else: sent_.append(self._decompose(i))
        return sent_
    
    def onehot_to_sent(self, encoded_sent):
        chars = []
        for c in encoded_sent:
            if len(c) == 1:
                idx = c[0] - self.number_begin
                if idx == 10:
                    chars.append(' ')
                elif 0 <= idx < 10:
                    chars.append(str(idx))
                else:
                    chars.append(None)
            elif len(c) == 3:
                cho, jung, jong = c
                if (0 <= cho < self.jung_begin) and (self.jung_begin <= jung < self.jong_begin) and (self.jong_begin <= jong < self.number_begin):
                    chars.append(compose(chosung_list[cho], jungsung_list[jung - self.jung_begin], jongsung_list[jong - self.jong_begin]))
                else:
                    chars.append(None)
            else:
                chars.append(None)
        return ''.join(chars)
        
    def _normalize(self, sent):
        import re
        regex = re.compile('[^ㄱ-ㅎㅏ-ㅣ가-힣 0-9]')
        sent = regex.sub(' ', sent)
        sent = doublespace_pattern.sub(' ', sent)
        return sent
        
    def _compose(self, cho, jung, jong):
        return chr(kor_begin + chosung_base * cho + jungsung_base * jung + jong)

    def _decompose(self, i):
        if (jaum_begin <= i <= jaum_end):
            return (i - jaum_begin,)
        if (moum_begin <= i <= moum_end):
            return (i - moum_begin,)
        i -= kor_begin
        cho  = i // chosung_base
        jung = ( i - cho * chosung_base ) // jungsung_base 
        jong = ( i - cho * chosung_base - jung * jungsung_base )    
        return (cho, self.jung_begin + jung, self.jong_begin + jong)