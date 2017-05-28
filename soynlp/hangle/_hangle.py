import re

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
    if remove_repeat > 0:
        doc = repeatchars_pattern.sub('\\1' * remove_repeat, doc)

    f = ''    
    for c in doc:
        i = ord(c)
        
        if (c == ' ') or (is_korean(i)) or (english and is_english(i)) or (number and is_number(i)) or (punctuation and is_punctuation(i)):
            f += c            
        elif c in remains:
            f += c        
        else:
            f += ' '
            
    return doublespace_pattern.sub(' ', f).strip()

def split_jamo(c):    
    i = ord(c)
    
    if not is_korean(i):
        return None
    elif is_jaum(i):
        return [c, ' ', ' ']
    elif is_moum(i):
        return [' ', c, ' ']
    
    i -= kor_begin
    
    cho  = i // chosung_base
    jung = ( i - cho * chosung_base ) // jungsung_base 
    jong = ( i - cho * chosung_base - jung * jungsung_base )
    
    return [chosung_list[cho], jungsung_list[jung], jongsung_list[jong]]

def is_korean(i):
    i = to_base(i)
    return (kor_begin <= i <= kor_end) or (jaum_begin <= i <= jaum_end) or (moum_begin <= i <= moum_end)

def is_number(i):
    i = to_base(i)
    return (i >= 48 and i <= 57)

def is_english(i):
    i = to_base(i)
    return (i >= 97 and i <= 122) or (i >= 65 and i <= 90)

def is_punctuation(i):
    i = to_base(i)
    return (i == 33 or i == 34 or i == 39 or i == 44 or i == 46 or i == 63 or i == 96)

def is_jaum(i):
    i = to_base(i)
    return (jaum_begin <= i <= jaum_end)

def is_moum(i):
    i = to_base(i)
    return (moum_begin <= i <= moum_end)

def to_base(c):
    if type(c) == str:
        return ord(c)
    elif type(c) == int:
        return c
    else:
        raise TypeError

def combine_jamo(chosung, jungsung, jongsung):
    return chr(kor_begin + chosung_base * chosung_list.index(chosung) + jungsung_base * jungsung_list.index(jungsung) + jongsung_list.index(jongsung))


class ConvolutionalNN_Encoder:
        
    def __init__(self, vocabs={}):
        self.vocabs = vocabs
        
        self.jungsung_hot_begin = 31
        self.jongsung_hot_begin = 52
        self.symbol_hot_begin = 83

        self.cvocabs_ = [' ', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄸ', 'ㄹ', 
                   'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅃ', 
                   'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 
                   'ㅎ', 'ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ',
                   'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 
                   'ㅢ', 'ㅣ']
        self.cvocabs = {}
        self.cvocabs = {c:len(cvocabs) + 1 for c in cvocabs_}

        # svocabs_ = ['.',  ',',  '?',  '!',  '-', ':', 
        #            '0',  '1',  '2',  '3',  '4',  '5',  '6',  '7',  '8',  '9']
        # svocabs = {}
        # svocabs = {s:len(svocabs) + symbol_hot_begin for s in svocabs_}


    def encode_vocab(self, words, unknown=-1, blank=0, input_length=64):
        if len(words) > input_length:
            words = words[:input_length]
        return [self.vocabs[w] if w in self.vocabs else unknown for w in words] + [blank] * (input_length - len(words))

    def encode_jamo_onehot(self, chars, input_length=64, as_ndarray=False):
        ints = []
        return ints

    def encode_jamo_threehot(self, chars, input_length=64, as_ndarray=False):
        raise NotImplemented

'''
def _to_code(c):
    if c == ' ':
        return [0, 0, 0]
    if c in svocabs:
        return [svocabs[c], 0, 0]

    i = ord(c)

    if not is_korean(i):
        return [-1, -1, -1]
    elif is_jaum(i):
        return [i - jaum_begin, 0, 0]
    elif is_moum(i):
        return [0, i - moum_begin + jongsung_hot_begin, 0]

    i -= kor_begin

    cho  = i // chosung_base
    jung = ( i - cho * chosung_base ) // jungsung_base 
    jong = ( i - cho * chosung_base - jung * jungsung_base )
    return [cvocabs[chosung_list[cho]], cvocabs[jungsung_list[jung]] + jungsung_hot_begin, cvocabs[jongsung_list[jong]] + jongsung_hot_begin]

def _to_symbol(cjc):
    print([cjc[0], cjc[1] - jungsung_hot_begin, cjc[2] - jongsung_hot_begin])
    return [cvocabs_[cjc[0]], cvocabs_[cjc[1] - jungsung_hot_begin + jungsung_hot_begin - 1], cvocabs_[cjc[2]  - jongsung_hot_begin]]
'''


