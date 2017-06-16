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

    f = []    
    for c in doc:
        if (c == ' '):
            f.append(c)
            continue
        i = to_base(c)
        if (kor_begin <= i <= kor_end) or (jaum_begin <= i <= jaum_end) or (moum_begin <= i <= moum_end):
            f.append(c)
            continue
        if (english) and (i >= 97 and i <= 122) or (i >= 65 and i <= 90):
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

def character_is_jaum(c):
    return (jaum_begin <= to_base(c) <= jaum_end)

def character_is_moum(c):
    return (moum_begin <= to_base(c) <= moum_end)

def to_base(c):
    if type(c) == str:
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