# -*- encoding:utf8 -*-

from soynlp.hangle import compose, decompose
from ._conjugation import conjugate, conjugate_chat

class Lemmatizer:
    def __init__(self, stems, endings, predefined=None):
        self._stems = stems
        self._endings = endings
        self._initialize()
        if predefined:
            self._predefined.update(predefined)

    def _initialize(self):
        self._predefined = {'불어':('붇다', '불다'),
                            '그래':('그렇다',)
                           }

    def lemmatize(self, word, check_only_stem=False):
        candidates = set()
        for i in range(1, len(word)+1):
            l, r = word[:i], word[i:]
            for stem, ending in lemma_candidate(l, r, self._predefined):
                if stem in self._stems:
                    if check_only_stem:
                        candidates.add((stem, ending))
                    elif ending in self._endings:
                        candidates.add((stem, ending))
        return candidates

    def candidates(self, word):
        candidates = set()
        for i in range(1, len(word) + 1):
            l = word[:i]
            r = word[i:]
            candidates.update(self.lemma_candidate(l, r, self._predefined))
        return candidates

def debug_message(message, l, r):
    print('{}: {} + {}'.format(message, l, r))

def lemma_candidate_chat(l, r, predefined=None, debug=False):
    def add_lemma(stem, ending):
        candidates.add((stem, ending))

    def character_is_emoticon(c):
        return c in set('ㄷㅂㅅㅇㅋㅎ')

    candidates = lemma_candidate(l, r, predefined, debug)
    l_last = decompose(l[-1])

    # 어미가 ㄷ, ㅂ, ㅅ, ㅇ, ㅋ, ㅎ 일 경우,
    # (아닏, 아닙, 아닛, 아닝, 아닠, 아닣)
    # (그랟, 그랩, 그랫, 그랭, 그랰, 그랳)
    if not r and character_is_emoticon(l_last[2]):
        l_ = l[:-1] + compose(l_last[0], l_last[1], ' ')
        if debug:
            debug_message('마지막 종성이 이모티콘으로 의심되는 경우', l_, '()')
        candidates.update(lemma_candidate(l_, r, predefined, debug))

    return candidates

def lemma_candidate(l, r, predefined=None, debug=False):
    def add_lemma(stem, ending):
        candidates.add((stem, ending))

    candidates = {(l, r)}
    word = l + r

    l_last = decompose(l[-1])
    l_last_ = compose(l_last[0], l_last[1], ' ')
    l_front = l[:-1]
    r_first = decompose(r[0]) if r else ('', '', '')
    r_first_ = compose(r_first[0], r_first[1], ' ') if r else ' '
    r_end = r[1:]

    # ㄷ 불규칙 활용: 깨달 + 아 -> 깨닫 + 아
    if l_last[2] == 'ㄹ' and r_first[0] == 'ㅇ':
        l_stem = l_front + compose(l_last[0], l_last[1], 'ㄷ')
        add_lemma(l_stem, r)
        if debug:
            debug_message('ㄷ 불규칙 활용', l_stem, r)

    # 르 불규칙 활용: 굴 + 러 -> 구르 + 어
    if (l_last[2] == 'ㄹ') and (r_first_ == '러' or r_first_ == '라'):
        l_stem = l_front + compose(l_last[0], l_last[1], ' ') + '르'
        r_canon = compose('ㅇ', r_first[1], r_first[2]) + r_end
        add_lemma(l_stem, r_canon)
        if debug:
            debug_message('르 불규칙 활용', l_stem, r_canon)

    # ㅂ 불규칙 활용: 더러 + 워서 -> 더럽 + 어서
    if (l_last[2] == ' '):
        l_stem = l_front + compose(l_last[0], l_last[1], 'ㅂ')
        if (r_first_ == '워' or r_first_ == '와'):
            r_canon = compose('ㅇ', 'ㅏ' if r_first_ == '와' else 'ㅓ', r_first[2] if r_first[2] else ' ') + r_end
        elif (r_end and r_end[0] =='려'):
            r_canon = compose('ㅇ', 'ㅜ', r_first[2] if r_first[2] else ' ') + r_end
        else:
            r_canon = r
        add_lemma(l_stem, r_canon)
        if debug:
            debug_message('ㅂ 불규칙 활용', l_stem, r_canon)

    # 어미의 첫글자가 종성일 경우 (-ㄴ, -ㄹ, -ㅁ-, -ㅂ, -ㅆ)
    # 입 + 니다 -> 이 + ㅂ니다
    if l_last[2] == 'ㄴ' or l_last[2] == 'ㄹ' or l_last[2] == 'ㅁ' or l_last[2] == 'ㅂ' or l_last[2] == 'ㅆ':
        for jongsung in ' ㄹㅂㅎ':
            if l_last[2] == jongsung:
                continue
            l_stem = l_front + compose(l_last[0], l_last[1], jongsung)
            r_canon = l_last[2] + r
            add_lemma(l_stem, r_canon)
            if debug:
                debug_message('어미의 첫글자가 종성일 경우 (%s)'%jongsung, l_stem, r_canon)

    # ㅅ 불규칙 활용: 부 + 어 -> 붓 + 어
    # exception : 벗 + 어 -> 벗어
    if (l_last[2] == ' ' and l[-1] != '벗') and (r_first[0] == 'ㅇ'):
        l_stem = l_front + compose(l_last[0], l_last[1], 'ㅅ')
        add_lemma(l_stem, r)
        if debug:
            debug_message('ㅅ 불규칙 활용', l_stem, r)

    # 우 불규칙 활용: 똥퍼 + '' -> 똥푸 + 어
    if l_last_ == '퍼':
        l_stem = l_front + '푸'
        r_canon = compose('ㅇ', l_last[1], l_last[2]) + r
        add_lemma(l_stem, r_canon)
        if debug:
            debug_message('우 불규칙 활용 (퍼)', l_stem, r_canon)

    # 우 불규칙 활용: 줬 + 어 -> 주 + 었어
    if l_last[1] == 'ㅝ':
        l_stem = l_front + compose(l_last[0], 'ㅜ', ' ')
        r_canon = compose('ㅇ', 'ㅓ', l_last[2]) + r
        add_lemma(l_stem, r_canon)
        if debug:
            debug_message('우 불규칙 활용', l_stem, r_canon)

    # 오 불규칙 활용: 왔 + 어 -> 오 + 았어
    if l_last[1] == 'ㅘ':
        l_stem = l_front + compose(l_last[0], 'ㅗ', ' ')
        r_canon = compose('ㅇ', 'ㅏ', l_last[2]) + r
        add_lemma(l_stem, r_canon)
        if debug:
            debug_message('오 불규칙 활용', l_stem, r_canon)

    # ㅡ 탈락 불규칙 활용: 꺼 + '' -> 끄 + 어 / 텄 + 어 -> 트 + 었어
    if (l_last[1] == 'ㅓ' or l_last[1] == 'ㅏ'):
        l_stem = l_front + compose(l_last[0], 'ㅡ', ' ')
        r_canon = compose('ㅇ', l_last[1], l_last[2]) + r
        add_lemma(l_stem, r_canon)
        if debug:
            debug_message('ㅡ 탈락 불규칙 활용 (꺼)', l_stem, r_canon)

    # ㅡ 탈락 불규칙 활용: 모 + 았다 -> 모으 + 았다
    if l_last[2] == ' ' and r_first[0] == 'ㅇ' and (r_first[1] == 'ㅏ' or r_first[1] == 'ㅓ'):
        l_stem = l + '으'
        r_canon = r
        add_lemma(l_stem, r_canon)
        if debug:
            debug_message('ㅡ 탈락 불규칙 활용 (모으)', l_stem, r_canon)

    # 거라, 너라 불규칙 활용
    # '-거라/-너라'를 어미로 취급하면 규칙 활용
    # if (l[-1] == '가') and (r and (r[0] == '라' or r[:2] == '거라')):
    #    # TODO

    # 러 불규칙 활용: 이르 + 러 -> 이르다
    # if (r_first[0] == 'ㄹ' and r_first[1] == 'ㅓ'):
    #     if self.is_stem(l):
    #         # TODO

    # 여 불규칙 활용
    # 하 + 였다 -> 하 + 았다 -> 하다: '였다'를 어미로 취급하면 규칙 활용

    # 여 불규칙 활용 (2)
    # 했 + 다 -> 하 + 았다 / 해 + 라니깐 -> 하 + 아라니깐 / 했 + 었다 -> 하 + 았었다
    if l_last[0] == 'ㅎ' and l_last[1] == 'ㅐ':
        l_stem = l_front + '하'
        r_canon = compose('ㅇ', 'ㅏ', l_last[2]) + r
        add_lemma(l_stem, r_canon)
        if debug:
            debug_message('여 불규칙 활용', l_stem, r_canon)

    # ㅎ (탈락) 불규칙 활용
    if (l_last[2] == ' ' or l_last[2] == 'ㄴ' or l_last[2] == 'ㄹ' or l_last[2] == 'ㅂ' or l_last[2] == 'ㅆ'):
        # 파라 + 면 -> 파랗 + 면
        if (l_last[1] == 'ㅏ' or l_last[1] == 'ㅓ'):
            l_stem = l_front + compose(l_last[0], l_last[1], 'ㅎ')
            r_canon = r if l_last[2] == ' ' else l_last[2] + r
            add_lemma(l_stem, r_canon)
            if debug:
                debug_message('ㅎ 탈락 불규칙 활용', l_stem, r_canon)
        # ㅎ (축약) 불규칙 할용
        # 시퍼렜 + 다 -> 시퍼렇 + 었다, 파랬 + 다 -> 파랗 + 았다
        if (l_last[1] == 'ㅐ') or (l_last[1] == 'ㅔ'):
            # exception : 그렇 + 아 -> 그래
            if len(l) >= 2 and l[-2] == '그' and l_last[0] == 'ㄹ':
                l_stem = l_front + '렇'
            else:
                l_stem = l_front + compose(l_last[0], 'ㅓ' if l_last[1] == 'ㅔ' else 'ㅏ', 'ㅎ')
            r_canon = compose('ㅇ', 'ㅓ' if l_last[1] == 'ㅔ' else 'ㅏ', l_last[2]) + r
            add_lemma(l_stem, r_canon)
            if debug:
                debug_message('ㅎ 축약 불규칙 활용', l_stem, r_canon)

    # 이었 -> 였 규칙활용
    # 좋아졌 + 어 -> 좋아지 + 었어, 좋아졋 + 던 -> 좋아지 + 었던, 좋아져 + 서 -> 좋아지 + 어서
    # 였 + 어 -> 이 + 었어
    # 종성 ㅆ 을 ㅅ 으로 쓰는 경우도 고려 (자주 등장하는 맞춤법 오류)
    if ((l_last[2] == 'ㅆ' or l_last[2] == 'ㅅ' or l_last[2] == ' ') and
        (l_last[1] == 'ㅕ')):

        # except: -었 -> 이 + 었 (x) // -였-> 이 + 었 (o) // -졌 -> 지 + 었 (o) // -젔 -> 지 + 었
        if ((l_last[0] == 'ㅇ') and (l_last[1] == 'ㅕ')) or not (l_last[0] == 'ㅇ'):
            l_stem = l_front + compose(l_last[0], 'ㅣ', ' ')
            r_canon = compose('ㅇ', 'ㅓ', l_last[2])+ r
            add_lemma(l_stem, r_canon)
            if debug:
                debug_message('이었 -> 였 규칙 활용', l_stem, r_canon)

    ## Pre-defined set
    if predefined and (l, r) in predefined:
        for stem in predefined[(l, r)]:
            candidates.add(stem)
            if debug:
                debug_message('Predefined', l_stem, r_canon)

    # check whether lemma is conjugatable
    candidates_ = set()
    for stem, eomi in candidates:
        if not eomi:
            continue
        # hard rule
        if decompose(eomi[0])[2] == 'ㅎ':
            continue
        surfaces = conjugate(stem, eomi)
        if word in surfaces:
            candidates_.add((stem, eomi))
    return candidates_