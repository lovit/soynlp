# -*- encoding:utf8 -*-

from soynlp.hangle import compose, decompose

class Lemmatizer:
    def __init__(self, roots, surfacial_eomis, predefined=None):
        self._roots = roots
        self._surfacial_eomis = surfacial_eomis
        self._initialize()
        if predefined:
            self._predefined.update(predefined)

    def _initialize(self):
        self._predefined = {'불어':('붇다', '불다'),
                            '그래':('그렇다',)
                           }

    def is_root(self, w): return w in self._roots
    def is_surfacial_eomi(self, w): return w in self._surfacial_eomis

    def lemmatize(self, word, check_if_r_is_unknown=False):
        candidates = set()
        for i in range(1, len(word)+1):
            l, r = word[:i], word[i:]
            if not check_if_r_is_unknown and not self.is_surfacial_eomi(r):
                continue
            candidates.update(self._candidates(l, r))
        return candidates

    def candidates(self, word):
        candidates = set()
        for i in range(1, len(word) + 1):
            l = word[:i]
            r = word[i:]
            candidates.update(self._candidates(l, r))
        return candidates

    def _candidates(self, l, r):
        candidates = set()
        if self.is_root(l):
            candidates.add((l, r))

        l_last = decompose(l[-1])
        l_last_ = compose(l_last[0], l_last[1], ' ')
        r_first = decompose(r[0]) if r else ('', '', '')
        r_first_ = compose(r_first[0], r_first[1], ' ') if r else ' '

        # ㄷ 불규칙 활용: 깨닫 + 아 -> 깨달아
        if l_last[2] == 'ㄹ' and r_first[0] == 'ㅇ':
            l_root = l[:-1] + compose(l_last[0], l_last[1], 'ㄷ')
            if self.is_root(l_root):
                candidates.add((l_root, r))

        # 르 불규칙 활용: 굴 + 러 -> 구르다
        if (l_last[2] == 'ㄹ') and (r_first_ == '러' or (r_first_ == '라')):
            l_root = l[:-1] + compose(l_last[0], l_last[1], ' ') + '르'
            r_canon = compose('ㅇ', r_first[1], r_first[2]) + r[1:]
            if self.is_root(l_root):
                candidates.add((l_root, r_canon))

        # ㅂ 불규칙 활용: 더러 + 워서 -> 더럽다
        if (l_last[2] == ' ') and (r_first_ == '워' or r_first_ == '와'):
            l_root = l[:-1] + compose(l_last[0], l_last[1], 'ㅂ')
            r_canon = compose('ㅇ', 'ㅏ' if r_first_ == '와' else 'ㅓ', r_first[2]) + r[1:]
            if self.is_root(l_root):
                candidates.add((l_root, r_canon))

#         # 어미의 첫글자가 종성일 경우 (-ㄴ, -ㄹ, -ㅂ, -ㅅ)
#         # 입 + 니다 -> 입니다
        if l_last[2] == 'ㄴ' or l_last[2] == 'ㄹ' or l_last[2] == 'ㅂ' or l_last[2] == 'ㅆ':
            l_root = l[:-1] + compose(l_last[0], l_last[1], ' ')
            r_canon = l_last[2] + r
            if self.is_root(l_root):
                candidates.add((l_root, r_canon))

#         # ㅅ 불규칙 활용: 부 + 었다 -> 붓다
#         # exception : 벗 + 어 -> 벗어
        if (l_last[2] == ' ' and l[-1] != '벗') and (r_first[0] == 'ㅇ'):
            l_root = l[:-1] + compose(l_last[0], l_last[1], 'ㅅ')
            if self.is_root(l_root):
                candidates.add((l_root, r))

        # 우 불규칙 활용: 똥퍼 + '' -> 똥푸다
        if l_last_ == '퍼':
            l_root = l[:-1] + '푸'
            r_canon = compose('ㅇ', l_last[1], l_last[2]) + r
            if self.is_root(l_root):
                candidates.add((l_root, r_canon))

        # 우 불규칙 활용: 줬 + 어 -> 주다
        if l_last[1] == 'ㅝ':
            l_root = l[:-1] + compose(l_last[0], 'ㅜ', ' ')
            r_canon = compose('ㅇ', 'ㅓ', l_last[2]) + r
            if self.is_root(l_root):
                candidates.add((l_root, r_canon))

        # 오 불규칙 활용: 왔 + 어 -> 오다
        if l_last[1] == 'ㅘ':
            l_root = l[:-1] + compose(l_last[0], 'ㅗ', ' ')
            r_canon = compose('ㅇ', 'ㅏ', l_last[2]) + r
            if self.is_root(l_root):
                candidates.add((l_root, r_canon))

        # ㅡ 탈락 불규칙 활용: 꺼 + '' -> 끄다 / 텄 + 어 -> 트다
        if (l_last[1] == 'ㅓ' or l_last[1] == 'ㅏ'):
            l_root = l[:-1] + compose(l_last[0], 'ㅡ', ' ')
            r_canon = compose('ㅇ', l_last[1], l_last[2]) + r
            if self.is_root(l_root):
                candidates.add((l_root, r_canon))

        # 거라, 너라 불규칙 활용
        # '-거라/-너라'를 어미로 취급하면 규칙 활용
        # if (l[-1] == '가') and (r and (r[0] == '라' or r[:2] == '거라')):
        #    # TODO

        # 러 불규칙 활용: 이르 + 러 -> 이르다
        # if (r_first[0] == 'ㄹ' and r_first[1] == 'ㅓ'):
        #     if self.is_root(l):
        #         # TODO

        # 여 불규칙 활용
        # 하 + 였다 -> 하 + 았다 -> 하다: '였다'를 어미로 취급하면 규칙 활용

        # ㅎ (탈락) 불규칙 활용
        # 파라 + 면 -> 파랗다
        if (l_last[2] == ' ' or l_last[2] == 'ㄴ' or l_last[2] == 'ㄹ' or l_last[2] == 'ㅂ' or l_last[2] == 'ㅆ'):
            l_root = l[:-1] + compose(l_last[0], l_last[1], 'ㅎ')
            r_canon = r if l_last[2] == ' ' else l_last[2] + r
            if self.is_root(l_root):
                candidates.add((l_root, r_canon))

        # ㅎ (축약) 불규칙 할용
        # 시퍼렜 + 다 -> 시퍼렇다, 파랬 + 다 -> 파랗다, 파래 + '' -> 파랗다
        if (l_last[1] == 'ㅐ') or (l_last[1] == 'ㅔ'):
            # exception : 그렇 + 아 -> 그래
            if len(l) >= 2 and l[-2] == '그' and l_last[0] == 'ㄹ':
                l_root = l[:-1] + '렇'
            else:
                l_root = l[:-1] + compose(l_last[0], 'ㅓ' if l_last[1] == 'ㅔ' else 'ㅏ', 'ㅎ')
            r_canon = compose('ㅇ', 'ㅓ' if l_last[1] == 'ㅔ' else 'ㅏ', l_last[2]) + r
            if self.is_root(l_root):
                candidates.add((l_root, r_canon))

        ## Pre-defined set
        if (l, r) in self._predefined:
            for root in self._predefined[(l, r)]:
                candidates.add(root)

        return candidates