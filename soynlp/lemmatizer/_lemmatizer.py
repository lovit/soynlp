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

    def lemmatize(self, word):
        candidates = set()
        for i in range(1, len(word)+1):
            l, r = word[:i], word[i:]
            if not self.is_surfacial_eomi(r):
                continue
            candidates.update(self._candidates(l, r))
        return candidates

    def _candidates(self, l, r):
        candidates = set()
        if self.is_root(l):
            candidates.add(l + '다')

        l_last = decompose(l[-1])
        l_last_ = compose(l_last[0], l_last[1], ' ')
        r_first = decompose(r[0]) if r else ('', '', '')
        r_first_ = compose(r_first[0], r_first[1], ' ') if r else ' '

        ## 1. 어간이 바뀌는 불규칙 활용
        # 1.1. ㄷ 불규칙 활용: 깨닫 + 아 -> 깨달아
        if l_last[2] == 'ㄹ' and r_first[0] == 'ㅇ':
            l_root = l[:-1] + compose(l_last[0], l_last[1], 'ㄷ')
            if self.is_root(l_root):
                candidates.add(l_root + '다')

        # 1.2. 르 불규칙 활용: 굴 + 러 -> 구르다
        if (l_last[2] == 'ㄹ') and (r_first_ == '러' or (r_first_ == '라')):
            l_root = l[:-1] + compose(l_last[0], l_last[1], ' ') + '르'
            if self.is_root(l_root):
                candidates.add(l_root + '다')

        # 1.3. ㅂ 불규칙 활용: 더러 + 워서 -> 더럽다
        if (l_last[2] == ' ') and (r_first_ == '워'):
            l_root = l[:-1] + compose(l_last[0], l_last[1], 'ㅂ')
            if self.is_root(l_root):
                candidates.add(l_root + '다')

        # 1.3. ㅂ 불규칙 활용: 도 + 왔다 -> 돕다
        if (l == '도' or l == '고') and (r_first_ == '와'):
            l_root = compose(l_last[0], l_last[1], 'ㅂ')
            if self.is_root(l_root):
                candidates.add(l_root + '다')

        # 1.3. (추가) ㅂ 추가 불규칙: 입 + 니다 -> 이다, 합 + 니다 -> 하다
        if l_last[2] == 'ㅂ':
            l_root = compose(l_last[0], l_last[1], ' ')
            if self.is_root(l_root):
                candidates.add(l_root + '다')

        # 1.4. ㅅ 불규칙 활용: 부 + 었다 -> 붓다
        if (l_last[2] == ' ') and (r_first[0] == 'ㅇ'):
            l_root = l[:-1] + compose(l_last[0], l_last[1], 'ㅅ')
            if self.is_root(l_root):
                candidates.add(l_root + '다')

        # 1.5. 우 불규칙 활용: 똥퍼 + '' -> 똥푸다
        if l_last_ == '퍼':
            l_root = l[:-1] + '푸'
            if self.is_root(l_root):
                candidates.add(l_root + '다')

        # 1.5. 우 불규칙 활용: 줬 + 어 -> 주다
        if l_last[1] == 'ㅝ':
            l_root = l[:-1] + compose(l_last[0], 'ㅜ', ' ')
            if self.is_root(l_root):
                candidates.add(l_root + '다')

        # 1.6. ㅡ 탈락 불규칙 활용: 꺼 + '' -> 끄다 / 텄 + 어 -> 트다
        if (l_last[1] == 'ㅓ' or l_last[1] == 'ㅏ'):
            l_root = l[:-1] + compose(l_last[0], 'ㅡ', ' ')
            if self.is_root(l_root):
                candidates.add(l_root + '다')

        ## 2. 어미가 바뀌는 불규칙 활용
        # 2.1. 거라 불규칙 활용
        if (l[-1] == '가') and (r and (r[0] == '라' or r[:2] == '거라')):
            candidates.add(l + '다')

        # 2.2. 너라 불규칙 활용
        # 2.2.1: 규칙활용: 돌아오 + 너라 -> 돌아오다, 돌아오 + 라고 -> 돌아오다
        # 2.2.2: 돌아 + 왔다 -> 돌아오다
        if (l_last[1] == 'ㅘ'):
            l_root = l[:-1] + compose(l_last[0], 'ㅗ', ' ')
            if self.is_root(l_root):
                candidates.add(l_root + '다')

        # 2.3. 러 불규칙 활용: 이르 + 러 -> 이르다
        if (r_first[0] == 'ㄹ' and r_first[1] == 'ㅓ'):
            if self.is_root(l):
                candidates.add(l + '다')

        # 2.4. 여 불규칙 활용
        # 하 + 였다 -> 하 + 았다 -> 하다: '였다'를 어미로 넣으면 되는 문제

        # 2.5. 오 불규칙 활용
        # 달 + 아라 -> 다오, 걸 + 어라 -> 거오: 문어체적 표현에 자주 등장하며 구어체에서는 거의 없음
        # 생략

        ## 3. 어간과 어미가 모두 바뀌는 불규칙 활용
        # 3.1. ㅎ 불규칙 활용
        # 3.1.1: 파라 + 면 -> 파랗다
        if (l_last[2] == ' '):
            l_root = l[:-1] + compose(l_last[0], l_last[1], 'ㅎ')
            if self.is_root(l_root):
                candidates.add(l_root + '다')

        # 3.1.2. 시퍼렜 + 다 -> 시퍼렇다, 파랬 + 다 -> 파랗다, 파래 + '' -> 파랗다
        if (l_last[1] == 'ㅐ') or (l_last[1] == 'ㅔ'):
            l_root = l[:-1] + compose(l_last[0], 'ㅓ' if l_last[1] == 'ㅔ' else 'ㅏ', 'ㅎ')
            if self.is_root(l_root):
                candidates.add(l_root + '다')

        # (추가) 3.2 어미가 ㄴ인 경우: 간 + '' -> 가다, 푸른 + '' -> 푸르다, 
        # 한 + '' -> 하다, 이른 + '' -> 이르다
        if (not r) and (l_last[2] == 'ㄴ' or l_last[2] == 'ㄹ'):
            l_root = l[:-1] + compose(l_last[0], l_last[1], ' ')
            if self.is_root(l_root):
                candidates.add(l_root + '다')
            # 노란 -> 노랗다
            l_root = l[:-1] + compose(l_last[0], l_last[1], 'ㅎ')
            if self.is_root(l_root):
                candidates.add(l_root + '다')

        ## Pre-defined set
        if l+r in self._predefined:
            for root in self._predefined[l+r]:
                candidates.add(root)

        return candidates

    def load(self, filename):
        with open(filename, encoding='utf-8') as fp:
            params = json.load(fp)
            self._roots = set(params['roots'])
            self._surfacial_eomis = set(params['surfacial_eomis'])
            self._predefined = {surfacial:tuple(canonicals)
                for surfacial, canonicals in params['predefined'].items()}

    def save(self, filename):
        with open(filename, 'w', encoding='utf-8') as fp:
            params = {
                'roots':list(self._roots),
                'surfacial_eomis':list(self._surfacial_eomis),
                'predefined': {surfacial:list(canonicals) 
                    for surfacial, canonicals in self._predefined.items()}
            }
            json.dump(params, fp, ensure_ascii=False, indent=2)