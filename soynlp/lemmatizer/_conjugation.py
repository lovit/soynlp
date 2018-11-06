# -*- encoding:utf8 -*-

from soynlp.hangle import compose, decompose

def conjugate(stem, ending):

    assert ending # ending must be inserted

    l_len = len(stem)
    l_last = decompose(stem[-1])
    l_last_ = stem[-1]
    r_first = decompose(ending[0])
    r_first_ = compose(r_first[0], r_first[1], ' ') if r_first[1] != ' ' else ending[0]

    candidates = set()
    
    # ㄷ 불규칙 활용: 깨달 + 아 -> 깨달아
    if l_last[2] == 'ㄷ' and r_first[0] == 'ㅇ':
        l = stem[:-1] + compose(l_last[0], l_last[1], 'ㄹ')
        candidates.add(l + ending)

    # 르 불규칙 활용: 구르 + 어 -> 굴러
    if (l_last_ == '르') and (r_first_ == '아' or r_first_ == '어') and l_len >= 2:
        c0, c1, c2 = decompose(stem[-2])
        l = stem[:-2] + compose(c0, c1, 'ㄹ')
        r = compose('ㄹ', r_first[1], r_first[2]) + ending[1:]
        candidates.add(l + r)

    # ㅂ 불규칙 활용:
    # (모음조화) 더럽 + 어 -> 더러워 / 곱 + 아 -> 고와 
    # (모음조화가 깨진 경우) 아름답 + 아 -> 아름다워 / (-답, -꼽, -깝, -롭)
    if (l_last[2] == 'ㅂ'):
        l = stem[:-1] + compose(l_last[0], l_last[1], ' ')
        if (r_first_ == '어' or r_first_ == '아'):
            if l_len >= 2 and (l_last_ == '답' or l_last_ == '곱' or l_last_ == '깝' or l_last_ == '롭'):
                c1 = 'ㅝ'
            elif r_first[1] == 'ㅗ':
                c1 = 'ㅘ'
            elif r_first[1] == 'ㅜ':
                c1 = 'ㅝ'
            elif r_first_ == '어':
                c1 = 'ㅝ'
            else: # r_first_ == '아'
                c1 = 'ㅘ'
            r = compose('ㅇ', c1, r_first[2]) + ending[1:]
            candidates.add(l + r)
        elif r_first[0] == 'ㅇ': # 돕 + 울까 = 도울까, 답 + 울까 = 다울까
            candidates.add(l + ending)

    # 어미의 첫글자가 종성일 경우 (-ㄴ, -ㄹ, -ㅂ, -ㅆ)
    # 이 + ㅂ니다 -> 입니다
    if l_last[2] == ' ' and r_first[1] == ' ' and (r_first[0] == 'ㄴ' or r_first[0] == 'ㄹ' or r_first[0] == 'ㅂ' or r_first[0] == 'ㅆ'):
        l = stem[:-1] + compose(l_last[0], l_last[1], r_first[0])
        r = ending[1:]
        candidates.add(l + r)

    # ㅅ 불규칙 활용: 붓 + 어 -> 부어
    # exception : 벗 + 어 -> 벗어    
    if (l_last[2] == 'ㅅ') and (r_first[0] == 'ㅇ'):
        if stem[-1] == '벗':
            l = stem
        else:
            l = stem[:-1] + compose(l_last[0], l_last[1], ' ')
        candidates.add(l + ending)

    # 우 불규칙 활용: 푸 + 어 -> 퍼 / 주 + 어 -> 줘
    if l_last[1] == 'ㅜ' and l_last[2] == ' ' and r_first[0] == 'ㅇ' and r_first[1] == 'ㅓ':
        if l_last_ == '푸':
            l = '퍼'
        else:
            l = stem[:-1] + compose(l_last[0], 'ㅝ', r_first[2])
        r = ending[1:]
        candidates.add(l + r)

    # 오 활용: 오 + 았어 -> 왔어
    if l_last[1] == 'ㅗ' and l_last[2] == ' ' and r_first[0] == 'ㅇ' and r_first[1] == 'ㅏ':
        l = stem[:-1] + compose(l_last[0], 'ㅘ', r_first[2])
        r = ending[1:]
        candidates.add(l + r)

    # ㅡ 탈락 불규칙 활용: 끄 + 어 -> 꺼 / 트 + 었다 -> 텄다
    if (l_last_ == '끄' or l_last_ == '크' or l_last_ == '트') and (r_first[0] == 'ㅇ') and (r_first[1] == 'ㅓ'):
        l = stem[:-1] + compose(l_last[0], r_first[1], r_first[2])
        r = ending[1:]
        candidates.add(l + r)

    # 거라, 너라 불규칙 활용
    # '-거라/-너라'를 어미로 취급하면 규칙 활용
    if ending[:2] == '어라' or ending[:2] == '아라':
        if l_last[1] == 'ㅏ':            
            r = '거' + ending[1:]
        elif l_last[1] == 'ㅗ':
            r = '너' + ending[1:]
        else:
            r = ending
        candidates.add(stem + r)

    # 러 불규칙 활용: 이르 + 어 -> 이르러 / 이르 + 었다 -> 이르렀다
    if l_last_ == '르' and r_first[0] == 'ㅇ' and r_first[1] == 'ㅓ':
        r = compose('ㄹ', r_first[1], r_first[2]) + ending[1:]
        candidates.add(stem + r)

    # 여 불규칙 활용
    # 하 + 았다 -> 하였다 / 하 + 었다 -> 하였다
    if l_last_ == '하' and r_first[0] == 'ㅇ' and (r_first[1] == 'ㅏ' or r_first[1] == 'ㅓ'):
        # case 1
        r = compose(r_first[0], 'ㅕ', r_first[2]) + ending[1:]
        candidates.add(stem + r)
        # case 2
        l = stem[:-1] + compose('ㅎ', 'ㅐ', r_first[2])
        r = ending[1:]
        candidates.add(l + r)

    # ㅎ (탈락) 불규칙 활용
    # 파라 + 면 -> 파랗다 / 동그랗 + ㄴ -> 동그란
    if l_last[2] == 'ㅎ' and l_last_ != '좋' and not (r_first[1] == 'ㅏ' or r_first[1] == 'ㅓ'):
        if r_first[1] == ' ':
            l = l = stem[:-1] + compose(l_last[0], l_last[1], r_first[0])
        else:
            l = stem[:-1] + compose(l_last[0], l_last[1], ' ')
        if r_first_ == '으':
            r = ending[1:]
        elif r_first[1] == ' ':            
            r = ''
        else:
            r = ending
        candidates.add(l + r)

    # ㅎ (축약) 불규칙 할용
    # 파랗 + 았다 -> 파랬다 / 시퍼렇 + 었다 -> 시퍼렜다
    if l_last[2] == 'ㅎ' and l_last_ != '좋' and (r_first[1] == 'ㅏ' or r_first[1] == 'ㅓ'):
        l = stem[:-1] + compose(l_last[0], 'ㅐ' if r_first[1] == 'ㅏ' else 'ㅔ', r_first[2])
        r = ending[1:]
        candidates.add(l + r)

    # ㅎ + 네 불규칙 활용
    # ㅎ 탈락과 ㅎ 유지 모두 맞음
    if l_last[2] == 'ㅎ' and r_first[0] == 'ㄴ' and r_first[1] != ' ':
        candidates.add(stem + ending)

    # 이었 -> 였 규칙활용
    if ending[0] == '었' and l_last[1] == 'ㅣ' and l_last[2] == ' ':
        candidates.add(stem[:-1] + compose(l_last[0], 'ㅕ', 'ㅆ') + ending[1:])
        if l_last[0] == 'ㅇ':
            candidates.add(stem + ending)

    if not candidates and r_first[1] != ' ':
        candidates.add(stem + ending)

    return candidates

def _conjugate_stem(stem):

    l_len = len(stem)
    l_last = decompose(stem[-1])
    l_last_ = stem[-1]

    candidates = {stem}

    # ㄷ 불규칙 활용: 깨달 + 아 -> 깨달아
    if l_last[2] == 'ㄷ':
        l = stem[:-1] + compose(l_last[0], l_last[1], 'ㄹ')
        candidates.add(l)

    # 르 불규칙 활용: 구르 + 어 -> 굴러
    if (l_last_ == '르') and l_len >= 2:
        c0, c1, c2 = decompose(stem[-2])
        l = stem[:-2] + compose(c0, c1, 'ㄹ')
        candidates.add(l)

    # ㅂ 불규칙 활용:
    # (모음조화) 더럽 + 어 -> 더러워 / 곱 + 아 -> 고와
    # (모음조화가 깨진 경우) 아름답 + 아 -> 아름다워 / (-답, -꼽, -깝, -롭)
    if (l_last[2] == 'ㅂ'):
        l = stem[:-1] + compose(l_last[0], l_last[1], ' ')
        candidates.add(l)

    # 어미의 첫글자가 종성일 경우 (-ㄴ, -ㄹ, -ㅂ, -ㅆ)
    # 이 + ㅂ니다 -> 입니다
    if l_last[2] == ' ':
        candidates.add(stem[:-1] + compose(l_last[0], l_last[1], 'ㄴ'))
        candidates.add(stem[:-1] + compose(l_last[0], l_last[1], 'ㄹ'))
        candidates.add(stem[:-1] + compose(l_last[0], l_last[1], 'ㅂ'))
        candidates.add(stem[:-1] + compose(l_last[0], l_last[1], 'ㅆ'))

    # ㅅ 불규칙 활용: 붓 + 어 -> 부어
    # exception : 벗 + 어 -> 벗어
    if (l_last[2] == 'ㅅ') and stem[-1] != '벗':
        candidates.add(stem[:-1] + compose(l_last[0], l_last[1], ' '))

    # 우 불규칙 활용: 푸 + 어 -> 퍼 / 주 + 어 -> 줘
    if l_last[1] == 'ㅜ' and l_last[2] == ' ':
        if l_last_ == '푸':
            l = '퍼'
        else:
            candidates.add(stem[:-1] + compose(l_last[0], 'ㅝ', ' '))
            candidates.add(stem[:-1] + compose(l_last[0], 'ㅝ', 'ㅆ'))

    # 오 활용: 오 + 았어 -> 왔어
    if l_last[1] == 'ㅗ' and l_last[2] == ' ':
        candidates.add(stem[:-1] + compose(l_last[0], 'ㅘ', ' '))
        candidates.add(stem[:-1] + compose(l_last[0], 'ㅘ', 'ㅆ'))

    # ㅡ 탈락 불규칙 활용: 끄 + 어 -> 꺼 / 트 + 었다 -> 텄다
    if (l_last_ == '끄' or l_last_ == '크' or l_last_ == '트'):
        candidates.add(stem[:-1] + compose(l_last[0], 'ㅓ', ' '))
        candidates.add(stem[:-1] + compose(l_last[0], 'ㅓ', 'ㅆ'))

    # 거라, 너라 불규칙 활용
    # '-거라/-너라'를 어미로 취급하면 규칙 활용

    # 러 불규칙 활용: 이르 + 어 -> 이르러 / 이르 + 었다 -> 이르렀다

    # 여 불규칙 활용
    # 하 + 았다 -> 하였다 / 하 + 었다 -> 하였다
    # 하 + 았다 -> 했다
    if l_last_ == '하':
        candidates.add(stem[:-1] + '해')
        candidates.add(stem[:-1] + '했')

    # ㅎ (탈락) 불규칙 활용
    # 파라 + 면 -> 파랗다 / 동그랗 + ㄴ -> 동그란
    if l_last[2] == 'ㅎ' and l_last_ != '좋':
        candidates.add(stem[:-1] + compose(l_last[0], l_last[1], ' '))
        candidates.add(stem[:-1] + compose(l_last[0], l_last[1], 'ㄴ'))
        candidates.add(stem[:-1] + compose(l_last[0], l_last[1], 'ㄹ'))
        # candidates.add(stem[:-1] + compose(l_last[0], l_last[1], 'ㅂ'))
        candidates.add(stem[:-1] + compose(l_last[0], l_last[1], 'ㅆ'))

    # ㅎ (축약) 불규칙 할용
    # 파랗 + 았다 -> 파랬다 / 시퍼렇 + 었다 -> 시퍼렜다
    if l_last[2] == 'ㅎ' and l_last_ != '좋':
        candidates.add(stem[:-1] + compose(l_last[0], 'ㅐ', 'ㅆ'))
        # candidates.add(stem[:-1] + compose(l_last[0], 'ㅔ', 'ㅆ'))

    # ㅎ + 네 불규칙 활용
    # ㅎ 탈락과 ㅎ 유지 모두 맞음

    # 이었 -> 였 규칙활용
    if l_last[1] == 'ㅣ' and l_last[2] == ' ':
        candidates.add(stem[:-1] + compose(l_last[0], 'ㅕ', 'ㅆ'))

    return candidates