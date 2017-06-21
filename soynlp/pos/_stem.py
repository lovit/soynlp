def conjugate(v, e):
    # Develop code
    verbs = {
        '깨닫', '불', '묻', '눋', '겯', '믿', '묻', '뜯'
    }

    eomis = {
        '아', '어나다', '어'
    }

    def is_verb(w): return w in verbs
    def is_eomi(w): return w in eomis
    
    v_ = decompose(v[-1])
    e_ = decompose(e[0])
    
    # https://namu.wiki/w/한국어/불규칙%20활용
    ## 1. 어간이 바뀌는 불규칙 활용
    # 1.1. ㄷ 불규칙 활용: 깨닫 + 아 -> 꺠달아
    if (v_[2] == 'ㄹ' and e_[0] == 'ㅇ'):
        # 규칙 활용
        if v == '믿' or v == '묻' or v == '뜯':
            return (v, e)
        canonicalv = v[:-1] + compose(v_[0], v_[1], 'ㄷ')
        if is_verb(canonicalv) and is_eomi(e):
            return (canonicalv, e)
    
    # 1.2. 르 불규칙 활용
    
    
    # 1.3. ㅂ 불규칙 활용
    
    
    # 1.4. ㅅ 불규칙 활용
    
    
    # 1.5. 우 불규칙 활용
    
    
    # 1.6. ㅡ 탈락 불규칙 활용
    
    
    ## 2. 어미가 바뀌는 불규칙 활용
    # 2.1. 거라 불규칙 활용
    
    
    # 2.2. 너라 불규칙 활용
    
    
    # 2.3. 러 불규칙 활용
    
    
    # 2.4. 여 불규칙 활용
    
    
    # 2.5. 오 불규칙 활용
    
    
    ## 3. 어간과 어미가 모두 바뀌는 불규칙 활용
    # 3.1. ㅎ 불규칙 활용
    
    
    # 4. 활용이 불완전한 동사
    
    
    ## 5. 헷갈리기 쉬운 불규칙 활용
    # 5.1. 이르다
    
    
    # 5.2. 붇다, 붓다, 불다
    
    
    ## 6. 체언    
    # 6.1. 복수형
    
    
    ## 7. 높임법
    # 7.1. 조사
    
    
    return (v, e)