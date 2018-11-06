from soynlp.hangle import decompose
from soynlp.lemmatizer import conjugate

def conjugate_as_present(stem):
    """기본형을 현재형으로 활용하여 말이 되면 동사, 아니면 형용사
    먹다 -> 먹는다, 파랗다 -> 파란다 (o)
    먹다 -> 먹는다고, 파랗다 -> 파란다고 (o) # -다고* 가 붙은 모든 어미
    먹다 -> 먹는, 파랗다 -> 파란 (x) 상태를 나타내는 '-는'은 혼동될 수 있음
    """

    eomis_0 = ['ㄴ다', 'ㄴ다고', '고있는']
    eomis_1 = ['는다', '는다고', '고있는']

    cho, jung, jong = decompose(stem[-1])
    if jong == ' ':
        return _conjugate(stem, eomis_0)
    else:
        return _conjugate(stem, eomis_1)

def conjugate_as_imperative(stem):
    """기본형을 명령형으로 활용하여 말이 되면 동사, 아니면 형용사
    먹다 -> 먹어라, 파랗다 -> 파래라 (o)
    먹다 -> 먹어, 파랗다 -> 파래 (x) 상태를 나타내는 '-어'는 혼동될 수 있음
    """

    eomis_0 = ['어라']
    eomis_1 = ['아라']

    cho, jung, jong = decompose(stem[-1])
    if jung == 'ㅓ' or jung == 'ㅕ':
        return _conjugate(stem, eomis_0)
    else:
        return _conjugate(stem, eomis_1)

def conjugate_as_pleasure(stem):
    """기본형을 청유형으로 활용하여 말이 되면 동사, 아니면 형용사
    먹다 -> 먹자, 파랗다 -> 파랗자 (o)
    먹다 -> 먹을까?, 파랗다 -> 파랄까? (x) 의문형과 혼동될 수 있음
    """

    eomis = ['자', 'ㄹ까', 'ㄹ까봐', '까', '까봐', '을까', '을까봐']
    return _conjugate(stem, eomis)

def _conjugate(stem, eomis):
    return {surface for eomi in eomis
            for surface in conjugate(stem, eomi)}

def rule_classify(stem):
    # 되/하는 동사/형용사가 모두 될 수 있기 때문에 이용하지 않음
    adj_suffixs = {'같','답','롭','만하','스럽','시럽','이','아니'}
    verb_suffixs = {'거리','당하','당허','시키'}

    lastone = stem[-1]
    lasttwo = stem[-2:]
    if (lastone in adj_suffixs) or (lasttwo in adj_suffixs):
        return 'Adjective'
    elif (lastone in verb_suffixs) or (lasttwo in verb_suffixs):
        return 'Verb'
    return None