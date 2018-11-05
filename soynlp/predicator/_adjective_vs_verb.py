def conjugate_as_present(stem):
    """기본형을 현재형으로 활용하여 말이 되면 동사, 아니면 형용사
    먹다 -> 먹는다, 파랗다 -> 파란다 (o)
    먹다 -> 먹는다고, 파랗다 -> 파란다고 (o) # -다고* 가 붙은 모든 어미
    먹다 -> 먹는, 파랗다 -> 파란 (x) 상태를 나타내는 '-는'은 혼동될 수 있음
    """
    raise NotImplemented

def conjugate_as_imperative(stem):
    """기본형을 명령형으로 활용하여 말이 되면 동사, 아니면 형용사
    먹다 -> 먹어라, 파랗다 -> 파래라 (o)
    먹다 -> 먹어, 파랗다 -> 파래 (x) 상태를 나타내는 '-어'는 혼동될 수 있음
    """
    raise NotImplemented

def conjugate_as_pleasure(stem):
    """기본형을 청유형으로 활용하여 말이 되면 동사, 아니면 형용사
    먹다 -> 먹자, 파랗다 -> 파랗자 (o)
    먹다 -> 먹을까?, 파랗다 -> 파랄까? (x) 의문형과 혼동될 수 있음
    """
    raise NotImplemented