from soynlp.hangle import decompose
from soynlp.lemmatizer import conjugate


def conjugate_as_present(stem: str) -> set[str]:
    """기본형을 현재형으로 활용하여 말이 되면 동사, 아니면 형용사

    먹다 -> 먹는다, 파랗다 -> 파란다 (o)
    먹다 -> 먹는다고, 파랗다 -> 파란다고 (o)
    먹다 -> 먹는, 파랗다 -> 파란 (x) 상태를 나타내는 '-는'은 혼동될 수 있음
    """
    eomis_0 = ["ㄴ다", "ㄴ다고", "고있는"]
    eomis_1 = ["는다", "는다고", "고있는"]

    _, _, jong = decompose(stem[-1])  # type: ignore[misc]
    if jong == " ":
        return _conjugate(stem, eomis_0)
    else:
        return _conjugate(stem, eomis_1)


def conjugate_as_imperative(stem: str) -> set[str]:
    """기본형을 명령형으로 활용하여 말이 되면 동사, 아니면 형용사

    먹다 -> 먹어라, 파랗다 -> 파래라 (o)
    먹다 -> 먹어, 파랗다 -> 파래 (x) 상태를 나타내는 '-어'는 혼동될 수 있음
    """
    eomis_0 = ["어라"]
    eomis_1 = ["아라"]

    _, jung, _ = decompose(stem[-1])  # type: ignore[misc]
    if jung in ("ㅓ", "ㅕ"):
        return _conjugate(stem, eomis_0)
    else:
        return _conjugate(stem, eomis_1)


def conjugate_as_pleasure(stem: str) -> set[str]:
    """기본형을 청유형으로 활용하여 말이 되면 동사, 아니면 형용사

    먹다 -> 먹자, 파랗다 -> 파랗자 (o)
    먹다 -> 먹을까?, 파랗다 -> 파랄까? (x) 의문형과 혼동될 수 있음
    """
    eomis = ["자", "ㄹ까", "ㄹ까봐", "까", "까봐", "을까", "을까봐"]
    return _conjugate(stem, eomis)


def _conjugate(stem: str, eomis: list[str]) -> set[str]:
    return {surface for eomi in eomis for surface in conjugate(stem, eomi)}


def rule_classify(stem: str) -> str | None:
    """접미사 규칙으로 형용사/동사 분류. 되/하는 동사/형용사 모두 가능하므로 제외."""
    adj_suffixes = {"같", "답", "롭", "만하", "스럽", "시럽", "이", "아니"}
    verb_suffixes = {"거리", "당하", "당허", "시키"}

    last_one = stem[-1]
    last_two = stem[-2:]
    if (last_one in adj_suffixes) or (last_two in adj_suffixes):
        return "Adjective"
    elif (last_one in verb_suffixes) or (last_two in verb_suffixes):
        return "Verb"
    return None
