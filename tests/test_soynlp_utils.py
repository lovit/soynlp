import pytest
from soynlp.utils import EojeolCounter, LRGraph


def test_lrgraph_construct():
    sents = ['이것은 예문이다', '이것도 예문이고', '저것은 예시이다']
    lrgraph = LRGraph(sents=sents)

    assert lrgraph._lr ==  {
        '이': {'것은': 1, '것도': 1},
        '이것': {'은': 2, '도': 2},
        '이것은': {'': 3},
        '예': {'문이다': 1, '문이고': 1, '시이다': 1},
        '예문': {'이다': 2, '이고': 2},
        '예문이': {'다': 3, '고': 3},
        '예문이다': {'': 4},
        '이것도': {'': 3},
        '예문이고': {'': 4},
        '저': {'것은': 1},
        '저것': {'은': 2},
        '저것은': {'': 3},
        '예시': {'이다': 2},
        '예시이': {'다': 3},
        '예시이다': {'': 4}
    }

    assert lrgraph._rl == {
        '것은': {'이': 1, '저': 1},
        '것도': {'이': 1},
        '은': {'이것': 2, '저것': 2},
        '도': {'이것': 2},
        '문이다': {'예': 1},
        '문이고': {'예': 1},
        '시이다': {'예': 1},
        '이다': {'예문': 2, '예시': 2},
        '이고': {'예문': 2},
        '다': {'예문이': 3, '예시이': 3},
        '고': {'예문이': 3}
    }

    lrgraph = LRGraph(sents=sents, l_max_length=3)

    assert lrgraph._lr == {
        '이': {'것은': 1, '것도': 1},
        '이것': {'은': 2, '도': 2},
        '이것은': {'': 3},
        '예': {'문이다': 1, '문이고': 1, '시이다': 1},
        '예문': {'이다': 2, '이고': 2},
        '예문이': {'다': 3, '고': 3},
        '이것도': {'': 3},
        '저': {'것은': 1},
        '저것': {'은': 2},
        '저것은': {'': 3},
        '예시': {'이다': 2},
        '예시이': {'다': 3}
    }

    assert lrgraph._rl == {
        '것은': {'이': 1, '저': 1},
        '것도': {'이': 1},
        '은': {'이것': 2, '저것': 2},
        '도': {'이것': 2},
        '문이다': {'예': 1},
        '문이고': {'예': 1},
        '시이다': {'예': 1},
        '이다': {'예문': 2, '예시': 2},
        '이고': {'예문': 2},
        '다': {'예문이': 3, '예시이': 3},
        '고': {'예문이': 3}
    }


def test_lrgraph_add_pair():
    lrgraph = LRGraph(l_max_length=3, r_max_length=3)
    lrgraph.add_lr_pair('abc', 'de')
    assert lrgraph._lr == {'abc': {'de': 1}}
    lrgraph.add_lr_pair('abcd', 'de')
    assert lrgraph._lr == {'abc': {'de': 1}}


def test_lrgraph_add_eojeol():
    lrgraph = LRGraph(l_max_length=3, r_max_length=3)
    lrgraph.add_eojeol('abcde')
    assert lrgraph._lr == {'ab': {'cde': 1}, 'abc': {'de': 1}}
    lrgraph.add_eojeol('abcde', count=3)
    assert lrgraph._lr == {'ab': {'cde': 4}, 'abc': {'de': 4}}

    lrgraph = LRGraph(l_max_length=3, r_max_length=4)
    lrgraph.add_eojeol('abcde')
    assert lrgraph._lr == {'a': {'bcde': 1}, 'ab': {'cde': 1}, 'abc': {'de': 1}}
    lrgraph.add_eojeol('abcde', count=3)
    assert lrgraph._lr == {'a': {'bcde': 4}, 'ab': {'cde': 4}, 'abc': {'de': 4}}


def test_lrgraph_discount_lr_pair():
    lrgraph = LRGraph(l_max_length=3, r_max_length=4)
    lrgraph.add_eojeol('abcde', count=4)
    assert lrgraph._lr == {'a': {'bcde': 4}, 'ab': {'cde': 4}, 'abc': {'de': 4}}
    assert lrgraph._rl == {'bcde': {'a': 4}, 'cde': {'ab': 4}, 'de': {'abc': 4}}
    lrgraph.discount_lr_pair('ab', 'cde', 3)
    assert lrgraph._lr == {'a': {'bcde': 4}, 'ab': {'cde': 1}, 'abc': {'de': 4}}
    assert lrgraph._rl == {'bcde': {'a': 4}, 'cde': {'ab': 1}, 'de': {'abc': 4}}


def test_lrgraph_get_r():
    lrgraph = LRGraph(l_max_length=3, r_max_length=4)
    lrgraph.add_eojeol('이것은', 1)
    lrgraph.add_eojeol('이것도', 2)
    lrgraph.add_eojeol('이것이', 3)
    lrgraph.add_eojeol('이것을', 4)
    assert lrgraph.get_r('이것', topk=3) == [('을', 4), ('이', 3), ('도', 2)]
    assert lrgraph.get_r('이것', topk=2) == [('을', 4), ('이', 3)]
    assert lrgraph.get_r('이것', topk=-1) == [('을', 4), ('이', 3), ('도', 2), ('은', 1)]


def test_lrgraph_get_l():
    lrgraph = LRGraph(l_max_length=3, r_max_length=4)
    lrgraph.add_eojeol('너의', 1)
    lrgraph.add_eojeol('나의', 2)
    lrgraph.add_eojeol('모두의', 3)
    lrgraph.add_eojeol('시작의', 4)
    assert lrgraph.get_l('의', topk=3) == [('시작', 4), ('모두', 3), ('나', 2)]
    assert lrgraph.get_l('의', topk=2) == [('시작', 4), ('모두', 3)]
    assert lrgraph.get_l('의', topk=-1) == [('시작', 4), ('모두', 3), ('나', 2), ('너', 1)]


def test_lrgraph_to_eojeol_counter():
    lrgraph = LRGraph(l_max_length=3, r_max_length=4)
    lrgraph.add_eojeol('너의', 1)
    lrgraph.add_eojeol('나의', 2)
    lrgraph.add_eojeol('모두의', 3)
    lrgraph.add_eojeol('시작의', 4)
    assert sorted(lrgraph.to_EojeolCounter().items(), key=lambda x:x[1]) == [('너의', 1), ('나의', 2), ('모두의', 3), ('시작의', 4)]


def test_eojeol_counter():
    sents = ['이것은 어절 입니다', '이것은 예문 입니다', '이것도 예문 이고요']
    eojeol_counter = EojeolCounter(sents=sents)
    assert sorted(eojeol_counter.items()) == [('어절', 1), ('예문', 2), ('이것도', 1), ('이것은', 2), ('이고요', 1), ('입니다', 2)]

    lrgraph = eojeol_counter.to_lrgraph()
    assert lrgraph.get_r('이것') == [('은', 2), ('도', 1)]

    eojeol_counter.remove_eojeols('이것은')
    assert sorted(eojeol_counter.items()) == [('어절', 1), ('예문', 2), ('이것도', 1), ('이고요', 1), ('입니다', 2)]

    eojeol_counter.remove_eojeols({'이것도', '입니다'})
    assert sorted(eojeol_counter.items()) == [('어절', 1), ('예문', 2), ('이고요', 1)]
