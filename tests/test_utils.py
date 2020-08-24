import os
import sys
soynlp_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.insert(0, soynlp_path)
from soynlp.utils import EojeolCounter, LRGraph


def test_lrgraph_construct():
    sents = ['이것은 예문이다', '이것도 예문이고', '저것은 예문이다']
    lrgraph = LRGraph(sents=sents)

    assert lrgraph._lr == {
         '예': {'문이고': 1, '문이다': 2},
         '예문': {'이고': 1, '이다': 2},
         '예문이': {'고': 1, '다': 2},
         '예문이고': {'': 1},
         '예문이다': {'': 2},
         '이': {'것도': 1, '것은': 1},
         '이것': {'도': 1, '은': 1},
         '이것도': {'': 1},
         '이것은': {'': 1},
         '저': {'것은': 1},
         '저것': {'은': 1},
         '저것은': {'': 1}
    }

    assert lrgraph._rl == {
         '것도': {'이': 1},
         '것은': {'이': 1, '저': 1},
         '고': {'예문이': 1},
         '다': {'예문이': 2},
         '도': {'이것': 1},
         '문이고': {'예': 1},
         '문이다': {'예': 2},
         '은': {'이것': 1, '저것': 1},
         '이고': {'예문': 1},
         '이다': {'예문': 2}
    }

    lrgraph = LRGraph(sents=sents, max_l_length=3)

    assert lrgraph._lr == {
         '예': {'문이고': 1, '문이다': 2},
         '예문': {'이고': 1, '이다': 2},
         '예문이': {'고': 1, '다': 2},
         '이': {'것도': 1, '것은': 1},
         '이것': {'도': 1, '은': 1},
         '이것도': {'': 1},
         '이것은': {'': 1},
         '저': {'것은': 1},
         '저것': {'은': 1},
         '저것은': {'': 1}
    }

    assert lrgraph._rl == {
         '것도': {'이': 1},
         '것은': {'이': 1, '저': 1},
         '고': {'예문이': 1},
         '다': {'예문이': 2},
         '도': {'이것': 1},
         '문이고': {'예': 1},
         '문이다': {'예': 2},
         '은': {'이것': 1, '저것': 1},
         '이고': {'예문': 1},
         '이다': {'예문': 2}
    }


def test_lrgraph_add_pair():
    lrgraph = LRGraph(max_l_length=3, max_r_length=3)
    lrgraph.add_lr_pair('abc', 'de')
    assert lrgraph._lr == {'abc': {'de': 1}}
    lrgraph.add_lr_pair('abcd', 'de')
    assert lrgraph._lr == {'abc': {'de': 1}}


def test_lrgraph_add_eojeol():
    lrgraph = LRGraph(max_l_length=3, max_r_length=3)
    lrgraph.add_eojeol('abcde')
    assert lrgraph._lr == {'ab': {'cde': 1}, 'abc': {'de': 1}}
    lrgraph.add_eojeol('abcde', count=3)
    assert lrgraph._lr == {'ab': {'cde': 4}, 'abc': {'de': 4}}

    lrgraph = LRGraph(max_l_length=3, max_r_length=4)
    lrgraph.add_eojeol('abcde')
    assert lrgraph._lr == {'a': {'bcde': 1}, 'ab': {'cde': 1}, 'abc': {'de': 1}}
    lrgraph.add_eojeol('abcde', count=3)
    assert lrgraph._lr == {'a': {'bcde': 4}, 'ab': {'cde': 4}, 'abc': {'de': 4}}


def test_lrgraph_discount_lr_pair():
    lrgraph = LRGraph(max_l_length=3, max_r_length=4)
    lrgraph.add_eojeol('abcde', count=4)
    assert lrgraph._lr == {'a': {'bcde': 4}, 'ab': {'cde': 4}, 'abc': {'de': 4}}
    assert lrgraph._rl == {'bcde': {'a': 4}, 'cde': {'ab': 4}, 'de': {'abc': 4}}
    lrgraph.discount_lr_pair('ab', 'cde', 3)
    assert lrgraph._lr == {'a': {'bcde': 4}, 'ab': {'cde': 1}, 'abc': {'de': 4}}
    assert lrgraph._rl == {'bcde': {'a': 4}, 'cde': {'ab': 1}, 'de': {'abc': 4}}


def test_lrgraph_get_r():
    lrgraph = LRGraph(max_l_length=3, max_r_length=4)
    lrgraph.add_eojeol('이것은', 1)
    lrgraph.add_eojeol('이것도', 2)
    lrgraph.add_eojeol('이것이', 3)
    lrgraph.add_eojeol('이것을', 4)
    assert lrgraph.get_r('이것', topk=3) == [('을', 4), ('이', 3), ('도', 2)]
    assert lrgraph.get_r('이것', topk=2) == [('을', 4), ('이', 3)]
    assert lrgraph.get_r('이것', topk=-1) == [('을', 4), ('이', 3), ('도', 2), ('은', 1)]


def test_lrgraph_get_l():
    lrgraph = LRGraph(max_l_length=3, max_r_length=4)
    lrgraph.add_eojeol('너의', 1)
    lrgraph.add_eojeol('나의', 2)
    lrgraph.add_eojeol('모두의', 3)
    lrgraph.add_eojeol('시작의', 4)
    assert lrgraph.get_l('의', topk=3) == [('시작', 4), ('모두', 3), ('나', 2)]
    assert lrgraph.get_l('의', topk=2) == [('시작', 4), ('모두', 3)]
    assert lrgraph.get_l('의', topk=-1) == [('시작', 4), ('모두', 3), ('나', 2), ('너', 1)]


def test_lrgraph_to_eojeol_counter():
    lrgraph = LRGraph(max_l_length=3, max_r_length=4)
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
