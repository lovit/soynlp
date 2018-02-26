# -*- encoding:utf8 -*-
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

sys.path.append('../')
import soynlp

from soynlp.pos.tagset import tagset
import pprint
class MyPrettyPrinter(pprint.PrettyPrinter):
	def format(self, _object, context, maxlevels, level):
		if isinstance(_object, unicode):
			return "'%s'" % _object.encode('utf8'), True, False
		elif isinstance(_object, str):
			_object = unicode(_object,'utf8')
			return "'%s'" % _object.encode('utf8'), True, False
		return pprint.PrettyPrinter.format(self, _object, context, maxlevels, level)

pp = MyPrettyPrinter()
pp.pprint(tagset)

from soynlp.pos import Dictionary
from soynlp.pos import LRTemplateMatcher
from soynlp.pos import LREvaluator
from soynlp.pos import SimpleTagger
from soynlp.pos import UnknowLRPostprocessor

pos_dict = {
    'Adverb': {'너무', '매우'},
    'Noun': {'너무너무너무', '아이오아이', '아이', '노래', '오', '이', '고양'},
    'Josa': {'는', '의', '이다', '입니다', '이', '이는', '를', '라', '라는'},
    'Verb': {'하는', '하다', '하고'},
    'Adjective': {'예쁜', '예쁘다'},
    'Exclamation': {'우와'}
}

dictionary = Dictionary(pos_dict)
pp.pprint(dictionary.pos_dict)
pp.pprint(dictionary.get_pos('아이오아이'))
pp.pprint(dictionary.get_pos('이'))
pp.pprint(dictionary.word_is_tag('아이오아이', 'Noun'))
pp.pprint(dictionary.word_is_tag('아이오아이', '명사'))
# sent = u'너무너무너무는아이오아이의노래입니다!!'
#
# pos_dict = {
#     'Adverb': {u'너무', u'매우'},
#     'Noun': {u'너무너무너무', u'아이오아이', u'아이', u'노래', u'오', u'이', u'고양'},
#     'Josa': {u'는', u'의', u'이다', u'입니다', u'이', u'이는', u'를', u'라', u'라는'},
#     'Verb': {u'하는', u'하다', u'하고'},
#     'Adjective': {u'예쁜', u'예쁘다'},
#     'Exclamation': {u'우와'}
# }
#
# dictionary = Dictionary(pos_dict)
# generator = LRTemplateMatcher(dictionary)
# pp.pprint(dict(zip(range(len(generator.generate(sent))), generator.generate(sent))))

sent = '너무너무너무는아이오아이의노래입니다!!'

pos_dict = {
    'Adverb': {'너무', '매우'},
    'Noun': {'너무너무너무', '아이오아이', '아이', '노래', '오', '이', '고양'},
    'Josa': {'는', '의', '이다', '입니다', '이', '이는', '를', '라', '라는'},
    'Verb': {'하는', '하다', '하고'},
    'Adjective': {'예쁜', '예쁘다'},
    'Exclamation': {'우와'}
}



dictionary = Dictionary(pos_dict)
generator = LRTemplateMatcher(dictionary)
pp.pprint(dict(zip(range(len(generator.generate(sent))), generator.generate(sent))))

evaluator = LREvaluator()
postprocessor = UnknowLRPostprocessor()

tagger = SimpleTagger(generator, evaluator, postprocessor)
pp.pprint(tagger.tag(sent))
pp.pprint(SimpleTagger(generator, evaluator).tag(sent))
tags, debugs = tagger.tag(sent, debug=True)
pp.pprint(tags)
pp.pprint(debugs)
