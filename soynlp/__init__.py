__title__ = 'soynlp'
__version__ = '0.0.493'
__author__ = 'Lovit'
__license__ = 'GPL v3'
__copyright__ = 'Copyright 2017 Lovit'

from . import hangle
from . import normalizer
from . import noun
from . import predicator
from . import postagger
from . import tokenizer
from . import vectorizer
from . import word
from . import utils

# for compatibility
from .utils import DoublespaceLineCorpus

__all__ = [
    # modules
    'hangle',
    'normalizer',
    'noun',
    'predicator',
    'pos',
    'tokenizer',
    'vectorizer',
    'word',
    'utils',
    # for compatibility with ver <= 0.0.45
    'DoublespaceLineCorpus'
]
