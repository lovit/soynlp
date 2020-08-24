from .about import __title__
from .about import __version__
from .about import __author__
from .about import __license__
from .about import __copyright__

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
