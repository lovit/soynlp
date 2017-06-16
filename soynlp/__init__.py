__title__ = 'soynlp'
__version__ = '0.0.21'
__author__ = 'Lovit'
__license__ = 'GPL v3'
__copyright__ = 'Copyright 2017 Lovit'

from . import word
from . import tokenizer
from . import noun
from . import hangle
from .utils import get_available_memory
from .utils import get_process_memory
from .utils import DoublespaceLineCorpus