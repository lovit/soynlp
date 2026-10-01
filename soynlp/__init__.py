from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("soynlp")
except PackageNotFoundError:
    __version__ = "0.0.0.dev0"

__title__ = "soynlp"
__author__ = "Lovit"
__license__ = "LGPL"
__copyright__ = "Copyright 2017 Lovit"

from . import hangle, lemmatizer, normalizer, noun, pos, postagger, predicator, tokenizer, utils, vectorizer, word
from .artifacts import Artifact, load_artifact, load_matrix, save_artifact, save_matrix
from .utils import DoublespaceLineCorpus

__all__ = [
    "Artifact",
    "save_artifact",
    "load_artifact",
    "save_matrix",
    "load_matrix",
    "hangle",
    "lemmatizer",
    "normalizer",
    "noun",
    "pos",
    "postagger",
    "predicator",
    "tokenizer",
    "utils",
    "vectorizer",
    "word",
    "DoublespaceLineCorpus",
]
