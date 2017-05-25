from description import __version__, __author__
from setuptools import setup, find_packages

setup(
   name="soynlp",
   version=__version__,
   author=__author__,
   author_email='soy.lovit@gmail.com',
   url='https://github.com/lovit/soynlp',
   description="Unsupervised Korean Natural Language Processing Toolkits",
   long_description="""It contains unsupervised word extraction, tokenizers and noun extractors. 
   These algorithms are not depending training corpus but extract patterns from data by theirselves.

   Current version has follows
   - Word extraction
     - Cohesion score
     - Branching Entropy
     - Accessor Variety
   - Tokenizers
     - RegexTokenizer
     - LTokenizer
     - MaxScoreTokenizer
   - Noun extractor
     - LRNounExtractor
     

   Following packages are helpful
   - krwordrank: Unsupervised Korean word/keyword extractor
     - https://github.com/lovit/KR-WordRank
     - pip install krwordrank
   - soyspacing: Korean spacing error corrector
     - https://github.com/lovit/soyspacing
     - pip install soyspacing
   """,
   install_requires=["numpy>=1.12.1", "psutil>=5.0.1"],
   keywords = ['korean natural language processing'],
   packages=find_packages(),
   package_data={'soynlp':['trained_models/*']}
)