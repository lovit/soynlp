from description import __version__, __author__
from setuptools import setup, find_packages

setup(
   name="soynlp",
   version=__version__,
   author=__author__,
   author_email='soy.lovit@gmail.com',
   url='https://github.com/lovit/soynlp',
   description="Unsupervised Korean Natural Language Processing Toolkits",
   long_description="""Python library for Korean Natural Language Processing. 
   
   It supports followings.
       1. Unsupervised word extractor
       2. Word score based tokenizer
       3. Noun extractor
       4. Dictionary based part of speech tagger
       5. Text normalizer

   Details are written at https://github.com/lovit/soynlp
   """,
   install_requires=["numpy>=1.12.1", "psutil>=5.0.1"],
   keywords = ['korean natural language processing'],
   packages=find_packages(),
   package_data={'soynlp':['trained_models/*', 'pos/dictionary/*.txt', 'pos/dictionary/*/*.txt']}
)