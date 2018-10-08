import soynlp
import setuptools
from setuptools import setup, find_packages


with open('README.md', encoding='utf-8') as fh:
    long_description = fh.read()

setuptools.setup(
    name="soynlp",
    version=soynlp.__version__,
    author=soynlp.__author__,
    author_email='soy.lovit@gmail.com',
    description="Unsupervised Korean Natural Language Processing Toolkits",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/lovit/soynlp',
    packages=setuptools.find_packages(),
    package_data={
        'soynlp':[
            'trained_models/*',
            'pos/dictionary/*.txt',
            'pos/dictionary/*/*.txt'
        ]
    },
    keywords = [
        'korean-nlp',
        'korean-text-processing',
        'nlp',
        'tokenizer',
        'postagging',
        'word-extraction'
    ],
    install_requires=[
        "numpy>=1.12.1",
        "psutil>=5.0.1"
    ],
    classifiers=(
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ),
)