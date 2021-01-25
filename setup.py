import os
from setuptools import setup, find_packages


with open(os.path.join(os.path.dirname(__file__), "README.md"), encoding="utf-8") as f:
    long_description = f.read()


def get_about():
    about = {}
    basedir = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(basedir, "soynlp", "about.py")) as f:
        exec(f.read(), about)
    return about


def requirements():
    requirements_path = os.path.join(os.path.dirname(__file__), "requirements.txt")
    with open(requirements_path, encoding="utf-8") as f:
        return f.read().splitlines()


about = get_about()
setup(
    name=about["__name__"],
    version=about["__version__"],
    author=about["__author__"],
    description=about["__description__"],
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lovit/soynlp",
    install_requires=requirements(),
    packages=find_packages(),
    entry_points={
        "console_scripts": ["soynlp=soynlp.cli:main"],
    },
    package_data={
        "soynlp": [
            "trained_models/*",
            "pos/dictionary/*.txt",
            "pos/dictionary/*/*.txt",
            "postagger/dictionary/default/*/*.txt",
            "noun/*.txt",
            "noun/pretrained_models/*",
            "lemmatizer/dictionary/default/*/*.txt",
        ]
    },
    keywords=[
        "korean-nlp",
        "korean-text-processing",
        "nlp",
        "tokenizer",
        "postagging",
        "word-extraction",
    ],
    classifiers=(
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ),
)
