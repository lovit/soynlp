# Soy Korean Natural Language Processing Toolkits

- pure Python code
- author: Lovit (Hyunjoong Kim)

It contains unsupervised word extraction, tokenizers and noun extractors. These algorithms are not depending training corpus but extract patterns from data by theirselves.

Current version has follows

- Word extraction
	- WordExtractor: It contains three word scoring methods; 
		(1) Cohesion score
		(2) Branching Entropy
		(3) Accessor Variety
- Tokenizers
	- RegexTokenizer
	- LTokenizer
	- MaxScoreTokenizer	
- Noun extractor
	- LRNounExtractor 

Following packages are helpful

krwordrank: Unsupervised Korean word/keyword extractor
- https://github.com/lovit/KR-WordRank
- pip install krwordrank

soyspacing: Korean spacing error corrector
- https://github.com/lovit/soyspacing
- pip install soyspacing

## Setup

	pip install soynlp

## Requires

- numpy 
- psutil
