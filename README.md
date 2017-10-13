# soynlp

한국어 분석을 위한 pure python code 입니다. 


비지도학습 기반의 단어 추출 / 토크나이저 / 명사 추출기를 포함하고 있습니다. 


It contains unsupervised word extraction, tokenizers and noun extractors. These algorithms are not depending training corpus but extract patterns from data by theirselves.


현재 버전에서 제공하는 기능은 다음과 같습니다. 

- Word extraction
	- WordExtractor: It contains three word scoring methods; 
		1. Cohesion score
		1. Branching Entropy
		1. Accessor Variety
- Tokenizers
	- RegexTokenizer
	- LTokenizer
	- MaxScoreTokenizer	
- Noun extractor
	- LRNounExtractor 
	- NewsNounExtractor


아래 패키지은 soynlp와 독립적으로 작업하고 있는 (1) 키워드 추출기, (2) 띄어쓰기 오류 교정기 입니다. 

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

## notes

- [slide files](notes/unskonlp.pdf)에 알고리즘들의 원리 및 설명을 적어뒀습니다. 데이터야놀자에서 발표했던 자료입니다. 