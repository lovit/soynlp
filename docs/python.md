---
title: Python 사용법
description: 명사·단어 추출, 토큰화와 리소스 저장·재사용의 기본 흐름을 설명합니다.
section: 사용 가이드
permalink: /python/
---

# Python 사용법

공개 모듈에서 기능을 import해 사용합니다.
아래 코드는 저장소를 설치한 환경에서 실행하며, 각 절의 예제는 독립적으로 실행할 수 있습니다.

## 명사 추출

현재 코퍼스 API는 문장을 반복해서 읽을 수 있는 입력을 받아 명사별 빈도·점수를 반환합니다.
작은 예제에서 학습 흐름을 확인하고 실제 데이터에 맞게 임계값을 조정하세요.

```python
from soynlp.noun import LRNounExtractor

sentences = ["한국어를 공부합니다", "한국어는 재미있습니다"] * 20
nouns = LRNounExtractor(verbose=False).extract(sentences)

for word, score in sorted(nouns.items()):
    print(word, score.frequency, score.score)
```

기존 v1의 `train() → extract()`와 v2의 `train_extract()`도 제공합니다.
같은 클래스 이름에 기존·현재 사용법이 있으므로 [호환성 안내](COMPATIBILITY.md)를 확인하세요.

## 단어 추출

master의 사용법은 먼저 코퍼스를 학습하고 점수를 추출합니다.
각 단어의 Scores에는 빈도·결합도·접근자 다양도·분기 엔트로피가 포함됩니다.

```python
from soynlp.word import WordExtractor

sentences = ["한국어를 공부합니다", "한국어는 재미있습니다"] * 20
extractor = WordExtractor(min_frequency=2, verbose_points=0)
extractor.train(sentences)
words = extractor.extract()

for word, score in sorted(words.items()):
    print(word, score.cohesion_forward)
```

현재 코퍼스 API인 `WordExtractor(verbose=False).extract(sentences, min_frequency=2)`는
cohesion/accessor_variety/branching_entropy 매핑을 반환합니다.
생성자와 반환값이 다르므로 두 경로의 옵션을 섞지 마세요.

## 추출한 명사로 토큰화

`LTokenizer`에 명사 점수를 전달하면 어절의 왼쪽 부분을 점수에 따라 분리합니다.

```python
from soynlp.noun import LRNounExtractor
from soynlp.tokenizer import LTokenizer

sentences = ["한국어를 공부합니다", "한국어는 재미있습니다"] * 20
nouns = LRNounExtractor(verbose=False).extract(sentences)
scores = {word: value.score for word, value in nouns.items()}
tokenizer = LTokenizer(scores=scores)

print(tokenizer.tokenize("한국어를 공부합니다"))
```

문자 위치를 포함한 Token 객체가 필요하면 `tokenize(..., return_words=False)`를 **키워드 인자**로 전달합니다.
기존 Boolean 위치 인자는 호환 API의 의미로 처리합니다.

## 명사 리소스 저장과 재사용

JSON Artifact는 종류·스키마 버전·파라미터·데이터를 함께 저장합니다.
아래 예제는 임시 디렉터리를 사용하므로 여러 번 실행할 수 있습니다.

```python
import tempfile
from pathlib import Path

from soynlp import Artifact, load_artifact, save_artifact
from soynlp.noun import LRNounExtractor
from soynlp.services import text_processor

sentences = ["한국어를 공부합니다", "한국어는 재미있습니다"] * 20
nouns = LRNounExtractor(verbose=False).extract(sentences)

with tempfile.TemporaryDirectory() as directory:
    path = Path(directory) / "nouns.json"
    save_artifact(Artifact("nouns", nouns), path)
    resource = load_artifact(path, expected_kind="nouns")
    tokenize = text_processor("tokenize", "l", resource=resource)
    print(tokenize("한국어를 공부합니다"))
```

실제 작업에서는 보관할 파일 경로를 지정합니다.
이미 존재하는 파일은 `overwrite=True`가 있어야 교체됩니다.
JSON을 읽으면 데이터는 JSON 객체·배열이며 namedtuple 인스턴스로 자동 복원되지 않습니다.
CLI도 같은 리소스를 `tokenize --method l --resource nouns.json`으로 읽습니다.

## 다음 단계

{% assign source_ref = site.github.build_revision | default: site.docs_source_ref %}
- [전체 Python 실행 예제]({{ site.repository_url }}/blob/{{ source_ref }}/examples/refactor_demo.py)는 bigram, 품사, 문맥 행렬, PMI·SVD와 행렬 저장까지 연결합니다.
- [CLI 안내](CLI.md)에서 전체 기능과 옵션, 저장 형식을 확인할 수 있습니다.
- [YAML 파이프라인](pipeline.md)으로 여러 단계를 설정 파일에 묶을 수 있습니다.
