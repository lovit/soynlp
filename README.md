# soynlp

soynlp는 코퍼스에서 학습하는 **비지도 한국어 자연어 처리** Python 패키지입니다.
부분 문자열의 빈도·결합도·접근자 다양도와 LR-Graph를 이용해 명사와 단어를 추출하고,
정규화, 토큰화, 표제어 분석, 활용, 품사 분석과 벡터화를 제공합니다.

`refactor-2026`은 기존 Python 사용법을 유지하면서 기능별 CLI, YAML 파이프라인,
재사용 가능한 리소스 저장 형식과 개발 환경을 정비하는 브랜치입니다.
이 브랜치의 새 CLI를 이용하려면 이 저장소를 설치해야 합니다.

## 시작하기

Python **3.12 이상**이 필요합니다. 개발과 검증은 Python 3.12와 `uv`를 사용합니다.

```sh
uv sync --python 3.12 --locked
uv run --python 3.12 soynlp --help
uv run --python 3.12 soynlp tokenize --text '한국어를 공부합니다'
uv run --python 3.12 soynlp conjugate --text '먹' --eomi '었다'
uv run --python 3.12 soynlp pipeline run -c examples/text_pipeline.yaml
uv run --python 3.12 python examples/refactor_demo.py
```

일반 Python 환경에서는 저장소 디렉터리에서 `uv pip install .`로 설치할 수 있습니다.
이모지 제거 기능에는 `uv pip install '.[emoji]'` 또는 `uv sync --extra emoji`가 필요합니다.

## Python API

```python
from soynlp.noun import LRNounExtractor
from soynlp.tokenizer import RegexTokenizer
from soynlp.word import WordExtractor

sentences = ["한국어를 공부합니다", "한국어는 재미있습니다"] * 20
nouns = LRNounExtractor(verbose=False).extract(sentences)
tokens = RegexTokenizer().tokenize(sentences[0])

# master의 train → extract 사용법과 알고리즘
words = WordExtractor(min_frequency=2, verbose_points=0)
words.train(sentences)
scores = words.extract()
```

| 기능 | 공개 모듈 |
| --- | --- |
| 한글·자모·문자열 거리 | `soynlp.hangle` |
| 정규화 | `soynlp.normalizer` |
| 명사 추출 | `soynlp.noun` |
| 단어·bigram 추출 | `soynlp.word` |
| 토큰화·어절 패턴 학습 | `soynlp.tokenizer` |
| 표제어 분석·활용 | `soynlp.lemmatizer` |
| 용언·어미·어간 추출 | `soynlp.predicator` |
| 뉴스·채팅 품사 추출 및 태깅 | `soynlp.pos`, `soynlp.postagger` |
| 어절 집계·LR-Graph | `soynlp.utils`, `soynlp.core` |
| 단어 빈도·문맥 행렬 | `soynlp.vectorizer` |
| 버전 있는 JSON 리소스·NPZ | `soynlp.artifacts` |
| 기능별 공통 연산 | `soynlp.services` |
| YAML 실행·검증 | `soynlp.pipeline` |

BoW는 **단어 빈도(TF)** 행렬입니다. TF-IDF, NER와 새로운 MaxScoreTagger 알고리즘은 이번 범위에 포함하지 않습니다.

## 안내와 검증

- [한글 사용자 가이드 사이트](https://lovit.github.io/soynlp/)
- [설치·빠른 시작](docs/getting-started.md), [Python 사용법](docs/python.md), [YAML 파이프라인](docs/pipeline.md)
- [전체 CLI와 저장 형식](docs/CLI.md)
- [Python 호환성과 API 차이](docs/COMPATIBILITY.md)
- [실행 가능한 Python 예제](examples/refactor_demo.py)
- [구현 결과·기능별 커밋과 머지](REFACTOR_DELIVERY.md)
- [작업 기록](REFACTOR_PROGRESS.md), [기존 진단](REFACTOR_AUDIT.md)
- [Codex 작업 지침](.codex/WORKFLOW.md)

```sh
uv run --python 3.12 pytest
uv run --python 3.12 ruff check .
uv run --python 3.12 ruff format --check .
uv run --python 3.12 pyright
uv build
uv run --python 3.12 python scripts/verify_distribution.py
```
