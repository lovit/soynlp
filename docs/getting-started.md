---
title: 설치·빠른 시작
description: Python 3.12와 uv로 refactor-2026을 설치하고 첫 명령을 실행합니다.
section: 시작하기
permalink: /getting-started/
---

# 설치·빠른 시작

Python **3.12 이상**이 필요합니다. 이 문서의 실행 예제와 개발 검증은 Python 3.12와 `uv`를 사용합니다.
`uv` 설치는 [공식 설치 안내](https://docs.astral.sh/uv/getting-started/installation/)를 참고하세요.

## 저장소 설치

{% assign source_ref = site.github.build_revision | default: site.docs_source_ref %}
문서에서 사용하는 코드를 받으려면 저장소를 복제하고 **이 문서의 기준 커밋**으로 이동합니다.
검토 중인 문서 브랜치에서도 새 기능을 그대로 실행할 수 있습니다.

```sh
git clone https://github.com/lovit/soynlp.git
cd soynlp
git checkout {{ source_ref }}
uv sync --python 3.12 --locked
```

위 `checkout`은 문서와 일치하는 커밋을 사용합니다. 이후 개발 브랜치의 업데이트를 받으려면
`git switch refactor-2026`으로 전환합니다.
기존 PyPI 배포본에는 이 문서의 새 CLI가 없을 수 있으므로 이 가이드는 저장소 설치를 기준으로 합니다.

## CLI로 한 문장 처리

아래 명령은 모두 저장소 루트에서 실행합니다.

```sh
uv run --python 3.12 soynlp --help
uv run --python 3.12 soynlp tokenize --text '한국어를 공부합니다'
uv run --python 3.12 soynlp normalize --method repeat --text '한국어ㅋㅋㅋㅋ'
uv run --python 3.12 soynlp conjugate --text '먹' --eomi '었다'
```

`soynlp`와 `python -m soynlp`는 같은 CLI 진입점입니다.

```sh
uv run --python 3.12 python -m soynlp tokenize --text '한국어를 공부합니다'
```

문장 결과는 기본 stdout에 JSONL로 출력합니다.
`--input`으로 TXT·JSONL 파일 또는 표준 입력(`-`)을 읽고 `--output`으로 파일에 저장할 수 있습니다.
기존 출력 파일을 교체하려면 `--overwrite`를 지정합니다.

## Python에서 호출

```python
from soynlp.noun import LRNounExtractor
from soynlp.tokenizer import RegexTokenizer

sentences = ["한국어를 공부합니다", "한국어는 재미있습니다"] * 20
nouns = LRNounExtractor(verbose=False).extract(sentences)
tokens = RegexTokenizer().tokenize(sentences[0])
print(nouns)
print(tokens)
```

`.py` 파일로 저장한 뒤 `uv run --python 3.12 python 파일명.py`로 실행합니다.
[Python 사용법](python.md)에서는 추출 결과로 토큰화하고 JSON 리소스를 저장하는 흐름을 설명합니다.

## 예제 파일로 실행

저장소에는 ID와 텍스트를 담은 작은 JSONL 코퍼스, YAML 설정과 Python 예제가 있습니다.

```sh
uv run --python 3.12 soynlp tokenize --input examples/data/demo.jsonl --input-format jsonl --offsets
uv run --python 3.12 soynlp pipeline validate -c examples/text_pipeline.yaml
uv run --python 3.12 soynlp pipeline run -c examples/text_pipeline.yaml
uv run --python 3.12 python examples/refactor_demo.py
```

작은 예제는 실행 흐름을 확인하기 위한 데이터입니다.
실제 명사·단어 추출은 충분한 코퍼스에서 빈도·점수 임계값을 조정해 사용하세요.

## 선택적 이모지 기능

이모지 제거에는 `emoji` 의존성이 필요합니다.

```sh
uv sync --python 3.12 --locked --extra emoji
uv run --python 3.12 soynlp normalize --method emoji --text '한국어 😀'
```

이미 Python 가상 환경을 사용하는 경우 저장소 루트에서 `uv pip install .`로 설치할 수 있습니다.
이모지 기능까지 설치하려면 `uv pip install '.[emoji]'`를 사용합니다.
