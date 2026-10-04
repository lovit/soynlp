---
title: 호환성 안내
description: master의 공개 사용법과 현재 코퍼스 API, 반환 형식·모델 파일의 차이를 설명합니다.
section: 기존 사용자
permalink: /compatibility/
---

# Python API 호환성

호환성 기준은 `master`의 공개 모듈 import, 학습·추출 사용법, tuple 결과와 기존 모델 파일입니다.
실제 master `503eaee`에서 만든 결과와 pickle 모델을 `tests/unit/fixtures/compat`에 보관하고 비교합니다.

## 유지되는 사용법

- `soynlp` 루트의 모듈 속성, 메타데이터와 `DoublespaceLineCorpus`.
- noun의 LRNounExtractor, LRNounExtractor_v2, NewsNounExtractor, NounScore와 조사 특징 추출.
- word의 WordExtractor, Bigram과 기존 WordExtractor pickle save/load.
- tokenizer의 Regex/L/MaxScore/NounMatch와 MaxLRScoreTokenizer, NounLMatchTokenizer, normalize.
- utils의 DoublespaceLineCorpus, sort_by_alphabet, EojeolCounter와 기존 LRGraph 생성·변경·저장 사용법.
- NounScore와 Predicator의 tuple 해체·인덱싱·`_asdict()` 동작. EomiScore도 기존 namedtuple입니다.
- 이전 CLI pipeline 옵션과 기존 YAML 태스크.

단어·명사 추출은 과거와 현재 알고리즘이 다르므로 기존 구현을 `soynlp/_legacy`에 유지합니다.
공개 모듈에서 어댑터를 통해 호출하며, 사용자 코드는 `_legacy`를 직접 import할 필요가 없습니다.
리소스 경로와 Python 3.12 실행 오류를 수정했고 진행 상태 출력은 logging으로 연결했습니다.

```python
from soynlp.noun import LRNounExtractor, LRNounExtractor_v2
from soynlp.word import WordExtractor

sentences = ["한국어를 공부합니다", "한국어는 재미있습니다"] * 20

# 기존 v1 train/extract와 3필드 NounScore_v1
noun_v1 = LRNounExtractor(verbose=False)
noun_v1.train(sentences, min_noun_frequency=2)
old_nouns = noun_v1.extract(min_noun_frequency=2)

# 기존 v2 알고리즘과 공개 NounScore(빈도, 점수)
old_v2 = LRNounExtractor_v2(verbose=False).train_extract(sentences)

words = WordExtractor(min_frequency=2, verbose_points=0)
words.train(sentences)
old_words = words.extract()
```

## 코퍼스 입력 API

```python
from soynlp.noun import LRNounExtractor
from soynlp.word import WordExtractor

sentences = ["한국어를 공부합니다", "한국어는 재미있습니다"] * 20
nouns = LRNounExtractor(verbose=False).extract(sentences)
result = WordExtractor(verbose=False).extract(sentences, min_frequency=2)
cohesion = result["cohesion"]
```

`LRNounExtractor.extract(corpus)`는 현재 알고리즘입니다.
`train()` 또는 기존 생성자 인자를 사용하면 기존 v1 모드로 전환합니다.
새 생성자 길이 이름은 `max_l_length`, `max_r_length`이며 기존 이름은 `max_left_length`, `max_right_length`입니다.
기존 생성자 위치 인자도 지원합니다.

`WordExtractor.train() → extract()`는 기존 Scores 반환값입니다.
`WordExtractor.extract(corpus)`는 cohesion/accessor_variety/branching_entropy 매핑을 반환하는 코퍼스 API입니다.
기존 생성자 옵션으로 만든 객체에는 기존 알고리즘이 적용되므로 두 경로의 옵션을 섞지 마세요.
CLI에서는 `extract words --method legacy/corpus`로 이 차이를 명시합니다.

토큰화의 기존 `flatten`·`compose_compound`·`debug` 옵션과 Boolean 위치 인자는 기존 반환 형태입니다.
새 위치 정보를 얻으려면 **키워드** `return_words=False`를 사용합니다.
기본 RegexTokenizer는 버전 숫자와 실제 문자 위치를 처리하는 현재 구현입니다.
기존 Regex 분할이 필요하면 `flatten=True/False` 또는 기존 위치 인자를 지정합니다.

`soynlp.utils.LRGraph`는 기존 생성·인스턴스 load 사용법입니다.
현재 생성·클래스 load와 `from_sents`는 `soynlp.core.LRGraph`를 사용합니다.
EojeolCounter는 기존 coverage API와 길이 인자 별칭을 지원합니다.
공개 EojeolCounter는 기존 v2 명사 추출에도 전달할 수 있습니다.

`PredicatorExtractor.train_extract()/extract()`의 반환 순서는 **(adjectives, verbs)**입니다.
입력 nouns가 필요하며 `adjectives`, `verbs`, `eomis` 등 생성자 인자는 어휘 집합입니다.
어간은 `먹`, `예쁘`처럼 `다`를 제거한 형태로 전달합니다.

## 새 저장 API

```python
from soynlp import Artifact, load_artifact, save_artifact

resource = Artifact("nouns", {"한국어": {"frequency": 40, "score": 1.0}})
# 새 경로를 지정합니다. 이미 존재하면 명시적으로 overwrite=True를 사용합니다.
save_artifact(resource, "/tmp/my-soynlp-nouns.json")
loaded = load_artifact("/tmp/my-soynlp-nouns.json", expected_kind="nouns")
```

JSON을 읽으면 data는 JSON 객체·배열이며 원래 Python namedtuple 인스턴스로 자동 복원되지 않습니다.
토큰화·사전·태깅 서비스는 해당 데이터를 읽어 재사용합니다.
기존 pickle은 WordExtractor의 기존 모델에만 사용합니다.

## 검증 범위와 남은 결정

실제 코퍼스 실행, master 결과·모델 비교, 워커 수에 따른 명사 결과 일치,
Python/CLI/YAML 리소스 재사용, 빈 행렬과 어휘 순서, 설치된 배포 파일의 내장 사전을 검증합니다.
macOS에서 로컬 검증했으며 Linux·macOS CI를 설정했습니다.
패키지 회귀 CI의 원격 실행 결과는 구현 기록을 참고하세요.

리팩토링 기능 구현은 이슈·PR·원격 push 없이 기능 브랜치와 merge commit으로 진행했습니다.
이번 문서 사이트는 `docs/soynlp-doc` 브랜치를 push해 GitHub Pages에서 검토합니다.
사용자 검토 후 `refactor-2026`에 merge commit으로 병합하고 문서 소스를 복구합니다.
CLI와 JSON 형식은 이 브랜치의 구현 후보입니다. 사용자 검토 후 공개 릴리스 버전과 이식 방식을 결정할 수 있습니다.
Python 3.12 최소 버전 변경은 이 브랜치의 기존 요구 사항이며 master의 과거 Python 환경 전체를 지원하지 않습니다.
NER·새 MaxScoreTagger 알고리즘·TF-IDF는 이번 작업에 포함하지 않습니다.
