---
title: CLI 안내
description: 전체 명령, 문장·코퍼스 입출력과 리소스 재사용 방법을 안내합니다.
section: 사용 가이드
permalink: /cli/
---

# 기능별 CLI

`soynlp`와 `python -m soynlp`는 같은 진입점입니다.
아래 명령은 `uv run --python 3.12 soynlp` 뒤에 붙입니다.
`--help`는 각 명령과 하위 명령에서 사용할 수 있습니다.

## 명령 체계

| 명령 | 하위 명령 또는 `--method` | 입력과 결과 |
| --- | --- | --- |
| `normalize` | text(기본), jamo, hangle-emoji, emoji, repeat, spaces, characters, padding | 문장 → 정규화 문자열 |
| `tokenize` | regex(기본), l, max-score, noun-match, max-lr-score, noun-l-match, legacy-normalize | 문장 → 토큰; legacy-normalize는 기존 정규화 문자열 |
| `tag` | korean(기본), lr-max-score | 문장 → 품사 결과 |
| `lemmatize` | filtered(기본), candidates, chat | 어절 → 어간·어미 후보 |
| `conjugate` | normal(기본), chat, present, imperative, pleasure | 어간 → 활용형 |
| `hangle` | decompose(기본), compose, jamo, classify, distance, encode, decode | 문자·문장 → 자모·분류·거리·인코딩 |
| `extract nouns` | lr(기본), lr-v1, lr-v2, news | 코퍼스 → 명사 빈도·점수 |
| `extract words` | legacy(기본), corpus | 코퍼스 → 단어 점수 |
| `extract bigrams` | frequency(기본), pmi, mikolov | 코퍼스 → bigram 빈도·점수 |
| `extract pos` | news(기본), chat, combined | 코퍼스 → 품사별 추출 결과 |
| `extract predicators` | default | 코퍼스 → 형용사·동사·표제어 |
| `extract eomis`, `extract stems`, `extract features` | default | 코퍼스 + seed 사전·명사 리소스 → 추출 결과 |
| `corpus` | convert, count-eojeols | 문서 변환 또는 어절 빈도 |
| `train` | patterns | 어절 패턴·LR 통계, 선택적으로 HITS 점수 |
| `dictionary` | build, merge, inspect, export | POS 사전 생성·결합·통계·TSV 내보내기 |
| `vectorize` | bow, contexts | 코퍼스 → NPZ 행렬·어휘 |
| `matrix` | pmi, svd, similar | 저장된 행렬 변환·행 유사도 |
| `lrgraph` | build, neighbors | 코퍼스 → 그래프 또는 이웃 빈도 |
| `pipeline` | run, validate, tasks | YAML 실행·정적 검증·태스크 목록 |

예시 단위 변환과 코퍼스 집계는 같은 입력 옵션을 사용합니다.
집계 명령은 모든 예시의 통계를 결합해 하나의 리소스를 만듭니다.
반복 읽기가 필요한 입력은 임시 파일에 보관합니다. 병렬 알고리즘은 작업 분배를 위해 문장 목록을 만들 수 있습니다.

## 공통 입출력

- `--text '문장'` 또는 `--input corpus.txt` 중 하나를 지정합니다. `--input -`는 표준 입력입니다.
- TXT는 한 줄에 한 예시이며 빈 줄도 예시입니다. JSONL은 한 줄에 한 객체이며 `--input-format jsonl`을 지정합니다.
- JSONL의 텍스트 필드는 기본 `text`이며 `--text-key body`처럼 바꿀 수 있습니다. 텍스트 필드는 문자열이어야 합니다.
- 문장 결과는 JSONL로 출력하며 ID와 다른 메타데이터를 유지합니다. 기본 결과 키는 `normalized`, `tokens`, `tags`, `lemmas`, `forms`입니다. `hangle`은 `result`입니다.
- `--result-key`로 결과 키를 바꿀 수 있습니다. `--result-key text`는 원문 필드를 교체합니다.
- 결과는 기본 stdout입니다. `-o/--output`으로 파일에 저장합니다. 기존 파일은 `--overwrite`가 있어야 교체합니다.
- 행렬은 `-o result.npz`가 필요합니다. 로그는 stderr이며 `--verbose`는 최상위 명령 앞에 지정합니다.
- 종료 코드는 성공 `0`, 입출력·실행 실패 `1`, 인자·설정 검증 실패 `2`입니다.

```sh
uv run --python 3.12 soynlp normalize --method repeat --text '한국어ㅋㅋㅋㅋ'
uv run --python 3.12 soynlp tokenize --input examples/data/demo.jsonl --input-format jsonl --offsets
uv run --python 3.12 soynlp lemmatize --text '먹었다'
uv run --python 3.12 soynlp hangle --method compose --text 'ㄱ ㅏ ㄴ'
uv run --python 3.12 soynlp hangle --method distance --text '가나다' --other '가너다'
```

`--parameters`는 알고리즘 옵션의 JSON 객체입니다.
정규화·토큰화·태깅에서는 생성자 옵션을 사용합니다.
예: `normalize --method repeat --parameters '{"max_repeat":3}'`.
표제어 후보·활용은 해당 함수의 옵션입니다.

`tokenize --offsets`는 regex/l/max-score/noun-match에서 문자 위치를 출력합니다.
max-lr-score는 기존 `(word, L/R)` 형식이며 위치 출력은 지원하지 않습니다.
`hangle encode`는 `ConvolutionHangleEncoder.sent_to_onehot`의 문자별 인덱스 배열을 반환합니다.
`decode`에는 이 배열을 JSON 문자열로 입력합니다.
`conjugate normal/chat`은 `--eomi`가 필요합니다.
filtered 표제어 분석은 내장 어간·어미 사전을 사용하며 `--dictionary`로 교체할 수 있습니다.

## 추출과 리소스 재사용

추출 옵션은 `--parameters` 안의 `constructor`, `train`, `extract` 객체로 구분합니다.
`lr` 명사와 `corpus` 단어, bigram은 별도 `train` 단계 없이 `extract` 옵션을 사용합니다.

```sh
uv run --python 3.12 soynlp extract nouns --input examples/data/demo.jsonl --input-format jsonl -o nouns.json
uv run --python 3.12 soynlp tokenize --method l --resource nouns.json --text '한국어를 공부합니다'
uv run --python 3.12 soynlp dictionary build --resource nouns.json --tag Noun -o dictionary.json
uv run --python 3.12 soynlp tag --dictionary dictionary.json --text '한국어를 공부합니다'
uv run --python 3.12 soynlp extract words --input examples/data/demo.jsonl --input-format jsonl --parameters '{"constructor":{"min_frequency":1,"verbose_points":0}}' -o words.json
uv run --python 3.12 soynlp tokenize --method l --resource words.json --text '한국어를 공부합니다'
```

단어 리소스의 `score`는 기본 토큰화 가중치입니다. `--score-field cohesion_forward` 등으로 변경할 수 있습니다.
`corpus` 단어 결과에서는 `cohesion.leftside`처럼 중첩 필드도 선택할 수 있습니다.
해당 점수가 없는 문맥 통계 전용 항목은 토큰화 가중치에서 제외합니다.

어미·어간 추출에는 `--dictionary`가 필요합니다. 사전은 POS별 단어 목록이며
어미 추출에는 Stem/Verb/Adjective, 어간 추출에는 이 항목과 Eomi seed가 필요합니다.
`extract features`에는 명사 빈도·점수가 있는 `--resource`가 필요합니다.
용언 추출은 기본 어간·어미 사전을 사용합니다. `--resource` 명사와 `--dictionary` seed를 전달할 수 있습니다.
`PredicatorExtractor`의 Python 반환 순서는 **형용사, 동사**이며 JSON 결과는 이름으로 구분합니다.

```sh
uv run --python 3.12 soynlp train patterns --input examples/data/demo.jsonl --input-format jsonl --parameters '{"constructor":{"min_frequency":1},"hits":{"max_iter":2}}' -o patterns.json
uv run --python 3.12 soynlp tokenize --method max-lr-score --resource patterns.json --text '한국어를 공부합니다'
uv run --python 3.12 soynlp lrgraph build --input examples/data/demo.jsonl --input-format jsonl -o lrgraph.json
uv run --python 3.12 soynlp lrgraph neighbors --resource lrgraph.json --word '한국어' --direction right
```

patterns 옵션은 `constructor`, `train`, `hits`입니다. HITS 옵션을 생략하면 그래프와 어휘만 저장합니다.
패턴 리소스는 max-lr-score 토큰화와 lr-max-score 태깅에서 재사용할 수 있습니다.
`train tagger`는 제공하지 않습니다. `KoreanPOSTagger.train(sentences=...)`는 코퍼스 통계를 학습하지 않고 추가 사전을 등록하는 기존 API입니다.

## 벡터화와 행렬

```sh
uv run --python 3.12 soynlp vectorize contexts --input examples/data/demo.jsonl --input-format jsonl --parameters '{"min_tf":1}' -o contexts.npz
uv run --python 3.12 soynlp vectorize bow --input examples/data/demo.jsonl --input-format jsonl --vocabulary contexts.npz -o bow.npz
uv run --python 3.12 soynlp matrix pmi --input contexts.npz -o pmi.npz
uv run --python 3.12 soynlp matrix svd --input pmi.npz --parameters '{"n_components":2}' -o embeddings.npz
uv run --python 3.12 soynlp matrix similar --input embeddings.npz --query '한국어를' --topk 3
```

BoW는 TF 행렬이며 TF-IDF가 아닙니다. 기본 토큰화는 공백 분리입니다.
`--tokenizer-method`와 `--tokenizer-resource`로 토큰화를 바꿀 수 있습니다.
`--vocabulary`는 vocabulary JSON 리소스 또는 기존 NPZ 메타데이터를 읽습니다.
문맥 행렬도 같은 어휘 순서를 재사용할 수 있습니다.

SVD는 행 임베딩 `embeddings.npz`와 열 임베딩 `embeddings-columns.npz`를 저장합니다.
각 메타데이터의 parameters에 특잇값을 보관합니다. 기본 seed는 0입니다.
PMI 입력은 음수가 아닌 빈도이며, SVD 차원은 열 수보다 작아야 합니다.
문맥 행의 이름은 어휘이고 BoW 행 이름은 입력 순서의 `document:0`, `document:1` 등입니다.

## 저장 형식

JSON 리소스는 `schema_version`, `kind`, `parameters`, `data`를 포함합니다.
현재 버전은 `1`이며 지원하지 않는 버전과 잘못된 kind는 거부합니다.
빈도와 점수를 이름 있는 필드로 보관합니다. Bigram은 tuple 키를 잃지 않도록 레코드 배열로 저장합니다.

NPZ는 같은 이름의 `.json` 메타데이터와 함께 보관합니다.
메타데이터에는 shape, sparse/dense 형식, 행·열 어휘와 SHA-256이 들어갑니다.
파일과 메타데이터가 맞지 않으면 읽기를 거부합니다.
각 파일은 임시 파일을 거쳐 게시하지만 두 파일의 교체 전체가 하나의 파일시스템 트랜잭션은 아닙니다.

`WriteJson`은 문서 객체의 배열을 JSONL로 저장합니다.
명사·단어 점수 같은 매핑은 `WriteArtifact`를 사용합니다. `WriteJson`에 매핑을 전달하면 오류가 발생합니다.
새 CLI 리소스는 pickle을 사용하지 않습니다. 기존 WordExtractor 모델만 기존 save/load 경로로 호환합니다.

## 파이프라인

```sh
uv run --python 3.12 soynlp pipeline tasks
uv run --python 3.12 soynlp pipeline validate -c examples/text_pipeline.yaml
uv run --python 3.12 soynlp pipeline run -c examples/text_pipeline.yaml
```

기존 `pipeline -c config.yaml`과 `pipeline --config_file config.yaml`도 유지합니다.
상대 경로는 실행 디렉터리를 기준으로 합니다.

새 태스크는 ReadDocuments/WriteDocuments, ReadArtifact/WriteArtifact,
TextNormalize/TextTokenize/Tag/Lemmatize/Conjugate/Hangle, Extract/CountEojeols/TrainPatterns,
BuildDictionary/MergeDictionary/InspectDictionary/ExportDictionary,
BuildLRGraph/LRGraphNeighbors, Vectorize/PMI/SVD/Similar, ReadMatrix/WriteMatrix입니다.
Extract의 kind와 method는 CLI와 같습니다. TextTokenize의 `resource_key`는 이전 태스크가 만든 Artifact를 참조합니다.
기존 Normalize/Tokenize/ExtractNoun/ExtractWord 등도 유지합니다.

`validate`는 태스크 이름·인자 스키마·입력 키 연결과 지원 방법을 확인합니다.
태스크를 실행하거나 리소스 파일을 열지 않으므로 실제 데이터·파일·행렬의 검사는 실행 시 수행합니다.
Python에서는 `Pipeline.validate(config, initial_keys={...})`와 `Pipeline()(config, parameters={...})`로 초기 입력을 전달할 수 있습니다.
