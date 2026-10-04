---
title: YAML 파이프라인
description: YAML로 문서 읽기, 정규화, 토큰화와 결과 저장을 연결합니다.
section: 사용 가이드
permalink: /pipeline/
---

# YAML 파이프라인

여러 처리 단계를 YAML 파일의 `pipeline` 목록에 순서대로 작성합니다.
각 단계는 태스크 `name`과 인자 `args`로 구성하며,
이전 단계가 만든 데이터를 키로 참조합니다.

## 문장 처리 예제

{% assign source_ref = site.github.build_revision | default: site.docs_source_ref %}
저장소의 [예제 설정]({{ site.repository_url }}/blob/{{ source_ref }}/examples/text_pipeline.yaml)은
[JSONL 코퍼스]({{ site.repository_url }}/blob/{{ source_ref }}/examples/data/demo.jsonl)를 읽고
정규화·토큰화한 결과를 stdout에 출력합니다.

```yaml
pipeline:
  - name: ReadDocuments
    args:
      path: examples/data/demo.jsonl
      format: jsonl
  - name: TextNormalize
    args:
      method: text
      result_key: normalized
  - name: TextTokenize
    args:
      method: regex
      text_key: normalized
      result_key: tokens
      offsets: true
  - name: WriteDocuments
    args:
      path: '-'
      format: jsonl
```

`text` 원문을 유지하면서 `normalized`, `tokens` 필드를 추가합니다.
입력 문서의 ID와 메타데이터도 결과에 남습니다.
`offsets: true`는 토큰의 시작·끝 문자 위치를 포함합니다.

## 검증과 실행

저장소 루트에서 실행합니다.

```sh
uv run --python 3.12 soynlp pipeline validate -c examples/text_pipeline.yaml
uv run --python 3.12 soynlp pipeline run -c examples/text_pipeline.yaml
uv run --python 3.12 soynlp pipeline tasks
```

`validate`는 태스크 이름, 인자 스키마, 지원 방법과 입력 키 연결을 검사합니다.
태스크를 실행하거나 입력 파일을 읽지 않으므로 실제 파일·데이터·행렬의 검사는 실행 시 수행합니다.

상대 경로는 **YAML 파일 위치가 아닌 실행 디렉터리**를 기준으로 합니다.
기존 `pipeline -c config.yaml`과 `pipeline --config_file config.yaml`도 사용할 수 있습니다.

## 작업별 태스크

| 작업 | 태스크 |
| --- | --- |
| 문서 입출력 | ReadDocuments, WriteDocuments |
| 리소스 입출력 | ReadArtifact, WriteArtifact |
| 문장 처리 | TextNormalize, TextTokenize, Tag, Lemmatize, Conjugate, Hangle |
| 코퍼스 분석 | Extract, CountEojeols, TrainPatterns |
| 사전 관리 | BuildDictionary, MergeDictionary, InspectDictionary, ExportDictionary |
| LR-Graph | BuildLRGraph, LRGraphNeighbors |
| 벡터화·행렬 | Vectorize, PMI, SVD, Similar, ReadMatrix, WriteMatrix |

`Extract`의 kind와 method는 [CLI 안내](CLI.md)의 추출 명령과 같습니다.
`TextTokenize`의 `resource_key`는 이전 태스크가 만든 Artifact를 참조합니다.
기존 Normalize, Tokenize, ExtractNoun, ExtractWord 태스크도 유지합니다.

## 결과 저장

문서 객체 배열은 `WriteDocuments` 또는 기존 `WriteJson`으로 JSONL에 저장합니다.
명사·단어 점수 같은 매핑은 `WriteArtifact`, 행렬은 `WriteMatrix`를 사용합니다.
`WriteJson`에 매핑을 전달하면 문서 형식 오류가 발생합니다.

파일 출력 시 기존 파일 교체에는 `overwrite: true`가 필요합니다.
리소스와 행렬의 저장 형식은 [CLI의 저장 형식 안내](CLI.md#저장-형식)를 참고하세요.

## Python에서 초기 입력 전달

```python
from soynlp.configs import from_dict
from soynlp.pipeline import Pipeline

config = from_dict(
    {
        "pipeline": [
            {
                "name": "TextNormalize",
                "args": {"method": "repeat", "result_key": "normalized"},
            }
        ]
    }
)
parameters = {"corpus": [{"id": 1, "text": "한국어ㅋㅋㅋㅋ"}]}

report = Pipeline.validate(config, initial_keys={"corpus"})
result = Pipeline()(config, parameters=parameters)
print(report)
print(result["corpus"])
```
