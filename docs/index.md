---
title: 한글 사용자 가이드
description: soynlp의 설치, Python API, CLI와 YAML 파이프라인 사용법을 안내합니다.
section: 소개
permalink: /
home: true
---

# 코퍼스로 배우는<br>한국어 자연어 처리

soynlp는 비지도 한국어 자연어 처리 Python 패키지입니다.
코퍼스의 빈도와 결합도, 문맥 통계로 단어·명사를 추출하고
정규화, 토큰화, 표제어 분석과 벡터화를 제공합니다.

> 이 문서는 **refactor-2026 개발 브랜치**를 기준으로 합니다.
> 새 CLI와 YAML 태스크는 이 브랜치의 저장소를 설치해야 사용할 수 있습니다.
> [설치·빠른 시작](getting-started.md)에서 시작하세요.

<div class="entry-grid">
  <a class="entry-card" href="{{ '/python/' | relative_url }}">
    <strong>Python</strong>
    <span>코드에서 추출기를 호출하고<br>결과를 다음 연산에 연결합니다.</span>
    <span class="entry-arrow" aria-hidden="true">→</span>
  </a>
  <a class="entry-card" href="{{ '/cli/' | relative_url }}">
    <strong>CLI</strong>
    <span>문장 하나부터 코퍼스 파일까지<br>명령어로 처리합니다.</span>
    <span class="entry-arrow" aria-hidden="true">→</span>
  </a>
  <a class="entry-card" href="{{ '/pipeline/' | relative_url }}">
    <strong>YAML 파이프라인</strong>
    <span>여러 처리 단계를 설정 파일로<br>묶어 반복 실행합니다.</span>
    <span class="entry-arrow" aria-hidden="true">→</span>
  </a>
</div>

## 어떤 일을 할 수 있나요?

| 작업 | 제공 기능 |
| --- | --- |
| 문장 처리 | 한글·자모 처리, 정규화, 토큰화, 태깅, 표제어 분석과 활용 |
| 코퍼스 분석 | 명사·단어·bigram·품사·용언·어미·어간·조사 특징 추출 |
| 학습과 사전 | 어절 집계, 패턴 학습, LR-Graph, POS 사전 생성·결합 |
| 벡터화 | 단어 빈도(BoW), 문맥 행렬, PMI, SVD와 행 유사도 |
| 작업 연결 | JSONL 문서, JSON 리소스, NPZ 행렬 저장·재사용, YAML 실행 |

BoW는 단어 빈도(TF) 행렬입니다. TF-IDF, NER와 새로운 MaxScoreTagger 알고리즘은 이번 범위에 포함하지 않습니다.

## 먼저 한 문장 처리하기

저장소 설치 후 루트 디렉터리에서 실행합니다.

```sh
uv run --python 3.12 soynlp tokenize --text '한국어를 공부합니다'
```

```json
{"text": "한국어를 공부합니다", "tokens": ["한국어를", "공부합니다"]}
```

하나의 문장은 `--text`, 파일은 `--input`으로 전달합니다.
전체 명령과 리소스 재사용 예제는 [CLI 안내](CLI.md)를 참고하세요.

## 기존 사용자를 위한 안내

master의 공개 import, 학습·추출 호출과 기존 모델 파일의 호환 경로를 제공합니다.
기존 알고리즘과 새 코퍼스 API의 차이, 토큰 반환 옵션은 [호환성 안내](COMPATIBILITY.md)에 정리했습니다.

{% assign source_ref = site.github.build_revision | default: site.docs_source_ref %}
개발 변경과 검증 결과는 저장소의
[구현 결과]({{ site.repository_url }}/blob/{{ source_ref }}/REFACTOR_DELIVERY.md)와
[작업 기록]({{ site.repository_url }}/blob/{{ source_ref }}/REFACTOR_PROGRESS.md)에서 확인할 수 있습니다.
