# REFACTOR.md - soynlp 리팩토링 항목

## 리팩토링 이력

| #   | 작업                         | 상태     | 간략 요약                                                          | 영향 코드                                                                            |
| --- | ---------------------------- | -------- | ------------------------------------------------------------------ | ------------------------------------------------------------------------------------ |
| 1   | 누락된 `__init__.py`         | **완료** | configs, core, pipeline 패키지에 `__init__.py` 추가 및 re-export   | `soynlp/{configs,core,pipeline}/__init__.py`                                         |
| 2   | `__version__` 노출           | **완료** | `importlib.metadata`로 버전 노출                                   | `soynlp/__init__.py`                                                                 |
| 3   | `ExtractNounTask` 구현       | **완료** | stub 함수 제거, `LRNounExtractor` 래핑으로 재작성                  | `soynlp/pipeline/tasks/extract_nouns.py`                                             |
| 4   | `LRGraph` 클래스 개선        | **완료** | `setdefault` 패턴, save/load 수정, `from_sents` 추가               | `soynlp/core/lrgraph.py`                                                             |
| 5   | `assert` → `if...raise`      | **완료** | write_json, write_text에 `FileExistsError`, lrgraph에 `ValueError` | `soynlp/pipeline/tasks/write_{json,text}.py`, `soynlp/core/lrgraph.py`               |
| 6   | `data/loader.py` 위치        | **완료** | `DoublespaceLineCorpus` 제거, `CorpusLoader` 도입, 데이터 이동     | `soynlp/utils/utils.py`, `soynlp/noun/lr.py`, `soynlp/word/`                         |
| 7   | Pipeline 동적 로딩           | **완료** | `TASK_REGISTRY` dict 도입, `importlib` 제거                        | `soynlp/pipeline/{pipeline.py,tasks/__init__.py}`                                    |
| 8   | `yaml.safe_load()` 교체      | **완료** | `yaml.full_load()` → `yaml.safe_load()`                            | `soynlp/configs/config.py`                                                           |
| 9   | 테스트 커버리지 확충         | **완료** | 3개 → 244개 unit + 14개 integration                                | `tests/` 전체                                                                        |
| 10  | `cli.py` 개선                | **완료** | `inspect.signature` 제거, 직접 디스패치                            | `soynlp/cli.py`                                                                      |
| 11  | `.gitignore` 업데이트        | **완료** | `.python-version`, `uv.lock`, `.idea/` 등 추가                     | `.gitignore`                                                                         |
| 12  | `__main__.py` 구조           | **완료** | 변경 불필요, 이미 표준 패턴                                        | `soynlp/__main__.py`                                                                 |
| 13  | `predicator` 모듈 포팅       | **완료** | 용언 추출기 포팅                                                   | `soynlp/predicator/`                                                                 |
| 14  | `pos` 모듈 포팅              | **완료** | 도메인별 품사 추출 포팅                                            | `soynlp/pos/`                                                                        |
| 15  | `ner` 모듈 포팅              | **완료** | 규칙 기반 개체명 인식 (스켈레톤)                                   | `soynlp/ner/`                                                                        |
| 16  | `postagger` 포팅             | **완료** | `_lrtagger.py`, `_pos_extractor.py` 포함 전체 포팅                 | `soynlp/postagger/`                                                                  |
| 17  | `normalizer/_normalizer`     | **완료** | 저수준 정규화 함수 포팅                                            | `soynlp/normalizer/_normalizer.py`                                                   |
| 18  | `tokenizer` builder          | **완료** | `EojeolPatternTrainer` 포팅                                        | `soynlp/tokenizer/_tokenizer_builder.py`                                             |
| 19  | `noun/lr.py` docstring       | **완료** | LRNounExtractor docstring + 사용 예제 보강                         | `soynlp/noun/lr.py`                                                                  |
| 20  | `tokenizer` docstring        | **완료** | Token namedtuple + 토크나이저 docstring 보강                       | `soynlp/tokenizer/tokenizer.py`                                                      |
| 21  | `trained_models` 포팅        | **완료** | 명사 예측용 사전 학습 모델 파일                                    | `soynlp/trained_models/`                                                             |
| 22  | LRGraph 통합                 | **완료** | `utils/utils.py`의 LRGraph 제거, `core/lrgraph.py`로 일원화        | `soynlp/core/lrgraph.py`, `soynlp/utils/`                                            |
| 23  | Pipeline Task 4종 추가       | **완료** | NormalizeTask, ExtractWordTask, TokenizeTask, Task 레지스트리 갱신 | `soynlp/pipeline/tasks/`                                                             |
| 24  | hangle 모듈 포팅             | **완료** | Py2 호환 제거, type hints 추가                                     | `soynlp/hangle/`                                                                     |
| 25  | lemmatizer 모듈 포팅         | **완료** | Py2 호환 제거, type hints 추가                                     | `soynlp/lemmatizer/`                                                                 |
| 26  | postagger 모듈 포팅          | **완료** | namedtuple→dataclass, Py2 제거                                     | `soynlp/postagger/`                                                                  |
| 27  | vectorizer 모듈 포팅         | **완료** | Py2 호환 제거, type hints 추가                                     | `soynlp/vectorizer/`                                                                 |
| 28  | type hints 추가              | **완료** | Python 3.12 스타일 type hints (11개 파일, 116개 함수)              | 전체                                                                                 |
| 29  | private module import 수정   | **완료** | hangle public import로 변경                                        | `soynlp/pos/_chat_pos.py`, `soynlp/hangle/__init__.py`                               |
| 30  | integration test 추가        | **완료** | 14개 integration test (pytest assert 형식)                         | `tests/integration/`                                                                 |
| 31  | 레거시 usage 테스트 삭제     | **완료** | 삭제된 data/2016-10-20.zip 참조 테스트 4개 제거                    | `tests/unit/test_{lrnounextractor,nountokenizer,tokenizers,wordextractor}.py`        |
| 32  | integration 속도 최적화      | **완료** | session-scoped fixture로 중복 연산 제거 (~200s→~118s)              | `tests/integration/conftest.py`, 4개 테스트 파일                                     |
| 33  | word_context_matrix 비결정성 | **완료** | similar words를 구조적 검증으로 변경                               | `tests/integration/test_word_context_matrix.py`                                      |
| 34  | cli.py 단위 테스트           | **완료** | argparse 동작 테스트 4개                                           | `tests/unit/test_cli.py`                                                             |
| 35  | pytest.mark.parametrize 적용 | **완료** | for-loop 기반 → parametrize 변환 (5개 파일)                        | `tests/unit/test_{lrnounextractor,nountokenizer,tokenizers,wordextractor,bigram}.py` |

| 36 | 중복 테스트 파일 삭제 | **완료** | `tests/pipeline/`, `tests/test_sanity.py` 삭제 | `tests/` |
| 37 | `typing.Optional`/`Union` → pipe 문법 | **완료** | 3개 파일에서 `X \| None` 문법으로 통일, `Callable` → `collections.abc` | `soynlp/pipeline/tasks/{extract_nouns,normalize}.py`, `soynlp/normalizer/normalizer.py` |
| 38 | `print()` → `logging` 마이그레이션 | **완료** | 109개 print → logging 전환 (17개 파일) | `soynlp/` 전체 (predicator, lemmatizer, vectorizer, pos, postagger, tokenizer 등) |
| 39 | integration test 구조 개편 | **완료** | 자기 완결적 예시 디렉토리 + 자동 탐색 구조로 개편 | `tests/integration/` |
| 40 | `from __future__ import annotations` 제거 | **완료** | Python 3.12에서 불필요한 future import 제거 | `soynlp/word/` |
| 41 | `.format()`/`%` → f-string 변환 | **완료** | 구형 포매팅 문법을 f-string으로 일괄 변환 | `soynlp/` 전체 |
| 42 | regex raw string 적용 | **완료** | 정규식 패턴에 `r""` raw string 적용 | `soynlp/` 전체 |
| 43 | dataclass `slots=True` 추가 | **완료** | 모든 dataclass에 `slots=True` 추가 | `soynlp/` 전체 |
