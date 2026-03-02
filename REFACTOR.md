# REFACTOR.md - soynlp 리팩토링 항목

## 리팩토링 이력

| 작업                         | 상태     | 간략 요약                                                                                              | 영향 코드                                                              |
| ---------------------------- | -------- | ------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------- |
| 1. 누락된 `__init__.py`      | **완료** | configs, core, pipeline 패키지에 `__init__.py` 추가 및 re-export                                       | `soynlp/{configs,core,pipeline}/__init__.py`                           |
| 2. `__version__` 노출        | **완료** | `importlib.metadata`로 버전 노출                                                                       | `soynlp/__init__.py`                                                   |
| 3. `ExtractNounTask` 구현    | **완료** | stub 함수 제거, `LRNounExtractor` 래핑으로 재작성                                                      | `soynlp/pipeline/tasks/extract_nouns.py`                               |
| 4. `LRGraph` 클래스 개선     | **완료** | `setdefault` 패턴, `save→_lr`, `load→@classmethod`, `+= e` → `+= 1` 수정, `from_sents` 추가            | `soynlp/core/lrgraph.py`                                               |
| 5. `assert` → `if...raise`   | **완료** | write_json, write_text에 `FileExistsError`, lrgraph에 `ValueError`                                     | `soynlp/pipeline/tasks/write_{json,text}.py`, `soynlp/core/lrgraph.py` |
| 6. `data/loader.py` 위치     | **완료** | `DoublespaceLineCorpus` 제거, `CorpusLoader` (JSONL/TXT) 도입, 데이터 `tests/integration/data/`로 이동 | `soynlp/utils/utils.py`, `soynlp/noun/lr.py`, `soynlp/word/`           |
| 7. Pipeline 동적 로딩        | **완료** | `TASK_REGISTRY` dict 도입, `importlib` 제거                                                            | `soynlp/pipeline/{pipeline.py,tasks/__init__.py}`                      |
| 8. `yaml.safe_load()` 교체   | **완료** | `yaml.full_load()` → `yaml.safe_load()`                                                                | `soynlp/configs/config.py`                                             |
| 9. 테스트 커버리지 확충      | **완료** | 3개 → 76개 (LRGraph, Task, Config, Pipeline 통합 등)                                                   | `tests/unit/` 전체                                                     |
| 10. `cli.py` 개선            | **완료** | `inspect.signature` 제거, 직접 디스패치                                                                | `soynlp/cli.py`                                                        |
| 11. `.gitignore` 업데이트    | **완료** | `.python-version`, `uv.lock`, `.idea/`, `.vscode/`, `.DS_Store` 추가                                   | `.gitignore`                                                           |
| 12. `__main__.py` 구조       | **완료** | 변경 불필요, 이미 표준 패턴                                                                            | `soynlp/__main__.py`                                                   |
| 13. `predicator` 모듈 포팅   | **완료** | 용언 추출기 (PredicatorExtractor, EomiExtractor, StemExtractor)                                        | `soynlp/predicator/`                                                   |
| 14. `pos` 모듈 포팅          | **완료** | 도메인별 품사 추출 (NewsPOSExtractor, ChatPOSExtractor)                                                | `soynlp/pos/`                                                          |
| 15. `ner` 모듈 포팅          | **완료** | 규칙 기반 개체명 인식 (스켈레톤, NotImplementedError)                                                  | `soynlp/ner/`                                                          |
| 16. `postagger` 미포팅       | **완료** | `_lrtagger.py`, `_pos_extractor.py` 포팅                                                               | `soynlp/postagger/`                                                    |
| 17. `normalizer/_normalizer` | **완료** | 저수준 정규화 함수 모음 포팅                                                                           | `soynlp/normalizer/_normalizer.py`                                     |
| 18. `tokenizer` builder      | **완료** | `EojeolPatternTrainer` 커스텀 토크나이저 학습기                                                        | `soynlp/tokenizer/_tokenizer_builder.py`                               |
| 19. `noun/lr.py` docstring   | **완료** | LRNounExtractor docstring + 사용 예제 보강                                                             | `soynlp/noun/lr.py`                                                    |
| 20. `tokenizer` docstring    | **완료** | Token namedtuple + 토크나이저 docstring 보강                                                           | `soynlp/tokenizer/tokenizer.py`                                        |
| 21. `trained_models` 포팅    | **완료** | 명사 예측용 사전 학습 모델 파일                                                                        | `soynlp/trained_models/`                                               |
| LRGraph 통합                 | **완료** | `utils/utils.py`의 LRGraph 제거, `core/lrgraph.py`로 일원화                                            | `soynlp/core/lrgraph.py`, `soynlp/utils/{__init__,utils}.py`           |
| NormalizeTask 신규           | **완료** | `TextNormalizer.build_normalizer()` 래핑 pipeline task                                                 | `soynlp/pipeline/tasks/normalize.py`                                   |
| ExtractWordTask 신규         | **완료** | `WordExtractor` 래핑 pipeline task                                                                     | `soynlp/pipeline/tasks/extract_words.py`                               |
| TokenizeTask 신규            | **완료** | 4종 토크나이저(regex, max_score, noun_match, l_tokenizer) pipeline task                                | `soynlp/pipeline/tasks/tokenize.py`                                    |
| Task 레지스트리 갱신         | **완료** | 4개 신규 task를 import 및 `__all__` 등록                                                               | `soynlp/pipeline/tasks/__init__.py`                                    |
| hangle 모듈 포팅             | **완료** | Py2 호환 제거, type hints 추가, normalize 제거 (normalizer로 이동 완료)                                | `soynlp/hangle/`                                                       |
| lemmatizer 모듈 포팅         | **완료** | Py2 호환 제거, type hints 추가, 활용(conjugation) 로직 유지                                            | `soynlp/lemmatizer/`                                                   |
| postagger 모듈 포팅          | **완료** | namedtuple→dataclass(LR), Py2 제거, \_lrtagger/\_pos_extractor 제외                                    | `soynlp/postagger/`                                                    |
| vectorizer 모듈 포팅         | **완료** | Py2 호환 제거, type hints 추가, BaseVectorizer + word_context_matrix                                   | `soynlp/vectorizer/`                                                   |

---

## 완료 항목

### 1. 누락된 `__init__.py` 파일

`soynlp/configs/__init__.py`, `soynlp/core/__init__.py`, `soynlp/pipeline/__init__.py` 생성.
각 패키지의 주요 클래스/함수를 re-export.

### 2. `soynlp/__init__.py`에 `__version__` 추가

`importlib.metadata.version("soynlp")`으로 `pyproject.toml`의 버전을 자동 노출.

### 3. `ExtractNounTask` 구현

stub 함수 4개(`noun_candidates_from_lrgraph`, `select_nouns_from_candidates`, `extract_compounds`, `postprocessing`)를 삭제하고 `LRNounExtractor.extract()`를 직접 호출하도록 재작성. `parameters["corpus"]` → `parameters["nouns"]` (dict[str, NounScore]).

### 4. `LRGraph` 클래스 개선 (`soynlp/core/lrgraph.py`)

- `add_lr_pair()`: `setdefault` 패턴으로 KeyError 방지
- `save()`: `_lr_origin` 대신 `_lr` 저장 (freeze 후 데이터 유실 방지)
- `load()`: 인스턴스 메서드 → `@classmethod` 팩토리
- `corpus_to_lrgraph()`: `+= e` → `+= 1` 빈도 버그 수정
- `from_sents()` classmethod 추가 (sents로부터 LRGraph 직접 생성)
- `freeze()`: `copy.deepcopy`로 현재 `_lr` 상태 보존
- `max_l_length`, `max_r_length` 속성 추가 및 입력 검증

### 5. 에러 처리 방식 개선

- `WriteJsonTask`, `WriteTextTask`: `assert` → `if os.path.exists: raise FileExistsError`
- `corpus_to_lrgraph()`: `assert` → `if...raise ValueError`

### 7. Pipeline의 동적 task 로딩 방식

- `importlib.import_module` + `getattr(f"{name}Task")` 패턴을 `TASK_REGISTRY` dict로 교체
- `soynlp/pipeline/tasks/__init__.py`에 `TASK_REGISTRY: dict[str, type[Task]]` 추가
- `pipeline.py`에서 `TASK_REGISTRY.get()`으로 조회, 없으면 `ValueError` (사용 가능한 task 목록 포함)
- YAML `name` 값 변경 없음 (PascalCase, `Task` 접미사 없음)

### 8. `yaml.safe_load()` 교체

`soynlp/configs/config.py`에서 `yaml.full_load()` → `yaml.safe_load()`.

### 9. 테스트 커버리지 확충

기존 3개 → 76개 테스트 (slow 제외). 추가된 테스트 파일:

- `tests/unit/test_lrgraph.py` — LRGraph 전체 (init, add, remove, save/load, freeze/reset, from_sents)
- `tests/unit/test_config.py` — Config, TaskConfig, from_yaml, from_dict
- `tests/unit/pipeline/tasks/test_write_json.py` — 정상 쓰기, FileExistsError, ValueError
- `tests/unit/pipeline/tasks/test_write_text.py` — 동일 패턴
- `tests/unit/pipeline/tasks/test_dummy.py` — parameters 통과 확인
- `tests/unit/pipeline/tasks/test_normalize.py` — 정규화, 필드 보존, 커스텀 키
- `tests/unit/pipeline/tasks/test_extract_nouns.py` — 누락 키 에러 (slow: 실제 추출)
- `tests/unit/pipeline/tasks/test_extract_words.py` — 누락 키 에러 (slow: cohesion 추출)
- `tests/unit/pipeline/tasks/test_tokenize.py` — 4종 토크나이저, namedtuple score, 에러 케이스
- `tests/unit/pipeline/test_pipeline.py` — Dummy, ReadText→WriteText, ReadText→Normalize→WriteText

### 10. `cli.py` 개선

- `inspect.signature` 기반 동적 파라미터 추출 제거
- `run()` 헬퍼 함수 제거, `set_defaults(func=...)` 패턴 제거
- `subparsers(dest="command")` + `if args.command == "pipeline"` 직접 디스패치로 교체

### 11. `.gitignore` 업데이트

`.python-version`, `uv.lock`, `.idea/`, `.vscode/`, `.DS_Store` 추가.

### LRGraph 통합 (utils → core)

`soynlp/utils/utils.py`의 LRGraph 클래스를 제거하고 `soynlp/core/lrgraph.py`로 일원화.
`EojeolCounter._to_lrgraph()`가 core LRGraph를 생성하도록 수정.
`soynlp/utils/__init__.py`에서 `from soynlp.core.lrgraph import LRGraph` re-export (하위 호환성).

### 신규 Pipeline Task 4종

- **NormalizeTask** (`soynlp/pipeline/tasks/normalize.py`): `TextNormalizer.build_normalizer()` 래핑. corpus의 각 example text를 정규화.
- **ExtractWordTask** (`soynlp/pipeline/tasks/extract_words.py`): `WordExtractor` 래핑. cohesion, accessor_variety, branching_entropy 추출.
- **TokenizeTask** (`soynlp/pipeline/tasks/tokenize.py`): `regex`, `max_score`, `noun_match`, `l_tokenizer` 4종 지원. `score_field`로 NounScore 등의 속성 접근.
- Task 레지스트리 (`soynlp/pipeline/tasks/__init__.py`) 갱신.

### 12. `__main__.py` 구조

- 변경 불필요. 이미 `from soynlp.cli import main; main()` 표준 패턴.

### hangle 모듈 포팅

`soynlp/hangle/` 모듈을 `refactoring` 브랜치에서 포팅:

- Python 2 호환 코드 제거 (`sys.version_info`, `reload`, `unicode`)
- `normalize()` 함수 제외 (deprecated, `soynlp/normalizer/`로 이동 완료)
- type hints 추가 (`decompose() → tuple[str, str, str] | None` 등)
- `_hangle.py` (decompose, compose, character*is*\*, ConvolutionHangleEncoder)
- `_distance.py` (levenshtein, jamo_levenshtein, cosine_distance, jaccard_distance)
- 테스트: `tests/unit/test_hangle.py` (32개)

### lemmatizer 모듈 포팅

`soynlp/lemmatizer/` 모듈을 `refactoring` 브랜치에서 포팅:

- Python 2 호환 코드 제거
- type hints 추가
- `_conjugation.py` (conjugate, conjugate_chat, \_conjugate_stem)
- `_lemmatizer.py` (Lemmatizer, lemma_candidate, lemma_candidate_chat)
- 사전 데이터 (`dictionary/`, `tag/`) 그대로 복사
- 테스트: `tests/unit/test_lemmatizer.py` (14개)

### postagger 모듈 포팅

`soynlp/postagger/` 모듈을 `refactoring` 브랜치에서 포팅:

- `LR` namedtuple → `@dataclass(frozen=True, slots=True)` 변환
- Python 2 호환 코드 전체 제거
- `_lrtagger.py`, `_pos_extractor.py` 제외 (미포팅 의존성)
- `_dictionary.py` (Dictionary: load/save/get_pos/add_words/remove_words)
- `_template.py` (LR, EojeolTemplateMatcher, LRTemplateMatcher)
- `_evaluator.py` (SimpleEojeolEvaluator, LREvaluator)
- `_tagger.py` (SimpleTagger, UnknowLRPostprocessor)
- 사전 데이터 (`dictionary/`, `tagset/`) 그대로 복사
- 테스트: `tests/unit/test_postagger.py` (14개)

### vectorizer 모듈 포팅

`soynlp/vectorizer/` 모듈을 `refactoring` 브랜치에서 포팅:

- type hints 추가
- `_vectorizer.py` (BaseVectorizer: fit, transform, save, load)
- `_word_context.py` (sent_to_word_contexts_matrix)
- 테스트: `tests/unit/test_vectorizer.py` (10개)

---

## 미착수 항목

### 13. `predicator` 모듈 포팅

- `refactoring` 브랜치의 신규 모듈을 Py3.12 스타일로 포팅
- Py2 호환 코드 제거, type hints 추가, `is` → `==` 비교 수정, f-string 적용
- `PredicatorExtractor`, `EomiExtractor`, `StemExtractor`, `Predicator` namedtuple
- `_adjective_vs_verb.py`: 형용사/동사 활용 규칙 판별 (현재형/명령형/청유형)
- 4개 파일: `_predicator.py`, `_eomi.py`, `_stem.py`, `_adjective_vs_verb.py`

### 14. `pos` 모듈 포팅

- `refactoring` 브랜치의 신규 모듈을 Py3.12 스타일로 포팅
- `NewsPOSExtractor` (뉴스), `ChatPOSExtractor` (대화체 — NewsPOSExtractor 상속)
- `load_default_adverbs()`, `stem_to_adverb()` 부사 유틸리티
- `LRNounExtractor_v2` → `LRNounExtractor`로 대체 (v2 미구현)
- 3개 파일: `_news_pos.py`, `_chat_pos.py`, `_adverb.py`

### 15. `ner` 모듈 포팅

- 스켈레톤 프레임워크 포팅: `is_date()`, `is_number()`, `is_monetary()` 등 6개 함수
- 모두 `NotImplementedError` 상태

### 16. `postagger` 미포팅 파일

- `_lrtagger.py`: `LREvaluator` + `LRMaxScoreTagger` (L-R 평가 프로파일 기반 태거)
- `_pos_extractor.py`: `POSExtractor` — 명사+용언 통합 추출 (`LRNounExtractor` + `PredicatorExtractor`)
- Py2 코드 정리, type hints 추가, `__init__.py`에 등록

### 17. `normalizer/_normalizer.py` 포팅

- Py2 호환 코드 제거 (`sys.version_info`, `reload` 등)
- 함수형 정규화 API: `normalize()`, `emoticon_normalize()`, `repeat_normalize()`, `only_hangle()` 등
- `normalize_sent_for_lrgraph()` — predicator 모듈 의존 함수
- 기존 클래스 기반 API (`TextNormalizer`)와 병행 export

### 18. `tokenizer/_tokenizer_builder.py` 포팅

- `EojeolPatternTrainer`: 코퍼스 기반 커스텀 토크나이저 학습
- `train()`, `_scan_vocabulary()`, `_build_graph()`, `save()`/`load()`, `train_hits()`
- Py2 호환 코드 제거, type hints 추가

### 21. `trained_models/` 포팅

- `noun_predictor_sejong`, `noun_predictor_sejong_logisticregression` 바이너리 파일 복사

### 19. `noun/lr.py` docstring 보강

- `refactoring` 브랜치에서 `LRNounExtractor` 클래스에 docstring + 사용 예제 추가
- `extract()` 메서드에 파라미터 문서화, `custom_exclude_function` 인자 추가
- 함수 구조는 동일, 문서화 차이

### 20. `tokenizer/tokenizer.py` docstring 보강

- `Token` namedtuple 추가 (`word`, `begin`, `end`, `score`, `length`, `eojeol_id`)
- 각 토크나이저 클래스(`RegexTokenizer`, `LTokenizer`, `MaxScoreTokenizer`, `NounMatchTokenizer`)에 docstring + 사용 예제
- API 변경 없음, 문서화 차이

### 21. `trained_models/` 포팅

- `noun_predictor_sejong`, `noun_predictor_sejong_logisticregression` (바이너리 모델 파일)
- 명사 예측용 사전 학습 로지스틱 회귀 모델
- `noun/lr.py`에서 참조 가능

### 19. `noun/lr.py` docstring 보강

`refactoring` 브랜치의 docstring을 포팅:

- `LRNounExtractor` 클래스: Examples 섹션 추가 (학습, 명사 확인, 명사 토크나이저 사용)
- `extract()`: Args 상세화, custom_exclude_function 예제, Note 섹션 (명사 추출 원리 설명)
- `decompose_compound()`: Args, Returns, Examples 추가
- `predict()`: Args 상세화, Examples 추가 (debug 모드, word_features 직접 전달)
- `get_noun_tokenizer()`: Examples 추가 (학습 → 토크나이저 사용 흐름)
- `prepare_r_features()`: Args, Returns 상세화

### 20. `tokenizer/tokenizer.py` docstring 보강

`refactoring` 브랜치의 docstring을 포팅:

- `RegexTokenizer`: 클래스 docstring + `tokenize()` 메서드 Examples 추가
- `LTokenizer`: tolerance 있는/없는 경우 Examples, Token 반환 예제
- `MaxScoreTokenizer`: pretrained scores / WordExtractor 학습 Examples
- `NounMatchTokenizer`: noun_scores, noun_set, concat_compound, must_be_L Examples

### 테스트 커버리지 보강

기존에 테스트가 없던 3개 모듈에 테스트 추가 (총 43개):

- `tests/unit/test_ner.py` (6개): 스켈레톤 함수 NotImplementedError 검증
- `tests/unit/test_predicator.py` (20개): Predicator/EomiScore namedtuple, rule*classify, conjugate_as*\*, PredicatorExtractor 초기화/분류
- `tests/unit/test_pos.py` (17개): stem_to_adverb, load_default_adverbs, NewsPOSExtractor/ChatPOSExtractor 초기화

### 6. `data/loader.py` 위치 및 구조

- `DoublespaceLineCorpus` 클래스 제거 (double-space-separated text 포맷 폐기)
- `CorpusLoader` 클래스 신규 도입 (`soynlp/utils/utils.py`): JSONL, TXT 두 가지 포맷 지원
- `EojeolCounter`에 dict item 지원 추가 (`text_key` 파라미터)
- 4개 모듈 참조 업데이트: `noun/lr.py`, `word/word.py`, `word/ngram.py`, `tokenizer/tokenizer.py`
- 데이터 파일을 `tests/integration/data/`로 이동
- integration test 예제 추가: `extract_frequent_nouns`, `cohesion_tokenize`

### 포팅 완료 순서 (의존관계 기준)

1. `normalizer/_normalizer.py` → 2. `predicator/` → 3. `postagger/_lrtagger.py`, `_pos_extractor.py` → 4. `pos/` → 5. `tokenizer/_tokenizer_builder.py` → 6. `ner/` → 7. `trained_models/`
