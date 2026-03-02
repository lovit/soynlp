# REFACTOR.md - soynlp 리팩토링 항목

## 리팩토링 이력

| 작업                       | 상태     | 간략 요약                                                                                   | 영향 코드                                                              |
| -------------------------- | -------- | ------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| 1. 누락된 `__init__.py`    | **완료** | configs, core, pipeline 패키지에 `__init__.py` 추가 및 re-export                            | `soynlp/{configs,core,pipeline}/__init__.py`                           |
| 2. `__version__` 노출      | **완료** | `importlib.metadata`로 버전 노출                                                            | `soynlp/__init__.py`                                                   |
| 3. `ExtractNounTask` 구현  | **완료** | stub 함수 제거, `LRNounExtractor` 래핑으로 재작성                                           | `soynlp/pipeline/tasks/extract_nouns.py`                               |
| 4. `LRGraph` 클래스 개선   | **완료** | `setdefault` 패턴, `save→_lr`, `load→@classmethod`, `+= e` → `+= 1` 수정, `from_sents` 추가 | `soynlp/core/lrgraph.py`                                               |
| 5. `assert` → `if...raise` | **완료** | write_json, write_text에 `FileExistsError`, lrgraph에 `ValueError`                          | `soynlp/pipeline/tasks/write_{json,text}.py`, `soynlp/core/lrgraph.py` |
| 6. `data/loader.py` 위치   | 미착수   | 패키지 외부 데이터 로더 이동 필요                                                           | `data/loader.py`                                                       |
| 7. Pipeline 동적 로딩      | 미착수   | registry 패턴 도입 검토                                                                     | `soynlp/pipeline/pipeline.py`                                          |
| 8. `yaml.safe_load()` 교체 | **완료** | `yaml.full_load()` → `yaml.safe_load()`                                                     | `soynlp/configs/config.py`                                             |
| 9. 테스트 커버리지 확충    | **완료** | 3개 → 76개 (LRGraph, Task, Config, Pipeline 통합 등)                                        | `tests/unit/` 전체                                                     |
| 10. `cli.py` 개선          | 미착수   | `inspect.signature` 방식 교체 검토                                                          | `soynlp/cli.py`                                                        |
| 11. `.gitignore` 업데이트  | **완료** | `.python-version`, `uv.lock`, `.idea/`, `.vscode/`, `.DS_Store` 추가                        | `.gitignore`                                                           |
| 12. `__main__.py` 구조     | 미착수   | 구조 정리 검토                                                                              | `soynlp/__main__.py`                                                   |
| LRGraph 통합               | **완료** | `utils/utils.py`의 LRGraph 제거, `core/lrgraph.py`로 일원화                                 | `soynlp/core/lrgraph.py`, `soynlp/utils/{__init__,utils}.py`           |
| NormalizeTask 신규         | **완료** | `TextNormalizer.build_normalizer()` 래핑 pipeline task                                      | `soynlp/pipeline/tasks/normalize.py`                                   |
| ExtractWordTask 신규       | **완료** | `WordExtractor` 래핑 pipeline task                                                          | `soynlp/pipeline/tasks/extract_words.py`                               |
| TokenizeTask 신규          | **완료** | 4종 토크나이저(regex, max_score, noun_match, l_tokenizer) pipeline task                     | `soynlp/pipeline/tasks/tokenize.py`                                    |
| Task 레지스트리 갱신       | **완료** | 4개 신규 task를 import 및 `__all__` 등록                                                    | `soynlp/pipeline/tasks/__init__.py`                                    |

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

---

## 미착수 항목

### 6. `data/loader.py` 위치 및 구조

- `data/` 디렉토리는 패키지 외부에 있어 `soynlp` 패키지에서 접근이 어려움
- 데이터 로딩 유틸리티를 `soynlp` 패키지 내로 이동하거나, `package_data`로 등록 필요
- `zip` 파일 (`2016-10-20.zip`) 처리 로직이 없음

### 7. Pipeline의 동적 task 로딩 방식

- `Pipeline._load_tasks()`에서 `importlib.import_module("soynlp.pipeline.tasks")`로 하드코딩된 모듈 경로 사용
- task 이름으로 `f"{task_config.name}Task"` 패턴을 강제하고 있음 — 확장성이 제한적
- entry point 기반 플러그인 시스템이나 registry 패턴 도입 검토

### 10. `cli.py`의 `_run()` 함수

- `inspect.signature`로 파라미터를 동적으로 추출하는 방식은 취약함
- argparse의 `Namespace`에서 직접 필요한 인자를 추출하는 것이 더 명확

### 12. `__main__.py` 구조

- 현재 `soynlp/__main__.py`가 `cli.main`을 호출하는 단순 래퍼
- `if __name__ == "__main__":` 가드가 있으나 `from soynlp.cli import main` 줄이 모듈 로드 시 항상 실행됨 — 이는 문제는 아니지만 구조 정리 검토
