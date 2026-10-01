# `refactor-2026` 진단 보고서

작성일: 2026-09-30 · 기준: `master` `503eaee`, `refactor-2026` `13028c6` · 검증 환경: macOS, Python 3.12.13

## 목적과 판정 기준

soynlp는 별도의 정답 학습 데이터 없이 한국어 코퍼스의 통계적 패턴으로 단어·명사·품사 등을 추출하고, 정규화·토크나이징·표제화·벡터화하는 패키지다. `master`의 README는 동일한 도메인의 문서를 모아 추출기를 학습하는 사용법을 설명한다. 이번 리팩토링은 그 기능을 유지하면서 패키지 진입점, 설치 및 사용법, CLI와 파이프라인, 코드 품질을 개선하는 작업으로 해석했다.

이 보고서의 호환성 기준은 **Python 3.12 이상에서 `master`의 공개 import·호출 방식과 주요 결과를 유지하는 것**이다. 구버전 Python 지원 복원은 대상에 넣지 않았다. 두 브랜치는 2020년의 공통 커밋에서 갈라졌고 `master`에만 있는 커밋이 2개 있으므로, 리팩토링 이력과 현재 양쪽 소스를 함께 대조했다. 현재 테스트의 통과 여부만으로 `master`와의 호환성을 판정하지 않았다.

## 확인된 작업과 검증 결과

| 영역 | `refactor-2026`에서 확인한 내용 |
| --- | --- |
| 패키징·진입점 | `setup.py`에서 `pyproject.toml`/Hatchling/`uv`로 이동, Python 3.12+ 요구, `soynlp` 콘솔 스크립트와 `python -m soynlp` 추가. |
| 기능 구성 | LR-Graph와 추출·정규화·품사·표제화·벡터화 모듈을 재구성하고 YAML 파이프라인 및 태스크 레지스트리를 추가. 기존 `master`에 CLI는 없었다. |
| 품질 작업 | 타입 힌트, 로깅, 다중 프로세스 처리, `dataclass`, 리팩토링 단위·통합 테스트를 추가. `REFACTOR.md`의 47개 항목과 `todo.md`는 모두 완료로 표시돼 있다. |
| Codex 설정 | 루트 `AGENTS.md`와 `.codex/WORKFLOW.md`를 추가. 읽기 전용 신규 Codex 세션이 `AGENTS.md`를 자동 로드하고 연결된 문서의 Python·브랜치 규칙을 읽는 것을 확인했다. |

검증 결과:

- `uv run --python 3.12 pytest -q`: **416 passed, 201 warnings** (약 145초). 이 중 통합 테스트는 9개다. 경고 1건은 pytest의 제너레이터 `parametrize` 사용 중단 예고, 200건은 통합 예제가 deprecated 정규화 함수를 호출한 결과다.
- CI에 고정된 Ruff 0.9.4: `ruff check .` 및 `ruff format --check .` 통과. CI에 고정된 Pyright 1.1.393: **2 errors** (`soynlp/noun/lr.py:313`, `:855`).
- 현재 `uv` 개발 환경에 설치된 Ruff 0.16.9: **129 errors**; Markdown 코드 블록을 포함한 포맷 검사에서 `soynlp/README.md` 1개 파일 불일치. Pyright 1.1.414: **4 errors**(기존 명사 모듈 2건과 `soynlp/utils/pmi.py:71`의 타입 오류 2건). 이 수치는 도구 버전 차이를 포함하며 129건을 기능 결함으로 세지 않았다.
- 휠 빌드와 별도 Python 3.12 가상환경 설치·import·CLI 실행 성공. 휠에는 명사 모델, 사전 등 점검한 4종의 리소스가 포함됐다. `soynlp --help`, `--version`, `python -m soynlp --version`, 더미 파이프라인 실행도 성공했다.
- 양쪽 브랜치를 같은 Python 3.12 환경에서 실행한 표본 비교에서는 `decompose`, `compose`, 기존 `normalize`, `RegexTokenizer`, `LTokenizer`의 결과가 일치했다. 전체 알고리즘의 결과 동등성을 증명하는 검사는 아니다.

## 남은 문제와 근거

### P1 — 공개 API 호환성

`master` 소스를 임시 디렉터리에서 같은 Python 3.12 의존성으로 실행해 기존 이름과 `NounScore`의 튜플 동작을 확인했다. 현재 브랜치에서 같은 접근을 실행하면 아래 이름이 없다.

| 기존 사용 경로 | 현재 상태와 영향 |
| --- | --- |
| `import soynlp; soynlp.noun`, `soynlp.hangle`, `soynlp.DoublespaceLineCorpus`, `soynlp.__author__` | 루트 `__init__.py`의 기존 노출이 사라졌다. 하위 모듈을 직접 import하는 경로는 별도로 작동할 수 있지만, 기존 루트 접근은 실패한다. |
| `soynlp.noun.NewsNounExtractor`, `LRNounExtractor_v2`, `extract_domain_pos_features` | 공개 이름과 일부 구현 파일이 제거됐다. 새 `LRNounExtractor`가 각 사용 사례를 대체하는지 검증되지 않았다. |
| `soynlp.tokenizer.MaxLRScoreTokenizer`, `NounLMatchTokenizer`, `normalize` | 기존 공개 이름이 없다. |
| `soynlp.utils.DoublespaceLineCorpus`, `sort_by_alphabet`; `soynlp.word.Bigram` | 기존 공개 이름이 없다. `CorpusLoader`, `BigramExtractor`가 있지만 호출 방식이 같지는 않다. |
| `NounScore` | `master`의 namedtuple은 `tuple(NounScore(2, 0.5)) == (2, 0.5)`였으나 현재 dataclass는 `TypeError`를 낸다. `Token`, `Predicator` 등 다른 namedtuple 전환도 같은 호환성 검토가 필요하다. |

기존 공개 이름의 실패를 검증하는 회귀 테스트는 현재 테스트에서 찾지 못했다. 이름을 복원할지, 호환 래퍼를 둘지, 변경을 공식적으로 허용할지는 사용 사례별 결정이 필요하다. 정규화 함수와 `soynlp.word.pmi`에는 이미 deprecation 래퍼가 있어 적용 가능한 선례가 있다.

### P1 — 사용 문서와 실제 호출 방식 불일치

- 루트 `README.md`는 아직 “Renewing ...”과 Poetry 설치를 안내하지만, 현재 패키지 설정은 `uv`/Hatchling이다. 새 사용자는 설치 경로와 패키지 목적을 루트 문서에서 확인하기 어렵다.
- `soynlp/README.md`의 파이프라인 YAML은 `filepath`와 `remove_repeat`을 사용한다. 실제 태스크 인자는 `path`와 `remove_repeatchar`이며, 안내된 설정을 로드하면 `MissingValueError: missing value for field "path"`가 발생한다.
- 같은 가이드의 `BigramExtractor().train(...)`은 존재하지 않는 메서드이고, `extract(score_type=...)`도 현재 시그니처와 맞지 않는다. `PredicatorExtractor(adjective_stems=...)` 역시 현재 생성자가 받지 않는다.
- 가이드가 `NounScore`, `Predicator`, `Token`을 namedtuple이라고 설명하지만 현재는 dataclass다. `CLAUDE.md`에는 삭제된 `data/loader.py`와 이미 구현된 `ExtractNounTask`를 골격 코드로 설명하는 부분이 남아 있다. `REFACTOR.md`의 테스트 수와 삭제된 파일 경로도 현재 상태와 맞지 않는다.

### P1 — 타입 검사 실패 및 도구 버전 차이

- CI 버전 Pyright의 2건: `decompose_compound()`에서 `self.nouns`가 `None`일 가능성과 `MaxScoreTokenizer`에 `dict[str, int]`를 전달하는 타입 불일치. 런타임 결함 여부는 별도 재현이 필요하지만, 현재 타입 검사 게이트는 통과하지 않는다.
- 개발 의존성은 `ruff>=0.9.4`, `pyright>=1.1.393`으로 열려 있고 CI의 pre-commit 버전은 고정돼 있다. 현재 로컬 검사와 CI 검사 결과가 달라 재현 가능한 품질 기준을 정하기 어렵다.

### P2 — 검증·출시 범위의 결정 필요

- `pyproject.toml` 버전은 `0.0.0-dev`이고 `master`는 `0.0.493`이다. 리팩토링 브랜치의 개발 버전으로는 동작하지만, 배포 버전과 기존 사용자 업그레이드 정책이 정해져 있지 않다.
- CLI는 현재 `pipeline -c`만 제공한다. 패키지의 추출·정규화 기능을 CLI의 개별 명령으로도 지원할지는 요구사항이 확정되지 않았다. 현재 CLI 테스트는 인자 파싱과 디스패치 중심이다.
- 현재 통합 테스트는 새 파이프라인과 예제 9개를 검증하지만, `master`의 공개 경로와 결과를 체계적으로 비교하는 테스트는 없다. pytest 중단 예고 경고와 deprecated 예제 호출도 정리 대상이다.

## 다음 의사결정 제안

1. **호환성 계약 확정:** 위 공개 API별로 유지·별칭/래퍼 제공·의도적 변경 중 하나를 정한다. 기존 추출기처럼 기능 자체가 사라진 경우는 포팅과 대체 가능성을 따로 평가한다.
2. **사용자 진입점 정리:** 루트 설치 문서와 실행 가능한 예제를 먼저 맞추고, CLI에 `pipeline` 외 명령이 필요한지 정한다.
3. **품질 게이트 정렬:** CI Pyright 오류를 해결한 뒤 Ruff/Pyright의 로컬·CI 버전 정책을 하나로 맞춘다. 그 후 필요한 호환성 테스트를 추가한다.
4. **출시 기준 결정:** 버전 번호, 지원 Python 범위(현재 판정 기준은 3.12+), 변경 안내와 배포 절차를 정한다.

이 보고서는 진단 결과와 선택지를 기록한 것이다. 기존 공개 API 전체의 성능·결과 동등성, 삭제된 예전 노트북·데이터의 사용자 영향, 모든 사전·모델 파일의 의미적 정확성은 아직 검증하지 않았다. 코드 수정과 문서 개정은 위 의사결정 후 별도 작업으로 진행한다.
