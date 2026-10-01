# refactor-2026 구현 결과

## 제공 범위

soynlp의 목적은 코퍼스 통계로 한국어의 단어·명사·문맥을 추출하고,
정규화·토큰화·표제어 분석·품사 처리·벡터화를 제공하는 것이다.
이번 구현은 master의 공개 사용법을 유지하면서 Python, CLI, YAML에서 기능을 사용할 수 있도록 정리했다.

- `.codex/WORKFLOW.md`와 루트 `AGENTS.md`에 작업 지침을 정리했다.
- LR-Graph 변환, 사전 어댑터, LR 태거, 뉴스 품사 추출과 병렬 명사 추출의 실행 결함을 수정했다.
- master의 공개 import, 생성자·메서드 호출, tuple 반환값과 기존 WordExtractor 모델 읽기를 복원했다.
  기존 알고리즘은 `soynlp/_legacy`에 보관하고 공개 모듈의 어댑터에서 연결한다.
- 문장 변환·코퍼스 추출·사전·패턴 학습·벡터화·행렬·LR-Graph 기능을 CLI로 제공한다.
  전체 명령과 입출력 예시는 [CLI 안내](docs/CLI.md)에 있다.
- Python 서비스 함수를 CLI와 YAML 태스크에서 공유한다.
  파이프라인의 기존 실행 옵션을 유지하고 태스크 이름·인자·의존 키 검증을 추가했다.
- JSONL 문서, 버전 있는 JSON Artifact, 어휘·shape·checksum을 포함한 NPZ 저장 형식을 제공한다.
  추출 결과를 토큰화·태깅·사전·행렬 작업에서 재사용할 수 있다.
- 도구 버전과 `uv.lock`을 고정하고 Linux·macOS CI 및 배포 설치 검증을 추가했다.

기존 사용법과 현재 사용법의 차이는 [호환성 안내](docs/COMPATIBILITY.md)에 기록했다.
실행 예제는 [Python 예제](examples/refactor_demo.py)와 [YAML 예제](examples/text_pipeline.yaml)이다.

## 최종 검증

Python 3.12, macOS에서 다음 결과를 확인했다.

| 검증 | 결과 |
| --- | --- |
| `uv run --python 3.12 pytest -q` | 505개 통과, 202개 경고, 160.28초 |
| `uv run --python 3.12 ruff check .` | 통과 |
| `uv run --python 3.12 ruff format --check .` | 173개 파일 통과 |
| `uv run --python 3.12 pyright` | 오류·경고 0개 |
| `examples/refactor_demo.py` | 명사·단어·bigram·품사·리소스 재사용·PMI/SVD 예제 통과 |
| `examples/text_pipeline.yaml` | 검증·실행 통과, 문서 ID·메타데이터와 토큰 위치 유지 |
| `docs/CLI.md`의 shell 예제 | 임시 디렉터리에서 23개 명령 통과 |
| `uv build` | wheel 및 sdist 생성 |
| `scripts/verify_distribution.py` | wheel·sdist 각각 격리 환경 설치, 내장 사전·모델 및 CLI 확인 |

테스트 경고에는 기존 의존성의 deprecation 경고와 master 정규식의 FutureWarning이 포함된다.
호환성 검증은 master 결과 fixture, 기존 pickle 모델, 공개 import·호출·반환 타입과 실행 회귀를 포함한다.
실제 코퍼스의 추출 품질과 임계값은 사용 데이터에 맞춰 평가해야 한다.
Linux·macOS CI 설정은 작성했으며 원격 CI 실행 결과는 아직 없다.

## 기능 브랜치와 merge commit

기준 커밋은 `13028c6`이다. 기능별 별도 worktree에서 구현·검증·커밋한 뒤
`refactor-2026`에 `--no-ff`로 병합했다. 모든 기능 브랜치를 로컬에 유지했다.
이슈·PR 생성과 원격 push는 하지 않았다.

| 기능 브랜치 | 기능 커밋 | 병합 커밋 |
| --- | --- | --- |
| `feature/project-guidance` | `623abda` | `895f359` |
| `feature/runtime-repair` | `8bf10f3` | `a940b9b` |
| `feature/public-api-compat` | `d1e8411` | `16a4d1b` |
| `feature/artifact-io` | `b9cf192` | `7629560` |
| `feature/cli-text` | `5d8486d` | `240ce40` |
| `feature/cli-corpus` | `eaf7e2c` | `f7d1bc8` |
| `feature/cli-matrix` | `43e837a` | `c637579` |
| `feature/pipeline-services` | `8f68238` | `d4d7ee0` |
| `feature/quality-packaging` | `570857e` | `c40cb23` |
| `feature/compat-positional` | `55c22c2` | `af49fc5` |
| `feature/cli-resource-reuse` | `2d7eccc` | `45e585a` |
| `feature/noun-score-compat` | `13d788a` | `bc8db4c` |

`feature/usage-docs`는 이 결과 기록, 사용 문서와 실행 예제를 별도 커밋으로 제공한다.
자기 자신의 커밋 ID를 파일에 포함하지 않으며 아래 명령으로 마지막 문서 병합까지 확인할 수 있다.

```sh
git log --first-parent --merges --format='%h %p %s' origin/refactor-2026..refactor-2026
git branch --list 'feature/*'
```

각 기능의 변경 내용과 당시 검증은 [구현 기록](REFACTOR_PROGRESS.md)에 있다.
[기존 진단](REFACTOR_AUDIT.md)은 구현 전 상태를 다루므로 현재 상태 판단에는 이 문서를 사용한다.

## 이후 결정할 일

- 이 결과를 검토한 뒤 기능 커밋을 이슈·PR로 이식할지 결정한다.
- 원격 반영 후 CI 결과를 확인하고 배포 버전과 릴리스 방식을 결정한다.
- 사용 코퍼스에 맞춰 추출 임계값과 성능을 평가한다.

NER, 새 MaxScoreTagger 알고리즘과 TF-IDF 추가는 이번 범위에 포함하지 않았다.
`KoreanPOSTagger.train()`은 기존 추가 사전 등록 API이므로 코퍼스를 학습하는 `train tagger` 명령을 만들지 않았다.
새 리소스는 JSON을 사용하며 기존 WordExtractor의 pickle save/load는 호환 경로로 유지한다.
NPZ와 JSON 메타데이터는 각각 임시 파일을 거쳐 저장한다. 두 파일 교체가 하나의 트랜잭션은 아니며
중단으로 파일이 어긋나면 checksum 검증으로 읽기를 거부한다.
