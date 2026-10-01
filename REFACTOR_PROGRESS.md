# 리팩토링 구현 기록

Base: `refactor-2026` (`13028c6`). Python 3.12 + uv.

## 완료 기준

- master 공개 import·호출·반환값과 저장 파일 호환성 복원
- LR-Graph·품사 추출·LR 태거 실행 결함 복구
- 버전 있는 JSON 리소스, JSONL 문장 결과, NPZ 행렬의 저장·재사용
- 기능별 CLI 및 대응 YAML 태스크, 설정 검증
- 문서 예제·품질 검사·휠/소스 배포 설치 검증

NER·MaxScoreTagger 새 알고리즘과 TF-IDF 추가는 범위 밖이다.

## 작업 방식

각 기능은 별도 worktree/feature 브랜치에서 검증하고 merge commit으로 병합한다.
이슈·PR·원격 push 없이 로컬 코드와 기록을 제공한다. Git 이력에서 각 merge commit의 두 번째 부모가 기능 커밋이다.

## 기능별 기록

| 브랜치 | 작업 | 검증 |
| --- | --- | --- |
| feature/project-guidance | Codex 지침과 기존 진단 보고서, 구현 기록 정리 | 지침·현재 base 확인 |
| feature/runtime-repair | LR-Graph 변환과 사전 어댑터 및 순서 의존 명사 판정 복구 | 전체 423개 통과 후 영향 예제 재검증 통과; 실행 회귀 8개와 병렬 회귀 8개 통과 |
| feature/public-api-compat | 기존 명사·단어·bigram 알고리즘과 토큰 반환 옵션, 모델 파일 및 공개 import 복원 | 전체 431개 통과; 최종 호환 회귀 8개 통과; 복원 파일 Ruff 검사 통과 |
| feature/artifact-io | 공통 JSON Artifact, 스트리밍 TXT·JSONL 문서 및 NPZ 행렬과 검증 가능한 메타데이터 저장 | 저장·호환 회귀 18개 통과; 새 저장 모듈 Ruff·Pyright 통과 |
| feature/cli-text | 정규화·토큰화·태깅·표제어·활용·한글 CLI와 공통 문장 처리 서비스 구현 | 기존 CLI와 새 기능 32개 테스트 통과; 새 코드 Ruff·Pyright 통과 |
