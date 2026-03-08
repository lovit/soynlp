# Branch Rules

## 브랜치 생성 원칙

- GitHub 이슈 번호를 기반으로 브랜치를 생성한다.
- 형식: `feature/{이슈번호}` (예: `feature/1`, `feature/42`)
- 이슈와 무관한 작업은 `feature/짧은-설명` 형식 사용 가능.

## 작업 흐름

1. GitHub에 이슈 생성
2. 해당 이슈 번호로 worktree 생성: `git worktree add feature/{번호} -b feature/{번호}`
3. **해당 worktree 디렉터리에서 Claude Code 실행**
4. 파일 편집 및 커밋
5. 작업 완료 후 PR 생성

## Claude Code와 worktree

- Claude Code는 실행된 디렉터리를 기준으로 파일을 읽고 편집한다.
- **반드시 feature worktree 디렉터리 안에서 `claude`를 실행해야 한다.**
- main worktree에서 실행하면 편집이 main 브랜치에 적용된다.

```bash
# 이슈 #1 작업 시작
git worktree add feature/1 -b feature/1
cd feature/1
claude  # ← 여기서 실행해야 함
```

## 예시

```bash
# 이슈 #42에 대한 작업
git worktree add feature/42 -b feature/42
cd feature/42
claude
```
