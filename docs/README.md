# 문서 사이트 개발

GitHub Pages 소스는 지정한 브랜치의 `/docs`이며 Jekyll로 빌드한다.
사이트 주소는 <https://lovit.github.io/soynlp/>이다.
이 파일은 사이트 빌드에서 제외한다.

## 로컬 빌드와 미리보기

Ruby 3.3.4와 Bundler를 사용한다. macOS 시스템 Ruby는 변경하지 않는다.
`mise`를 사용한다면 저장소 루트에서 다음과 같이 실행할 수 있다.

```sh
MISE_RUBY_COMPILE=false mise install ruby@3.3.4
cd docs
mise exec ruby@3.3.4 -- bundle config set --local path vendor/bundle
mise exec ruby@3.3.4 -- bundle install
mise exec ruby@3.3.4 -- bundle exec jekyll build --safe --strict_front_matter
mise exec ruby@3.3.4 -- bundle exec jekyll serve --safe --host 127.0.0.1
```

브라우저에서 `http://127.0.0.1:4000/soynlp/`를 연다.
운영과 같은 `/soynlp` 경로에서 메뉴·본문 링크·CSS를 확인한다.
Ruby와 Gem 의존성은 패키지의 Python 환경과 별개이다.

## 문서 작성

- 한글 Markdown에 title, description, section, permalink front matter를 작성한다.
- 탐색 메뉴는 `_data/navigation.yml`, 공통 화면은 `_layouts/default.html`에서 관리한다.
- 문서 사이 링크는 Markdown 파일 경로를 사용한다. Jekyll이 페이지 URL로 변환한다.
- 레이아웃·HTML에서 사이트 경로는 `relative_url` 필터를 사용한다.
- 저장소 파일 링크는 `site.github.build_revision`을 사용해 배포 커밋과 연결한다.
- `_site`, `vendor/bundle`, `.bundle`과 캐시는 커밋하지 않는다. Gemfile.lock은 추적한다.

## 이번 검토와 병합

`docs/soynlp-doc`의 `/docs`로 Pages 소스를 전환해 사용자 검토를 진행한다.
검토 기간에는 기존 공개 URL에 검토본이 표시된다.
사용자 검토 완료 후 `refactor-2026`에 `--no-ff`로 병합하고 push한 뒤
Pages 소스를 `refactor-2026`의 `/docs`로 복구한다.
브랜치는 유지하며 이슈·PR은 생성하지 않는다.
