# soynlp 리팩토링 TODO

## 파일 이름 변경 (\_prefix 제거)

- [ ] `hangle/_hangle.py` → `hangle/hangle.py`
- [ ] `hangle/_distance.py` → `hangle/distance.py`
- [ ] `lemmatizer/_lemmatizer.py` → `lemmatizer/lemmatizer.py`
- [ ] `lemmatizer/_conjugation.py` → `lemmatizer/conjugation.py`
- [ ] `normalizer/_normalizer.py` → `normalizer/normalizer.py`
- [ ] `noun/_josa.py` → `noun/josa.py`
- [ ] `pos/_adverb.py` → `pos/adverb.py`
- [ ] `pos/_chat_pos.py` → `pos/chat_pos.py`
- [ ] `pos/_news_pos.py` → `pos/news_pos.py`
- [ ] `postagger/_dictionary.py` → `postagger/dictionary.py`
- [ ] `postagger/_evaluator.py` → `postagger/evaluator.py`
- [ ] `postagger/_lrtagger.py` → `postagger/lrtagger.py`
- [ ] `postagger/_maxscore.py` → `postagger/maxscore.py`
- [ ] `postagger/_pos_extractor.py` → `postagger/pos_extractor.py`
- [ ] `postagger/_tagger.py` → `postagger/tagger.py`
- [ ] `postagger/_template.py` → `postagger/template.py`
- [ ] `predicator/_adjective_vs_verb.py` → `predicator/adjective_vs_verb.py`
- [ ] `predicator/_eomi.py` → `predicator/eomi.py`
- [ ] `predicator/_predicator.py` → `predicator/predicator.py`
- [ ] `predicator/_stem.py` → `predicator/stem.py`
- [ ] `tokenizer/_tokenizer_builder.py` → `tokenizer/tokenizer_builder.py`
- [ ] `vectorizer/_vectorizer.py` → `vectorizer/vectorizer.py`
- [ ] `vectorizer/_word_context.py` → `vectorizer/word_context.py`

## High Priority — 코드 품질

- [ ] `NounScore` namedtuple → dataclass (slots=True) (`noun/lr.py`)
- [ ] `Token` namedtuple → dataclass (slots=True) (`tokenizer/tokenizer.py`)
- [ ] `ScoreTable`, `Table` namedtuple → dataclass (`postagger/_lrtagger.py`)
- [ ] `Predicator` namedtuple → dataclass (`predicator/_predicator.py`)
- [ ] broad `except Exception: continue` 패턴 구체화 (`pos/_news_pos.py` 등)
- [ ] private 함수 `_` prefix 누락: `noun/lr.py`의 `load_features()` → `_load_features()`

## Medium Priority — 코드 품질

- [ ] `postagger/_tagger.py`: `UnknowLRPostprocessor` 오타 → `UnknownLRPostprocessor`
- [ ] `postagger/_lrtagger.py`: `make_scoretable()` 파라미터 10개 → 리팩토링
- [ ] `pos/_chat_pos.py`: 매직 값 (`"업"`, `"닿"`, `"땋"`) 상수화 및 주석 추가
- [ ] `hangle/_hangle.py`: 내부 상수 (`kor_begin` 등) `_` prefix 추가
- [ ] `hangle/_hangle.py`: `ConvolutionHangleEncoder.jamo_to_idx` 동적 생성으로 전환
- [ ] `vectorizer/_vectorizer.py`: 내부 상태 변수 관계 명확화

## Type Hints 추가

- [ ] `pos/_news_pos.py`: public 메서드 전체 타입 어노테이션
- [ ] `pos/_chat_pos.py`: public 메서드 전체 타입 어노테이션
- [ ] `postagger/_tagger.py`: 파라미터 및 반환값 타입 어노테이션
- [ ] `postagger/_lrtagger.py`: 파라미터 및 반환값 타입 어노테이션
- [ ] `postagger/_dictionary.py`: 반환값 타입 어노테이션
- [ ] `noun/lr.py`: 복잡한 dict 타입 구체화
- [ ] `core/lrgraph.py`: `_lr`, `_rl` 내부 타입 어노테이션
- [ ] `vectorizer/_vectorizer.py`: 메서드 타입 어노테이션
