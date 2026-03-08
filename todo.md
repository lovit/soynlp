# soynlp 리팩토링 TODO

## 파일 이름 변경 (\_prefix 제거)

- [x] `hangle/_hangle.py` → `hangle/hangle.py`
- [x] `hangle/_distance.py` → `hangle/distance.py`
- [x] `lemmatizer/_lemmatizer.py` → `lemmatizer/lemmatizer.py`
- [x] `lemmatizer/_conjugation.py` → `lemmatizer/conjugation.py`
- [x] `normalizer/_normalizer.py` → `normalizer/normalizer.py` (병합)
- [x] `noun/_josa.py` → `noun/josa.py`
- [x] `pos/_adverb.py` → `pos/adverb.py`
- [x] `pos/_chat_pos.py` → `pos/chat_pos.py`
- [x] `pos/_news_pos.py` → `pos/news_pos.py`
- [x] `postagger/_dictionary.py` → `postagger/dictionary.py`
- [x] `postagger/_evaluator.py` → `postagger/evaluator.py`
- [x] `postagger/_lrtagger.py` → `postagger/lrtagger.py`
- [x] `postagger/_maxscore.py` → `postagger/maxscore.py`
- [x] `postagger/_pos_extractor.py` → `postagger/pos_extractor.py`
- [x] `postagger/_tagger.py` → `postagger/tagger.py`
- [x] `postagger/_template.py` → `postagger/template.py`
- [x] `predicator/_adjective_vs_verb.py` → `predicator/adjective_vs_verb.py`
- [x] `predicator/_eomi.py` → `predicator/eomi.py`
- [x] `predicator/_predicator.py` → `predicator/predicator.py`
- [x] `predicator/_stem.py` → `predicator/stem.py`
- [x] `tokenizer/_tokenizer_builder.py` → `tokenizer/tokenizer_builder.py`
- [x] `vectorizer/_vectorizer.py` → `vectorizer/vectorizer.py`
- [x] `vectorizer/_word_context.py` → `vectorizer/word_context.py`

## High Priority — 코드 품질

- [x] `NounScore` namedtuple → dataclass (slots=True) (`noun/lr.py`)
- [x] `Token` namedtuple → dataclass (slots=True) (`tokenizer/tokenizer.py`)
- [x] `ScoreTable`, `Table` namedtuple → dataclass (`postagger/lrtagger.py`)
- [x] `Predicator` namedtuple → dataclass (`predicator/predicator.py`)
- [x] broad `except Exception: continue` 패턴 구체화 (`pos/news_pos.py` 등)
- [x] private 함수 `_` prefix 누락: `noun/lr.py`의 `load_features()` → `_load_features()`

## Medium Priority — 코드 품질

- [x] `postagger/tagger.py`: `UnknowLRPostprocessor` 오타 → `UnknownLRPostprocessor`
- [x] `postagger/lrtagger.py`: `make_scoretable()` 파라미터 10개 → `_make_scoretable(Table)` 리팩토링
- [x] `pos/chat_pos.py`: 매직 값 (`"업"`, `"닿"`, `"땋"`) 상수화 및 주석 추가
- [x] `hangle/hangle.py`: 내부 상수 (`kor_begin` 등) `_` prefix 추가
- [x] `hangle/hangle.py`: `ConvolutionHangleEncoder.jamo_to_idx` 동적 생성으로 전환
- [x] `vectorizer/vectorizer.py`: 내부 상태 변수 관계 명확화

## Type Hints 추가

- [x] `pos/news_pos.py`: public 메서드 전체 타입 어노테이션
- [x] `pos/chat_pos.py`: public 메서드 전체 타입 어노테이션
- [x] `postagger/tagger.py`: 파라미터 및 반환값 타입 어노테이션
- [x] `postagger/lrtagger.py`: 파라미터 및 반환값 타입 어노테이션
- [x] `postagger/dictionary.py`: 반환값 타입 어노테이션
- [x] `noun/lr.py`: 복잡한 dict 타입 구체화
- [x] `core/lrgraph.py`: `_lr`, `_rl` 내부 타입 어노테이션
- [x] `vectorizer/vectorizer.py`: 메서드 타입 어노테이션

## 완료

모든 항목 완료 ✓
