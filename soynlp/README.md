# soynlp 기능 가이드

soynlp는 비지도 학습 기반 한국어 자연어 처리 패키지입니다.

---

## 목차

- [hangle](#hangle--한글-자모-처리)
- [noun](#noun--명사-추출)
- [word](#word--단어-추출)
- [pos](#pos--품사-추출)
- [postagger](#postagger--품사-태깅)
- [predicator](#predicator--술어-추출)
- [lemmatizer](#lemmatizer--표제화)
- [tokenizer](#tokenizer--토크나이징)
- [normalizer](#normalizer--텍스트-정규화)
- [vectorizer](#vectorizer--벡터화)
- [core](#core--lr-graph)
- [pipeline](#pipeline--파이프라인)

---

## hangle — 한글 자모 처리

한글 음절/자모 분해·합성, 문자 타입 판별, 문자열 거리 계산을 제공합니다.

```python
from soynlp.hangle import decompose, compose, character_is_korean

cho, jung, jong = decompose('한')  # ('ㅎ', 'ㅏ', 'ㄴ')
char = compose('ㅎ', 'ㅏ', 'ㄴ')   # '한'
character_is_korean('한')           # True
```

**주요 API**

| 함수/상수                                        | 설명                           |
| ------------------------------------------------ | ------------------------------ |
| `decompose(char)`                                | 음절 → (초성, 중성, 종성) 분해 |
| `compose(cho, jung, jong)`                       | 자모 → 음절 합성               |
| `character_is_korean(c)`                         | 한글 문자 판정                 |
| `character_is_complete_korean(c)`                | 완성 음절 판정                 |
| `character_is_jaum(c)` / `character_is_moum(c)`  | 자음/모음 판정                 |
| `levenshtein(a, b)`                              | 편집 거리                      |
| `jamo_levenshtein(a, b)`                         | 자모 단위 편집 거리            |
| `chosung_list`, `jungsung_list`, `jongsung_list` | 자모 상수 리스트               |

---

## noun — 명사 추출

LR-Graph 기반 비지도 학습으로 코퍼스에서 명사를 추출합니다.

```python
from soynlp.noun import LRNounExtractor

extractor = LRNounExtractor(verbose=True)
nouns = extractor.extract(sentences)
# {'아이오아이': NounScore(frequency=127, score=0.98), ...}

tokenizer = extractor.get_noun_tokenizer()
tokens = tokenizer.tokenize('아이오아이는 아이돌 그룹입니다')
```

**주요 API**

| 클래스/타입       | 설명                            |
| ----------------- | ------------------------------- |
| `LRNounExtractor` | LR-Graph 기반 명사 추출기       |
| `NounScore`       | `(frequency, score)` namedtuple |

---

## word — 단어 추출

응집도(Cohesion), 분지 엔트로피(Branching Entropy), 접근 다양성(Accessor Variety)으로 단어 경계를 판별합니다.

```python
from soynlp.word import WordExtractor, pmi

extractor = WordExtractor()
extractor.train(sentences)
words = extractor.extract(min_frequency=5)

from soynlp.word import BigramExtractor
bigram = BigramExtractor()
bigram.train(sentences)
scores = bigram.extract(score_type='pmi')
```

**주요 API**

| 클래스/타입        | 설명                                      |
| ------------------ | ----------------------------------------- |
| `WordExtractor`    | 단어 추출기 (cohesion, branching entropy) |
| `BigramExtractor`  | 바이그램 추출기 (frequency, pmi, mikolov) |
| `pmi()`            | Pointwise Mutual Information 계산         |
| `CohesionScore`    | 응집도 점수 dataclass                     |
| `BranchingEntropy` | 분지 엔트로피 dataclass                   |

---

## pos — 품사 추출

코퍼스에서 도메인별로 명사·동사·형용사·부사를 자동으로 추출합니다.

```python
from soynlp.pos import NewsPOSExtractor, ChatPOSExtractor

# 뉴스 도메인
extractor = NewsPOSExtractor(verbose=True)
result = extractor.train_extract(sentences, min_noun_score=0.3)
# result.nouns, result.verbs, result.adjectives, result.adverbs

# 채팅 도메인 (축약형·신조어 처리)
chat_extractor = ChatPOSExtractor(verbose=True)
result = chat_extractor.train_extract(sentences)
```

**주요 API**

| 클래스                   | 설명                                              |
| ------------------------ | ------------------------------------------------- |
| `NewsPOSExtractor`       | 뉴스 도메인 품사 추출기                           |
| `ChatPOSExtractor`       | 채팅 도메인 품사 추출기 (`NewsPOSExtractor` 상속) |
| `load_default_adverbs()` | 기본 부사 사전 로드                               |

---

## postagger — 품사 태깅

POS 사전을 이용해 각 어절을 L(어근) + R(어미/조사)로 분해하고 태그를 부여합니다.

```python
from soynlp.postagger import LRMaxScoreTagger, Dictionary

tagger = LRMaxScoreTagger(sents=sentences, dictionary_word_mincount=3)
tagged = tagger.pos('자연어 처리를 공부합니다')
# [('자연어', 'Noun'), ('처리를', ...), ...]
```

**주요 API**

| 클래스             | 설명                                      |
| ------------------ | ----------------------------------------- |
| `SimpleTagger`     | 사전 기반 단순 태거                       |
| `LRMaxScoreTagger` | LR-Graph + coherence/droprate 스코링 태거 |
| `MaxScoreTagger`   | 최고 점수 선택 태거                       |
| `Dictionary`       | POS 사전 (JSON 저장/로드)                 |
| `POSExtractor`     | 명사+용언 통합 추출기                     |

---

## predicator — 술어 추출

코퍼스에서 동사·형용사와 어미(ending)를 비지도 방식으로 추출합니다.

```python
from soynlp.predicator import PredicatorExtractor

extractor = PredicatorExtractor(
    nouns={'명사': 100},
    adjective_stems={'예쁘': 50},
)
verbs, adjectives = extractor.extract(sentences)
# {'하다': Predicator(frequency=500, lemma={'하'}), ...}
```

**주요 API**

| 클래스/함수                 | 설명                            |
| --------------------------- | ------------------------------- |
| `PredicatorExtractor`       | 술어 추출기                     |
| `EomiExtractor`             | 어미 추출기                     |
| `StemExtractor`             | 어간 추출기                     |
| `conjugate_as_present()`    | 현재형 활용형 생성              |
| `conjugate_as_imperative()` | 명령형 활용형 생성              |
| `Predicator`                | `(frequency, lemma)` namedtuple |

---

## lemmatizer — 표제화

활용형 단어에서 기본형(lemma)을 복원합니다.

```python
from soynlp.lemmatizer import Lemmatizer

lemmatizer = Lemmatizer(
    stems={'예쁘다', '먹다'},
    endings={'어', '었다', '는', '고'}
)
results = lemmatizer.lemmatize('예뻤다')  # {('예쁘다', '었다')}
candidates = lemmatizer.candidates('먹었다')
```

**주요 API**

| 함수/클래스               | 설명             |
| ------------------------- | ---------------- |
| `Lemmatizer`              | 표제화기         |
| `lemma_candidate(word)`   | 표제형 후보 추출 |
| `conjugate(stem, ending)` | 활용형 생성      |

---

## tokenizer — 토크나이징

정규식·점수 기반·명사 매칭 등 다양한 토크나이저를 제공합니다.

```python
from soynlp.tokenizer import RegexTokenizer, LTokenizer, MaxScoreTokenizer

# 정규식 기반
tokenizer = RegexTokenizer()
tokens = tokenizer.tokenize('한글입니다123abc')

# L 기반 (명사 사전 필요)
ltokenizer = LTokenizer(scores={'아이오아이': 0.98, '아이돌': 0.85})
tokens = ltokenizer.tokenize('아이오아이는 아이돌입니다')

# 최대 점수 기반
ms_tokenizer = MaxScoreTokenizer(scores={'자연어': 0.9, '처리': 0.8})
tokens = ms_tokenizer.tokenize('자연어처리')
```

**주요 API**

| 클래스                 | 설명                                                      |
| ---------------------- | --------------------------------------------------------- |
| `RegexTokenizer`       | 정규식 기반 토크나이저                                    |
| `LTokenizer`           | L(좌측) 기반 토크나이저                                   |
| `MaxScoreTokenizer`    | 최대 점수 기반 토크나이저                                 |
| `NounMatchTokenizer`   | 명사 매칭 토크나이저                                      |
| `EojeolPatternTrainer` | 어절 패턴 학습기                                          |
| `Token`                | `(word, begin, end, score, length, eojeol_id)` namedtuple |

---

## normalizer — 텍스트 정규화

반복 문자, 이모티콘, 특수문자 등을 정규화합니다.

```python
from soynlp.normalizer import normalize, repeat_normalize, emoticon_normalize

normalize('ㅋㅋㅋㅋㅋㅋ진짜!!!')           # 'ㅋㅋ진짜!'
repeat_normalize('와아아아아', n_repeats=2)  # '와아아'
emoticon_normalize('ㅠㅠㅠㅠ', n_repeats=2) # 'ㅠㅠ'

from soynlp.normalizer import only_hangle, remove_doublespace
only_hangle('한글 abc 123')  # '한글  '
remove_doublespace('한글  공백')  # '한글 공백'
```

**주요 API**

| 함수/클래스                           | 설명                                                         |
| ------------------------------------- | ------------------------------------------------------------ |
| `normalize(text, ...)`                | 통합 정규화 (alphabet/number/punctuation/symbol/repeat 옵션) |
| `repeat_normalize(text, n_repeats)`   | 반복 문자 정규화                                             |
| `emoticon_normalize(text, n_repeats)` | 이모티콘 정규화                                              |
| `only_hangle(text)`                   | 한글만 유지                                                  |
| `only_hangle_number(text)`            | 한글·숫자만 유지                                             |
| `remove_doublespace(text)`            | 중복 공백 제거                                               |
| `TextNormalizer`                      | 컴포넌트 조합형 정규화 클래스                                |

---

## vectorizer — 벡터화

문서를 TF-IDF 희소 행렬로 변환하거나 단어-문맥 행렬을 생성합니다.

```python
from soynlp.vectorizer import BaseVectorizer, sent_to_word_contexts_matrix

vectorizer = BaseVectorizer(min_tf=2, max_df=0.9)
X = vectorizer.fit_transform(documents)  # scipy.sparse.csr_matrix

matrix, idx2vocab = sent_to_word_contexts_matrix(
    sentences, windows=3, min_tf=5
)
```

**주요 API**

| 클래스/함수                      | 설명                |
| -------------------------------- | ------------------- |
| `BaseVectorizer`                 | TF-IDF 벡터화기     |
| `sent_to_word_contexts_matrix()` | 단어-문맥 행렬 생성 |

---

## core — LR-Graph

한국어 어절을 L(좌측)·R(우측)로 분해하는 이분 그래프 자료구조입니다.

```python
from soynlp.core import LRGraph, corpus_to_lrgraph

lrgraph = LRGraph.from_sents(['자연어 처리는 어렵습니다'])
lrgraph.get_r('자연어', topk=5)  # [('처리는', 1), ...]
lrgraph.get_l('처리는', topk=5)  # [('자연어', 1), ...]

lrgraph2 = corpus_to_lrgraph(sentences)
```

**주요 API**

| 메서드                      | 설명                         |
| --------------------------- | ---------------------------- |
| `LRGraph.from_sents(sents)` | 문장 리스트에서 LRGraph 생성 |
| `add_eojeol(eojeol, freq)`  | 어절 추가                    |
| `get_r(L, topk)`            | L에 연결된 R 목록 조회       |
| `get_l(R, topk)`            | R에 연결된 L 목록 조회       |
| `save(path)` / `load(path)` | 저장/로드                    |
| `corpus_to_lrgraph(sents)`  | 코퍼스 → LRGraph 변환        |

---

## pipeline — 파이프라인

YAML 설정 파일로 여러 작업을 체인처럼 연결해 실행합니다.

```yaml
# pipeline.yaml
pipeline:
  - name: ReadText
    args:
      filepath: input.txt
  - name: Normalize
    args:
      remove_repeat: 2
  - name: ExtractNoun
    args:
      min_noun_score: 0.3
  - name: WriteJson
    args:
      filepath: output.jsonl
```

```python
from soynlp.pipeline import Pipeline

Pipeline.run('pipeline.yaml')
```

```bash
uv run soynlp pipeline -c pipeline.yaml
```

**등록된 Task 목록**

| Task 이름     | 설명             |
| ------------- | ---------------- |
| `ReadText`    | 텍스트 파일 읽기 |
| `ReadJson`    | JSONL 파일 읽기  |
| `WriteText`   | 텍스트 파일 쓰기 |
| `WriteJson`   | JSONL 파일 쓰기  |
| `Normalize`   | 텍스트 정규화    |
| `Tokenize`    | 토크나이징       |
| `ExtractNoun` | 명사 추출        |
| `ExtractWord` | 단어 추출        |
| `Dummy`       | 테스트용 no-op   |

---

## 모듈 간 의존 관계

```
normalizer ──→ hangle
noun       ──→ tokenizer, core
pos        ──→ noun, predicator, tokenizer, lemmatizer
postagger  ──→ tokenizer
predicator ──→ hangle, lemmatizer, normalizer, core
lemmatizer ──→ hangle
pipeline   ──→ normalizer, tokenizer, noun, word
```

## 일반적인 사용 패턴

**명사 추출 → 토크나이징**

```python
extractor = LRNounExtractor()
nouns = extractor.extract(sentences)
tokenizer = extractor.get_noun_tokenizer()
tokens = tokenizer.tokenize(text)
```

**품사 추출 → 태깅**

```python
pos_extractor = NewsPOSExtractor()
pos_dict = pos_extractor.train_extract(sentences)
# pos_dict를 postagger에 활용
```
