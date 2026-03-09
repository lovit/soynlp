import pytest

from soynlp.vectorizer import BaseVectorizer, sent_to_word_contexts_matrix


class TestBaseVectorizer:
    @pytest.fixture
    def docs(self):
        return [
            "나는 학생 입니다",
            "나는 사과를 먹었다",
            "학생이 사과를 먹었다",
            "나는 나는 학생 학생",
        ]

    def test_fit(self, docs):
        vec = BaseVectorizer(min_tf=0, verbose=False)
        vec.fit(docs)
        assert vec.n_vocabs > 0
        assert "나는" in vec.vocabulary_

    def test_transform(self, docs):
        vec = BaseVectorizer(min_tf=0, verbose=False)
        vec.fit(docs)
        x = vec.transform(docs)
        assert x.shape[0] == len(docs)  # type: ignore[index]
        assert x.shape[1] == vec.n_vocabs  # type: ignore[index]

    def test_fit_transform(self, docs):
        vec = BaseVectorizer(min_tf=0, verbose=False)
        x = vec.fit_transform(docs)
        assert x.shape[0] == len(docs)  # type: ignore[index]

    def test_encode_decode_list(self, docs):
        vec = BaseVectorizer(min_tf=0, verbose=False)
        vec.fit(docs)
        encoded = vec.encode_a_doc_to_list(docs[0])
        decoded = vec.decode_from_list(encoded)
        for word in decoded:
            assert word in vec.vocabulary_

    def test_encode_decode_bow(self, docs):
        vec = BaseVectorizer(min_tf=0, verbose=False)
        vec.fit(docs)
        bow = vec.encode_a_doc_to_bow(docs[0])
        decoded = vec.decode_from_bow(bow)
        assert isinstance(decoded, dict)

    def test_vocabs(self, docs):
        vec = BaseVectorizer(min_tf=0, verbose=False)
        vec.fit(docs)
        vocabs = vec.vocabs()
        assert len(vocabs) == vec.n_vocabs

    def test_min_tf_filter(self, docs):
        vec = BaseVectorizer(min_tf=2, verbose=False)
        vec.fit(docs)
        # Words appearing only once should be filtered
        assert vec.n_vocabs > 0
        for word in vec.idx2vocab:
            # all remaining words should appear at least twice across docs
            count = sum(doc.split().count(word) for doc in docs)
            assert count >= 2


_VECTORIZER_DOCS = [
    "나는 학생 입니다",
    "나는 사과를 먹었다",
    "학생이 사과를 먹었다",
    "나는 나는 학생 학생",
] * 200


class TestBaseVectorizerMultiprocessing:
    def test_fit_multi_equals_single(self):
        """n_workers=4로 fit한 vocabulary가 단일 프로세스와 동일하다."""
        vec_single = BaseVectorizer(min_tf=0, verbose=False)
        vec_single.fit(_VECTORIZER_DOCS, n_workers=1)

        vec_multi = BaseVectorizer(min_tf=0, verbose=False)
        vec_multi.fit(_VECTORIZER_DOCS, n_workers=4)

        assert vec_single.vocabulary_ == vec_multi.vocabulary_

    def test_fit_transform_multi(self):
        """n_workers=4 fit_transform이 에러 없이 동작하고 같은 모양의 행렬을 반환한다."""
        vec_single = BaseVectorizer(min_tf=0, verbose=False)
        x_single = vec_single.fit_transform(_VECTORIZER_DOCS)

        vec_multi = BaseVectorizer(min_tf=0, verbose=False)
        x_multi = vec_multi.fit_transform(_VECTORIZER_DOCS, n_workers=4)

        assert x_single.shape == x_multi.shape  # type: ignore[index]


_WORD_CONTEXT_DOCS = ["a b c d e f"] * 200


class TestSentToWordContextsMatrix:
    def test_basic(self):
        sents = ["a b c d e"] * 20
        x, idx2vocab = sent_to_word_contexts_matrix(sents, windows=2, min_tf=1, verbose=False)
        assert x.shape[0] == x.shape[1]  # type: ignore[index]  # square matrix
        assert len(idx2vocab) == x.shape[0]  # type: ignore[index]

    def test_min_tf_filter(self):
        sents = ["a b c"] * 20 + ["x y z"]
        x, idx2vocab = sent_to_word_contexts_matrix(sents, windows=1, min_tf=10, verbose=False)
        # x, y, z appear only once, should be filtered
        assert "x" not in idx2vocab
        assert "a" in idx2vocab

    def test_dynamic_weight(self):
        sents = ["a b c d e"] * 20
        x1, _ = sent_to_word_contexts_matrix(sents, windows=2, min_tf=1, dynamic_weight=False, verbose=False)
        x2, _ = sent_to_word_contexts_matrix(sents, windows=2, min_tf=1, dynamic_weight=True, verbose=False)
        # With dynamic weight, some values should be smaller
        assert x1.sum() >= x2.sum()

    def test_n_workers_equals_single(self):
        """n_workers=2로 실행해도 어휘와 행렬 합계가 단일 프로세스와 동일하다."""
        x1, idx1 = sent_to_word_contexts_matrix(_WORD_CONTEXT_DOCS, windows=2, min_tf=1, verbose=False, n_workers=1)
        x2, idx2 = sent_to_word_contexts_matrix(_WORD_CONTEXT_DOCS, windows=2, min_tf=1, verbose=False, n_workers=2)
        assert set(idx1) == set(idx2)
        assert abs(x1.sum() - x2.sum()) < 1e-6  # type: ignore[operator]

    def test_n_workers_lambda_fallback(self):
        """lambda tokenizer는 pickle 불가이므로 n_workers가 무시되어 정상 동작한다."""
        sents = ["a b c d e"] * 20
        x, idx2vocab = sent_to_word_contexts_matrix(
            sents, windows=2, min_tf=1, tokenizer=lambda x: x.split(), verbose=False, n_workers=2
        )
        assert x.shape[0] == x.shape[1]  # type: ignore[index]
