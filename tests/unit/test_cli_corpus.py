import json

import pytest

from soynlp.artifacts import Artifact, load_artifact, save_artifact
from soynlp.cli import main
from soynlp.services.corpus import ReplayCorpus
from soynlp.utils import EojeolCounter


@pytest.fixture
def corpus(tmp_path):
    path = tmp_path / "corpus.txt"
    path.write_text(
        "아이오아이가 아이돌입니다\n아이오아이는 아이돌 그룹입니다\n한국어를 공부합니다\n한국어는 재미있습니다\n" * 20
    )
    return str(path)


@pytest.fixture
def seeds(tmp_path):
    path = tmp_path / "dictionary.json"
    save_artifact(
        Artifact("dictionary", {"Noun": ["한국어"], "Verb": ["먹", "공부하"], "Eomi": ["다", "었다"], "Josa": ["는", "를"]}),
        path,
    )
    return str(path)


def test_count_and_dictionary_reuse(corpus, tmp_path, capsys):
    counts = tmp_path / "counts.json"
    dictionary = tmp_path / "dictionary.json"
    main(["corpus", "count-eojeols", "--input", corpus, "-o", str(counts)])
    assert load_artifact(counts).data == dict(EojeolCounter(open(corpus).read().splitlines()).items())
    main(["dictionary", "build", "--resource", str(counts), "-o", str(dictionary)])
    main(["dictionary", "inspect", "--resource", str(dictionary)])
    assert json.loads(capsys.readouterr().out)["data"]["entries"] > 0
    main(["tokenize", "--method", "noun-match", "--resource", str(dictionary), "--text", "한국어를"])
    assert json.loads(capsys.readouterr().out)["tokens"]


def test_word_resource_feeds_tokenizer(corpus, tmp_path, capsys):
    words = tmp_path / "words.json"
    main(
        [
            "extract",
            "words",
            "--input",
            corpus,
            "--parameters",
            '{"constructor":{"min_frequency":2,"verbose_points":0}}',
            "-o",
            str(words),
        ]
    )
    result = load_artifact(words, expected_kind="words")
    assert "cohesion_forward" in result.data["한국어"]
    main(["tokenize", "--method", "l", "--resource", str(words), "--text", "한국어를 공부합니다"])
    assert json.loads(capsys.readouterr().out)["tokens"][0] == "한국어"


@pytest.mark.parametrize("method", ["lr", "lr-v1", "lr-v2", "news"])
def test_noun_methods(method, corpus, capsys):
    main(["extract", "nouns", "--method", method, "--input", corpus])
    assert isinstance(json.loads(capsys.readouterr().out)["data"], dict)


@pytest.mark.parametrize("method", ["frequency", "pmi", "mikolov"])
def test_bigram_methods(method, corpus, capsys):
    main(["extract", "bigrams", "--method", method, "--input", corpus])
    result = json.loads(capsys.readouterr().out)
    assert result["data"] and "frequency" in result["data"][0]


@pytest.mark.parametrize("kind", ["eomis", "stems", "features", "predicators"])
def test_grammar_extraction(kind, corpus, seeds, tmp_path, capsys):
    resource = tmp_path / "nouns.json"
    save_artifact(Artifact("nouns", {"한국어": {"frequency": 40, "score": 1.0}}), resource)
    main(["extract", kind, "--input", corpus, "--dictionary", seeds, "--resource", str(resource)])
    result = json.loads(capsys.readouterr().out)
    assert result["kind"] == kind and isinstance(result["data"], dict)


@pytest.mark.parametrize("method", ["news", "chat", "combined"])
def test_pos_methods(method, corpus, capsys):
    main(["extract", "pos", "--method", method, "--input", corpus])
    assert isinstance(json.loads(capsys.readouterr().out)["data"], dict)


def test_patterns_and_graph_reuse(corpus, tmp_path, capsys):
    main(["train", "patterns", "--input", corpus])
    assert json.loads(capsys.readouterr().out)["data"]["lrgraph"]
    graph = tmp_path / "lrgraph.json"
    main(["lrgraph", "build", "--input", corpus, "-o", str(graph)])
    main(["lrgraph", "neighbors", "--resource", str(graph), "--word", "한국어"])
    assert ["를", 20] in json.loads(capsys.readouterr().out)["data"]


def test_merge_export_and_conversion(corpus, seeds, tmp_path, capsys):
    merged = tmp_path / "merged.json"
    main(["dictionary", "merge", seeds, seeds, "-o", str(merged)])
    assert load_artifact(merged).data["Verb"] == ["공부하", "먹"]
    main(["dictionary", "export", "--resource", str(merged)])
    assert "Verb\t먹\n" in capsys.readouterr().out
    main(["corpus", "convert", "--input", corpus])
    assert len(capsys.readouterr().out.splitlines()) == 80


def test_replay_corpus_keeps_newlines():
    with ReplayCorpus(iter([{"text": "한국어\n공부"}])) as corpus:
        assert len(corpus) == 1
        assert list(corpus) == list(corpus) == ["한국어\n공부"]


def test_pattern_resource_reused_by_tokenizer_and_tagger(corpus, tmp_path, capsys):
    resource = tmp_path / "patterns.json"
    main(
        [
            "train",
            "patterns",
            "--input",
            corpus,
            "--parameters",
            '{"constructor":{"min_frequency":2},"train":{"n_workers":2},"hits":{"max_iter":2}}',
            "-o",
            str(resource),
        ]
    )
    artifact = load_artifact(resource)
    assert artifact.data["left_scores"] and artifact.data["right_scores"]
    for command, method, key in [("tokenize", "max-lr-score", "tokens"), ("tag", "lr-max-score", "tags")]:
        main([command, "--method", method, "--resource", str(resource), "--text", "한국어를 공부합니다"])
        assert json.loads(capsys.readouterr().out)[key]


def test_corpus_word_scores_can_be_reused(corpus, tmp_path, capsys):
    resource = tmp_path / "words.json"
    main(
        [
            "extract",
            "words",
            "--method",
            "corpus",
            "--input",
            corpus,
            "--parameters",
            '{"extract":{"min_frequency":2}}',
            "-o",
            str(resource),
        ]
    )
    main(["tokenize", "--method", "l", "--resource", str(resource), "--text", "한국어를 공부합니다"])
    assert json.loads(capsys.readouterr().out)["tokens"]


def test_predicator_seed_dictionary_controls_candidates(tmp_path):
    from soynlp.services.corpus import extract_corpus

    dictionary = tmp_path / "seeds.json"
    save_artifact(Artifact("dictionary", {"Verb": ["먹"], "Adjective": [], "Eomi": ["었다", "다"]}), dictionary)
    result = extract_corpus(["먹었다 예뻤다"] * 3, "predicators", dictionary=str(dictionary), resource=Artifact("nouns", {}))
    assert "먹었다" in result.data["verbs"]
    assert result.data["adjectives"] == {}
