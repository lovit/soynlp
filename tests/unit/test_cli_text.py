import json

import pytest

from soynlp.artifacts import Artifact, save_artifact
from soynlp.cli import main
from soynlp.services import TEXT_METHODS, text_processor


@pytest.fixture
def nouns(tmp_path):
    path = tmp_path / "nouns.json"
    save_artifact(Artifact("nouns", {"한국어": {"frequency": 3, "score": 1.0}}), path)
    return str(path)


@pytest.mark.parametrize("method", TEXT_METHODS["normalize"])
def test_normalizers(method, capsys):
    main(["normalize", "--method", method, "--text", "한국어  공부ㅋㅋㅋ"])
    result = json.loads(capsys.readouterr().out)
    assert result["text"] == "한국어  공부ㅋㅋㅋ"
    assert isinstance(result["normalized"], str)


@pytest.mark.parametrize("method", TEXT_METHODS["tokenize"])
def test_tokenizer_methods(method, nouns, capsys):
    main(["tokenize", "--method", method, "--resource", nouns, "--text", "한국어를 공부합니다"])
    result = json.loads(capsys.readouterr().out)
    assert result["tokens"]


@pytest.mark.parametrize("method", TEXT_METHODS["tag"])
def test_tag_methods(method, capsys):
    main(["tag", "--method", method, "--text", "한국어를 공부합니다"])
    result = json.loads(capsys.readouterr().out)
    assert isinstance(result["tags"], list)


@pytest.mark.parametrize("method", TEXT_METHODS["lemmatize"])
def test_lemma_methods(method, capsys):
    main(["lemmatize", "--method", method, "--text", "먹었다"])
    result = json.loads(capsys.readouterr().out)
    assert ["먹", "었다"] in result["lemmas"]


@pytest.mark.parametrize("method", TEXT_METHODS["conjugate"])
def test_conjugation_methods(method, capsys):
    args = ["conjugate", "--method", method, "--text", "먹"]
    if method in {"normal", "chat"}:
        args += ["--eomi", "었다"]
    main(args)
    assert json.loads(capsys.readouterr().out)["forms"]


def test_hangle_operations():
    assert text_processor("hangle", "decompose")("간") == [("ㄱ", "ㅏ", "ㄴ")]
    assert text_processor("hangle", "compose")("ㄱ ㅏ ㄴ") == "간"
    assert text_processor("hangle", "jamo")("간")
    assert text_processor("hangle", "classify")("간")[0]["complete_korean"] is True
    assert text_processor("hangle", "distance", other="간")("간")["levenshtein"] == 0
    encoded = text_processor("hangle", "encode")("간 1")
    assert text_processor("hangle", "decode")(json.dumps(encoded)) == "간 1"


def test_jsonl_metadata_offsets_and_output_guard(tmp_path, capsys):
    source = tmp_path / "source.jsonl"
    source.write_text(json.dumps({"id": 7, "body": "한국어를", "meta": {"source": "test"}}, ensure_ascii=False) + "\n")
    output = tmp_path / "tokens.jsonl"
    args = [
        "tokenize",
        "--input",
        str(source),
        "--input-format",
        "jsonl",
        "--text-key",
        "body",
        "--offsets",
        "-o",
        str(output),
    ]
    main(args)
    result = json.loads(output.read_text())
    assert result["id"] == 7 and result["meta"] == {"source": "test"}
    assert result["tokens"][0]["begin"] == 0
    assert capsys.readouterr().out == ""
    with pytest.raises(SystemExit) as error:
        main(args)
    assert error.value.code == 1
    main(args + ["--overwrite"])


def test_stdin_and_error_codes(tmp_path, capsys, monkeypatch):
    from io import StringIO

    monkeypatch.setattr("sys.stdin", StringIO("한국어\n공부\n"))
    main(["normalize", "--input", "-"])
    assert len(capsys.readouterr().out.splitlines()) == 2
    with pytest.raises(SystemExit) as error:
        main(["tokenize", "--input", str(tmp_path / "missing.txt")])
    assert error.value.code == 1
    with pytest.raises(SystemExit) as error:
        main(["normalize", "--text", "text", "--parameters", '{"unknown_option": true}'])
    assert error.value.code == 2
    with pytest.raises(SystemExit) as error:
        main(["normalize", "--text", "text", "--input", "-"])
    assert error.value.code == 2
