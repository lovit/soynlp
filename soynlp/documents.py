"""Streaming text and JSONL documents with metadata preservation."""

import json
import sys
from collections.abc import Iterable, Iterator, Mapping
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from .artifacts import atomic_output, json_value


def read_documents(path: str | Path, *, format: str = "txt", text_key: str = "text") -> Iterator[dict[str, Any]]:
    if format not in {"txt", "jsonl"}:
        raise ValueError("Document format must be txt or jsonl")
    context = nullcontext(sys.stdin) if str(path) == "-" else Path(path).open(encoding="utf-8")
    with context as stream:
        for number, line in enumerate(stream, 1):
            if format == "txt":
                yield {text_key: line.rstrip("\r\n")}
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Line {number}: invalid JSON: {error.msg}") from error
            if not isinstance(item, dict) or not isinstance(item.get(text_key), str):
                raise ValueError(f"Line {number}: expected an object with string field {text_key!r}")
            yield item


def write_documents(
    documents: Iterable[dict[str, Any]],
    path: str | Path = "-",
    *,
    format: str = "jsonl",
    text_key: str = "text",
    overwrite: bool = False,
) -> None:
    if format not in {"txt", "jsonl"}:
        raise ValueError("Document format must be txt or jsonl")
    if isinstance(documents, Mapping):
        raise TypeError("Expected document records; use save_artifact() for mappings of extraction scores")
    context = nullcontext(sys.stdout) if str(path) == "-" else atomic_output(path, overwrite=overwrite)
    with context as stream:
        for number, item in enumerate(documents, 1):
            if not isinstance(item, dict):
                raise TypeError(f"Document {number}: expected an object")
            if format == "jsonl":
                line = json.dumps(json_value(item), ensure_ascii=False, allow_nan=False)
            else:
                line = item.get(text_key)
                if not isinstance(line, str) or "\n" in line or "\r" in line:
                    raise ValueError(f"Document {number}: expected a single-line string field {text_key!r}")
            stream.write(line + "\n")
