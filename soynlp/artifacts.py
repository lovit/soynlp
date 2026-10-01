"""Versioned resources and matrices shared by Python, CLI and pipelines."""

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from scipy import sparse

SCHEMA_VERSION = 1


def json_value(value: Any) -> Any:
    """Convert public NLP result objects without discarding score fields."""
    if is_dataclass(value) and not isinstance(value, type):
        return json_value(asdict(value))
    as_dictionary = getattr(value, "_asdict", None)
    if callable(as_dictionary):
        return json_value(as_dictionary())
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("JSON object keys must be strings; represent tuple keys as explicit records")
        return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, (set, frozenset)):
        return [json_value(item) for item in sorted(value)]
    if isinstance(value, (tuple, list)):
        return [json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


@dataclass(frozen=True)
class Artifact:
    kind: str
    data: Any
    parameters: dict[str, Any] | None = None
    schema_version: int = SCHEMA_VERSION

    def as_dict(self) -> dict[str, Any]:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"Unsupported artifact schema_version: {self.schema_version!r}")
        if not isinstance(self.kind, str) or not self.kind:
            raise ValueError("Artifact kind must be a nonempty string")
        if self.parameters is not None and not isinstance(self.parameters, dict):
            raise ValueError("Artifact parameters must be an object")
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "parameters": json_value(self.parameters or {}),
            "data": json_value(self.data),
        }

    @classmethod
    def from_dict(cls, value: Any, expected_kind: str | None = None) -> "Artifact":
        if not isinstance(value, dict) or set(value) != {"schema_version", "kind", "parameters", "data"}:
            raise ValueError("Artifact requires schema_version, kind, parameters and data")
        if type(value["schema_version"]) is not int or value["schema_version"] != SCHEMA_VERSION:
            raise ValueError(f"Unsupported artifact schema_version: {value['schema_version']!r}")
        result = cls(**value)
        result.as_dict()
        if expected_kind is not None and result.kind != expected_kind:
            raise ValueError(f"Expected artifact kind {expected_kind!r}, got {result.kind!r}")
        return result


@contextmanager
def atomic_output(path: str | Path, *, overwrite: bool = False, binary: bool = False) -> Iterator[Any]:
    """Publish a complete file; the default also protects against concurrent writers."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not overwrite and target.exists():
        raise FileExistsError(f"Output already exists: {target}")
    fd, name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(fd, "wb" if binary else "w", encoding=None if binary else "utf-8") as stream:
            yield stream
            stream.flush()
            os.fsync(stream.fileno())
        if overwrite:
            os.replace(name, target)
        else:
            os.link(name, target)
    finally:
        Path(name).unlink(missing_ok=True)


def save_artifact(artifact: Artifact, path: str | Path, *, overwrite: bool = False) -> None:
    payload = json.dumps(artifact.as_dict(), ensure_ascii=False, allow_nan=False, sort_keys=True)
    with atomic_output(path, overwrite=overwrite) as stream:
        stream.write(payload + "\n")


def load_artifact(path: str | Path, *, expected_kind: str | None = None) -> Artifact:
    with Path(path).open(encoding="utf-8") as stream:
        return Artifact.from_dict(json.load(stream), expected_kind)


def _vocabulary(value: list[str] | None, length: int, name: str) -> None:
    if value is not None and (
        not isinstance(value, list)
        or not all(isinstance(word, str) for word in value)
        or len(value) != length
        or len(set(value)) != length
    ):
        raise ValueError(f"{name} must contain {length} distinct strings")


def save_matrix(
    path: str | Path,
    matrix: Any,
    *,
    vocabulary: list[str] | None = None,
    row_vocabulary: list[str] | None = None,
    parameters: dict[str, Any] | None = None,
    kind: str = "matrix",
    overwrite: bool = False,
) -> None:
    """Save NPZ plus a matching JSON sidecar; readers detect interrupted writes."""
    target = Path(path)
    if target.suffix != ".npz":
        raise ValueError("Matrix output must have an .npz extension")
    sidecar = target.with_suffix(".json")
    if not overwrite:
        for candidate in (target, sidecar):
            if candidate.exists():
                raise FileExistsError(f"Output already exists: {candidate}")
    if len(matrix.shape) != 2:
        raise ValueError("Expected a two-dimensional matrix")
    _vocabulary(vocabulary, matrix.shape[1], "vocabulary")
    _vocabulary(row_vocabulary, matrix.shape[0], "row_vocabulary")
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(suffix=".npz", dir=target.parent)
    os.close(fd)
    try:
        if sparse.issparse(matrix):
            sparse.save_npz(name, sparse.csr_matrix(matrix))
            matrix_format = "csr"
        else:
            np.savez_compressed(name, matrix=np.asarray(matrix))
            matrix_format = "dense"
        data = {
            "shape": list(matrix.shape),
            "format": matrix_format,
            "vocabulary": vocabulary,
            "row_vocabulary": row_vocabulary,
            "sha256": hashlib.sha256(Path(name).read_bytes()).hexdigest(),
        }
        # Validate and serialize before changing either destination.
        payload = json.dumps(Artifact(kind, data, parameters).as_dict(), ensure_ascii=False, allow_nan=False)
        if overwrite:
            os.replace(name, target)
        else:
            os.link(name, target)
        try:
            with atomic_output(sidecar, overwrite=overwrite) as stream:
                stream.write(payload + "\n")
        except Exception:
            if not overwrite:
                target.unlink(missing_ok=True)
            raise
    finally:
        Path(name).unlink(missing_ok=True)


def load_matrix(path: str | Path) -> tuple[Any, Artifact]:
    target = Path(path)
    metadata = load_artifact(target.with_suffix(".json"))
    data = metadata.data
    if not isinstance(data, dict) or data.get("format") not in {"csr", "dense"}:
        raise ValueError("Invalid matrix metadata")
    if hashlib.sha256(target.read_bytes()).hexdigest() != data.get("sha256"):
        raise ValueError("Matrix and metadata do not match")
    if data["format"] == "csr":
        matrix = sparse.load_npz(target).tocsr()
    else:
        with np.load(target, allow_pickle=False) as archive:
            matrix = archive["matrix"]
    if list(matrix.shape) != data.get("shape"):
        raise ValueError("Matrix shape does not match metadata")
    _vocabulary(data.get("vocabulary"), matrix.shape[1], "vocabulary")
    _vocabulary(data.get("row_vocabulary"), matrix.shape[0], "row_vocabulary")
    return matrix, metadata
