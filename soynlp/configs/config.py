from dataclasses import dataclass

import dacite
import yaml


@dataclass(slots=True)
class TaskConfig:
    name: str
    args: dict


@dataclass(slots=True)
class Config:
    pipeline: list[TaskConfig]


def from_yaml(path: str) -> Config:
    try:
        with open(path, encoding="utf-8") as file:
            data = yaml.safe_load(file)
    except yaml.YAMLError as error:
        raise ValueError(f"Invalid YAML: {error}") from error
    return from_dict(data)  # type: ignore


def from_dict(data: dict) -> Config:
    try:
        return dacite.from_dict(data_class=Config, data=data, config=dacite.Config(strict=True))
    except dacite.DaciteError as error:
        raise ValueError(f"Invalid pipeline configuration: {error}") from error
