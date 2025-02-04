from dataclasses import dataclass

import dacite
import yaml


@dataclass
class TaskConfig:
    name: str
    args: dict


@dataclass
class Config:
    pipeline: list[TaskConfig]


def from_yaml(path: str) -> Config:
    with open(path) as file:
        data = yaml.full_load(file)
    return from_dict(data)  # type: ignore


def from_dict(data: dict) -> Config:
    return dacite.from_dict(data_class=Config, data=data)  # type: ignore
