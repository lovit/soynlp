from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar, get_args


@dataclass
class TaskArgs:
    pass


TaskArgsType = TypeVar("TaskArgsType", bound=TaskArgs)


class Task(ABC, Generic[TaskArgsType]):
    def __init__(self, args: TaskArgsType):
        self._args = args

    @classmethod
    def args(cls) -> type[TaskArgs]:
        generic_type = cls.__orig_bases__[0]  # type: ignore[attr-defined]
        return get_args(generic_type)[0]

    @abstractmethod
    def __call__(self, parameters: dict) -> dict:
        raise NotImplementedError
