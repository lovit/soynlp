from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar, get_args


@dataclass(slots=True)
class TaskArgs:
    pass


TaskArgsType = TypeVar("TaskArgsType", bound=TaskArgs)


class Task(ABC, Generic[TaskArgsType]):
    def __init__(self, args: TaskArgsType) -> None:
        self._args = args

    @classmethod
    def args(cls) -> type[TaskArgsType]:
        generic_type = cls.__orig_bases__[0]  # type: ignore[attr-defined]
        return get_args(generic_type)[0]

    @classmethod
    def validate_args(cls, args: TaskArgsType) -> None:
        """Check configuration without creating an algorithm or reading files."""

    @classmethod
    def required_keys(cls, args: TaskArgsType) -> set[str]:
        keys = {getattr(args, "in_key")} if hasattr(args, "in_key") else set()
        for name in ("resource_key", "dictionary_key"):
            value = getattr(args, name, None)
            if value:
                keys.add(value)
        if hasattr(args, "scores_key") and getattr(args, "tokenizer_type", None) != "regex":
            keys.add(getattr(args, "scores_key"))
        return keys

    @classmethod
    def produced_keys(cls, args: TaskArgsType) -> set[str]:
        return {getattr(args, "out_key")} if hasattr(args, "out_key") else set()

    @abstractmethod
    def __call__(self, parameters: dict) -> dict:
        raise NotImplementedError
