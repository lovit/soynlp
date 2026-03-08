import os
from dataclasses import dataclass

from soynlp.pipeline.tasks.task import Task, TaskArgs


@dataclass(slots=True)
class WriteTextTaskArgs(TaskArgs):
    path: str
    text_key: str = "text"
    in_key: str = "corpus"


class WriteTextTask(Task[WriteTextTaskArgs]):
    def __call__(self, parameters: dict) -> dict:
        if self._args.in_key not in parameters:
            raise ValueError(f"Not found `{self._args.in_key}` in `parameters`")
        examples = parameters[self._args.in_key]

        if os.path.exists(self._args.path):
            raise FileExistsError(f"Already exist `{self._args.path}`")
        os.makedirs(os.path.dirname(os.path.abspath(self._args.path)), exist_ok=True)

        with open(self._args.path, "w", encoding="utf-8") as file:
            for example in examples:
                text = example[self._args.text_key]
                file.write(f"{text}\n")

        return parameters
