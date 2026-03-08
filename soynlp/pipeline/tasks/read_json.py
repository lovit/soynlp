import json
from dataclasses import dataclass

from soynlp.pipeline.tasks.task import Task, TaskArgs


@dataclass(slots=True)
class ReadJsonTaskArgs(TaskArgs):
    path: str
    out_key: str = "corpus"


class ReadJsonTask(Task[ReadJsonTaskArgs]):
    def __call__(self, parameters: dict) -> dict:
        with open(self._args.path, encoding="utf-8") as f:
            examples = [json.loads(line.strip()) for line in f]
        return {self._args.out_key: examples}
