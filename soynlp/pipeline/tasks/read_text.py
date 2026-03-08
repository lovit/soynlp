from dataclasses import dataclass

from soynlp.pipeline.tasks.task import Task, TaskArgs


@dataclass(slots=True)
class ReadTextTaskArgs(TaskArgs):
    path: str
    text_key: str = "text"
    out_key: str = "corpus"


class ReadTextTask(Task[ReadTextTaskArgs]):
    def __call__(self, parameters: dict) -> dict:
        with open(self._args.path, encoding="utf-8") as f:
            examples = [{self._args.text_key: line.strip()} for line in f]
        return {self._args.out_key: examples}
