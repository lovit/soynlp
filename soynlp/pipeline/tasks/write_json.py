from dataclasses import dataclass

from soynlp.documents import write_documents
from soynlp.pipeline.tasks.task import Task, TaskArgs


@dataclass(slots=True)
class WriteJsonTaskArgs(TaskArgs):
    path: str
    in_key: str = "corpus"
    overwrite: bool = False


class WriteJsonTask(Task[WriteJsonTaskArgs]):
    def __call__(self, parameters: dict) -> dict:
        if self._args.in_key not in parameters:
            raise ValueError(f"Not found `{self._args.in_key}` in `parameters`")
        write_documents(parameters[self._args.in_key], self._args.path, overwrite=self._args.overwrite)
        return parameters
