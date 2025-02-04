from dataclasses import dataclass

from soynlp.pipeline.tasks.task import Task, TaskArgs


@dataclass
class DummyTaskArgs(TaskArgs):
    name: str = "Dummy Task"


class DummyTask(Task[DummyTaskArgs]):
    def __call__(self, parameters: dict) -> dict:
        print(f"Called in DummyTask({self._args.name})")
        return parameters
