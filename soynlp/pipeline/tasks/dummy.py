import logging
from dataclasses import dataclass

from soynlp.pipeline.tasks.task import Task, TaskArgs

logger = logging.getLogger(__name__)


@dataclass
class DummyTaskArgs(TaskArgs):
    name: str = "Dummy Task"


class DummyTask(Task[DummyTaskArgs]):
    def __call__(self, parameters: dict) -> dict:
        logger.info(f"Called in DummyTask({self._args.name})")
        return parameters
