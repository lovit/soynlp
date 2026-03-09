import dacite

from soynlp.configs.config import Config, from_yaml
from soynlp.pipeline.tasks import TASK_REGISTRY, Task


class Pipeline:
    def __call__(self, config: Config) -> dict:
        tasks: list[Task] = self._load_tasks(config)
        parameters: dict = {}

        for task in tasks:
            result = task(parameters)
            if not isinstance(result, dict):
                raise TypeError(
                    f"Task {type(task).__name__!r} returned {type(result).__name__!r}, expected dict. "
                    "All Task.__call__() implementations must return a dict."
                )
            parameters |= result
        return parameters

    def _load_tasks(self, config: Config) -> list[Task]:
        tasks = []
        for task_config in config.pipeline:
            task_class = TASK_REGISTRY.get(task_config.name)
            if task_class is None:
                available = ", ".join(sorted(TASK_REGISTRY))
                raise ValueError(f"Unknown task: {task_config.name!r}. Available: {available}")
            task_args = dacite.from_dict(data_class=task_class.args(), data=task_config.args)  # type: ignore
            task = task_class(task_args)
            tasks.append(task)
        return tasks

    @classmethod
    def run(cls, config_file: str) -> None:
        config = from_yaml(config_file)
        pipeline = Pipeline()
        pipeline(config)
