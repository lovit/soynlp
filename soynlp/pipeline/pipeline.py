import importlib

import dacite

from soynlp.configs.config import Config, from_yaml
from soynlp.pipeline.tasks import Task


class Pipeline:
    def __call__(self, config: Config):
        tasks: list[Task] = self._load_tasks(config)
        parameters: dict = {}

        for task in tasks:
            parameters |= task(parameters)
        return parameters

    def _load_tasks(self, config: Config) -> list[Task]:
        tasks = []
        for task_config in config.pipeline:
            task_module = importlib.import_module("soynlp.pipeline.tasks")
            task_class: type[Task] = getattr(task_module, f"{task_config.name}Task")
            task_args = dacite.from_dict(data_class=task_class.args(), data=task_config.args)  # type: ignore
            task = task_class(task_args)
            tasks.append(task)
        return tasks

    @classmethod
    def run(cls, config_file: str):
        config = from_yaml(config_file)
        pipeline = Pipeline()
        pipeline(config)
