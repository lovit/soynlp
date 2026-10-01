import dacite

from soynlp.configs.config import Config, from_yaml
from soynlp.pipeline.tasks import TASK_REGISTRY, Task


class Pipeline:
    def __call__(self, config: Config, parameters: dict | None = None) -> dict:
        parameters = dict(parameters or {})
        self.validate(config, set(parameters))
        tasks: list[Task] = self._load_tasks(config)
        for task in tasks:
            result = task(parameters)
            if not isinstance(result, dict):
                raise TypeError(
                    f"Task {type(task).__name__!r} returned {type(result).__name__!r}, expected dict. All Task.__call__() implementations must return a dict."
                )
            parameters |= result
        return parameters

    @classmethod
    def _task_specs(cls, config: Config):
        for position, task_config in enumerate(config.pipeline, 1):
            task_class = TASK_REGISTRY.get(task_config.name)
            if task_class is None:
                raise ValueError(f"Unknown task: {task_config.name!r}. Available: {', '.join(sorted(TASK_REGISTRY))}")
            try:
                args = dacite.from_dict(data_class=task_class.args(), data=task_config.args, config=dacite.Config(strict=True))
                task_class.validate_args(args)
            except (dacite.DaciteError, ValueError, TypeError) as error:
                raise ValueError(f"Task {position} ({task_config.name}): {error}") from error
            yield task_class, args

    @classmethod
    def validate(cls, config: Config, initial_keys: set[str] | None = None) -> list[dict]:
        """Validate task schemas and dependency keys without executing any task."""
        available = set(initial_keys or set())
        report = []
        for position, (task_class, args) in enumerate(cls._task_specs(config), 1):
            required = task_class.required_keys(args)
            produced = task_class.produced_keys(args)
            missing = required - available
            if missing:
                raise ValueError(f"Task {position} ({task_class.__name__}): unavailable input keys: {sorted(missing)}")
            available |= produced
            report.append({"task": task_class.__name__, "inputs": sorted(required), "outputs": sorted(produced)})
        return report

    def _load_tasks(self, config: Config) -> list[Task]:
        return [task_class(args) for task_class, args in self._task_specs(config)]

    @classmethod
    def run(cls, config_file: str) -> None:
        cls()(from_yaml(config_file))
