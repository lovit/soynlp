from soynlp.pipeline.tasks.dummy import DummyTask  # noqa F401
from soynlp.pipeline.tasks.extract_nouns import ExtractNounTask  # noqa F401
from soynlp.pipeline.tasks.extract_words import ExtractWordTask  # noqa F401
from soynlp.pipeline.tasks.normalize import NormalizeTask  # noqa F401
from soynlp.pipeline.tasks.read_json import ReadJsonTask  # noqa F401
from soynlp.pipeline.tasks.read_text import ReadTextTask  # noqa F401
from soynlp.pipeline.tasks.task import Task  # noqa F401
from soynlp.pipeline.tasks.tokenize import TokenizeTask  # noqa F401
from soynlp.pipeline.tasks.write_json import WriteJsonTask  # noqa F401
from soynlp.pipeline.tasks.write_text import WriteTextTask  # noqa F401


__all__ = (
    "Task",
    "TASK_REGISTRY",
    "DummyTask",
    "ExtractNounTask",
    "ExtractWordTask",
    "NormalizeTask",
    "ReadJsonTask",
    "ReadTextTask",
    "TokenizeTask",
    "WriteJsonTask",
    "WriteTextTask",
)

TASK_REGISTRY: dict[str, type[Task]] = {
    "Dummy": DummyTask,
    "ExtractNoun": ExtractNounTask,
    "ExtractWord": ExtractWordTask,
    "Normalize": NormalizeTask,
    "ReadJson": ReadJsonTask,
    "ReadText": ReadTextTask,
    "Tokenize": TokenizeTask,
    "WriteJson": WriteJsonTask,
    "WriteText": WriteTextTask,
}
