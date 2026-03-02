from soynlp.pipeline.tasks.dummy import DummyTask, DummyTaskArgs


class TestDummyTask:
    def test_passthrough(self):
        task = DummyTask(DummyTaskArgs(name="test"))
        params = {"key": "value", "number": 42}
        result = task(params)
        assert result == params

    def test_default_name(self):
        task = DummyTask(DummyTaskArgs())
        assert task._args.name == "Dummy Task"
