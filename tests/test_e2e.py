from typing import Callable
from unittest.mock import MagicMock
from inspect_ai import Task, eval as inspect_eval
import pytest

class TestEndToEndInspectRuns:
    """
    A test class for tests which simulate an entire Inspect eval run
    """
    @pytest.mark.weave_hooks_disabled   
    def test_weave_init_not_called_on_run_start_when_disabled(self, patched_weave_evaluation_hooks: dict[str, MagicMock], hello_world_eval: Callable[[], Task]) -> None:
        # Given
        weave_init = patched_weave_evaluation_hooks["weave_init"]
        # When
        inspect_eval(hello_world_eval, model="mockllm/model")

        # Then
        assert isinstance(weave_init, MagicMock)
        weave_init.assert_not_called()

    def test_weave_init_called_on_run_start(self, patched_weave_evaluation_hooks: dict[str, MagicMock], hello_world_eval: Callable[[], Task]) -> None:
        # Given
        weave_init = patched_weave_evaluation_hooks["weave_init"]

        # When
        inspect_eval(hello_world_eval, model="mockllm/model")

        # Then
        assert isinstance(weave_init, MagicMock)
        weave_init.assert_called_once()

    def test_weave_evaluation_finalised_with_exception_on_error(self, patched_weave_evaluation_hooks: dict[str, MagicMock], error_eval: Callable[[], Task]) -> None:
        # Given
        weave_evaluation_logger = patched_weave_evaluation_hooks["weave_evaluation_logger"]
        weave_evaluation_logger.finish = MagicMock()
        weave_evaluation_logger._is_finalized = False

        # When
        inspect_eval(error_eval, model="mockllm/model")

        # Then
        assert weave_evaluation_logger.finish.call_args_list[0][1]["exception"].error == "RuntimeError('Simulated failure')"

    def test_weave_evaluation_logger_created_on_task_start(self, patched_weave_evaluation_hooks: dict[str, MagicMock], hello_world_eval: Callable[[], Task]) -> None:
        # Given
        weave_evaluation_logger = patched_weave_evaluation_hooks["weave_evaluation_logger"]

        # When
        eval_logs = inspect_eval(hello_world_eval, model="mockllm/model")

        # Then
        assert isinstance(weave_evaluation_logger, MagicMock)
        assert len(eval_logs) == 1
        run_id = eval_logs[0].eval.run_id
        task_id = eval_logs[0].eval.task_id
        eval_id = eval_logs[0].eval.eval_id
        weave_evaluation_logger.assert_called_once_with(
            name="hello_world",
            dataset="test_dataset",
            model="mockllm__model",
            eval_attributes={
                "test": "test",
                "inspect_run_id": run_id,
                "inspect_task_id": task_id,
                "inspect_eval_id": eval_id,
            }
        )