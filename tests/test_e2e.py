from inspect_ai.log import EvalLog
from typing import Callable
import configparser
import os
from pathlib import Path
from unittest.mock import MagicMock


class TestEndToEndInspectRuns:
    """
    A test class for tests which simulate an entire Inspect eval run
    """

    def test_weave_init_not_called_on_run_start_when_disabled(self, inspect_eval_and_weave_mocks: dict[str, Callable[[], list[EvalLog]] | MagicMock], tmp_path: Path) -> None:

        # Given
        config = configparser.ConfigParser()
        config["default"] = {
            "entity": "test-entity",
            "project": "test-project",
            "mode": "disabled"
        }
        os.makedirs(tmp_path / "wandb")
        with open(tmp_path / "wandb" / "settings", "w") as f:
            config.write(f)
        os.chdir(tmp_path)

        # When

        inspect_eval = inspect_eval_and_weave_mocks["inspect_eval"]
        weave_init = inspect_eval_and_weave_mocks["weave_init"]
        inspect_eval()

        # Then
        assert isinstance(weave_init, MagicMock)
        weave_init.assert_not_called()

    def test_weave_init_called_on_run_start(self, inspect_eval_and_weave_mocks: dict[str, Callable[[], list[EvalLog]] | MagicMock], initialise_wandb: None) -> None:
        # Given
        inspect_eval = inspect_eval_and_weave_mocks["inspect_eval"]
        weave_init = inspect_eval_and_weave_mocks["weave_init"]

        # When
        inspect_eval()

        # Then
        assert isinstance(weave_init, MagicMock)
        weave_init.assert_called_once()

    def test_weave_evaluation_finalised_with_exception_on_error(self, inspect_eval_and_weave_mocks_with_error: dict[str, Callable[[], list[EvalLog]] | MagicMock], initialise_wandb: None) -> None:
        # Given
        inspect_eval = inspect_eval_and_weave_mocks_with_error["inspect_eval"]
        weave_evaluation_logger = inspect_eval_and_weave_mocks_with_error["weave_evaluation_logger"]
        assert isinstance(weave_evaluation_logger, MagicMock)
        weave_evaluation_logger.finish = MagicMock()
        weave_evaluation_logger._is_finalized = False

        # When
        inspect_eval()

        # Then
        print(weave_evaluation_logger.finish.call_args_list)
        assert weave_evaluation_logger.finish.call_args_list[0][1]["exception"].error == "RuntimeError('Simulated failure')"

    def test_weave_evaluation_logger_created_on_task_start(self, inspect_eval_and_weave_mocks: dict[str, Callable[[], list[EvalLog]] | MagicMock], initialise_wandb: None) -> None:
        # Given
        inspect_eval = inspect_eval_and_weave_mocks["inspect_eval"]
        weave_evaluation_logger = inspect_eval_and_weave_mocks["weave_evaluation_logger"]

        # When
        eval_logs = inspect_eval()

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