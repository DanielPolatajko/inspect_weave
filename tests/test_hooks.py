from inspect_ai.log import EvalLog
from typing import Callable
import configparser
import os
from pathlib import Path
from unittest.mock import MagicMock
from inspect_ai.hooks import SampleEnd, TaskEnd
from inspect_ai.model import ChatCompletionChoice, ModelOutput, ChatMessageAssistant
from inspect_ai.log import EvalSample
from inspect_weave.hooks import WeaveEvaluationHooks
from inspect_ai.scorer import Score
import pytest
import weave

class TestWeaveEvaluationHooks:

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

    def test_weave_finish_called_on_run_end(self, inspect_eval_and_weave_mocks: dict[str, Callable[[], list[EvalLog]] | MagicMock], initialise_wandb: None) -> None:
        # Given
        inspect_eval = inspect_eval_and_weave_mocks["inspect_eval"]
        weave_finish = inspect_eval_and_weave_mocks["weave_finish"]

        # When
        inspect_eval()

        # Then
        assert isinstance(weave_finish, MagicMock)
        weave_finish.assert_called_once()

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
        weave_evaluation_logger.assert_called_once_with(
            name=f"hello_world_{run_id}",
            dataset="test_dataset",
            model="mockllm__model"
        )

    @pytest.mark.asyncio
    async def test_weave_evaluation_logger_finish_called_on_task_end(self) -> None:
        # Given
        hooks = WeaveEvaluationHooks()
        hooks.weave_eval_logger = MagicMock(spec=weave.EvaluationLogger)

        # When
        await hooks.on_task_end(TaskEnd(
            run_id="test_run_id",
            eval_id="test_eval_id",
            log=MagicMock(spec=EvalLog)
        ))

        # Then
        hooks.weave_eval_logger.finish.assert_called_once()

    @pytest.mark.asyncio
    async def test_writes_eval_score_to_weave_on_sample_end(self) -> None:
        # Given
        hooks = WeaveEvaluationHooks()
        sample = SampleEnd(
            run_id="test_run_id",
            eval_id="test_eval_id",
            sample_id="test_sample_id",
            sample=EvalSample(
                id=1,
                epoch=1,
                input="test_input",
                target="test_output",
                scores={"test_score": Score(value=1.0)},
                output=ModelOutput(model="mockllm/model", choices=[ChatCompletionChoice(message=ChatMessageAssistant(content="test_output"))])
            )
        )

        mock_weave_eval_logger = MagicMock(spec=weave.EvaluationLogger)
        mock_score_logger = MagicMock(spec=weave.flow.eval_imperative.ScoreLogger)
        mock_weave_eval_logger.log_prediction.return_value = mock_score_logger
        hooks.weave_eval_logger = mock_weave_eval_logger

        # When
        await hooks.on_sample_end(sample)

        # Then
        mock_weave_eval_logger.log_prediction.assert_called_once_with(
            inputs={"input": "test_input"},
            output="test_output"
        )
        mock_score_logger.log_score.assert_called_once_with(
            scorer="test_score",
            score=1.0
        )
        mock_score_logger.finish.assert_called_once()