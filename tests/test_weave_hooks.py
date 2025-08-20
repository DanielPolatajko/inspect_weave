from inspect_ai.log import EvalLog
from unittest.mock import MagicMock
from inspect_ai.hooks import SampleEnd, TaskEnd, RunEnd
from inspect_ai.model import ChatCompletionChoice, ModelOutput, ChatMessageAssistant
from inspect_ai.log import EvalSample, EvalResults, EvalScore, EvalMetric, EvalSpec, EvalConfig, EvalDataset
from inspect_ai._eval.eval import EvalLogs
from inspect_wandb.hooks import WeaveEvaluationHooks
from inspect_ai.scorer import Score
import pytest
from datetime import datetime
from weave.evaluation.eval_imperative import ScoreLogger, EvaluationLogger
from inspect_wandb.config.settings import WeaveSettings
from weave.trace.weave_client import WeaveClient

@pytest.fixture(scope="function")
def task_end_eval_log() -> EvalLog:
    return EvalLog(
        eval=EvalSpec(
            run_id="test_run_id",
            task_id="test_task_id",
            created=datetime.now().isoformat(),
            task="test_task",
            dataset=EvalDataset(),
            model="mockllm/model",
            config=EvalConfig()
        ),
        results=EvalResults(
            total_samples=1,
            scores=[
                EvalScore(
                    name="test_score",
                    scorer="test_scorer",
                    metrics={"test_metric": EvalMetric(name="test_metric", value=1.0)}
                )
            ]
        )
    )

@pytest.fixture(scope="function")
def test_settings() -> WeaveSettings:
    return WeaveSettings(
        enabled=True,
        entity="test-entity",
        project="test-project",
        autopatch=False
    )
    

class TestWeaveEvaluationHooks:
    """
    Tests for individual hook functionalities
    """

    @pytest.mark.asyncio
    async def test_writes_eval_score_to_weave_on_sample_end(self, test_settings: WeaveSettings) -> None:
        # Given
        hooks = WeaveEvaluationHooks()
        hooks.settings = test_settings
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

        mock_weave_eval_logger = MagicMock(spec=EvaluationLogger)
        mock_score_logger = MagicMock(spec=ScoreLogger)
        mock_weave_eval_logger.log_prediction.return_value = mock_score_logger
        hooks.weave_eval_loggers["test_eval_id"] = mock_weave_eval_logger

        # When
        await hooks.on_sample_end(sample)

        # Then
        mock_weave_eval_logger.log_prediction.assert_called_once_with(
            inputs={"input": "test_input"},
            output="test_output",
            parent_call=None
        )
        mock_score_logger.log_score.assert_called_once_with(
            scorer="test_score",
            score=1.0,
        )
        mock_score_logger.finish.assert_called_once()

    @pytest.mark.asyncio
    async def test_writes_eval_score_to_weave_on_sample_end_with_metadata(self, test_settings: WeaveSettings) -> None:
        # Given
        hooks = WeaveEvaluationHooks()
        hooks.settings = test_settings
        sample = SampleEnd(
            run_id="test_run_id",
            eval_id="test_eval_id",
            sample_id="test_sample_id",
            sample=EvalSample(
                id=1,
                epoch=1,
                input="test_input",
                target="test_output",
                scores={"test_score": Score(value=1.0, metadata={"test": "test"})},
                output=ModelOutput(model="mockllm/model", choices=[ChatCompletionChoice(message=ChatMessageAssistant(content="test_output"))])
            )
        )

        mock_weave_eval_logger = MagicMock(spec=EvaluationLogger)
        mock_score_logger = MagicMock(spec=ScoreLogger)
        mock_weave_eval_logger.log_prediction.return_value = mock_score_logger
        hooks.weave_eval_loggers["test_eval_id"] = mock_weave_eval_logger

        # When
        await hooks.on_sample_end(sample)

        # Then
        mock_weave_eval_logger.log_prediction.assert_called_once_with(
            inputs={"input": "test_input"},
            output="test_output",
            parent_call=None
        )
        mock_score_logger.log_score.assert_called_once_with(
            scorer="test_score",
            score=1.0
        )
        mock_score_logger.finish.assert_called_once()

    @pytest.mark.asyncio
    async def test_writes_inspect_eval_summary_metrics_to_weave_on_task_end(self, task_end_eval_log: EvalLog, test_settings: WeaveSettings) -> None:
        # Given
        hooks = WeaveEvaluationHooks()
        hooks.settings = test_settings
        task_end = TaskEnd(
            run_id="test_run_id",
            eval_id="test_eval_id",
            log=task_end_eval_log
        )

        mock_weave_eval_logger = MagicMock(spec=EvaluationLogger)
        hooks.weave_eval_loggers["test_eval_id"] = mock_weave_eval_logger

        # When
        await hooks.on_task_end(task_end)

        # Then
        expected_summary = {
            "test_score": {
                "test_metric": 1.0
            }
        }
        mock_weave_eval_logger.log_summary.assert_called_once_with(
            expected_summary
        )

    @pytest.mark.asyncio
    async def test_passes_exception_to_weave_on_error_run_end(self, test_settings: WeaveSettings) -> None:
        # Given
        e = Exception("test_exception")
        hooks = WeaveEvaluationHooks()
        hooks.settings = test_settings
        hooks.weave_client = MagicMock(spec=WeaveClient)
        task_end = RunEnd(
            run_id="test_run_id",
            logs=EvalLogs([]),       
            exception=e
        )

        mock_weave_eval_logger = MagicMock(spec=EvaluationLogger)
        mock_weave_eval_logger.finish = MagicMock()
        mock_weave_eval_logger._is_finalized = False
        hooks.weave_eval_loggers["test_eval_id"] = mock_weave_eval_logger

        # When
        await hooks.on_run_end(task_end)

        # Then
        mock_weave_eval_logger.finish.assert_called_once_with(
            exception=e
        )
