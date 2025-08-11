from typing import Callable
from unittest.mock import MagicMock, patch
from inspect_ai import Task, eval as inspect_eval
from inspect_weave.config.settings import WeaveSettings, ModelsSettings, InspectWeaveSettings

class TestEndToEndInspectRuns:
    """
    A test class for tests which simulate an entire Inspect eval run
    """
    def test_weave_init_not_called_on_run_start_when_disabled(self, patched_weave_evaluation_hooks: dict[str, MagicMock], hello_world_eval: Callable[[], Task]) -> None:
        # Given - Mock settings loader to return disabled weave settings
        disabled_settings = InspectWeaveSettings(
            weave=WeaveSettings(enabled=False, entity="test-entity", project="test-project"),
            models=ModelsSettings(enabled=True, entity="test-entity", project="test-project")
        )
        
        weave_init = patched_weave_evaluation_hooks["weave_init"]
        
        # When
        with patch('inspect_weave.hooks.weave_hooks.SettingsLoader.parse_inspect_weave_settings', return_value=disabled_settings):
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
        sample_count = eval_logs[0].eval.dataset.samples
        epochs = eval_logs[0].eval.config.epochs
        epochs_reducer = eval_logs[0].eval.config.epochs_reducer
        fail_on_error = eval_logs[0].eval.config.fail_on_error
        sandbox_cleanup = eval_logs[0].eval.config.sandbox_cleanup
        log_samples = eval_logs[0].eval.config.log_samples
        log_realtime = eval_logs[0].eval.config.log_realtime
        log_images = eval_logs[0].eval.config.log_images
        score_display = eval_logs[0].eval.config.score_display

        weave_evaluation_logger.assert_called_once_with(
            name="hello_world",
            dataset="test_dataset",
            model="mockllm__model",
            eval_attributes={
                "test": "test",
                "inspect": {
                    "run_id": run_id,
                    "task_id": task_id,
                    "eval_id": eval_id,
                    'sample_count': sample_count, 
                    'epochs': epochs, 'epochs_reducer': epochs_reducer, 
                    'fail_on_error': fail_on_error, 'sandbox_cleanup': sandbox_cleanup, 
                    'log_samples': log_samples, 'log_realtime': log_realtime, 'log_images': log_images, 'score_display': score_display
                }
            }
        )