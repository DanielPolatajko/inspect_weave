from typing import Any
from inspect_ai.hooks import Hooks, RunEnd, RunStart, SampleEnd, TaskStart, TaskEnd
import weave
from weave.trace.settings import UserSettings
from inspect_weave.hooks.utils import format_model_name, format_score_types
from inspect_weave.config.settings_loader import SettingsLoader
from inspect_weave.config.settings import WeaveSettings
from logging import getLogger
from inspect_weave.weave_custom_overrides.custom_evaluation_logger import CustomEvaluationLogger
from inspect_weave.exceptions import WeaveEvaluationException
from weave.trace.context import call_context
from typing_extensions import override
from wandb.old.core import wandb_dir
from pathlib import Path

logger = getLogger(__name__)

class WeaveEvaluationHooks(Hooks):
    """
    Provides Inspect hooks for writing eval scores to the Weave Evaluations API.
    """

    weave_eval_logger: CustomEvaluationLogger | None = None
    settings: WeaveSettings | None = None

    @override
    async def on_run_start(self, data: RunStart) -> None:
        assert self.settings is not None
        weave.init(
            project_name=self.settings.project,
            settings=UserSettings(
                print_call_link=False
            )
        )

    @override
    async def on_run_end(self, data: RunEnd) -> None:
        if self.weave_eval_logger is not None:
            if not self.weave_eval_logger._is_finalized:
                if data.exception is not None:
                    self.weave_eval_logger.finish(exception=data.exception)
                elif errors := [eval.error for eval in data.logs]:
                    self.weave_eval_logger.finish(
                        exception=WeaveEvaluationException(
                            message="Inspect run failed", 
                            error="\n".join([error.message for error in errors if error is not None])
                        )
                    )
                else:
                    self.weave_eval_logger.finish()
        weave.finish()

    @override
    async def on_task_start(self, data: TaskStart) -> None:
        model_name = format_model_name(data.spec.model) 
        self.weave_eval_logger = CustomEvaluationLogger(
            name=data.spec.task,
            dataset=data.spec.dataset.name or "test_dataset", # TODO: set a default dataset name
            model=model_name,
            eval_attributes=self._get_eval_metadata(data)
        )
        assert self.weave_eval_logger._evaluate_call is not None
        call_context.set_call_stack([self.weave_eval_logger._evaluate_call]).__enter__()

    @override
    async def on_task_end(self, data: TaskEnd) -> None:
        assert self.weave_eval_logger is not None
        summary: dict[str, dict[str, int | float]] = {}
        if data.log and data.log.results:
            for score in data.log.results.scores:
                scorer_name = score.name
                if score.metrics:
                    summary[scorer_name] = {}
                    for metric_name, metric in score.metrics.items():
                        summary[scorer_name][metric_name] = metric.value
        self.weave_eval_logger.log_summary(summary)

    @override
    async def on_sample_end(self, data: SampleEnd) -> None:
        assert self.weave_eval_logger is not None
        sample_score_logger = self.weave_eval_logger.log_prediction(
            inputs={"input": data.sample.input},
            output=data.sample.output.completion
        )
        if data.sample.scores is not None:
            for k,v in data.sample.scores.items():
                score_metadata = (v.metadata or {}) | ({"explanation": v.explanation} if v.explanation is not None else {})
                with weave.attributes(score_metadata):
                    sample_score_logger.log_score(
                        scorer=k,
                        score=format_score_types(v.value)
                    )
            sample_score_logger.finish()

    @override
    def enabled(self) -> bool:
        settings_path = Path(wandb_dir()) / "inspect-weave-settings.yaml"
        self.settings = self.settings or SettingsLoader.parse_inspect_weave_settings(settings_path).weave
        return self.settings.enabled

    def _get_eval_metadata(self, data: TaskStart) -> dict[str, str | dict[str, Any]]:

        eval_metadata = data.spec.metadata or {}
        
        inspect_data = {
            "run_id": data.run_id,
            "task_id": data.spec.task_id,
            "eval_id": data.eval_id,
            "sample_count": data.spec.config.limit if data.spec.config.limit is not None else data.spec.dataset.samples
        }
        
        # Add task_args key-value pairs
        if data.spec.task_args:
            for key, value in data.spec.task_args.items():
                inspect_data[key] = value
        
        # Add config key-value pairs if config is not None
        if data.spec.config is not None:
            config_dict = data.spec.config.__dict__
            for key, value in config_dict.items():
                if value is not None:
                    inspect_data[key] = value
        
        eval_metadata["inspect"] = inspect_data
        
        return eval_metadata