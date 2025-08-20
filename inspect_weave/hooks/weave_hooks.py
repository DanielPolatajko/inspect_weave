from typing import Any
from inspect_ai.hooks import Hooks, RunEnd, RunStart, SampleEnd, SampleStart, TaskStart, TaskEnd
import weave
from weave.trace.settings import UserSettings
from inspect_weave.hooks.utils import format_model_name, format_score_types
from inspect_weave.config.settings_loader import SettingsLoader
from inspect_weave.config.settings import WeaveSettings
from logging import getLogger
from inspect_weave.weave_custom_overrides.autopatcher import get_inspect_patcher, CustomAutopatchSettings
from inspect_weave.weave_custom_overrides.custom_evaluation_logger import CustomEvaluationLogger
from inspect_weave.exceptions import WeaveEvaluationException
from weave.trace.weave_client import Call
from weave.trace.context import call_context
from typing_extensions import override

logger = getLogger(__name__)

class WeaveEvaluationHooks(Hooks):
    """
    Provides Inspect hooks for writing eval scores to the Weave Evaluations API.
    """

    weave_eval_loggers: dict[str, CustomEvaluationLogger] = {}
    settings: WeaveSettings | None = None
    sample_calls: dict[str, Call] = {}

    @override
    async def on_run_start(self, data: RunStart) -> None:
        # Ensure settings are loaded (in case enabled() wasn't called first)
        if self.settings is None:
            logger.info("Loading settings")
            self.settings = SettingsLoader.load_inspect_weave_settings().weave
        
        self.weave_client = weave.init(
            project_name=f"{self.settings.entity}/{self.settings.project}",
            settings=UserSettings(
                print_call_link=False
            )
        )
        if self.settings.autopatch:
            get_inspect_patcher(CustomAutopatchSettings().inspect).attempt_patch()

    @override
    async def on_run_end(self, data: RunEnd) -> None:
        # Finalize all active loggers
        for weave_eval_logger in self.weave_eval_loggers.values():
            if not weave_eval_logger._is_finalized:
                if data.exception is not None:
                    weave_eval_logger.finish(exception=data.exception)
                elif errors := [eval.error for eval in data.logs]:
                    weave_eval_logger.finish(
                        exception=WeaveEvaluationException(
                            message="Inspect run failed", 
                            error="\n".join([error.message for error in errors if error is not None])
                        )
                    )
                else:
                    weave_eval_logger.finish()
        
        # Clear the loggers dict
        self.weave_eval_loggers.clear()
        self.weave_client.finish(use_progress_bar=False)
        if self.settings is not None and self.settings.autopatch:
            get_inspect_patcher().undo_patch()

    @override
    async def on_task_start(self, data: TaskStart) -> None:
        model_name = format_model_name(data.spec.model) 
        weave_eval_logger = CustomEvaluationLogger(
            name=data.spec.task,
            dataset=data.spec.dataset.name or "test_dataset", # TODO: set a default dataset name
            model=model_name,
            eval_attributes=self._get_eval_metadata(data)
        )
        
        # Store logger with task_id as key
        self.weave_eval_loggers[data.eval_id] = weave_eval_logger
        
        assert weave_eval_logger._evaluate_call is not None
        call_context.push_call(weave_eval_logger._evaluate_call)

    @override
    async def on_task_end(self, data: TaskEnd) -> None:
        weave_eval_logger = self.weave_eval_loggers.get(data.eval_id)
        assert weave_eval_logger is not None
        
        summary: dict[str, dict[str, int | float]] = {}
        if data.log and data.log.results:
            for score in data.log.results.scores:
                scorer_name = score.name
                if score.metrics:
                    summary[scorer_name] = {}
                    for metric_name, metric in score.metrics.items():
                        summary[scorer_name][metric_name] = metric.value
        weave_eval_logger.log_summary(summary)

    @override
    async def on_sample_start(self, data: SampleStart) -> None:
        if self.settings is not None and self.settings.autopatch:
            self.sample_calls[data.sample_id] = self.weave_client.create_call(
                op="inspect-sample",
                inputs={"input": data.summary.input},
                attributes={"sample_id": data.sample_id, "epoch": data.summary.epoch},
                display_name="inspect-sample"
            )

    @override
    async def on_sample_end(self, data: SampleEnd) -> None:
        weave_eval_logger = self.weave_eval_loggers.get(data.eval_id)
        assert weave_eval_logger is not None
        
        sample_id = int(data.sample.id)
        epoch = data.sample.epoch
        input_value = data.sample.input
        with weave.attributes({"sample_id": sample_id, "epoch": epoch}):
            sample_score_logger = weave_eval_logger.log_prediction(
                inputs={"input": input_value},
                output=data.sample.output.completion,
                parent_call=self.sample_calls[data.sample_id] if self.settings is not None and self.settings.autopatch else None
            )
        if data.sample.scores is not None:
            for k,v in data.sample.scores.items():
                score_metadata = (v.metadata or {}) | ({"explanation": v.explanation} if v.explanation is not None else {})
                with weave.attributes(score_metadata):
                    sample_score_logger.log_score(
                        scorer=k,
                        score=format_score_types(v.value)
                    )

                    # Log various metrics to Weave
        try:
            # Total time
            if (
                hasattr(data.sample, "total_time")
                and data.sample.total_time is not None
            ):
                sample_score_logger.log_score(
                    scorer="total_time", score=data.sample.total_time
                )

            # Total tokens - model_usage is a dict of model_name -> ModelUsage
            if hasattr(data.sample, "model_usage") and data.sample.model_usage:
                # Get the first (and usually only) model's token usage
                for model_name, usage in data.sample.model_usage.items():
                    if usage.total_tokens is not None:
                        sample_score_logger.log_score(
                            scorer="total_tokens", score=usage.total_tokens
                        )
                        break  # Only log the first model's tokens

            # Number of tools from metadata - metadata is a dict
            if (
                hasattr(data.sample, "metadata")
                and data.sample.metadata
                and "Annotator Metadata" in data.sample.metadata
                and "Number of tools" in data.sample.metadata["Annotator Metadata"]
            ):
                sample_score_logger.log_score(
                    scorer="num_tool_calls",
                    score=int(
                        data.sample.metadata["Annotator Metadata"]["Number of tools"]
                    ),
                )

        except Exception as e:
            logger.error(f"Failed to log metrics to Weave: {e}")
            raise e

        sample_score_logger.finish()
        if self.settings is not None and self.settings.autopatch:
            self.weave_client.finish_call(self.sample_calls[data.sample_id], output=data.sample.output.completion)
            self.sample_calls.pop(data.sample_id)

    @override
    def enabled(self) -> bool:
        self.settings = self.settings or SettingsLoader.load_inspect_weave_settings().weave
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
