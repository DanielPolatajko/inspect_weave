from inspect_ai.hooks import Hooks, RunEnd, RunStart, SampleEnd, hooks, TaskStart, TaskEnd
import weave
from weave.trace.settings import UserSettings
from inspect_weave.utils import format_model_name, format_score_types, read_wandb_project_name_from_settings
from logging import getLogger
from inspect_weave.custom_evaluation_logger import CustomEvaluationLogger
from inspect_weave.exceptions import WeaveEvaluationException
from weave.trace.context import call_context
import json
import os
from datetime import datetime

logger = getLogger("WeaveEvaluationHooks")

@hooks(name="weave_evaluation_hooks", description="Integration hooks for writing evaluation results to Weave")
class WeaveEvaluationHooks(Hooks):
    """
    Provides Inspect hooks for writing eval scores to the Weave Evaluations API.
    """

    weave_eval_logger: CustomEvaluationLogger | None = None
    async def on_run_start(self, data: RunStart) -> None:
        project_name = read_wandb_project_name_from_settings(logger=logger)
        if project_name is None:
            return
        weave.init(
            project_name=project_name,
            settings=UserSettings(
                print_call_link=False
            )
        )

    async def on_run_end(self, data: RunEnd) -> None:
        # Dump entire run end data object to JSON file for debugging (recursive)
        try:
            os.makedirs("inspect_weave/logs", exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"/Users/matanshtepel/Documents/School Related/research/research-dev/inspect_evals/venv/lib/python3.13/site-packages/inspect_weave/logs/run_end_debug_{timestamp}.json"

            # Recursively convert data object to dict and serialize
            with open(filename, "w") as f:
                json.dump(self._recursive_dict(data), f, indent=2)

            logger.info(f"Run end debug data written to {filename}")
        except Exception as e:
            logger.error(f"Failed to write run end debug data: {e}")

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

    async def on_task_start(self, data: TaskStart) -> None:
        model_name = format_model_name(data.spec.model) 
        self.weave_eval_logger = CustomEvaluationLogger(
            name=data.spec.task,
            dataset=data.spec.dataset.name or "test_dataset", # TODO: set a default dataset name
            model=model_name,
            eval_attributes=self._get_eval_metadata(data)
        )
        call_context.set_call_stack([self.weave_eval_logger._evaluate_call]).__enter__()

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

    async def on_sample_end(self, data: SampleEnd) -> None:
        assert self.weave_eval_logger is not None
        sample_score_logger = self.weave_eval_logger.log_prediction(
            inputs={"input": data.sample.input},
            output=data.sample.output.completion
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

            # Total tokens - model_usage is a dict of model_name -> usage_dict
            if hasattr(data.sample, "model_usage") and data.sample.model_usage:
                # Get the first (and usually only) model's token usage
                for model_name, usage_dict in data.sample.model_usage.items():
                    if (
                        "total_tokens" in usage_dict
                        and usage_dict["total_tokens"] is not None
                    ):
                        sample_score_logger.log_score(
                            scorer="total_tokens", score=usage_dict["total_tokens"]
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

            # Which tools from metadata
            # ? I think this one is not displaying, afaik
            if (
                hasattr(data.sample, "metadata")
                and data.sample.metadata
                and "Annotator Metadata" in data.sample.metadata
                and "Tools" in data.sample.metadata["Annotator Metadata"]
            ):
                with weave.attributes(
                    {
                        "Which tools": str(
                            data.sample.metadata["Annotator Metadata"]["Tools"]
                        )
                    }
                ):
                    pass  # Just set the attribute

        except Exception as e:
            logger.error(f"Failed to log metrics to Weave: {e}")

        if data.sample.scores is not None:
            for k,v in data.sample.scores.items():
                score_metadata = (v.metadata or {}) | ({"explanation": v.explanation} if v.explanation is not None else {})
                with weave.attributes(score_metadata):
                    sample_score_logger.log_score(
                        scorer=k,
                        score=format_score_types(v.value)
                    )
        sample_score_logger.finish()

    def enabled(self) -> bool:
        # Will error if wandb project is not set
        if read_wandb_project_name_from_settings(logger=logger) is None:
            return False
        return True

    def _get_eval_metadata(self, data: TaskStart) -> dict[str, str]:
        eval_metadata = data.spec.metadata or {}
        eval_metadata["inspect_run_id"] = data.run_id
        eval_metadata["inspect_task_id"] = data.spec.task_id
        eval_metadata["inspect_eval_id"] = data.eval_id
        return eval_metadata