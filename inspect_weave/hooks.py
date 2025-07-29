from inspect_ai.hooks import Hooks, RunEnd, RunStart, SampleEnd, hooks, TaskStart, TaskEnd
import weave
from weave.trace.settings import UserSettings
from inspect_weave.utils import format_model_name, format_score_types, read_wandb_project_name_from_settings
from logging import getLogger
from inspect_weave.custom_evaluation_logger import CustomEvaluationLogger

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
        if self.weave_eval_logger is not None:
            if not self.weave_eval_logger._is_finalized:
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

    async def on_task_end(self, data: TaskEnd) -> None:
        assert self.weave_eval_logger is not None
        summary: dict[str, dict[str, dict[int, float]]] = {}
        if data.log and data.log.results:
            for score in data.log.results.scores:
                scorer_name = score.name
                if score.metrics:
                    summary[scorer_name] = {}
                    for metric_name, metric in score.metrics.items():
                        summary[scorer_name][metric_name] = metric.value
                        
        self.weave_eval_logger.log_summary(summary)
        self.weave_eval_logger.finish()

    async def on_sample_end(self, data: SampleEnd) -> None:
        assert self.weave_eval_logger is not None
        sample_score_logger = self.weave_eval_logger.log_prediction(
            inputs={"input": data.sample.input},
            output=data.sample.output.completion
        )
        if data.sample.scores is not None:
            for k,v in data.sample.scores.items():
                score_metadata = (v.metadata or {}) | ({"explanation": v.explanation} if v.explanation is not None else {})
                sample_score_logger.log_score(
                    scorer=k,
                    score=format_score_types(v.value),
                    metadata=score_metadata
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