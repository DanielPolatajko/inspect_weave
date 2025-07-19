
from inspect_ai.hooks import Hooks, RunEnd, RunStart, SampleEnd, hooks, TaskStart, TaskEnd
import weave
from weave.trace.settings import UserSettings
from inspect_weave.utils import format_model_name, format_score_types, read_wandb_project_name_from_settings
from logging import getLogger

logger = getLogger("WeaveEvaluationHooks")

@hooks(name="weave_evaluation_hooks", description="Integration hooks for writing evaluation results to Weave")
class WeaveEvaluationHooks(Hooks):
    """
    Provides Inspect hooks for writing eval scores to the Weave Evaluations API.
    """

    weave_eval_logger: weave.EvaluationLogger | None = None

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
        evaluation_name = f"{data.spec.task}_{data.spec.run_id}"
        self.weave_eval_logger = weave.EvaluationLogger(
            name=evaluation_name,
            dataset=data.spec.dataset.name or "test_dataset", # TODO: set a default dataset name
            model=model_name
        )

    async def on_task_end(self, data: TaskEnd) -> None:
        assert self.weave_eval_logger is not None
        self.weave_eval_logger.log_summary()
        self.weave_eval_logger.finish()

    async def on_sample_end(self, data: SampleEnd) -> None:
        assert self.weave_eval_logger is not None
        sample_score_logger = self.weave_eval_logger.log_prediction(
            inputs={"input": data.sample.input},
            output=data.sample.output.completion
        )
        if data.sample.scores is not None:
            for k,v in data.sample.scores.items():
                sample_score_logger.log_score(
                    scorer=k,
                    score=format_score_types(v.value)
                 )
                if v.metadata is not None and "category" in v.metadata:
                    sample_score_logger.log_score(
                        scorer=f"{k}_{v.metadata['category']}",
                        score=format_score_types(v.value)
                    )
            sample_score_logger.finish()

    def enabled(self) -> bool:
        # Will error if wandb project is not set
        if read_wandb_project_name_from_settings(logger=logger) is None:
            return False
        return True