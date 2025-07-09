
import os

from inspect_ai.hooks import Hooks, RunEnd, RunStart, SampleEnd, hooks, TaskStart, TaskEnd
import weave
from inspect_weave.utils import format_model_name

@hooks(name="weave_evaluation_hooks", description="Integration hooks for writing evaluation results to Weave")
class WeaveEvaluationHooks(Hooks):
    """
    Provides Inspect hooks for writing eval scores to the Weave Evaluations API.
    """

    weave_eval_logger: weave.EvaluationLogger | None = None

    async def on_run_start(self, data: RunStart) -> None:
        try:
            weave.init(os.environ["WEAVE_PROJECT_NAME"])
        except KeyError as e:
            raise ValueError("WEAVE_PROJECT_NAME is not set, must be set in the environment to use Weave Evaluation Hooks") from e

    async def on_run_end(self, data: RunEnd) -> None:
        weave.finish()

    async def on_task_start(self, data: TaskStart) -> None:
        model_name = format_model_name(data.spec.model) 
        evaluation_name = f"{data.spec.task}_{data.spec.run_id}"
        self.weave_eval_logger = weave.EvaluationLogger(name=evaluation_name, dataset=data.spec.dataset.name or "test_dataset", model=model_name)

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
                    score=v.value if not isinstance(v.value, str) and not isinstance(v.value, list) else {"score": str(v.value)}  # TODO: handle different score return types
                 )
                sample_score_logger.log_score(
                    scorer="correct_answer",
                    score=data.sample.target == v.value
                )
            sample_score_logger.finish()

    def enabled(self) -> bool:
        return True # TODO: find a reliable way to check if Weave is setup correctly