
import os

from inspect_ai.hooks import Hooks, RunEnd, RunStart, SampleEnd, hooks, TaskStart, TaskEnd
import weave

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
        self.weave_eval_logger = weave.EvaluationLogger(name=data.spec.task, dataset=data.spec.dataset.name or "test_dataset", model=data.spec.model)

    async def on_task_end(self, data: TaskEnd) -> None:
        assert self.weave_eval_logger is not None
        self.weave_eval_logger.log_summary()
        self.weave_eval_logger.finish()

    async def on_sample_end(self, data: SampleEnd) -> None:
        assert self.weave_eval_logger is not None
        sample_score_logger = self.weave_eval_logger.log_prediction(
            inputs={"input": data.summary.input},
            output=data.summary.target # TODO: this should be the model output, which is not available in the data.summary object
        )
        if data.summary.scores is not None:
            for k,v in data.summary.scores.items():
                sample_score_logger.log_score(
                    scorer=k,
                    score=v.value if not isinstance(v.value, str) and not isinstance(v.value, list) else {"score": v.value[0]}
                )
                sample_score_logger.log_score(
                    scorer="correct_answer",
                    score=data.summary.target == v.value
                )
            sample_score_logger.finish()

    async def enable(self) -> bool:
        return "WANDB_API_KEY" in os.environ