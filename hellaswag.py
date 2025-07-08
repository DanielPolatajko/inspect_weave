from inspect_ai import Task, task
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.scorer import choice
from inspect_ai.solver import multiple_choice, system_message
import weave
from weave.trace.weave_client import WeaveClient, Call
from dotenv import load_dotenv

import os

from inspect_ai.hooks import Hooks, RunEnd, RunStart, SampleEnd, hooks, SampleStart, TaskStart, TaskEnd

@hooks(name="weave_test_hooks", description="Weave test hooks")
class WeaveTestHooks(Hooks):

    weave_eval_logger: weave.EvaluationLogger | None = None
    client: WeaveClient | None = None
    sample_call: Call | None = None

    async def on_run_start(self, data: RunStart) -> None:
        self.client = weave.init("test-project")

    async def on_run_end(self, data: RunEnd) -> None:
        weave.finish()

    async def on_task_start(self, data: TaskStart) -> None:
        self.weave_eval_logger = weave.EvaluationLogger(name=data.spec.task, dataset=data.spec.dataset.name or "test_dataset", model=data.spec.model)

    async def on_task_end(self, data: TaskEnd) -> None:
        assert self.weave_eval_logger is not None
        self.weave_eval_logger.log_summary()
        self.weave_eval_logger.finish()

    async def on_sample_start(self, data: SampleStart) -> None:
        assert self.client is not None
        self.sample_call = self.client.create_call(
            op="process_sample",
            inputs={"input": data.summary.input},
        )

    async def on_sample_end(self, data: SampleEnd) -> None:
        assert self.client is not None
        assert self.sample_call is not None
        assert self.weave_eval_logger is not None
        self.client.finish_call(self.sample_call, output=data.summary.target)
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

load_dotenv()

SYSTEM_MESSAGE = """
Choose the most plausible continuation for the story.
"""

def record_to_sample(record):
    return Sample(
        input=record["ctx"],
        target=chr(ord("A") + int(record["label"])),
        choices=record["endings"],
        metadata=dict(
            source_id=record["source_id"]
        )
    )

@task
def hellaswag():
   
    # dataset
    dataset = hf_dataset(
        path="hellaswag",
        split="validation",
        sample_fields=record_to_sample,
        trust=True
    )

    # define task
    return Task(
        dataset=dataset,
        solver=[
          system_message(SYSTEM_MESSAGE),
          multiple_choice()
        ],
        scorer=choice(),
    )