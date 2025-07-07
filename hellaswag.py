from inspect_ai import Task, task
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.scorer import choice
from inspect_ai.solver import multiple_choice, system_message
import weave
from dotenv import load_dotenv

import os

from inspect_ai.hooks import Hooks, RunEnd, RunStart, SampleEnd, hooks

@hooks(name="weave_test_hooks", description="Weave test hooks")
class WeaveTestHooks(Hooks):

    weave_eval_logger: weave.EvaluationLogger | None = None

    async def on_run_start(self, data: RunStart) -> None:
        weave.init("test-project")
        self.weave_eval_logger = weave.EvaluationLogger()

    async def on_run_end(self, data: RunEnd) -> None:
        self.weave_eval_logger.finish()
        weave.finish()

    async def on_sample_end(self, data: SampleEnd) -> None:
        weave.init("test-project")
        sample_score_logger = self.weave_eval_logger.log_prediction(
            inputs=data.summary.input,
            output=data.summary.target # TODO: this should be the model output, which is not available in the data.summary object
        )
        for k,v in data.summary.scores.items():
            sample_score_logger.log_score(
                scorer=k,
                score=v.value
            )
            sample_score_logger.log_score(
                scorer="correct_answer",
                score=data.summary.target == v.value
            )
        sample_score_logger.finish()

    async def enable(self) -> bool:
        return "WANDB_API_KEY" in os.environ

load_dotenv()

weave.init("test-project")

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

@weave.op()
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