from inspect_ai import Task, task
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.scorer import choice, scorer, accuracy, stderr, Score
from inspect_ai.solver._task_state import TaskState, Target
from inspect_ai.solver import multiple_choice, system_message
from dotenv import load_dotenv


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

@scorer(metrics=[accuracy(), stderr()])
def includes(ignore_case: bool = True):

    async def score(state: TaskState, target: Target):

        # check for correct
        answer = state.output.completion
        target = target.text
        if ignore_case:
            correct = answer.lower().rfind(target.lower()) != -1
        else:
            correct = answer.rfind(target) != -1

        # return score
        return Score(
            value = 1 if correct else 0,
            answer=answer,
            metadata={
                "test": "test"
            }
        )

    return score

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
        scorer=includes(),
    )