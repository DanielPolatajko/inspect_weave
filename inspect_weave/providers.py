from inspect_weave.hooks import WeaveEvaluationHooks, LogHooks
from inspect_ai.hooks import hooks


@hooks(name="weave_evaluation_hooks", description="Integration hooks for writing evaluation results to Weave")
def weave_evaluation_hooks():
    return WeaveEvaluationHooks

@hooks(name="log_hooks", description="test log dataframe feature in Inspect")
def log_hooks():
    return LogHooks
