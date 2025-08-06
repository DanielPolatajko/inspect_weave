from inspect_weave.hooks import WeaveEvaluationHooks
from inspect_ai.hooks import hooks


@hooks(name="weave_evaluation_hooks", description="Integration hooks for writing evaluation results to Weave")
def weave_evaluation_hooks():
    return WeaveEvaluationHooks