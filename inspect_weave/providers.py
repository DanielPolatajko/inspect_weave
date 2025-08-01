from inspect_ai.hooks import hooks
from inspect_weave.hooks import WeaveEvaluationHooks, WandBModelHooks

@hooks(name="weave_evaluation_hooks", description="Weave evaluation integration")
def weave_evaluation_hooks():
    return WeaveEvaluationHooks

@hooks(name="wandb_models_hooks", description="Weights & Biases model integration")
def wandb_models_hooks():
    return WandBModelHooks