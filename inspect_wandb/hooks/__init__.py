from inspect_wandb.config.extras_manager import INSTALLED_EXTRAS
from inspect_wandb.hooks.model_hooks import WandBModelHooks

if INSTALLED_EXTRAS["weave"]:
    from inspect_wandb.hooks.weave_hooks import WeaveEvaluationHooks
    __all__ = ["WeaveEvaluationHooks", "WandBModelHooks"]
else:
    __all__ = ["WandBModelHooks"]