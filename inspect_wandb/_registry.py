from inspect_wandb.providers import wandb_models_hooks
from inspect_wandb.config.extras_manager import INSTALLED_EXTRAS
if INSTALLED_EXTRAS["weave"]:
    from inspect_wandb.providers import weave_evaluation_hooks

__all__ = ["wandb_models_hooks"]
if INSTALLED_EXTRAS["weave"]:
    __all__.append("weave_evaluation_hooks")