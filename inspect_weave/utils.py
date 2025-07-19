from weave.flow.eval_imperative import ScoreType
from inspect_ai.scorer import Value
from typing import Sequence, Mapping
from wandb.old.core import wandb_dir
from pathlib import Path
import configparser
from logging import Logger, getLogger

utils_logger = getLogger(__name__)

def format_model_name(model_name: str) -> str:
    return model_name.replace("/", "__").replace("-", "_").replace(".", "__").replace(":", "__").replace("@", "__")

def format_score_types(score_value: Value) -> ScoreType:
    if isinstance(score_value, str):
        return {"score": score_value}
    elif isinstance(score_value, int):
        return float(score_value)
    elif isinstance(score_value, Sequence):
        if len(score_value) != 1:
            raise ValueError("Sequence score cannot be passed to Weave")
        return {"score": score_value[0]}
    elif isinstance(score_value, Mapping):
        return dict(score_value)
    else:
        return score_value

def read_wandb_project_name_from_settings(logger: Logger | None = None) -> str | None:
    settings_path = Path(wandb_dir()) / "settings"
    if not settings_path.exists():
        raise ValueError("Wandb settings file not found, please run `wandb init` to set up a project")
    with open(settings_path, "r") as f:
        settings = configparser.ConfigParser()
        settings.read_file(f)
    if "default" in settings and "mode" in settings["default"] and settings["default"]["mode"] == "disabled":
        if logger is None:
            utils_logger.warning("Weave evaluation hooks are currently disabled. Please run `wandb init` to enable.")
        else:
            logger.warning("Weave evaluation hooks are currently disabled. Please run `wandb init` to enable.")
        return None
    return f"{settings['default']['entity']}/{settings['default']['project']}"