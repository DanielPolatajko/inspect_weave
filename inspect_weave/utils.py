from weave.flow.eval_imperative import ScoreType
from inspect_ai.scorer import Value
from typing import Sequence, Mapping
from wandb.old.core import wandb_dir
from pathlib import Path
import configparser

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

def read_wandb_project_name_from_settings() -> str:
    settings_path = Path(wandb_dir()) / "settings"
    if not settings_path.exists():
        raise ValueError("Wandb settings file not found, please run `wandb init` to set up a project")
    with open(settings_path, "r") as f:
        settings = configparser.ConfigParser()
        settings.read_file(f)
    return f"{settings['default']['entity']}/{settings['default']['project']}"