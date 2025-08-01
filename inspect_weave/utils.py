from typing import Any
from weave.flow.eval_imperative import ScoreType
from inspect_ai.scorer import Value
from typing import Sequence, Mapping
from wandb.old.core import wandb_dir
from pathlib import Path
import configparser
from logging import Logger, getLogger
import yaml
from inspect_weave.exceptions import WandBNotInitialisedException

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

def read_wandb_entity_and_project_name_from_settings(logger: Logger | None = None) -> tuple[str, str]:
    settings_path = Path(wandb_dir()) / "settings"
    if not settings_path.exists():
        raise WandBNotInitialisedException()
    with open(settings_path, "r") as f:
        settings = configparser.ConfigParser()
        settings.read_file(f)
    return settings['default']['entity'], settings['default']['project']

def parse_inspect_weave_settings() -> dict[str, Any]:
    settings_path = Path(wandb_dir()) / "inspect-weave-settings.yaml"
    if not settings_path.exists():
        entity, project_name = read_wandb_entity_and_project_name_from_settings()
        utils_logger.warning(f"Inspect Weave settings file not found, please add a `inspect-weave-settings.yaml` file to the wandb directory if you want to configure the inspect_weave hooks. Proceeding with default settings. Entity: {entity}, Project: {project_name}")
        return {
            "weave": {
                "enabled": True,
                "entity": entity,
                "project": project_name
            },
            "models": {
                "enabled": True,
                "entity": entity,
                "project": project_name
            }
        }
    with open(settings_path, "r") as f:
        settings = yaml.safe_load(f)
    return settings