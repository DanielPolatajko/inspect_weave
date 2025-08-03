from inspect_weave.config.settings import WeaveSettings, ModelsSettings, InspectWeaveSettings
from wandb.old.core import wandb_dir
from pathlib import Path
import yaml
from logging import getLogger
from inspect_weave.exceptions import WandBNotInitialisedException
import configparser

logger = getLogger(__name__)

class SettingsLoader:

    @classmethod
    def parse_inspect_weave_settings(cls, settings_path: Path | None = None) -> InspectWeaveSettings:
        if settings_path is None or not settings_path.exists():
            entity, project_name = cls._read_wandb_entity_and_project_name_from_settings()
            logger.warning(f"Inspect Weave settings file not found, please add a `inspect-weave-settings.yaml` file to the wandb directory if you want to configure the inspect_weave hooks. Proceeding with default settings. Entity: {entity}, Project: {project_name}")
            return InspectWeaveSettings(
                weave=WeaveSettings(entity=entity, project=project_name),
                models=ModelsSettings(entity=entity, project=project_name)
            )
        with open(settings_path, "r") as f:
            settings = yaml.safe_load(f)
        return InspectWeaveSettings(
            weave=WeaveSettings(**settings["weave"]),
            models=ModelsSettings(**settings["models"])
        )

    @staticmethod
    def _read_wandb_entity_and_project_name_from_settings() -> tuple[str, str]:
        settings_path = Path(wandb_dir()) / "settings"
        if not settings_path.exists():
            raise WandBNotInitialisedException()
        with open(settings_path, "r") as f:
            settings = configparser.ConfigParser()
            settings.read_file(f)
        return settings['default']['entity'], settings['default']['project']