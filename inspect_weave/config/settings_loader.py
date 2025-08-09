from inspect_weave.config.settings import WeaveSettings, ModelsSettings, InspectWeaveSettings
from wandb.old.core import wandb_dir
from pathlib import Path
import tomllib
from logging import getLogger
from inspect_weave.exceptions import WandBNotInitialisedException
import configparser

logger = getLogger(__name__)

class SettingsLoader:

    @classmethod
    def parse_inspect_weave_settings(cls, pyproject_path: Path | None = None) -> InspectWeaveSettings:
        """
        Load settings with this priority:
        1. Always start with wandb settings for entity/project
        2. Apply any customizations from user's pyproject.toml
        3. Default to hooks enabled, no files uploaded
        """
        # Always get base entity/project from wandb settings
        try:
            entity, project_name = cls._read_wandb_entity_and_project_name_from_settings()
            logger.info(f"Using wandb settings as base: Entity={entity}, Project={project_name}")
        except Exception as e:
            logger.error(f"Failed to read wandb settings. Please run 'wandb init' first. Error: {e}")
            raise
        
        # Load any customizations from user's pyproject.toml
        pyproject_config = cls._load_pyproject_config(pyproject_path)
        
        if pyproject_config:
            logger.info("Found [tool.inspect-weave] configuration in pyproject.toml")
        else:
            logger.info("No [tool.inspect-weave] configuration found. Using defaults: both hooks enabled, no files uploaded")
        
        # Build settings with defaults + customizations
        weave_config = pyproject_config.get("weave", {}) if pyproject_config else {}
        models_config = pyproject_config.get("models", {}) if pyproject_config else {}
        
        return InspectWeaveSettings(
            weave=WeaveSettings(
                enabled=weave_config.get("enabled", True),  # Default: enabled
                entity=entity,  # Always from wandb
                project=project_name  # Always from wandb
            ),
            models=ModelsSettings(
                enabled=models_config.get("enabled", True),  # Default: enabled
                entity=entity,  # Always from wandb
                project=project_name,  # Always from wandb
                files=models_config.get("files")  # Default: None (no files)
            )
        )
    
    @classmethod
    def _load_pyproject_config(cls, pyproject_path: Path | None = None) -> dict | None:
        """Load [tool.inspect-weave] section from user's pyproject.toml"""
        if pyproject_path is None:
            # Search for pyproject.toml in current directory and parents
            cwd = Path.cwd()
            for parent in [cwd, *cwd.parents]:
                candidate = parent / "pyproject.toml"
                if candidate.exists():
                    pyproject_path = candidate
                    break
            else:
                return None
        
        if not pyproject_path.exists():
            return None
            
        try:
            with open(pyproject_path, "rb") as f:
                data = tomllib.load(f)
        except (OSError, tomllib.TOMLDecodeError) as e:
            logger.warning(f"Failed to parse pyproject.toml at {pyproject_path}: {e}")
            return None
        
        return data.get("tool", {}).get("inspect-weave")

    @staticmethod
    def _read_wandb_entity_and_project_name_from_settings() -> tuple[str, str]:
        settings_path = Path(wandb_dir()) / "settings"
        if not settings_path.exists():
            raise WandBNotInitialisedException()
        with open(settings_path, "r") as f:
            settings = configparser.ConfigParser()
            settings.read_file(f)
        return settings['default']['entity'], settings['default']['project']