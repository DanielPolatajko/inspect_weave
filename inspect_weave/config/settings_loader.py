from inspect_weave.config.settings import WeaveSettings, ModelsSettings, InspectWeaveSettings
from logging import getLogger

logger = getLogger(__name__)

class SettingsLoader:

    @classmethod
    def load_inspect_weave_settings(cls) -> InspectWeaveSettings:
        """
        Load settings with this priority:
        1. Environment variables (both WANDB_* vars defined by W&B, and INSPECT_WEAVE_* vars defined by this package)
        2. Wandb settings file (for entity/project - handled by WandBSettingsSource)
        3. Initial settings (programmatic overrides)
        4. Pyproject.toml customizations
        5. Defaults if no other source provides values
        
        Note: The WandBSettingsSource will automatically read the wandb settings file.
        If no wandb settings are found, the settings creation will fail with validation errors
        for missing entity/project unless they are provided via environment variables.
        """
    
        # Simply create the settings - the sources are defined as part of the pydantic settings model
        return InspectWeaveSettings(
            weave=WeaveSettings.model_validate({}),
            models=ModelsSettings.model_validate({})
        )