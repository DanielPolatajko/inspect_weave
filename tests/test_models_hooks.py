import os
from pathlib import Path
import yaml
from inspect_weave.hooks.model_hooks import WandBModelHooks

class TestWandBModelHooks:
    """
    Tests for the WandBModelHooks class.
    """

    def test_enabled(self, write_inspect_weave_settings: None) -> None:
        """
        Test that the enabled method returns True when the settings are set to True.
        """
        hooks = WandBModelHooks()
        assert hooks.enabled() is True

    def test_enabled_returns_false_when_settings_are_set_to_false(self, tmp_path: Path) -> None:
        """
        Test that the enabled method returns False when the settings are set to False.
        """
        settings = {
            "weave": {
                "enabled": True
            },
            "models": {
                "enabled": False
            }
        }
        os.makedirs(tmp_path / "wandb")
        with open(tmp_path / "wandb" / "inspect-weave-settings.yaml", "w") as f:
            yaml.dump(settings, f)
        hooks = WandBModelHooks()
        assert hooks.enabled() is False