from inspect_wandb.hooks.model_hooks import WandBModelHooks
from inspect_wandb.config.settings import ModelsSettings
from unittest.mock import patch

class TestWandBModelHooks:
    """
    Tests for the WandBModelHooks class.
    """

    def test_enabled(self) -> None:
        """
        Test that the enabled method returns True when the settings are set to True.
        """
        hooks = WandBModelHooks()
        assert hooks.enabled()

    def test_enabled_returns_false_when_settings_are_set_to_false(self) -> None:
        """
        Test that the enabled method returns False when the settings are set to False.
        """
        # Mock the settings loader to return disabled models settings
        disabled_settings = ModelsSettings(
            enabled=False, 
            entity="test-entity", 
            project="test-project"
        )
        
        with patch('inspect_wandb.hooks.model_hooks.SettingsLoader.load_inspect_wandb_settings') as mock_loader:
            mock_loader.return_value.models = disabled_settings
            hooks = WandBModelHooks()
            assert not hooks.enabled()

    