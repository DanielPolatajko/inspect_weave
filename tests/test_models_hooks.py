from inspect_weave.hooks.model_hooks import WandBModelHooks
import pytest

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

    @pytest.mark.models_hooks_disabled
    def test_enabled_returns_false_when_settings_are_set_to_false(self) -> None:
        """
        Test that the enabled method returns False when the settings are set to False.
        """
        hooks = WandBModelHooks()
        assert not hooks.enabled()

    