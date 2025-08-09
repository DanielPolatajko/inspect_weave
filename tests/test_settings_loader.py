from inspect_weave.config.settings_loader import SettingsLoader
from inspect_weave.exceptions import WandBNotInitialisedException
from pathlib import Path
import os
import pytest


class TestSettingsLoader:
    def test_read_wandb_project_name_from_settings(self) -> None:
        # When
        project_name = SettingsLoader._read_wandb_entity_and_project_name_from_settings()

        # Then
        assert project_name == ("test-entity", "test-project")

    def test_read_wandb_project_name_from_settings_raises_error_if_settings_file_not_found(self, tmp_path: Path) -> None:
        # Given
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)

            # When
            with pytest.raises(WandBNotInitialisedException, match="wandb settings file not found. Please run `wandb init` to set up a project."):
                SettingsLoader._read_wandb_entity_and_project_name_from_settings()
        finally:
            # Restore original working directory to not break other tests
            os.chdir(original_cwd)
    
    def test_parse_settings_defaults_when_no_pyproject_toml(self) -> None:
        """Test that defaults are used when no pyproject.toml config exists"""
        # When
        settings = SettingsLoader.parse_inspect_weave_settings()
        
        # Then - should use wandb settings with defaults
        assert settings.weave.enabled is True
        assert settings.weave.entity == "test-entity"
        assert settings.weave.project == "test-project"
        assert settings.models.enabled is True
        assert settings.models.entity == "test-entity"
        assert settings.models.project == "test-project"
        assert settings.models.files is None
    
    def test_parse_settings_with_pyproject_toml_customizations(self, tmp_path: Path) -> None:
        """Test that pyproject.toml customizations override defaults"""
        # Given - create pyproject.toml with custom settings
        pyproject_content = """
[tool.inspect-weave.weave]
enabled = false

[tool.inspect-weave.models]
enabled = true
files = ["config.yaml", "results/"]
"""
        pyproject_path = tmp_path / "pyproject.toml"
        pyproject_path.write_text(pyproject_content)
        
        # When
        settings = SettingsLoader.parse_inspect_weave_settings(pyproject_path)
        
        # Then - should use wandb settings with pyproject customizations
        assert settings.weave.enabled is False  # Customized
        assert settings.weave.entity == "test-entity"  # From wandb
        assert settings.weave.project == "test-project"  # From wandb
        assert settings.models.enabled is True  # Customized
        assert settings.models.entity == "test-entity"  # From wandb
        assert settings.models.project == "test-project"  # From wandb
        assert settings.models.files == ["config.yaml", "results/"]  # Customized