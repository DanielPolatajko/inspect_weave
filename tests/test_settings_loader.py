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
        os.chdir(tmp_path)

        # When
        with pytest.raises(WandBNotInitialisedException, match="wandb settings file not found. Please run `wandb init` to set up a project."):
            SettingsLoader._read_wandb_entity_and_project_name_from_settings()