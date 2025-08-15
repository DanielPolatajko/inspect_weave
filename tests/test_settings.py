import pytest
import os
from unittest.mock import patch
from inspect_weave.config.settings import ModelsSettings, WeaveSettings, InspectWeaveSettings
from inspect_weave.config.wandb_settings_source import WandBSettingsSource
from inspect_weave.config.settings_loader import SettingsLoader


class TestWandBSettingsSource:
    """Test the WandBSettingsSource that reads from wandb settings file"""
    
    def test_wandb_settings_source_with_valid_file(self, tmp_path):
        """Test WandBSettingsSource with a valid wandb settings file"""
        # Create mock wandb settings file
        wandb_dir = tmp_path / "wandb"
        wandb_dir.mkdir()
        settings_file = wandb_dir / "settings"
        settings_content = """[default]
entity = source-test-entity
project = source-test-project
"""
        settings_file.write_text(settings_content)
        
        # Mock wandb_dir to return our test directory
        with patch('inspect_weave.config.wandb_settings_source.wandb_dir', return_value=str(wandb_dir)):
            source = WandBSettingsSource(ModelsSettings)
            result = source()
            
            assert result == {
                'WANDB_ENTITY': 'source-test-entity',
                'WANDB_PROJECT': 'source-test-project'
            }
    
    def test_wandb_settings_source_with_missing_file(self, tmp_path):
        """Test WandBSettingsSource when wandb settings file doesn't exist"""
        wandb_dir = tmp_path / "wandb"
        wandb_dir.mkdir()
        # Don't create settings file
        
        with patch('inspect_weave.config.wandb_settings_source.wandb_dir', return_value=str(wandb_dir)):
            source = WandBSettingsSource(ModelsSettings)
            result = source()
            
            # Should return empty dict when file doesn't exist
            assert result == {}
    
    def test_wandb_settings_source_with_invalid_file(self, tmp_path):
        """Test WandBSettingsSource with invalid settings file"""
        wandb_dir = tmp_path / "wandb"
        wandb_dir.mkdir()
        settings_file = wandb_dir / "settings"
        settings_file.write_text("invalid content")
        
        with patch('inspect_weave.config.wandb_settings_source.wandb_dir', return_value=str(wandb_dir)):
            source = WandBSettingsSource(ModelsSettings)
            result = source()
            
            # Should return empty dict when file is invalid
            assert result == {}
    
    def test_wandb_settings_source_caches_results(self, tmp_path):
        """Test that WandBSettingsSource caches the parsed results"""
        wandb_dir = tmp_path / "wandb"
        wandb_dir.mkdir()
        settings_file = wandb_dir / "settings"
        settings_content = """[default]
entity = cached-entity
project = cached-project
"""
        settings_file.write_text(settings_content)
        
        with patch('inspect_weave.config.wandb_settings_source.wandb_dir', return_value=str(wandb_dir)):
            source = WandBSettingsSource(ModelsSettings)
            
            # First call should read the file
            result1 = source()
            
            # Modify the file
            settings_file.write_text("[default]\nentity=modified\nproject=modified")
            
            # Second call should return cached results
            result2 = source()
            
            assert result1 == result2  # Should be the same (cached)
            assert result1['WANDB_ENTITY'] == 'cached-entity'  # Original value


class TestModelsSettings:
    """Test ModelsSettings pydantic model with priority order verification"""
    
    def test_default_values_with_mock_wandb(self, tmp_path):
        """Test default values when wandb settings are available"""
        # Create mock wandb settings
        wandb_dir = tmp_path / "wandb"
        wandb_dir.mkdir()
        settings_file = wandb_dir / "settings"
        settings_content = """[default]
entity = wandb-entity
project = wandb-project
"""
        settings_file.write_text(settings_content)
        
        with patch('inspect_weave.config.wandb_settings_source.wandb_dir', return_value=str(wandb_dir)):
            settings = ModelsSettings()
            
            assert settings.enabled is True
            assert settings.config is None
            assert settings.files is None
            assert settings.entity == "wandb-entity"  # From wandb settings
            assert settings.project == "wandb-project"  # From wandb settings

    def test_environment_variables_highest_priority(self, monkeypatch, tmp_path):
        """Test that environment variables take highest priority"""
        # Create mock wandb settings (should be overridden by env vars)
        wandb_dir = tmp_path / "wandb"
        wandb_dir.mkdir()
        settings_file = wandb_dir / "settings"
        settings_content = """[default]
entity = wandb-entity
project = wandb-project
"""
        settings_file.write_text(settings_content)
        
        # Set environment variables
        monkeypatch.setenv("INSPECT_WEAVE_MODELS_ENABLED", "false")
        monkeypatch.setenv("WANDB_PROJECT", "env-project")  # Uses alias
        monkeypatch.setenv("WANDB_ENTITY", "env-entity")    # Uses alias
        monkeypatch.setenv("INSPECT_WEAVE_MODELS_FILES", '["env-file1.txt", "env-file2.txt"]')
        
        with patch('inspect_weave.config.wandb_settings_source.wandb_dir', return_value=str(wandb_dir)):
            settings = ModelsSettings()
            
            assert settings.enabled is False  # Environment variable wins
            assert settings.project == "env-project"  # Environment variable wins over wandb
            assert settings.entity == "env-entity"  # Environment variable wins over wandb
            assert settings.files == ["env-file1.txt", "env-file2.txt"]

    def test_wandb_settings_middle_priority(self, tmp_path):
        """Test that wandb settings take middle priority"""
        # Create mock wandb settings
        wandb_dir = tmp_path / "wandb"
        wandb_dir.mkdir()
        settings_file = wandb_dir / "settings"
        settings_content = """[default]
entity = wandb-entity
project = wandb-project
"""
        settings_file.write_text(settings_content)
        
        with patch('inspect_weave.config.wandb_settings_source.wandb_dir', return_value=str(wandb_dir)):
            # Create settings with init values (should be overridden by wandb settings)
            settings = ModelsSettings(
                WANDB_PROJECT="init-project",
                WANDB_ENTITY="init-entity"
            )
            
            # Wandb settings should win over init settings
            assert settings.project == "wandb-project"
            assert settings.entity == "wandb-entity"
            assert settings.enabled is True  # Default value

    def test_init_settings_override_pyproject(self, tmp_path):
        """Test that init settings override pyproject.toml"""
        # Create pyproject.toml
        pyproject_content = """
[tool.inspect-weave.models]
enabled = false
files = ["toml-file1.txt", "toml-file2.txt"]
"""
        pyproject_path = tmp_path / "pyproject.toml"
        pyproject_path.write_text(pyproject_content)
        
        # Create empty wandb dir (no settings file)
        wandb_dir = tmp_path / "wandb"
        wandb_dir.mkdir()
        
        # Change to temp directory so pyproject.toml is found
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            with patch('inspect_weave.config.wandb_settings_source.wandb_dir', return_value=str(wandb_dir)):
                settings = ModelsSettings(
                    WANDB_PROJECT="init-project",
                    WANDB_ENTITY="init-entity",
                    enabled=True  # Should override pyproject.toml
                )
                
                assert settings.enabled is True  # Init settings win over pyproject
                assert settings.files == ["toml-file1.txt", "toml-file2.txt"]  # From pyproject.toml
                assert settings.project == "init-project"  # From init settings
                assert settings.entity == "init-entity"  # From init settings
        finally:
            os.chdir(original_cwd)

    def test_complete_priority_order(self, monkeypatch, tmp_path):
        """Test complete priority order: env vars > wandb > init settings > pyproject.toml"""
        # Create wandb settings (middle priority)
        wandb_dir = tmp_path / "wandb"
        wandb_dir.mkdir()
        settings_file = wandb_dir / "settings"
        settings_content = """[default]
entity = wandb-entity
project = wandb-project
"""
        settings_file.write_text(settings_content)
        
        # Create pyproject.toml (lowest priority)
        pyproject_content = """
[tool.inspect-weave.models]
enabled = false
files = ["toml-file.txt"]
"""
        pyproject_path = tmp_path / "pyproject.toml"
        pyproject_path.write_text(pyproject_content)
        
        # Set some env vars (highest priority)
        monkeypatch.setenv("INSPECT_WEAVE_MODELS_ENABLED", "true")
        monkeypatch.setenv("INSPECT_WEAVE_MODELS_FILES", '["env-file.txt"]')
        # Leave WANDB_PROJECT/ENTITY unset to test wandb settings
        
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            with patch('inspect_weave.config.wandb_settings_source.wandb_dir', return_value=str(wandb_dir)):
                settings = ModelsSettings(
                    WANDB_PROJECT="init-project",  # Should be overridden by wandb settings
                    WANDB_ENTITY="init-entity"     # Should be overridden by wandb settings
                )
                
                # Environment variables win over everything
                assert settings.enabled is True  # Env var overrides pyproject
                assert settings.files == ["env-file.txt"]  # Env var overrides pyproject
                
                # Wandb settings win over init settings
                assert settings.project == "wandb-project"  # Wandb wins over init
                assert settings.entity == "wandb-entity"  # Wandb wins over init
        finally:
            os.chdir(original_cwd)

    def test_config_field_serialization(self, monkeypatch, tmp_path):
        """Test that config field handles complex dictionary data"""
        # Create mock wandb settings
        wandb_dir = tmp_path / "wandb"
        wandb_dir.mkdir()
        settings_file = wandb_dir / "settings"
        settings_content = """[default]
entity = test-entity
project = test-project
"""
        settings_file.write_text(settings_content)
        
        config_json = '{"learning_rate": 0.001, "batch_size": 32, "nested": {"value": true}}'
        monkeypatch.setenv("INSPECT_WEAVE_MODELS_CONFIG", config_json)
        
        with patch('inspect_weave.config.wandb_settings_source.wandb_dir', return_value=str(wandb_dir)):
            settings = ModelsSettings()
            
            expected_config = {
                "learning_rate": 0.001,
                "batch_size": 32,
                "nested": {"value": True}
            }
            assert settings.config == expected_config

    def test_pyproject_toml_field_names(self, tmp_path):
        """Test that field names (entity, project) work correctly in pyproject.toml"""
        # Create pyproject.toml using field names
        pyproject_content = """
[tool.inspect-weave.models]
enabled = false
entity = "field-entity"
project = "field-project"
files = ["field-file.txt"]
"""
        pyproject_path = tmp_path / "pyproject.toml"
        pyproject_path.write_text(pyproject_content)
        
        # Create empty wandb dir (no settings file)
        wandb_dir = tmp_path / "wandb"
        wandb_dir.mkdir()
        
        # Change to temp directory so pyproject.toml is found
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            with patch('inspect_weave.config.wandb_settings_source.wandb_dir', return_value=str(wandb_dir)):
                settings = ModelsSettings()
                
                assert settings.enabled is False
                assert settings.entity == "field-entity"
                assert settings.project == "field-project"
                assert settings.files == ["field-file.txt"]
        finally:
            os.chdir(original_cwd)
    
    def test_pyproject_toml_alias_names(self, tmp_path):
        """Test that alias names (WANDB_ENTITY, WANDB_PROJECT) work correctly in pyproject.toml"""
        # Create pyproject.toml using alias names
        pyproject_content = """
[tool.inspect-weave.models]
enabled = false
WANDB_ENTITY = "alias-entity"
WANDB_PROJECT = "alias-project"
files = ["alias-file.txt"]
"""
        pyproject_path = tmp_path / "pyproject.toml"
        pyproject_path.write_text(pyproject_content)
        
        # Create empty wandb dir (no settings file)
        wandb_dir = tmp_path / "wandb"
        wandb_dir.mkdir()
        
        # Change to temp directory so pyproject.toml is found
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            with patch('inspect_weave.config.wandb_settings_source.wandb_dir', return_value=str(wandb_dir)):
                settings = ModelsSettings()
                
                assert settings.enabled is False
                assert settings.entity == "alias-entity"
                assert settings.project == "alias-project"
                assert settings.files == ["alias-file.txt"]
        finally:
            os.chdir(original_cwd)
    
    def test_pyproject_toml_field_vs_alias_consistency(self, tmp_path):
        """Test that field names and alias names produce the same result"""
        # Test with field names
        pyproject_content_field = """
[tool.inspect-weave.models]
entity = "test-entity"
project = "test-project"
"""
        pyproject_path = tmp_path / "pyproject.toml"
        pyproject_path.write_text(pyproject_content_field)
        
        wandb_dir = tmp_path / "wandb"
        wandb_dir.mkdir()
        
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            with patch('inspect_weave.config.wandb_settings_source.wandb_dir', return_value=str(wandb_dir)):
                settings_field = ModelsSettings()
                
            # Test with alias names
            pyproject_content_alias = """
[tool.inspect-weave.models]
WANDB_ENTITY = "test-entity"
WANDB_PROJECT = "test-project"
"""
            pyproject_path.write_text(pyproject_content_alias)
            
            with patch('inspect_weave.config.wandb_settings_source.wandb_dir', return_value=str(wandb_dir)):
                settings_alias = ModelsSettings()
                
            # Both should produce identical results
            assert settings_field.entity == settings_alias.entity == "test-entity"
            assert settings_field.project == settings_alias.project == "test-project"
            assert settings_field.enabled == settings_alias.enabled  # Should both be True (default)
        finally:
            os.chdir(original_cwd)


class TestWeaveSettings:
    """Test WeaveSettings pydantic model with priority order verification"""
    
    def test_default_values_with_mock_wandb(self, tmp_path):
        """Test default values when wandb settings are available"""
        # Create mock wandb settings
        wandb_dir = tmp_path / "wandb"
        wandb_dir.mkdir()
        settings_file = wandb_dir / "settings"
        settings_content = """[default]
entity = wandb-entity
project = wandb-project
"""
        settings_file.write_text(settings_content)
        
        with patch('inspect_weave.config.wandb_settings_source.wandb_dir', return_value=str(wandb_dir)):
            settings = WeaveSettings()
            
            assert settings.enabled is True
            assert settings.entity == "wandb-entity"
            assert settings.project == "wandb-project"

    def test_environment_variables_highest_priority(self, monkeypatch, tmp_path):
        """Test that environment variables take highest priority"""
        # Create mock wandb settings
        wandb_dir = tmp_path / "wandb"
        wandb_dir.mkdir()
        settings_file = wandb_dir / "settings"
        settings_content = """[default]
entity = wandb-entity
project = wandb-project
"""
        settings_file.write_text(settings_content)
        
        # Note: project/entity use WANDB_* aliases, not prefixed versions
        monkeypatch.setenv("INSPECT_WEAVE_WEAVE_ENABLED", "false")
        monkeypatch.setenv("WANDB_PROJECT", "env-weave-project")  # Uses alias
        monkeypatch.setenv("WANDB_ENTITY", "env-weave-entity")    # Uses alias
        
        with patch('inspect_weave.config.wandb_settings_source.wandb_dir', return_value=str(wandb_dir)):
            settings = WeaveSettings()
            
            assert settings.enabled is False
            assert settings.project == "env-weave-project"  # Env wins over wandb
            assert settings.entity == "env-weave-entity"    # Env wins over wandb

    def test_pyproject_toml_lowest_priority(self, tmp_path):
        """Test that pyproject.toml takes lowest priority"""
        pyproject_content = """
[tool.inspect-weave.weave]
enabled = false
"""
        pyproject_path = tmp_path / "pyproject.toml"
        pyproject_path.write_text(pyproject_content)
        
        # Create mock wandb settings
        wandb_dir = tmp_path / "wandb"
        wandb_dir.mkdir()
        settings_file = wandb_dir / "settings"
        settings_content = """[default]
entity = wandb-entity
project = wandb-project
"""
        settings_file.write_text(settings_content)
        
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            with patch('inspect_weave.config.wandb_settings_source.wandb_dir', return_value=str(wandb_dir)):
                settings = WeaveSettings()
                
                assert settings.enabled is False  # From pyproject.toml
                assert settings.project == "wandb-project"  # From wandb settings
                assert settings.entity == "wandb-entity"  # From wandb settings
        finally:
            os.chdir(original_cwd)

    def test_pyproject_toml_field_names(self, tmp_path):
        """Test that field names (entity, project) work correctly in pyproject.toml"""
        # Create pyproject.toml using field names
        pyproject_content = """
[tool.inspect-weave.weave]
enabled = false
entity = "field-entity"
project = "field-project"
"""
        pyproject_path = tmp_path / "pyproject.toml"
        pyproject_path.write_text(pyproject_content)
        
        # Create empty wandb dir (no settings file)
        wandb_dir = tmp_path / "wandb"
        wandb_dir.mkdir()
        
        # Change to temp directory so pyproject.toml is found
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            with patch('inspect_weave.config.wandb_settings_source.wandb_dir', return_value=str(wandb_dir)):
                settings = WeaveSettings()
                
                assert settings.enabled is False
                assert settings.entity == "field-entity"
                assert settings.project == "field-project"
        finally:
            os.chdir(original_cwd)
    
    def test_pyproject_toml_alias_names(self, tmp_path):
        """Test that alias names (WANDB_ENTITY, WANDB_PROJECT) work correctly in pyproject.toml"""
        # Create pyproject.toml using alias names
        pyproject_content = """
[tool.inspect-weave.weave]
enabled = false
WANDB_ENTITY = "alias-entity"
WANDB_PROJECT = "alias-project"
"""
        pyproject_path = tmp_path / "pyproject.toml"
        pyproject_path.write_text(pyproject_content)
        
        # Create empty wandb dir (no settings file)
        wandb_dir = tmp_path / "wandb"
        wandb_dir.mkdir()
        
        # Change to temp directory so pyproject.toml is found
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            with patch('inspect_weave.config.wandb_settings_source.wandb_dir', return_value=str(wandb_dir)):
                settings = WeaveSettings()
                
                assert settings.enabled is False
                assert settings.entity == "alias-entity"
                assert settings.project == "alias-project"
        finally:
            os.chdir(original_cwd)
    
    def test_pyproject_toml_field_vs_alias_consistency(self, tmp_path):
        """Test that field names and alias names produce the same result"""
        # Test with field names
        pyproject_content_field = """
[tool.inspect-weave.weave]
entity = "test-entity"
project = "test-project"
"""
        pyproject_path = tmp_path / "pyproject.toml"
        pyproject_path.write_text(pyproject_content_field)
        
        wandb_dir = tmp_path / "wandb"
        wandb_dir.mkdir()
        
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            with patch('inspect_weave.config.wandb_settings_source.wandb_dir', return_value=str(wandb_dir)):
                settings_field = WeaveSettings()
                
            # Test with alias names
            pyproject_content_alias = """
[tool.inspect-weave.weave]
WANDB_ENTITY = "test-entity"
WANDB_PROJECT = "test-project"
"""
            pyproject_path.write_text(pyproject_content_alias)
            
            with patch('inspect_weave.config.wandb_settings_source.wandb_dir', return_value=str(wandb_dir)):
                settings_alias = WeaveSettings()
                
            # Both should produce identical results
            assert settings_field.entity == settings_alias.entity == "test-entity"
            assert settings_field.project == settings_alias.project == "test-project"
            assert settings_field.enabled == settings_alias.enabled  # Should both be True (default)
        finally:
            os.chdir(original_cwd)


class TestInspectWeaveSettings:
    """Test the main InspectWeaveSettings composite model"""
    
    def test_composite_model_structure(self, tmp_path):
        """Test that composite model correctly contains both settings"""
        # Create mock wandb settings
        wandb_dir = tmp_path / "wandb"
        wandb_dir.mkdir()
        settings_file = wandb_dir / "settings"
        settings_content = """[default]
entity = test-entity
project = test-project
"""
        settings_file.write_text(settings_content)
        
        with patch('inspect_weave.config.wandb_settings_source.wandb_dir', return_value=str(wandb_dir)):
            settings = InspectWeaveSettings(
                weave=WeaveSettings(),
                models=ModelsSettings()
            )
            
            assert isinstance(settings.weave, WeaveSettings)
            assert isinstance(settings.models, ModelsSettings)
            assert settings.weave.project == "test-project"
            assert settings.models.project == "test-project"


class TestSettingsLoader:
    """Test the SettingsLoader integration with pydantic models and WandBSettingsSource"""
    
    @patch('inspect_weave.config.wandb_settings_source.wandb_dir')
    def test_load_inspect_weave_settings_success(self, mock_wandb_dir, tmp_path):
        """Test successful loading with wandb settings file"""
        # Create mock wandb settings file
        wandb_dir = tmp_path / "wandb"
        wandb_dir.mkdir()
        settings_file = wandb_dir / "settings"
        settings_content = """[default]
entity = test-entity
project = test-project
"""
        settings_file.write_text(settings_content)
        mock_wandb_dir.return_value = str(wandb_dir)
        
        # Load settings
        settings = SettingsLoader.load_inspect_weave_settings()
        
        assert isinstance(settings, InspectWeaveSettings)
        assert settings.weave.entity == "test-entity"
        assert settings.weave.project == "test-project"
        assert settings.models.entity == "test-entity"
        assert settings.models.project == "test-project"
        assert settings.weave.enabled is True
        assert settings.models.enabled is True

    @patch('inspect_weave.config.wandb_settings_source.wandb_dir')
    def test_load_inspect_weave_settings_with_env_override(self, mock_wandb_dir, tmp_path, monkeypatch):
        """Test that environment variables override wandb settings"""
        # Create mock wandb settings file
        wandb_dir = tmp_path / "wandb"
        wandb_dir.mkdir()
        settings_file = wandb_dir / "settings"
        settings_content = """[default]
entity = wandb-entity
project = wandb-project
"""
        settings_file.write_text(settings_content)
        mock_wandb_dir.return_value = str(wandb_dir)
        
        # Set environment variables
        monkeypatch.setenv("INSPECT_WEAVE_WEAVE_ENABLED", "false")
        monkeypatch.setenv("INSPECT_WEAVE_MODELS_ENABLED", "false")
        monkeypatch.setenv("WANDB_PROJECT", "env-project")
        monkeypatch.setenv("WANDB_ENTITY", "env-entity")
        
        settings = SettingsLoader.load_inspect_weave_settings()
        
        # Environment variables should override wandb settings
        assert settings.weave.entity == "env-entity"
        assert settings.weave.project == "env-project"
        assert settings.models.entity == "env-entity"
        assert settings.models.project == "env-project"
        
        # Environment variables should override enabled flags
        assert settings.weave.enabled is False
        assert settings.models.enabled is False

    @patch('inspect_weave.config.wandb_settings_source.wandb_dir')
    def test_load_inspect_weave_settings_with_pyproject_customization(self, mock_wandb_dir, tmp_path):
        """Test loading with pyproject.toml customizations"""
        # Create mock wandb settings file
        wandb_dir = tmp_path / "wandb"
        wandb_dir.mkdir()
        settings_file = wandb_dir / "settings"
        settings_content = """[default]
entity = wandb-entity
project = wandb-project
"""
        settings_file.write_text(settings_content)
        mock_wandb_dir.return_value = str(wandb_dir)
        
        # Create pyproject.toml with customizations
        pyproject_content = """
[tool.inspect-weave.weave]
enabled = false

[tool.inspect-weave.models]
enabled = true
files = ["model_config.yaml"]
"""
        pyproject_path = tmp_path / "pyproject.toml"
        pyproject_path.write_text(pyproject_content)
        
        # Change to temp directory
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            settings = SettingsLoader.load_inspect_weave_settings()
            
            # Wandb settings should be used for entity/project
            assert settings.weave.entity == "wandb-entity"
            assert settings.weave.project == "wandb-project"
            assert settings.models.entity == "wandb-entity"
            assert settings.models.project == "wandb-project"
            
            # Pyproject customizations should be applied
            assert settings.weave.enabled is False
            assert settings.models.enabled is True
            assert settings.models.files == ["model_config.yaml"]
        finally:
            os.chdir(original_cwd)

    @patch('inspect_weave.config.wandb_settings_source.wandb_dir')
    def test_wandb_settings_file_not_found(self, mock_wandb_dir, tmp_path):
        """Test error handling when wandb settings file is not found"""
        wandb_dir = tmp_path / "wandb"
        wandb_dir.mkdir()
        mock_wandb_dir.return_value = str(wandb_dir)
        # Don't create settings file
        
        with pytest.raises(Exception):  # Should raise validation error for missing entity/project
            SettingsLoader.load_inspect_weave_settings()




class TestPriorityOrderIntegration:
    """Integration tests for complete priority order across all settings with WandBSettingsSource"""
    
    @patch('inspect_weave.config.wandb_settings_source.wandb_dir')
    def test_complete_priority_integration(self, mock_wandb_dir, tmp_path, monkeypatch):
        """Test the complete priority order in a realistic scenario with WandBSettingsSource"""
        # 1. Create wandb settings (middle priority)
        wandb_dir = tmp_path / "wandb"
        wandb_dir.mkdir()
        settings_file = wandb_dir / "settings"
        settings_content = """[default]
entity = wandb-entity
project = wandb-project
"""
        settings_file.write_text(settings_content)
        mock_wandb_dir.return_value = str(wandb_dir)
        
        # 2. Create pyproject.toml (lowest priority)
        pyproject_content = """
[tool.inspect-weave.weave]
enabled = false

[tool.inspect-weave.models]
enabled = false
files = ["pyproject-file.yaml"]
config = {from = "pyproject"}
"""
        pyproject_path = tmp_path / "pyproject.toml"
        pyproject_path.write_text(pyproject_content)
        
        # 3. Set environment variables (highest priority)
        monkeypatch.setenv("INSPECT_WEAVE_WEAVE_ENABLED", "true")  # Override pyproject
        monkeypatch.setenv("INSPECT_WEAVE_MODELS_FILES", '["env-file.yaml"]')  # Override pyproject
        # Leave models.enabled unset to test pyproject fallback
        # Leave config unset to test pyproject fallback
        # Leave WANDB_PROJECT/ENTITY unset to test wandb settings
        
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            settings = SettingsLoader.load_inspect_weave_settings()
            
            # Environment variables should win (highest priority)
            assert settings.weave.enabled is True  # Env var overrides pyproject
            assert settings.models.files == ["env-file.yaml"]  # Env var overrides pyproject
            
            # Pyproject should be used when no env var (lowest priority)
            assert settings.models.enabled is False  # From pyproject (no env var set)
            assert settings.models.config == {"from": "pyproject"}  # From pyproject
            
            # Wandb settings should be used for entity/project (middle priority)
            assert settings.weave.entity == "wandb-entity"
            assert settings.weave.project == "wandb-project"
            assert settings.models.entity == "wandb-entity"
            assert settings.models.project == "wandb-project"
        finally:
            os.chdir(original_cwd)

    def test_validation_errors_without_wandb(self, tmp_path):
        """Test that validation errors are raised when no wandb settings and no env vars"""
        # Create empty wandb dir (no settings file)
        wandb_dir = tmp_path / "wandb"
        wandb_dir.mkdir()
        
        with patch('inspect_weave.config.wandb_settings_source.wandb_dir', return_value=str(wandb_dir)):
            # Missing required fields should raise validation error
            with pytest.raises(Exception):  # Pydantic validation error
                ModelsSettings()  # Missing project and entity
                
            with pytest.raises(Exception):  # Pydantic validation error
                WeaveSettings()  # Missing project and entity