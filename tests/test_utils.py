from inspect_weave.utils import format_model_name, format_score_types
import pytest
import re
from pathlib import Path
import configparser
from inspect_weave.utils import read_wandb_project_name_from_settings
import os

@pytest.mark.parametrize("model_name", [
    "google/vertex/gemini-2.0-flash",
    "anthropic/vertex/claude-3-5-sonnet-v2@20241022",
    "anthropic/bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0",
    "anthropic/claude-3-5-sonnet-latest",
    "hf/openai-community/gpt2"
])
def test_format_model_name_correctly_formats_inspect_valid_model_names(model_name: str) -> None:
    formatted_model_name = format_model_name(model_name)
    assert re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", formatted_model_name)


class TestFormatScoreTypes:
    """Test cases for format_score_types function."""
    
    def test_string_input(self):
        """Test that string input is wrapped in a score dict."""
        result = format_score_types("0.85")
        assert result == {"score": "0.85"}
    
    def test_int_input(self):
        """Test that int input is converted to float."""
        result = format_score_types(85)
        assert result == 85.0
        assert isinstance(result, float)
    
    def test_float_input(self):
        """Test that float input is returned as-is."""
        result = format_score_types(0.85)
        assert result == 0.85
    
    def test_single_element_sequence(self):
        """Test that single-element sequence is wrapped in a score dict."""
        result = format_score_types([0.85])
        assert result == {"score": 0.85}
    
    def test_single_element_tuple(self):
        """Test that single-element tuple is wrapped in a score dict."""
        result = format_score_types((0.85,))
        assert result == {"score": 0.85}
    
    def test_multi_element_sequence_raises_error(self):
        """Test that multi-element sequence raises ValueError."""
        with pytest.raises(ValueError, match="Sequence score cannot be passed to Weave"):
            format_score_types([0.85, 0.90])
    
    def test_empty_sequence_raises_error(self):
        """Test that empty sequence raises ValueError."""
        with pytest.raises(ValueError, match="Sequence score cannot be passed to Weave"):
            format_score_types([])
    
    def test_mapping_input(self):
        """Test that mapping input is converted to dict."""
        input_dict = {"accuracy": 0.85, "precision": 0.90}
        result = format_score_types(input_dict)
        assert result == input_dict
        assert isinstance(result, dict)
    
    def test_dict_with_score_key(self):
        """Test that dict with score key is returned as-is."""
        input_dict = {"score": 0.85}
        result = format_score_types(input_dict)
        assert result == input_dict
    
    def test_other_types_returned_as_is(self):
        """Test that other types (like bool) are converted to appropriate types."""
        assert format_score_types(True) == 1.0
        assert format_score_types(False) == 0.0
    
    def test_complex_nested_mapping(self):
        """Test that complex nested mappings are handled correctly."""
        input_dict = {
            "metrics": {
                "accuracy": 0.85,
                "precision": 0.90
            },
            "score": 0.87
        }
        result = format_score_types(input_dict)
        assert result == input_dict

def test_read_wandb_project_name_from_settings(tmp_path: Path) -> None:
    # Given
    config = configparser.ConfigParser()
    config["default"] = {
        "entity": "test-entity",
        "project": "test-project"
    }
    os.makedirs(tmp_path / "wandb")
    with open(tmp_path / "wandb" / "settings", "w") as f:
        config.write(f)
    os.chdir(tmp_path)

    # When
    project_name = read_wandb_project_name_from_settings()

    # Then
    assert project_name == "test-entity/test-project"

def test_read_wandb_project_name_from_settings_raises_error_if_settings_file_not_found(tmp_path: Path) -> None:
    # Given
    os.chdir(tmp_path)

    # When
    with pytest.raises(ValueError, match="Wandb settings file not found, please run `wandb init` to set up a project"):
        read_wandb_project_name_from_settings()

def test_read_wandb_project_name_from_settings_returns_none_if_mode_is_disabled(tmp_path: Path) -> None:
    # Given
    config = configparser.ConfigParser()
    config["default"] = {
        "entity": "test-entity",
        "project": "test-project",
        "mode": "disabled"
    }
    os.makedirs(tmp_path / "wandb")
    with open(tmp_path / "wandb" / "settings", "w") as f:
        config.write(f)
    os.chdir(tmp_path)

    # When
    project_name = read_wandb_project_name_from_settings()

    # Then
    assert project_name is None