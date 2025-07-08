from inspect_weave.utils import format_model_name
import pytest
import re

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