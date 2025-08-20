import pytest
import configparser
import os
from typing import Callable
from pathlib import Path
from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import exact
from inspect_ai.solver import generate, Solver, TaskState, Generate, solver
from unittest.mock import MagicMock
import inspect_ai.hooks._startup as hooks_startup_module
from unittest.mock import patch
from inspect_weave.providers import weave_evaluation_hooks
from pytest import TempPathFactory
from inspect_ai._util.registry import registry_find
from weave.evaluation.eval_imperative import EvaluationLogger

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


pytest_plugins = ["tests.conftest_weave_client"]

## Setup wandb directory and settings

@pytest.fixture(scope="session")
def wandb_path(tmp_path_factory: TempPathFactory) -> Path:
    """
    Returns the path to the wandb directory.
    """
    path = tmp_path_factory.mktemp("test")
    os.makedirs(path / "wandb")
    os.chdir(path)
    return path / "wandb"

@pytest.fixture(scope="session", autouse=True)
def initialise_wandb(wandb_path: Path) -> None:
    """
    Writes a wandb settings file to the tmp_path directory, and changes the current working directory to the tmp_path.
    """
    config = configparser.ConfigParser()
    config["default"] = {
        "entity": "test-entity",
        "project": "test-project"
    }
    with open(wandb_path / "settings", "w") as f:
        config.write(f)


## Mock wandb/weave client calls

@pytest.fixture(scope="function", autouse=True)
def patch_wandb_client():
    mock_config = MagicMock()
    mock_config.update = MagicMock()
    mock_summary = MagicMock()
    mock_summary.update = MagicMock()
    mock_log = MagicMock()
    mock_save = MagicMock()
    mock_wandb_init = MagicMock()
    with (
        patch("inspect_weave.hooks.model_hooks.wandb.init", mock_wandb_init),
        patch("inspect_weave.hooks.model_hooks.wandb.save", mock_save),
        patch("inspect_weave.hooks.model_hooks.wandb.config", mock_config),
        patch("inspect_weave.hooks.model_hooks.wandb.summary", mock_summary),
        patch("inspect_weave.hooks.model_hooks.wandb.log", mock_log)
    ):
        yield mock_wandb_init, mock_save, mock_config, mock_summary, mock_log

@pytest.fixture(scope="function")
def reset_inspect_ai_hooks():
    hooks_startup_module._registry_hooks_loaded = False
    yield
    # reload settings for every test
    hooks = registry_find(lambda x: x.type == "hooks")
    if hooks:
        for hook in hooks:
            hook.settings = None # type: ignore

@pytest.fixture(scope="function")
def patched_weave_evaluation_hooks(reset_inspect_ai_hooks: None):
    patched_evaluation_logger_class = MagicMock(spec=EvaluationLogger)
    patched_evaluation_logger_class.return_value = patched_evaluation_logger_class
    patched_evaluation_logger_class._is_finalized = False
    patched_evaluation_logger_class.finish = MagicMock(side_effect=lambda *args, **kwargs: setattr(patched_evaluation_logger_class, "_is_finalized", True))
    patched_evaluation_logger_class.log_summary = MagicMock()
    patched_evaluation_logger_class.log_prediction = MagicMock()
    patched_evaluation_logger_class._evaluate_call = MagicMock()

    with (
        patch("inspect_weave.hooks.weave_hooks.weave.init", MagicMock()) as weave_init,
        patch("inspect_weave.hooks.weave_hooks.weave.finish", MagicMock()) as weave_finish,
        patch("inspect_weave.hooks.weave_hooks.CustomEvaluationLogger", patched_evaluation_logger_class)
    ):
        weave_evaluation_hooks_instance = weave_evaluation_hooks() # type: ignore
        with patch("inspect_weave._registry.weave_evaluation_hooks", lambda: weave_evaluation_hooks_instance):
            yield {
                "weave_evaluation_hooks": weave_evaluation_hooks_instance,
                "weave_init": weave_init,
                "weave_finish": weave_finish,
                "weave_evaluation_logger": patched_evaluation_logger_class
            }

@pytest.fixture(scope="function")
def hello_world_eval() -> Callable[[], Task]:
    """
    Returns a mock Inspect eval plus a set of mocks that can be used to check that Weave was called correctly.
    """
    @task
    def hello_world():
        return Task(
            dataset=[
                Sample(
                    input="Just reply with Hello World",
                    target="Hello World",
                )
            ],
            solver=[generate()],
            scorer=exact(),
            metadata={"test": "test"}
        )

    return hello_world


@solver
def raise_error() -> Solver:
    """Raises an error."""
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        raise RuntimeError("Simulated failure")

    return solve

@pytest.fixture(scope="function")
def error_eval() -> Callable[[], Task]:
    """
    Returns a mock Inspect eval plus a set of mocks that can be used to check that Weave was called correctly.
    """
    @task
    def hello_world_with_error():
        return Task(
            dataset=[
                Sample(
                    input="Just reply with Hello World",
                    target="Hello World",
                )
            ],
            solver=[raise_error()],
            scorer=exact(),
            metadata={"test": "test"}
        )

    return hello_world_with_error


