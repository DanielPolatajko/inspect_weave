import pytest
import configparser
import os
from typing import Callable
from pathlib import Path
from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import exact
from inspect_ai.solver import generate
from inspect_ai import eval as inspect_eval
from inspect_ai.log import EvalLog
from unittest.mock import MagicMock
from inspect_weave.hooks import WeaveEvaluationHooks
from inspect_weave.custom_evaluation_logger import CustomEvaluationLogger
import inspect_ai.hooks._startup as hooks_startup_module
from unittest.mock import patch
from inspect_weave.providers import weave_evaluation_hooks


@pytest.fixture(scope="function")
def weave_evaluation_hooks_with_mocked_client_deps():
    with (
        patch("inspect_weave.hooks.weave.init", MagicMock()) as weave_init,
        patch("inspect_weave.hooks.weave.finish", MagicMock()) as weave_finish,
        patch("inspect_weave.hooks.CustomEvaluationLogger", MagicMock(spec=CustomEvaluationLogger)) as weave_evaluation_logger
    ):
        yield weave_evaluation_hooks(), weave_init, weave_finish, weave_evaluation_logger

@pytest.fixture(scope="function")
def register_hooks_for_testing(weave_evaluation_hooks_with_mocked_client_deps: tuple[WeaveEvaluationHooks, MagicMock, MagicMock, MagicMock]) -> dict[str, MagicMock]:
    """
    Override hook registration to ensure we are testing latest version of the package.
    """
    hooks_startup_module._load_registry_hooks = MagicMock(return_value=[weave_evaluation_hooks_with_mocked_client_deps[0]])
    hooks_startup_module._format_hook_for_printing = MagicMock(return_value="[bold]WeaveEvaluationHooks[/bold]: Integration hooks for writing evaluation results to Weave")

    return {
        "weave_init": weave_evaluation_hooks_with_mocked_client_deps[1],
        "weave_finish": weave_evaluation_hooks_with_mocked_client_deps[2],
        "weave_evaluation_logger": weave_evaluation_hooks_with_mocked_client_deps[3]
    }

@pytest.fixture(scope="function")
def inspect_eval_and_weave_mocks(register_hooks_for_testing: dict[str, MagicMock]) -> dict[str, Callable[[], list[EvalLog]] | MagicMock]:
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

    def inspect_eval_callable() -> list[EvalLog]:
        return inspect_eval(hello_world, model="mockllm/model")

    return {
        "inspect_eval": inspect_eval_callable,
    } | register_hooks_for_testing

@pytest.fixture(scope="function")
def initialise_wandb(tmp_path: Path) -> None:
    """
    Writes a wandb settings file to the tmp_path directory, and changes the current working directory to the tmp_path.
    """
    config = configparser.ConfigParser()
    config["default"] = {
        "entity": "test-entity",
        "project": "test-project"
    }
    os.makedirs(tmp_path / "wandb")
    with open(tmp_path / "wandb" / "settings", "w") as f:
        config.write(f)
    os.chdir(tmp_path)


