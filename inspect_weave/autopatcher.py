import importlib

import weave
from weave.integrations.patcher import SymbolPatcher, MultiPatcher
from weave.trace.autopatch import AutopatchSettings, IntegrationSettings
from pydantic import Field


inspect_patcher = MultiPatcher(
    [
        SymbolPatcher(
            lambda: importlib.import_module("inspect_ai._eval.run"),
            "task_run",
            weave.op(name="inspect_task"),
        )
    ]
)

def get_inspect_patcher(settings: IntegrationSettings | None = None) -> MultiPatcher:
    return inspect_patcher

class CustomAutopatchSettings(AutopatchSettings):
    inspect: IntegrationSettings = Field(default_factory=IntegrationSettings)

def autopatch_inspect(settings: CustomAutopatchSettings) -> None:
    get_inspect_patcher(settings.inspect).attempt_patch()

def reset_autopatch_inspect() -> None:
    get_inspect_patcher().undo_patch()