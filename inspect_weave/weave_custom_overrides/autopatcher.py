import importlib
# from typing import Callable, Any

import weave
from weave.integrations.patcher import SymbolPatcher, MultiPatcher
from weave.trace.autopatch import AutopatchSettings, IntegrationSettings
from pydantic import Field

# def add_weave_op_to_inspect_decorator(decorator: Callable[[Callable[..., Any]], Callable[..., Any]]) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
#     def new_decorator(func: Callable[..., Any]) -> Callable[..., Any]:
#         func_name = getattr(func, '__name__', 'unknown_function')
#         weave_op_kwargs = {'name': func_name}
        
#         # Extract name from @solver(name="...") closure
#         if hasattr(decorator, '__closure__') and decorator.__closure__:
#             closure_vars = decorator.__closure__
#             if hasattr(decorator, '__code__') and hasattr(decorator.__code__, 'co_freevars'):
#                 var_names = decorator.__code__.co_freevars
#                 for i, var_name in enumerate(var_names):
#                     if var_name == 'name' and i < len(closure_vars):
#                         closure_value = closure_vars[i].cell_contents
#                         if isinstance(closure_value, str):
#                             weave_op_kwargs['name'] = closure_value
#                             break
        
#         weave_decorated_func = weave.op(**weave_op_kwargs)(func)
#         return decorator(weave_decorated_func)
#     return new_decorator


inspect_patcher = MultiPatcher(
    [
        SymbolPatcher(
            lambda: importlib.import_module("inspect_ai.solver"),
            "solver",
            weave.op,
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