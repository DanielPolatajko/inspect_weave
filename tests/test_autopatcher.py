import pytest
import weave
from unittest.mock import patch
from inspect_ai import task, Task, eval
from inspect_ai.solver import generate, solver
from inspect_ai.scorer import exact
from inspect_ai.dataset import Sample
from inspect_weave.autopatcher import add_weave_op_to_inspect_decorator


def test_add_weave_op_to_named_solver():
    """Test that add_weave_op_to_inspect_decorator extracts name from @solver(name="...") and passes it to weave.op"""
    
    # Create a named solver to test with
    @solver(name="custom_solver_name")
    def test_solver():
        async def solve(state, generate):
            return state
        return solve
    
    # Mock weave.op to capture the arguments it's called with
    with patch('weave.op') as mock_weave_op:
        # Set up the mock to return a function that returns the input function
        mock_weave_op.return_value = lambda f: f
        
        # Apply our autopatcher function to the solver decorator
        # We need to get the actual decorator function that @solver(name="...") creates
        # The solver decorator with name creates a wrapper function
        original_solver_decorator = solver(name="custom_solver_name")
        patched_decorator = add_weave_op_to_inspect_decorator(original_solver_decorator)
        
        # Apply the patched decorator to a test function
        @patched_decorator
        def my_test_function():
            async def solve(state, generate):
                return state
            return solve
        
        # Verify that weave.op was called with the custom name
        mock_weave_op.assert_called_once_with(name="custom_solver_name")


def test_add_weave_op_to_unnamed_solver():
    """Test that add_weave_op_to_inspect_decorator uses function name when no custom name is provided"""
    
    # Create an unnamed solver
    @solver
    def my_function_name():
        async def solve(state, generate):
            return state
        return solve
    
    with patch('weave.op') as mock_weave_op:
        mock_weave_op.return_value = lambda f: f
        
        # Get the decorator created by @solver (without name)
        original_solver_decorator = solver
        patched_decorator = add_weave_op_to_inspect_decorator(original_solver_decorator)
        
        # Apply to a test function
        @patched_decorator  
        def test_function():
            async def solve(state, generate):
                return state
            return solve
        
        # Should use the function name as default
        mock_weave_op.assert_called_once_with(name="test_function")


@pytest.mark.skip_clickhouse_client
@pytest.mark.vcr(
    filter_headers=["authorization"],
    allowed_hosts=["api.wandb.ai", "localhost"],
)
def test_inspect_quickstart(
    client: weave.trace.weave_client.WeaveClient,
) -> None:
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

    eval(hello_world, model="mockllm/model")

    calls = list(client.calls())
    assert len(calls) == 6
    assert "inspect_task" in calls[1]._op_name