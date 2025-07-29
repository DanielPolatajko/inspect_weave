import pytest
import weave
from typing import Generator
from inspect_ai import task, Task, eval
from inspect_ai.solver import generate
from inspect_ai.scorer import exact
from inspect_ai.dataset import Sample


@pytest.fixture(scope="function")
def patch_inspect() -> Generator[None, None, None]:
    print("TODO: Implement")

@pytest.mark.skip_clickhouse_client
@pytest.mark.vcr(
    filter_headers=["authorization"],
    allowed_hosts=["api.wandb.ai", "localhost"],
)
def test_inspect_quickstart(
    client: weave.trace.weave_client.WeaveClient,
    patch_inspect: None,
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
    assert len(calls) == 1