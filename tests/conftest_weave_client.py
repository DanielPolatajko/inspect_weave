import logging
from weave.trace import autopatch, weave_client, weave_init
from weave.trace.context.call_context import set_call_stack
from weave.trace_server import trace_server_interface as tsi
from weave.trace_server_bindings import remote_http_trace_server
from weave.trace_server_bindings.caching_middleware_trace_server import (
    CachingMiddlewareTraceServer,
)
import typing
import weave
from typing import Callable

import pytest
from weave.trace_server.sqlite_trace_server import SqliteTraceServer
import os

import time
import urllib

import requests

import base64

from weave.trace_server import (
    external_to_internal_trace_server_adapter,
)

"""
This file is a pared-down version of the weave test client from the client lib, since it's pretty useful for testing.
We should write our own mock client when the chance arises.
"""


class TwoWayMapping:
    def __init__(self):
        self._ext_to_int_map = {}
        self._int_to_ext_map = {}

        # Useful for testing to ensure caching is working
        self.stats = {
            "ext_to_int": {
                "hits": 0,
                "misses": 0,
            },
            "int_to_ext": {
                "hits": 0,
                "misses": 0,
            },
        }

    def ext_to_int(self, key, default=None):
        if key not in self._ext_to_int_map:
            if default is None:
                raise ValueError(f"Key {key} not found")
            if default in self._int_to_ext_map:
                raise ValueError(f"Default {default} already in use")
            self._ext_to_int_map[key] = default
            self._int_to_ext_map[default] = key
            self.stats["ext_to_int"]["misses"] += 1
        else:
            self.stats["ext_to_int"]["hits"] += 1
        return self._ext_to_int_map[key]

    def int_to_ext(self, key, default):
        if key not in self._int_to_ext_map:
            if default is None:
                raise ValueError(f"Key {key} not found")
            if default in self._ext_to_int_map:
                raise ValueError(f"Default {default} already in use")
            self._int_to_ext_map[key] = default
            self._ext_to_int_map[default] = key
            self.stats["int_to_ext"]["misses"] += 1
        else:
            self.stats["int_to_ext"]["hits"] += 1
        return self._int_to_ext_map[key]


def b64(s: str) -> str:
    # Base64 encode the string
    return base64.b64encode(s.encode("ascii")).decode("ascii")


class DummyIdConverter(external_to_internal_trace_server_adapter.IdConverter):
    def __init__(self):
        self._project_map = TwoWayMapping()
        self._run_map = TwoWayMapping()
        self._user_map = TwoWayMapping()

    def ext_to_int_project_id(self, project_id: str) -> str:
        return self._project_map.ext_to_int(project_id, b64(project_id))

    def int_to_ext_project_id(self, project_id: str) -> typing.Optional[str]:
        return self._project_map.int_to_ext(project_id, b64(project_id))

    def ext_to_int_run_id(self, run_id: str) -> str:
        return self._run_map.ext_to_int(run_id, b64(run_id) + ":" + run_id)

    def int_to_ext_run_id(self, run_id: str) -> str:
        exp = run_id.split(":")[1]
        return self._run_map.int_to_ext(run_id, exp)

    def ext_to_int_user_id(self, user_id: str) -> str:
        return self._user_map.ext_to_int(user_id, b64(user_id))

    def int_to_ext_user_id(self, user_id: str) -> str:
        return self._user_map.int_to_ext(user_id, b64(user_id))


class TestOnlyUserInjectingExternalTraceServer(
    external_to_internal_trace_server_adapter.ExternalTraceServer
):
    def __init__(
        self,
        internal_trace_server: tsi.TraceServerInterface,
        id_converter: external_to_internal_trace_server_adapter.IdConverter,
        user_id: str,
    ):
        super().__init__(internal_trace_server, id_converter)
        self._user_id = user_id

    def call_start(self, req: tsi.CallStartReq) -> tsi.CallStartRes:
        req.start.wb_user_id = self._user_id
        return super().call_start(req)

    def calls_delete(self, req: tsi.CallsDeleteReq) -> tsi.CallsDeleteRes:
        req.wb_user_id = self._user_id
        return super().calls_delete(req)

    def call_update(self, req: tsi.CallUpdateReq) -> tsi.CallUpdateRes:
        req.wb_user_id = self._user_id
        return super().call_update(req)

    def feedback_create(self, req: tsi.FeedbackCreateReq) -> tsi.FeedbackCreateRes:
        req.wb_user_id = self._user_id
        return super().feedback_create(req)

    def cost_create(self, req: tsi.CostCreateReq) -> tsi.CostCreateRes:
        req.wb_user_id = self._user_id
        return super().cost_create(req)

    def actions_execute_batch(
        self, req: tsi.ActionsExecuteBatchReq
    ) -> tsi.ActionsExecuteBatchRes:
        req.wb_user_id = self._user_id
        return super().actions_execute_batch(req)

    def obj_create(self, req: tsi.ObjCreateReq) -> tsi.ObjCreateRes:
        req.obj.wb_user_id = self._user_id
        return super().obj_create(req)


def externalize_trace_server(
    trace_server: tsi.TraceServerInterface, user_id: str = "test_user"
) -> TestOnlyUserInjectingExternalTraceServer:
    return TestOnlyUserInjectingExternalTraceServer(
        trace_server,
        DummyIdConverter(),
        user_id,
    )


def check_server_up(host, port, num_retries=30) -> bool:
    """Check if a server is up and healthy at the given host:port.

    Args:
        host: The hostname to check
        port: The port to check
        num_retries: Number of retries to attempt (default 30)

    Returns:
        bool: True if server is healthy, False otherwise
    """
    base_url = f"http://{host}:{port}/"
    endpoint = "ping"

    return _check_server_health(
        base_url=base_url, endpoint=endpoint, num_retries=num_retries
    )


def _check_server_health(
    base_url: str, endpoint: str, num_retries: int = 1, sleep_time: int = 1
) -> bool:
    for _ in range(num_retries):
        try:
            response = requests.get(urllib.parse.urljoin(base_url, endpoint))
            if response.status_code == 200:
                return True
            time.sleep(sleep_time)
        except requests.exceptions.ConnectionError:
            time.sleep(sleep_time)

    print(
        f"Server not healthy @ {urllib.parse.urljoin(base_url, endpoint)}: no response"
    )
    return False

# Force testing to never report wandb sentry events
os.environ["WANDB_ERROR_REPORTING"] = "false"
TEST_ENTITY = "shawn"
def get_trace_server_flag(request):
    return "sqlite"


def pytest_sessionfinish(session, exitstatus):
    if exitstatus == pytest.ExitCode.NO_TESTS_COLLECTED:
        session.exitstatus = 0


# https://docs.pytest.org/en/7.1.x/example/simple.html#pytest-current-test-environment-variable
def get_test_name():
    return os.environ.get("PYTEST_CURRENT_TEST", " ").split(" ")[0]


class InMemoryWeaveLogCollector(logging.Handler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_records = {}

    def emit(self, record):
        curr_test = get_test_name()
        if curr_test not in self.log_records:
            self.log_records[curr_test] = []
        self.log_records[curr_test].append(record)

    def _get_logs(self, levelname: str):
        curr_test = get_test_name()
        logs = self.log_records.get(curr_test, [])

        return [
            record
            for record in logs
            if record.levelname == levelname
            and record.name.startswith("weave")
            # (Tim) For some reason that i cannot figure out, there is some test that
            # a) is trying to connect to the PROD trace server
            # b) seemingly doesn't fail
            # c) Logs these errors.
            # I would love to fix this, but I have not been able to pin down which test
            # is causing it and need to ship this PR, so I am just going to filter it out
            # for now.
            and not record.msg.startswith(
                "Task failed: HTTPError: 400 Client Error: Bad Request for url: https://trace.wandb.ai/"
            )
            # Exclude legacy
            and not record.name.startswith("weave.weave_server")
            and "legacy" not in record.name
        ]

    def get_error_logs(self):
        return self._get_logs("ERROR")

    def get_warning_logs(self):
        return self._get_logs("WARNING")


@pytest.fixture
def log_collector(request):
    handler = InMemoryWeaveLogCollector()
    logger = logging.getLogger()  # Get your specific logger here if needed
    logger.addHandler(handler)
    if hasattr(request, "param") and request.param == "warning":
        logger.setLevel(logging.WARNING)
    else:
        logger.setLevel(logging.ERROR)
    yield handler
    logger.removeHandler(handler)  # Clean up after the test


@pytest.fixture(autouse=True)
def logging_error_check(request, log_collector):
    yield
    if "disable_logging_error_check" in request.keywords:
        return
    error_logs = log_collector.get_error_logs()
    if error_logs:
        pytest.fail(
            f"Expected no errors, but found {len(error_logs)} error(s): {error_logs}"
        )


class TestOnlyFlushingWeaveClient(weave_client.WeaveClient):
    """
    A WeaveClient that automatically flushes after every method call.

    This subclass overrides the behavior of the standard WeaveClient to ensure
    that all write operations are immediately flushed to the underlying storage
    before any subsequent read operation is performed. This is particularly
    useful in testing scenarios where data is written and then immediately read
    back, as it eliminates potential race conditions or inconsistencies that
    might arise from delayed writes.

    The flush operation is applied to all public methods (those not starting with
    an underscore) except for the 'flush' method itself, to avoid infinite recursion.
    This aggressive flushing strategy may impact performance and should primarily
    be used in testing environments rather than production scenarios.

    Note: Due to this, the test suite essentially "blocks" on every operation. So
    if we are to test timing, this might not be the best choice. As an alternative,
    we could explicitly call flush() in every single test that reads data, before the
    read operation(s) are performed.
    """

    def set_autoflush(self, value: bool):
        self._autoflush = value

    def __getattribute__(self, name):
        self_super = super()
        attr = self_super.__getattribute__(name)

        if callable(attr) and name != "flush":

            def wrapper(*args, **kwargs):
                res = attr(*args, **kwargs)
                if self.__dict__.get("_autoflush", True):
                    self_super._flush()
                return res

            return wrapper

        return attr


def make_server_recorder(server: tsi.TraceServerInterface):  # type: ignore
    """A wrapper around a trace server that records all attribute access.

    This is extremely helpful for tests to assert that a certain series of
    attribute accesses happen (or don't happen), and in order. We will
    probably want to make this a bit more sophisticated in the future, but
    this is a pretty good start.

    For example, you can do something like the followng to assert that various
    read operations do not happen!

    ```pyth
    access_log = client.server.attribute_access_log
    assert "table_query" not in access_log
    assert "obj_read" not in access_log
    assert "file_content_read" not in access_log
    ```
    """

    class ServerRecorder(type(server)):  # type: ignore
        attribute_access_log: list[str]

        def __init__(self, server: tsi.TraceServerInterface):
            self.server = server
            self.attribute_access_log = []

        def __getattribute__(self, name):
            self_server = super().__getattribute__("server")
            access_log = super().__getattribute__("attribute_access_log")
            if name == "server":
                return self_server
            if name == "attribute_access_log":
                return access_log
            attr = self_server.__getattribute__(name)
            if name != "attribute_access_log":
                access_log.append(name)
            return attr

    return ServerRecorder(server)


@pytest.fixture
def get_sqlite_trace_server() -> Callable[[], TestOnlyUserInjectingExternalTraceServer]:
    def sqlite_trace_server_inner() -> TestOnlyUserInjectingExternalTraceServer:
        sqlite_server = SqliteTraceServer("file::memory:?cache=shared")
        sqlite_server.drop_tables()
        sqlite_server.setup_tables()
        return externalize_trace_server(sqlite_server, TEST_ENTITY)

    return sqlite_trace_server_inner


@pytest.fixture
def trace_server(
    request, get_sqlite_trace_server
) -> TestOnlyUserInjectingExternalTraceServer:
    trace_server_flag = get_trace_server_flag(request)
    if trace_server_flag == "sqlite":
        return get_sqlite_trace_server()
    else:
        # Once we split the trace server and client code, we can raise here.
        # For now, just return the sqlite trace server so we don't break existing tests.
        # raise ValueError(f"Invalid trace server: {trace_server_flag}")
        return get_sqlite_trace_server()


def create_client(
    request,
    trace_server,
    autopatch_settings: typing.Optional[autopatch.AutopatchSettings] = None,
    global_attributes: typing.Optional[dict[str, typing.Any]] = None,
) -> weave_init.InitializedClient:
    trace_server_flag = get_trace_server_flag(request)
    if trace_server_flag == "prod":
        # Note: this is only for local dev testing and should be removed
        return weave_init.init_weave("dev_testing")
    elif trace_server_flag == "http":
        server = remote_http_trace_server.RemoteHTTPTraceServer(trace_server_flag)
    else:
        server = trace_server

    # Removing this as it lead to passing tests that were not passing in prod!
    # Keeping off for now until it is the default behavior.
    # os.environ["WEAVE_USE_SERVER_CACHE"] = "true"
    caching_server = CachingMiddlewareTraceServer.from_env(server)
    client = TestOnlyFlushingWeaveClient(
        TEST_ENTITY, "test-project", make_server_recorder(caching_server)
    )
    inited_client = weave_init.InitializedClient(client)
    autopatch.autopatch(autopatch_settings)
    if global_attributes is not None:
        weave.trace.api._global_attributes = global_attributes

    return inited_client


@pytest.fixture
def zero_stack():
    with set_call_stack([]):
        yield


@pytest.fixture
def client(zero_stack, request, trace_server):
    """This is the standard fixture used everywhere in tests to test end to end
    client functionality"""
    inited_client = create_client(request, trace_server)
    try:
        yield inited_client.client
    finally:
        inited_client.reset()
        autopatch.reset_autopatch()