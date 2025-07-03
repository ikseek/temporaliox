import asyncio
from datetime import timedelta

import pytest
import pytest_asyncio
from temporalio import workflow
from temporalio.client import Client
from temporalio.worker import Worker

from temporaliox.activity import (
    activities_for_queue,
    decl,
)


# Activity declaration
@decl(task_queue="test-queue", start_to_close_timeout=timedelta(seconds=60))
def generate_report(user_name: str, action: str) -> str:
    pass


# Activity implementation
@generate_report.defn
async def generate_report_impl(user_name: str, action: str) -> str:
    return f"{action}, {user_name}!"


# Single string argument activity declaration
@decl(task_queue="test-queue", start_to_close_timeout=timedelta(seconds=60))
def process_message(message: str) -> str:
    pass


# Single string argument activity implementation
@process_message.defn
async def process_message_impl(message: str) -> str:
    return f"Processed: {message}"


# Workflow for testing
@workflow.defn()
class TestWorkflow:
    @workflow.run
    async def run(self, user_name: str, action: str) -> str:
        # Test regular execution
        result = await generate_report(user_name, action)

        # Test start method
        handle = generate_report.start(user_name, f"{action} again")
        result2 = await handle

        return f"{result}|{result2}"


# Workflow for testing with_options() functionality
@workflow.defn()
class TestWithOptionsWorkflow:
    @workflow.run
    async def run(self, user_name: str, action: str) -> str:
        # Test with_options() with timeout override
        result = await generate_report.with_options(
            start_to_close_timeout=timedelta(seconds=30)
        )(user_name, action)

        # Test with_options().start() with different timeout
        handle = generate_report.with_options(
            start_to_close_timeout=timedelta(seconds=45)
        ).start(user_name, f"{action} with with_options")
        result2 = await handle

        return f"{result}|{result2}"


@pytest_asyncio.fixture
async def temporal_client():
    """Create a Temporal client for testing."""
    client = await Client.connect("localhost:7233")
    yield client
    # Client doesn't need explicit cleanup


@pytest_asyncio.fixture
async def temporal_worker(temporal_client):
    """Create and start a Temporal worker for testing."""
    async with Worker(
        temporal_client,
        task_queue="test-queue",
        workflows=[TestWorkflow, TestWithOptionsWorkflow, TestSingleArgWorkflow],
        activities=activities_for_queue("test-queue"),
    ) as worker:
        yield worker


@pytest.mark.asyncio
async def test_activity_execution(temporal_client, temporal_worker):
    """Test that activities execute correctly through Temporal."""
    # Execute workflow
    result = await temporal_client.execute_workflow(
        TestWorkflow.run,
        args=["Alice", "Hello"],
        id=f"test-workflow-{asyncio.get_event_loop().time()}",
        task_queue="test-queue",
    )

    assert result == "Hello, Alice!|Hello again, Alice!"


@pytest.mark.asyncio
async def test_activity_with_different_args(temporal_client, temporal_worker):
    """Test activity with different arguments."""
    result = await temporal_client.execute_workflow(
        TestWorkflow.run,
        args=["Bob", "Goodbye"],
        id=f"test-workflow-{asyncio.get_event_loop().time()}",
        task_queue="test-queue",
    )

    assert result == "Goodbye, Bob!|Goodbye again, Bob!"


# Test workflow that only uses start
@workflow.defn()
class TestStartOnlyWorkflow:
    @workflow.run
    async def run(self, user_name: str) -> str:
        # Start multiple activities
        handle1 = generate_report.start(user_name, "First")
        handle2 = generate_report.start(user_name, "Second")
        handle3 = generate_report.start(user_name, "Third")

        # Wait for all
        results = await asyncio.gather(handle1, handle2, handle3)
        return "|".join(results)


@pytest.mark.asyncio
async def test_multiple_activity_starts(temporal_client):
    """Test starting multiple activities concurrently."""
    async with Worker(
        temporal_client,
        task_queue="test-queue",
        workflows=[TestStartOnlyWorkflow],
        activities=activities_for_queue("test-queue"),
    ):
        result = await temporal_client.execute_workflow(
            TestStartOnlyWorkflow.run,
            args=["Charlie"],
            id=f"test-workflow-{asyncio.get_event_loop().time()}",
            task_queue="test-queue",
        )

        assert result == "First, Charlie!|Second, Charlie!|Third, Charlie!"


# Workflow for testing single string argument activity
@workflow.defn()
class TestSingleArgWorkflow:
    @workflow.run
    async def run(self, message: str) -> str:
        # Test regular execution
        result = await process_message(message)

        # Test start method
        handle = process_message.start(f"{message} again")
        result2 = await handle

        return f"{result}|{result2}"


@pytest.mark.asyncio
async def test_single_str_activity_execution(temporal_client, temporal_worker):
    """Test activity with single string argument."""
    result = await temporal_client.execute_workflow(
        TestSingleArgWorkflow.run,
        args=["Hello World"],
        id=f"test-single-str-workflow-{asyncio.get_event_loop().time()}",
        task_queue="test-queue",
    )

    assert result == "Processed: Hello World|Processed: Hello World again"


@pytest.mark.asyncio
async def test_single_str_activity_with_different_messages(
    temporal_client, temporal_worker
):
    """Test single string activity with different messages."""
    result = await temporal_client.execute_workflow(
        TestSingleArgWorkflow.run,
        args=["Test Message"],
        id=f"test-single-str-workflow-{asyncio.get_event_loop().time()}",
        task_queue="test-queue",
    )

    assert result == "Processed: Test Message|Processed: Test Message again"


@pytest.mark.asyncio
async def test_with_options_functionality_integration(temporal_client, temporal_worker):
    """Test that with_options() functionality works in real Temporal workflows."""
    result = await temporal_client.execute_workflow(
        TestWithOptionsWorkflow.run,
        args=["David", "OptTest"],
        id=f"test-opt-workflow-{asyncio.get_event_loop().time()}",
        task_queue="test-queue",
    )

    assert result == "OptTest, David!|OptTest with with_options, David!"


@pytest.mark.asyncio
async def test_with_options_with_multiple_timeouts(temporal_client):
    """Test with_options() with multiple different timeout overrides."""
    async with Worker(
        temporal_client,
        task_queue="test-queue",
        workflows=[TestWithOptionsWorkflow],
        activities=activities_for_queue("test-queue"),
    ):
        result = await temporal_client.execute_workflow(
            TestWithOptionsWorkflow.run,
            args=["Eve", "MultiTimeout"],
            id=f"test-multi-timeout-workflow-{asyncio.get_event_loop().time()}",
            task_queue="test-queue",
        )

        assert result == "MultiTimeout, Eve!|MultiTimeout with with_options, Eve!"
