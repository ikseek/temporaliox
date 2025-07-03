# TemporalioX

A helper library for Temporal.io that enables separation of activity declarations from their implementations.

## Installation

```bash
pip install temporaliox
```

## Usage

### 1. Declare Activities

In your `activities.py` file, declare activities with their Temporal configuration:

```python
from temporaliox.activity import decl
from datetime import timedelta

@decl(task_queue="playwright", start_to_close_timeout=timedelta(seconds=60))
def generate_report(user_name: str, action: str) -> str:
    pass
```

### 2. Define Activity Implementations

In your `activities_def.py` file, provide the actual implementations:

```python
from .activities import generate_report

@generate_report.defn
def generate_report_def(user_name: str, action: str) -> str:
    return f"{action}, {user_name}!"
```

### 3. Use Activities in Workflows

In your `workflows.py` file, call activities as normal async functions:

```python
from temporalio import workflow
from .activities import generate_report

@workflow.defn()
class GenerateReport:
    @workflow.run
    async def run(self):
        # This will call temporalio.workflow.execute_activity under the hood
        report = await generate_report("Alice", "Hello")
        # report will be "Hello, Alice!"
        return report
```

### Starting Activities

You can also start activities asynchronously using the `.start()` method:

```python
# Start activity and get handle
handle = generate_report.start("Bob", "Goodbye")
# Wait for result later
result = await handle
```

### 4. Set Up Workers

Use `activities_for_queue()` to automatically collect all implemented activities for a specific task queue:

```python
from temporalio.worker import Worker
from temporaliox.activity import activities_for_queue

async def main():
    worker = Worker(
        client,
        task_queue="playwright",
        workflows=[GenerateReport],
        activities=activities_for_queue("playwright"),  # Automatically gets all activities for this queue
    )
    await worker.run()
```

The `activities_for_queue()` function will:
- Return all implemented activities for the specified queue
- Raise a `ValueError` if any declared activities are missing implementations
- Return an empty list if no activities exist for the queue

## Features

- **Separation of Concerns**: Declare activity signatures and Temporal configuration separately from implementations
- **Type Safety**: Full type hints support for better IDE experience
- **Automatic Registration**: Activities are automatically registered with Temporal using the `@stub.defn` decorator
- **Flexible Execution**: Support for both `await` (execute_activity) and `.start()` (start_activity) patterns
- **Queue-based Organization**: Activities are organized by task queue for easy worker setup

## How It Works

1. `@decl()` creates an `ActivityStub` that holds the activity configuration and registers it as undefined
2. `@stub.defn` decorates the implementation, registers it with Temporal, and moves it to the implemented registry
3. When called in a workflow, the stub uses `workflow.execute_activity` or `workflow.start_activity` with the configured options
4. `activities_for_queue()` collects all implemented activities for a queue, ensuring no undefined activities remain