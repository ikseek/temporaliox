from __future__ import annotations

import inspect
from collections import defaultdict
from dataclasses import dataclass
from datetime import timedelta
from functools import cached_property, wraps
from typing import Any, Callable, TypeVar, overload

from temporalio import activity as temporal_activity
from temporalio import workflow
from temporalio.common import Priority, RetryPolicy
from temporalio.workflow import (
    ActivityCancellationType,
    ActivityHandle,
    VersioningIntent,
)

__all__ = ["decl", "ActivityStub", "activities_for_queue"]

T = TypeVar("T", bound=Callable[..., Any])

_undefined_activities: defaultdict[str, set[str]] = defaultdict(set)
_activity_registry: defaultdict[str, list[Callable]] = defaultdict(list)


@dataclass(frozen=True)
class ActivityStub:
    name: str
    signature: inspect.Signature
    options: dict[str, Any]

    @staticmethod
    def from_func(
        func: Callable, task_queue: str, options: dict[str, Any]
    ) -> ActivityStub:
        name = func.__qualname__
        stub = ActivityStub(
            name=name,
            signature=inspect.signature(func),
            options={"task_queue": task_queue, **options},
        )
        _undefined_activities[task_queue].add(name)
        return stub

    def __str__(self) -> str:
        return self.name

    async def __call__(self, *args, **kwargs):
        return await workflow.execute_activity(
            self.name,
            arg=self._args_to_dict(*args, **kwargs),
            **self.options,
        )

    @cached_property
    def param_names(self) -> tuple[str, ...]:
        return tuple(self.signature.parameters.keys())

    def defn(self, impl_func: T) -> T:
        impl_sig = inspect.signature(impl_func)
        if impl_sig != self.signature:
            raise ValueError(
                f"Implementation signature {impl_sig} does not match "
                f"declaration signature {self.signature} for activity "
                f"'{self.name}'"
            )
        if inspect.iscoroutinefunction(impl_func):

            @wraps(impl_func)
            async def kwargs_unpacking_adapter(kwargs: dict):
                return await impl_func(**kwargs)

        else:

            @wraps(impl_func)
            def kwargs_unpacking_adapter(kwargs: dict):
                return impl_func(**kwargs)

        activity_impl = temporal_activity.defn(name=self.name)(kwargs_unpacking_adapter)

        queue_name = self.options["task_queue"]
        _undefined_activities[queue_name].discard(self.name)
        if not _undefined_activities[queue_name]:
            del _undefined_activities[queue_name]
        _activity_registry[queue_name].append(activity_impl)

        return activity_impl

    def start(self, *args, **kwargs) -> ActivityHandle:
        return workflow.start_activity(
            self.name,
            arg=self._args_to_dict(*args, **kwargs),
            **self.options,
        )

    def _args_to_dict(self, *args, **kwargs) -> dict[str, Any]:
        return {**dict(zip(self.param_names, args)), **kwargs}


@overload
def decl(
    *,
    task_queue: str,
    result_type: type | None = None,
    schedule_to_close_timeout: timedelta | None = None,
    schedule_to_start_timeout: timedelta | None = None,
    start_to_close_timeout: timedelta | None = None,
    heartbeat_timeout: timedelta | None = None,
    retry_policy: RetryPolicy | None = None,
    cancellation_type: ActivityCancellationType = None,
    activity_id: str | None = None,
    versioning_intent: VersioningIntent | None = None,
    summary: str | None = None,
    priority: Priority | None = None,
) -> Callable[[T], ActivityStub]:
    """
    Declare an activity with Temporal options.

    This overload provides IDE support for all Temporal activity options.
    All parameters match those in temporalio.workflow.execute_activity.

    Args:
        task_queue: Task queue name for the activity
        result_type: Expected return type (for type hints)
        schedule_to_close_timeout: Maximum time from scheduling to completion
        schedule_to_start_timeout: Maximum time from scheduling to start
        start_to_close_timeout: Maximum time for a single execution attempt
        heartbeat_timeout: Maximum time between heartbeats
        retry_policy: How to retry failed activities
        cancellation_type: How to handle cancellation
        activity_id: Unique identifier for this activity execution
        versioning_intent: Versioning behavior
        summary: Human-readable summary
        priority: Activity priority
    """
    ...


def decl(task_queue: str, **activity_options) -> Callable[[T], ActivityStub]:
    def decorator(func: T) -> ActivityStub:
        return ActivityStub.from_func(func, task_queue, activity_options)

    return decorator


def activities_for_queue(queue_name: str) -> list[Callable]:
    if _undefined_activities.get(queue_name):
        raise ValueError(
            f"Missing implementations for activities in queue '{queue_name}': "
            f"{', '.join(_undefined_activities[queue_name])}"
        )

    return _activity_registry.get(queue_name, [])
