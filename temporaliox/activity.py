from __future__ import annotations

import inspect
from collections import defaultdict
from dataclasses import dataclass
from datetime import timedelta
from functools import cached_property, update_wrapper
from typing import Any, Callable, TypeVar, overload

from temporalio import activity as temporal_activity
from temporalio import workflow
from temporalio.common import Priority, RetryPolicy
from temporalio.workflow import (
    ActivityCancellationType,
    ActivityHandle,
)

__all__ = ["decl", "ActivityDeclaration", "ActivityExecution", "activities_for_queue"]

T = TypeVar("T", bound=Callable[..., Any])

_UNPACKING_WRAPPER_ASSIGNMENTS = ("__module__", "__name__", "__qualname__", "__doc__")
_undefined_activities: defaultdict[str, set[str]] = defaultdict(set)
_activity_registry: defaultdict[str, list[Callable]] = defaultdict(list)


@dataclass(frozen=True)
class ActivityDeclaration:
    name: str
    signature: inspect.Signature
    defn_options: dict[str, Any]
    start_options: dict[str, Any]

    def __str__(self) -> str:
        return self.name

    async def __call__(self, *args, **kwargs):
        return await self.with_options()(*args, **kwargs)

    @staticmethod
    def create(
        func: Callable,
        task_queue: str,
        start_options: dict[str, Any],
        defn_options: dict[str, Any],
    ) -> ActivityDeclaration:
        name = func.__qualname__
        declaration = ActivityDeclaration(
            name=name,
            signature=inspect.signature(func),
            defn_options=defn_options,
            start_options={"task_queue": task_queue, **start_options},
        )
        _undefined_activities[task_queue].add(name)
        return declaration

    def defn(self, impl_func: T) -> T:
        impl_sig = inspect.signature(impl_func)
        if impl_sig != self.signature:
            raise ValueError(
                f"Implementation signature {impl_sig} does not match "
                f"declaration signature {self.signature} for activity "
                f"'{self.name}'"
            )
        activity_impl = _make_unary_temporal_activity(
            impl_func, name=self.name, **self.defn_options
        )

        queue_name = self.start_options["task_queue"]
        _undefined_activities[queue_name].discard(self.name)
        if not _undefined_activities[queue_name]:
            del _undefined_activities[queue_name]
        _activity_registry[queue_name].append(activity_impl)

        return activity_impl

    @overload
    def with_options(
        self,
        *,
        schedule_to_close_timeout: timedelta | None = None,
        schedule_to_start_timeout: timedelta | None = None,
        start_to_close_timeout: timedelta | None = None,
        heartbeat_timeout: timedelta | None = None,
        retry_policy: RetryPolicy | None = None,
        cancellation_type: ActivityCancellationType = None,
        summary: str | None = None,
        priority: Priority | None = None,
    ) -> ActivityExecution:
        """
        Create an ActivityExecution with custom runtime options.

        Allows overriding execution-time options that were set during declaration.
        This is useful for adjusting timeouts, retry policies, etc. based on
        runtime conditions.

        Args:
            schedule_to_close_timeout: Maximum time from scheduling to completion
            schedule_to_start_timeout: Maximum time from scheduling to start
            start_to_close_timeout: Maximum time for a single execution attempt
            heartbeat_timeout: Maximum time between heartbeats
            retry_policy: How to retry failed activities
            cancellation_type: How to handle cancellation
            summary: Human-readable summary for this execution
            priority: Activity priority for this execution

        Returns:
            ActivityExecution with custom options that can be called or started
        """

    def with_options(self, **overrides) -> ActivityExecution:
        return ActivityExecution(
            name=self.name,
            param_names=self._param_names,
            start_options={**self.start_options, **overrides},
        )

    def start(self, *args, **kwargs) -> ActivityHandle:
        return self.with_options().start(*args, **kwargs)

    @cached_property
    def _param_names(self) -> tuple[str, ...]:
        return tuple(self.signature.parameters.keys())


@dataclass(frozen=True)
class ActivityExecution:
    name: str
    param_names: tuple[str, ...]
    start_options: dict[str, Any]

    async def __call__(self, *args, **kwargs):
        return await workflow.execute_activity(
            self.name,
            arg=self._args_to_dict(*args, **kwargs),
            **self.start_options,
        )

    def start(self, *args, **kwargs) -> ActivityHandle:
        return workflow.start_activity(
            self.name,
            arg=self._args_to_dict(*args, **kwargs),
            **self.start_options,
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
    priority: Priority | None = None,
    no_thread_cancel_exception: bool = None,
) -> Callable[[T], ActivityDeclaration]:
    """
    Declare an activity with Temporal options.

    This overload provides IDE support for all Temporal activity options.
    These options set defaults that can be overridden at runtime using with_options().

    Declaration-time options (set defaults for all executions):
        task_queue: Task queue name for the activity (required)
        result_type: Expected return type (for type hints only, no runtime effect)
        schedule_to_close_timeout: Default maximum time from scheduling to completion
        schedule_to_start_timeout: Default maximum time from scheduling to start
        start_to_close_timeout: Default maximum time for a single execution attempt
        heartbeat_timeout: Default maximum time between heartbeats
        retry_policy: Default retry behavior for failed activities
        cancellation_type: Default cancellation handling behavior
        priority: Default activity priority
        no_thread_cancel_exception: Whether to disable thread cancellation

    Note: All start-time options (timeouts, retry policy, etc.) can be overridden
    at runtime using activity.with_options(option=value) for specific executions.
    """
    ...


def decl(
    task_queue: str, *, no_thread_cancel_exception=None, **start_options
) -> Callable[[T], ActivityDeclaration]:
    def decorator(func: T) -> ActivityDeclaration:
        defn_options = (
            {}
            if no_thread_cancel_exception is None
            else {"no_thread_cancel_exception": no_thread_cancel_exception}
        )
        return ActivityDeclaration.create(func, task_queue, start_options, defn_options)

    return decorator


def activities_for_queue(queue_name: str) -> list[Callable]:
    if _undefined_activities.get(queue_name):
        raise ValueError(
            f"Missing implementations for activities in queue '{queue_name}': "
            f"{', '.join(_undefined_activities[queue_name])}"
        )

    return _activity_registry.get(queue_name, [])


def _make_unary_temporal_activity(impl_func: Callable, **defn_options) -> Callable:
    if inspect.iscoroutinefunction(impl_func):

        async def unpack_kwargs(kwargs: dict):
            return await impl_func(**kwargs)

    else:

        def unpack_kwargs(kwargs: dict):
            return impl_func(**kwargs)

    # Do not assign annotations, the wrapper has different signature
    update_wrapper(unpack_kwargs, impl_func, assigned=_UNPACKING_WRAPPER_ASSIGNMENTS)
    return temporal_activity.defn(**defn_options)(unpack_kwargs)
