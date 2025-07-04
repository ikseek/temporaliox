import inspect
from collections import defaultdict
from dataclasses import dataclass, make_dataclass
from datetime import timedelta
from functools import cached_property, update_wrapper
from typing import Any, Callable, Optional, TypeVar, overload

from temporalio import activity as temporal_activity
from temporalio import workflow
from temporalio.common import Priority, RetryPolicy
from temporalio.workflow import (
    ActivityCancellationType,
    ActivityHandle,
)

__all__ = ["decl", "ActivityDeclaration", "ActivityExecution", "activities_for_queue"]

T = TypeVar("T", bound=Callable[..., Any])

_undefined_activities: defaultdict[str, set[str]] = defaultdict(set)
_activity_registry: defaultdict[str, list[Callable]] = defaultdict(list)


@dataclass(frozen=True)
class ActivityExecution:
    name: str
    param_names: tuple[str, ...]
    arg_type: type
    start_options: dict[str, Any]

    async def __call__(self, *args, **kwargs):
        return await workflow.execute_activity(
            self.name,
            arg=self._args_to_dataclass(*args, **kwargs),
            **self.start_options,
        )

    def start(self, *args, **kwargs) -> ActivityHandle:
        return workflow.start_activity(
            self.name,
            arg=self._args_to_dataclass(*args, **kwargs),
            **self.start_options,
        )

    def _args_to_dataclass(self, *args, **kwargs):
        """Convert args/kwargs to a dataclass instance."""
        # Combine positional and keyword arguments
        all_kwargs = {**dict(zip(self.param_names, args)), **kwargs}
        return self.arg_type(**all_kwargs)

    def _args_to_dict(self, *args, **kwargs) -> dict[str, Any]:
        """Legacy method for backward compatibility."""
        return {**dict(zip(self.param_names, args)), **kwargs}


@dataclass()
class ActivityDeclaration:
    signature: inspect.Signature
    defn_options: dict[str, Any]
    start_options: dict[str, Any]

    def __str__(self) -> str:
        return self.name

    async def __call__(self, *args, **kwargs):
        return await self.with_options()(*args, **kwargs)

    @property
    def name(self) -> str:
        return self.__qualname__

    @staticmethod
    def create(
        func: Callable,
        task_queue: str,
        start_options: dict[str, Any],
        defn_options: dict[str, Any],
    ) -> "ActivityDeclaration":
        declaration = ActivityDeclaration(
            signature=inspect.signature(func),
            defn_options=defn_options,
            start_options={"task_queue": task_queue, **start_options},
        )
        update_wrapper(declaration, func)
        _undefined_activities[task_queue].add(declaration.name)
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
            impl_func, arg_type=self.arg_type, name=self.name, **self.defn_options
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
        schedule_to_close_timeout: Optional[timedelta] = None,
        schedule_to_start_timeout: Optional[timedelta] = None,
        start_to_close_timeout: Optional[timedelta] = None,
        heartbeat_timeout: Optional[timedelta] = None,
        retry_policy: Optional[RetryPolicy] = None,
        cancellation_type: ActivityCancellationType = None,
        summary: Optional[str] = None,
        priority: Optional[Priority] = None,
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
            arg_type=self.arg_type,
            start_options={**self.start_options, **overrides},
        )

    def start(self, *args, **kwargs) -> ActivityHandle:
        return self.with_options().start(*args, **kwargs)

    @cached_property
    def _param_names(self) -> tuple[str, ...]:
        return tuple(self.signature.parameters.keys())

    @cached_property
    def arg_type(self) -> type:
        """Generate a dataclass type for activity arguments."""
        field_definitions = []
        for param_name, param in self.signature.parameters.items():
            param_type = (
                param.annotation if param.annotation != inspect.Parameter.empty else Any
            )

            if param.default != inspect.Parameter.empty:
                field_definitions.append((param_name, param_type, param.default))
            else:
                field_definitions.append((param_name, param_type))
        cls = make_dataclass(
            "arg_type",  # The actual class name
            field_definitions,
            frozen=True,
        )
        cls.__module__ = self.__module__
        cls.__qualname__ = f"{self.name}.{cls.__name__}"
        return cls


@overload
def decl(
    *,
    task_queue: str,
    result_type: Optional[type] = None,
    schedule_to_close_timeout: Optional[timedelta] = None,
    schedule_to_start_timeout: Optional[timedelta] = None,
    start_to_close_timeout: Optional[timedelta] = None,
    heartbeat_timeout: Optional[timedelta] = None,
    retry_policy: Optional[RetryPolicy] = None,
    cancellation_type: ActivityCancellationType = None,
    priority: Optional[Priority] = None,
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


def _make_unary_temporal_activity(
    impl_func: Callable, arg_type: type, **defn_options
) -> Callable:
    if inspect.iscoroutinefunction(impl_func):

        # no need to apply @wraps because we change the signature
        async def unpack_dataclass_to_kwargs(arg: arg_type):
            # Convert dataclass into args dict, non-recursively
            return await impl_func(**vars(arg))

    else:

        def unpack_dataclass_to_kwargs(arg: arg_type):
            # Convert dataclass into args dict, non-recursively
            return impl_func(**vars(arg))

    unpack_dataclass_to_kwargs.__name__ = impl_func.__name__
    unpack_dataclass_to_kwargs.__qualname__ = impl_func.__qualname__
    unpack_dataclass_to_kwargs.__module__ = impl_func.__module__

    return temporal_activity.defn(**defn_options)(unpack_dataclass_to_kwargs)
