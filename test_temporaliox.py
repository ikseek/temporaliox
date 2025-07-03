from datetime import timedelta

import pytest

from temporaliox.activity import (
    _activity_registry,
    _undefined_activities,
    activities_for_queue,
    decl,
)

# Test constants
TEST_QUEUE = "unit-test-queue"
CLASS_QUEUE = "unit-class-queue"
COMPLEX_QUEUE = "unit-complex-queue"
SIMPLE_QUEUE = "unit-test"


class TestActivityDeclaration:
    def test_activity_stub_creation(self):
        """Test that activity.decl creates a proper ActivityDeclaration."""

        @decl(task_queue=TEST_QUEUE, start_to_close_timeout=timedelta(seconds=30))
        def test_activity(name: str, value: int) -> str:
            pass

        assert hasattr(test_activity, "defn")
        assert hasattr(test_activity, "start")
        assert callable(test_activity)
        assert (
            test_activity.name
            == "TestActivityDeclaration.test_activity_stub_creation.<locals>."
            + "test_activity"
        )
        assert test_activity.options["task_queue"] == TEST_QUEUE
        assert test_activity.options["start_to_close_timeout"] == timedelta(seconds=30)

    def test_activity_stub_preserves_metadata(self):
        """Test that activity stub preserves function metadata in signature."""

        @decl(task_queue=TEST_QUEUE)
        def documented_activity(x: int) -> str:
            """This is a documented activity."""
            pass

        # With frozen dataclass, we store the signature instead of copying
        # function metadata
        assert (
            documented_activity.name
            == "TestActivityDeclaration.test_activity_stub_preserves_metadata."
            + "<locals>.documented_activity"
        )
        assert documented_activity.signature.parameters["x"].annotation is int
        assert documented_activity.signature.return_annotation is str


class TestActivityDefinition:
    def test_defn_with_matching_signature(self):
        """Test that defn accepts implementation with matching signature."""

        @decl(task_queue=TEST_QUEUE)
        def test_activity(name: str, value: int) -> str:
            pass

        def impl(name: str, value: int) -> str:
            return f"{name}: {value}"

        wrapper = test_activity.defn(impl)

        # Should return a function (the decorated wrapper)
        assert callable(wrapper)
        assert wrapper.__name__ == "impl"  # wraps preserves name

    def test_defn_validates_parameter_names(self):
        """Test that defn rejects implementation with different parameter names."""

        @decl(task_queue=TEST_QUEUE)
        def test_activity(user_name: str, action: str) -> str:
            pass

        with pytest.raises(
            ValueError,
            match="Implementation signature .* does not match declaration signature",
        ):

            @test_activity.defn
            def test_activity_impl(name: str, action: str) -> str:
                return f"{name}: {action}"

    def test_defn_validates_parameter_count(self):
        """Test that defn rejects implementation with different parameter count."""

        @decl(task_queue=TEST_QUEUE)
        def test_activity(name: str) -> str:
            pass

        with pytest.raises(
            ValueError,
            match="Implementation signature .* does not match declaration signature",
        ):

            @test_activity.defn
            def test_activity_impl(name: str, extra: str) -> str:
                return f"{name}: {extra}"

    def test_defn_validates_return_type(self):
        """Test that defn rejects implementation with different return type."""

        @decl(task_queue=TEST_QUEUE)
        def test_activity(name: str) -> str:
            pass

        with pytest.raises(
            ValueError,
            match="Implementation signature .* does not match declaration signature",
        ):

            @test_activity.defn
            def test_activity_impl(name: str) -> int:
                return 42

    def test_defn_creates_wrapper_for_dict_args(self):
        """Test that defn creates a wrapper that accepts dict arguments."""

        @decl(task_queue=TEST_QUEUE)
        def test_activity(name: str, value: int) -> str:
            pass

        @test_activity.defn
        def test_activity_impl(name: str, value: int) -> str:
            return f"{name}: {value}"

        # The returned function should be the wrapper
        assert callable(test_activity_impl)
        assert (
            test_activity_impl.__name__ == "test_activity_impl"
        )  # wraps preserves name


class TestArgumentConversion:
    def test_args_to_dict_positional(self):
        """Test conversion of positional arguments to dict."""

        @decl(task_queue=TEST_QUEUE)
        def test_activity(name: str, value: int) -> str:
            pass

        args_dict = test_activity._args_to_dict("Bob", 42)
        assert args_dict == {"name": "Bob", "value": 42}

    def test_args_to_dict_mixed(self):
        """Test conversion of mixed positional and keyword arguments."""

        @decl(task_queue=TEST_QUEUE)
        def test_activity(name: str, value: int) -> str:
            pass

        args_dict = test_activity._args_to_dict("Bob", value=42)
        assert args_dict == {"name": "Bob", "value": 42}

    def test_args_to_dict_all_kwargs(self):
        """Test conversion of all keyword arguments."""

        @decl(task_queue=TEST_QUEUE)
        def test_activity(name: str, value: int) -> str:
            pass

        args_dict = test_activity._args_to_dict(name="Bob", value=42)
        assert args_dict == {"name": "Bob", "value": 42}


class TestActivityOptions:
    def test_all_timeout_options(self):
        """Test that all timeout options are properly stored."""

        @decl(
            task_queue=TEST_QUEUE,
            schedule_to_close_timeout=timedelta(minutes=5),
            schedule_to_start_timeout=timedelta(minutes=1),
            start_to_close_timeout=timedelta(minutes=4),
            heartbeat_timeout=timedelta(seconds=30),
        )
        def test_activity(name: str) -> str:
            pass

        assert test_activity.options["schedule_to_close_timeout"] == timedelta(
            minutes=5
        )
        assert test_activity.options["schedule_to_start_timeout"] == timedelta(
            minutes=1
        )
        assert test_activity.options["start_to_close_timeout"] == timedelta(minutes=4)
        assert test_activity.options["heartbeat_timeout"] == timedelta(seconds=30)

    def test_retry_policy(self):
        """Test that retry policy is properly stored."""
        from temporalio.common import RetryPolicy

        retry_policy = RetryPolicy(
            maximum_attempts=3, initial_interval=timedelta(seconds=1)
        )

        @decl(task_queue=TEST_QUEUE, retry_policy=retry_policy)
        def test_activity(name: str) -> str:
            pass

        assert test_activity.options["retry_policy"] == retry_policy

    def test_additional_kwargs(self):
        """Test that additional keyword arguments are passed through."""

        @decl(task_queue=TEST_QUEUE, custom_option="custom_value", another_option=123)
        def test_activity(name: str) -> str:
            pass

        assert test_activity.options["custom_option"] == "custom_value"
        assert test_activity.options["another_option"] == 123


class TestAsyncActivities:
    def test_async_activity_definition(self):
        """Test that async activities are properly handled."""

        @decl(task_queue=TEST_QUEUE)
        def test_activity(name: str) -> str:
            pass

        @test_activity.defn
        async def test_activity_impl(name: str) -> str:
            return f"Hello, {name}!"

        # Should return a callable function
        assert callable(test_activity_impl)
        assert test_activity_impl.__name__ == "test_activity_impl"

    def test_sync_activity_definition(self):
        """Test that sync activities are properly handled."""

        @decl(task_queue=TEST_QUEUE)
        def test_activity(name: str) -> str:
            pass

        @test_activity.defn
        def test_activity_impl(name: str) -> str:
            return f"Hello, {name}!"

        # Should return a callable function
        assert callable(test_activity_impl)
        assert test_activity_impl.__name__ == "test_activity_impl"


class TestQualifiedNames:
    def test_activity_name_uses_qualname_for_module_functions(self):
        """Test that activity name uses __qualname__ for module-level functions."""

        @decl(task_queue=TEST_QUEUE)
        def module_function(name: str) -> str:
            pass

        assert (
            module_function.name
            == "TestQualifiedNames.test_activity_name_uses_qualname_for_module_"
            + "functions.<locals>.module_function"
        )

    def test_activity_name_uses_qualname_for_class_methods(self):
        """Test that activity name includes class name for class methods."""

        class TestClass:
            @staticmethod
            @decl(task_queue=TEST_QUEUE)
            def class_method(name: str) -> str:
                pass

        assert (
            TestClass.class_method.name
            == "TestQualifiedNames.test_activity_name_uses_qualname_for_class_"
            + "methods.<locals>.TestClass.class_method"
        )

    def test_activity_name_for_nested_class_methods(self):
        """Test that activity name includes full qualified name for nested classes."""

        class OuterClass:
            class InnerClass:
                @staticmethod
                @decl(task_queue=TEST_QUEUE)
                def nested_method(value: int) -> int:
                    pass

        expected_name = (
            "TestQualifiedNames.test_activity_name_for_nested_class_methods."
            + "<locals>.OuterClass.InnerClass.nested_method"
        )
        assert OuterClass.InnerClass.nested_method.name == expected_name


class TestStringRepresentations:
    def test_str_representation(self):
        """Test __str__ method shows just the activity name."""

        @decl(task_queue=TEST_QUEUE)
        def simple_activity(name: str) -> str:
            pass

        expected = (
            "TestStringRepresentations.test_str_representation.<locals>."
            + "simple_activity"
        )
        assert str(simple_activity) == expected

    def test_repr_representation_simple(self):
        """Test __repr__ method shows full details for simple case."""

        @decl(task_queue=TEST_QUEUE)
        def simple_activity(name: str) -> str:
            pass

        expected = (
            "ActivityDeclaration(name='TestStringRepresentations."
            + "test_repr_representation_simple.<locals>.simple_activity', "
            + "signature=<Signature (name: str) -> str>, "
            + "options={'task_queue': 'unit-test-queue'})"
        )
        assert repr(simple_activity) == expected

    def test_repr_representation_with_options(self):
        """Test __repr__ method shows all activity options."""
        from datetime import timedelta

        @decl(task_queue=COMPLEX_QUEUE, start_to_close_timeout=timedelta(seconds=60))
        def complex_activity(x: int, y: str) -> bool:
            pass

        expected = (
            "ActivityDeclaration(name='TestStringRepresentations."
            + "test_repr_representation_with_options.<locals>.complex_activity', "
            + "signature=<Signature (x: int, y: str) -> bool>, "
            + "options={'task_queue': 'unit-complex-queue', "
            + "'start_to_close_timeout': datetime.timedelta(seconds=60)})"
        )
        assert repr(complex_activity) == expected

    def test_str_representation_for_class_method(self):
        """Test __str__ shows qualified name for class methods."""

        class TestClass:
            @staticmethod
            @decl(task_queue=CLASS_QUEUE)
            def class_activity(value: int) -> str:
                pass

        expected = (
            "TestStringRepresentations.test_str_representation_for_class_method."
            + "<locals>.TestClass.class_activity"
        )
        assert str(TestClass.class_activity) == expected

    def test_repr_representation_preserves_dataclass_format(self):
        """Test that __repr__ uses standard dataclass format."""

        @decl(task_queue=SIMPLE_QUEUE)
        def test_activity() -> None:
            pass

        expected = (
            "ActivityDeclaration(name='TestStringRepresentations."
            + "test_repr_representation_preserves_dataclass_format.<locals>."
            + "test_activity', signature=<Signature () -> None>, "
            + "options={'task_queue': 'unit-test'})"
        )
        assert repr(test_activity) == expected


class TestActivityRegistry:
    def setup_method(self):
        """Save and clear the registry before each test."""
        # Save current state
        self._saved_activity_registry = {
            k: v.copy() for k, v in _activity_registry.items()
        }
        self._saved_undefined_activities = {
            k: v.copy() for k, v in _undefined_activities.items()
        }

        # Clear the registry for the test
        _activity_registry.clear()
        _undefined_activities.clear()

    def teardown_method(self):
        """Restore the registry after each test."""
        _activity_registry.clear()
        _undefined_activities.clear()

        # Restore previous state
        _activity_registry.update(self._saved_activity_registry)
        _undefined_activities.update(self._saved_undefined_activities)

    def test_activity_declaration_registers_in_registry(self):
        """Test that declaring an activity registers it in undefined activities."""

        @decl(task_queue="unit-test-registry")
        def registry_test_activity(name: str) -> str:
            pass

        activity_name = registry_test_activity.name
        assert "unit-test-registry" in _undefined_activities
        assert activity_name in _undefined_activities["unit-test-registry"]

        # Verify the activity stub itself has the correct properties
        assert registry_test_activity.options["task_queue"] == "unit-test-registry"

    def test_activity_implementation_registers_in_registry(self):
        """Test that defining an activity implementation updates the registry."""

        @decl(task_queue="unit-impl-test")
        def impl_test_activity(value: int) -> str:
            pass

        activity_name = impl_test_activity.name
        # Should be in undefined activities initially
        assert "unit-impl-test" in _undefined_activities
        assert activity_name in _undefined_activities["unit-impl-test"]

        @impl_test_activity.defn
        def impl_test_activity_impl(value: int) -> str:
            return f"Value: {value}"

        # Should be removed from undefined activities (queue deleted since empty)
        assert "unit-impl-test" not in _undefined_activities

        # Should be in activity registry under the queue
        assert "unit-impl-test" in _activity_registry
        assert len(_activity_registry["unit-impl-test"]) == 1
        assert callable(_activity_registry["unit-impl-test"][0])

    def test_activities_for_queue_returns_implementations(self):
        """Test that activities_for_queue returns list of implementations."""

        @decl(task_queue="unit-worker-queue")
        def worker_activity_1(x: int) -> int:
            pass

        @decl(task_queue="unit-worker-queue")
        def worker_activity_2(y: str) -> str:
            pass

        @decl(task_queue="unit-other-queue")
        def other_activity(z: bool) -> bool:
            pass

        # Define implementations
        @worker_activity_1.defn
        def worker_activity_1_impl(x: int) -> int:
            return x * 2

        @worker_activity_2.defn
        def worker_activity_2_impl(y: str) -> str:
            return f"Hello {y}"

        @other_activity.defn
        def other_activity_impl(z: bool) -> bool:
            return not z

        # Get activities for worker-queue
        worker_activities = activities_for_queue("unit-worker-queue")
        assert len(worker_activities) == 2

        # Verify all returned items are callable
        for activity in worker_activities:
            assert callable(activity)

        # Get activities for other-queue
        other_activities = activities_for_queue("unit-other-queue")
        assert len(other_activities) == 1
        assert callable(other_activities[0])

    def test_activities_for_queue_empty_when_no_activities(self):
        """Test that activities_for_queue returns empty list for unknown queue."""
        assert activities_for_queue("unit-nonexistent-queue") == []

    def test_activities_for_queue_raises_when_missing_implementations(self):
        """Test that activities_for_queue raises error for missing implementations."""

        @decl(task_queue="unit-incomplete-queue")
        def incomplete_activity_1(a: int) -> int:
            pass

        @decl(task_queue="unit-incomplete-queue")
        def incomplete_activity_2(b: str) -> str:
            pass

        # Only implement one of them
        @incomplete_activity_1.defn
        def incomplete_activity_1_impl(a: int) -> int:
            return a + 1

        # Should raise ValueError for missing implementation
        with pytest.raises(ValueError, match="Missing implementations for activities"):
            activities_for_queue("unit-incomplete-queue")

    def test_activities_for_queue_works_with_single_activity(self):
        """Test activities_for_queue works correctly with single activity."""

        @decl(task_queue="unit-single-queue")
        def single_activity(data: str) -> str:
            pass

        @single_activity.defn
        def single_activity_impl(data: str) -> str:
            return f"Processed: {data}"

        activities = activities_for_queue("unit-single-queue")
        assert len(activities) == 1
        assert callable(activities[0])

    def test_registry_tracks_multiple_queues(self):
        """Test that registry correctly tracks activities across multiple queues."""

        @decl(task_queue="unit-queue-a")
        def activity_a1(x: int) -> int:
            pass

        @decl(task_queue="unit-queue-a")
        def activity_a2(y: str) -> str:
            pass

        @decl(task_queue="unit-queue-b")
        def activity_b1(z: bool) -> bool:
            pass

        # Implement all activities
        @activity_a1.defn
        def activity_a1_impl(x: int) -> int:
            return x

        @activity_a2.defn
        def activity_a2_impl(y: str) -> str:
            return y

        @activity_b1.defn
        def activity_b1_impl(z: bool) -> bool:
            return z

        # Check each queue independently
        queue_a_activities = activities_for_queue("unit-queue-a")
        queue_b_activities = activities_for_queue("unit-queue-b")

        assert len(queue_a_activities) == 2
        assert len(queue_b_activities) == 1

    def test_registry_with_class_methods(self):
        """Test that registry works with class static methods."""

        class ActivityClass:
            @staticmethod
            @decl(task_queue="unit-class-method-queue")
            def class_static_activity(value: float) -> float:
                pass

        @ActivityClass.class_static_activity.defn
        def class_static_activity_impl(value: float) -> float:
            return value * 2.0

        activities = activities_for_queue("unit-class-method-queue")
        assert len(activities) == 1
        assert callable(activities[0])
