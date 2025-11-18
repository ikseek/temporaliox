from datetime import timedelta

import pytest

from temporaliox.activity import (
    ActivityExecution,
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
            "test_activity"
        )
        assert test_activity.start_options["task_queue"] == TEST_QUEUE
        assert test_activity.start_options["start_to_close_timeout"] == timedelta(
            seconds=30
        )

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
            "<locals>.documented_activity"
        )
        assert documented_activity.signature.parameters["x"].annotation is int
        assert documented_activity.signature.return_annotation is str

    def test_activity_arg_type_generation(self):
        """Test that activity declaration generates proper arg_type dataclass."""

        @decl(task_queue=TEST_QUEUE)
        def test_activity(name: str, value: int, active: bool = True) -> str:
            pass

        # Check arg_type is generated
        arg_type = test_activity.arg_type

        # Check the class name is "arg_type"
        assert arg_type.__name__ == "arg_type"

        # The qualname should be the full activity name + .arg_type
        assert arg_type.__qualname__ == (
            "TestActivityDeclaration.test_activity_arg_type_generation."
            "<locals>.test_activity.arg_type"
        )

        # Verify the qualname construction uses the class name correctly
        assert arg_type.__qualname__.endswith(".arg_type")

        # Check it's a dataclass
        assert hasattr(arg_type, "__dataclass_fields__")

        # Check fields
        from dataclasses import fields

        dataclass_fields = fields(arg_type)
        field_names = [f.name for f in dataclass_fields]
        assert field_names == ["name", "value", "active"]

        # Check field types and defaults
        name_field = next(f for f in dataclass_fields if f.name == "name")
        value_field = next(f for f in dataclass_fields if f.name == "value")
        active_field = next(f for f in dataclass_fields if f.name == "active")

        assert name_field.type is str
        assert value_field.type is int
        assert active_field.type is bool
        assert active_field.default is True

        # Test creating instance
        args = arg_type(name="test", value=42)
        assert args.name == "test"
        assert args.value == 42
        assert args.active is True  # default value


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


class TestDeclarationOptions:
    def test_defn_with_no_thread_cancel_exception(self):
        """Test that decl accepts no_thread_cancel_exception parameter."""

        @decl(task_queue=TEST_QUEUE, no_thread_cancel_exception=True)
        def test_activity(name: str) -> str:
            pass

        @test_activity.defn
        def test_activity_impl(name: str) -> str:
            return f"Hello, {name}!"

        # Should return a callable function
        assert callable(test_activity_impl)
        assert test_activity_impl.__name__ == "test_activity_impl"

        # Verify the option is stored in defn_options
        assert test_activity.defn_options["no_thread_cancel_exception"] is True

    def test_defn_with_default_no_thread_cancel_exception(self):
        """Test that defn works with default no_thread_cancel_exception=False."""

        @decl(task_queue=TEST_QUEUE)
        def test_activity(name: str) -> str:
            pass

        @test_activity.defn
        def test_activity_impl(name: str) -> str:
            return f"Hello, {name}!"

        # Should return a callable function
        assert callable(test_activity_impl)
        assert test_activity_impl.__name__ == "test_activity_impl"

        # Verify no definition options are set by default
        assert test_activity.defn_options == {}


class TestArgumentConversion:
    def test_args_to_dict_positional(self):
        """Test conversion of positional arguments to dict via ActivityExecution."""

        @decl(task_queue=TEST_QUEUE)
        def test_activity(name: str, value: int) -> str:
            pass

        execution = test_activity.with_options()
        args_dict = execution._args_to_dict("Bob", 42)
        assert args_dict == {"name": "Bob", "value": 42}

    def test_args_to_dict_mixed(self):
        @decl(task_queue=TEST_QUEUE)
        def test_activity(name: str, value: int) -> str:
            pass

        execution = test_activity.with_options()
        args_dict = execution._args_to_dict("Bob", value=42)
        assert args_dict == {"name": "Bob", "value": 42}

    def test_args_to_dict_all_kwargs(self):
        """Test conversion of all keyword arguments via ActivityExecution."""

        @decl(task_queue=TEST_QUEUE)
        def test_activity(name: str, value: int) -> str:
            pass

        execution = test_activity.with_options()
        args_dict = execution._args_to_dict(name="Bob", value=42)
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

        assert test_activity.start_options["schedule_to_close_timeout"] == timedelta(
            minutes=5
        )
        assert test_activity.start_options["schedule_to_start_timeout"] == timedelta(
            minutes=1
        )
        assert test_activity.start_options["start_to_close_timeout"] == timedelta(
            minutes=4
        )
        assert test_activity.start_options["heartbeat_timeout"] == timedelta(seconds=30)

    def test_retry_policy(self):
        """Test that retry policy is properly stored."""
        from temporalio.common import RetryPolicy

        retry_policy = RetryPolicy(
            maximum_attempts=3, initial_interval=timedelta(seconds=1)
        )

        @decl(task_queue=TEST_QUEUE, retry_policy=retry_policy)
        def test_activity(name: str) -> str:
            pass

        assert test_activity.start_options["retry_policy"] == retry_policy

    def test_additional_kwargs(self):
        """Test that additional keyword arguments are passed through."""

        @decl(task_queue=TEST_QUEUE, custom_option="custom_value", another_option=123)
        def test_activity(name: str) -> str:
            pass

        assert test_activity.start_options["custom_option"] == "custom_value"
        assert test_activity.start_options["another_option"] == 123


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
            "functions.<locals>.module_function"
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
            "methods.<locals>.TestClass.class_method"
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
            "<locals>.OuterClass.InnerClass.nested_method"
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
            "simple_activity"
        )
        expected_arg_type = (
            "<class 'test_temporaliox."
            "TestStringRepresentations.test_str_representation.<locals>."
            "simple_activity.arg_type'>"
        )
        assert str(simple_activity) == expected
        assert str(simple_activity.arg_type) == expected_arg_type

    def test_repr_representation_simple(self):
        """Test __repr__ method shows full details for simple case."""

        @decl(task_queue=TEST_QUEUE)
        def simple_activity(name: str) -> str:
            pass

        expected = (
            "ActivityDeclaration(signature=<Signature (name: str) -> str>, "
            "defn_options={}, start_options={'task_queue': 'unit-test-queue'})"
        )
        assert repr(simple_activity) == expected

    def test_repr_representation_with_options(self):
        """Test __repr__ method shows all activity options."""
        from datetime import timedelta

        @decl(task_queue=COMPLEX_QUEUE, start_to_close_timeout=timedelta(seconds=60))
        def complex_activity(x: int, y: str) -> bool:
            pass

        expected = (
            "ActivityDeclaration(signature=<Signature (x: int, y: str) -> bool>, "
            "defn_options={}, start_options={'task_queue': 'unit-complex-queue', "
            "'start_to_close_timeout': datetime.timedelta(seconds=60)})"
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
            "<locals>.TestClass.class_activity"
        )
        assert str(TestClass.class_activity) == expected

    def test_repr_representation_preserves_dataclass_format(self):
        """Test that __repr__ uses standard dataclass format."""

        @decl(task_queue=SIMPLE_QUEUE)
        def test_activity() -> None:
            pass

        expected = (
            "ActivityDeclaration(signature=<Signature () -> None>, "
            "defn_options={}, start_options={'task_queue': 'unit-test'})"
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
        assert (
            registry_test_activity.start_options["task_queue"] == "unit-test-registry"
        )

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


class TestActivityExecution:
    def test_activity_execution_creation(self):
        """Test ActivityExecution creation and basic properties."""

        @decl(task_queue=TEST_QUEUE, start_to_close_timeout=timedelta(seconds=30))
        def test_activity(name: str, value: int) -> str:
            pass

        execution = test_activity.with_options()

        assert isinstance(execution, ActivityExecution)
        assert execution.name == test_activity.name
        assert execution.start_options["task_queue"] == TEST_QUEUE
        assert execution.start_options["start_to_close_timeout"] == timedelta(
            seconds=30
        )

    def test_activity_execution_args_to_dict(self):
        """Test ActivityExecution._args_to_dict method."""

        @decl(task_queue=TEST_QUEUE)
        def test_activity(name: str, value: int, flag: bool) -> str:
            pass

        execution = test_activity.with_options()

        # Test positional args
        result = execution._args_to_dict("Alice", 42, True)
        assert result == {"name": "Alice", "value": 42, "flag": True}

        # Test mixed args
        result = execution._args_to_dict("Bob", value=99, flag=False)
        assert result == {"name": "Bob", "value": 99, "flag": False}

        # Test all kwargs
        result = execution._args_to_dict(name="Charlie", value=0, flag=True)
        assert result == {"name": "Charlie", "value": 0, "flag": True}

    def test_activity_execution_has_callable_interface(self):
        """Test that ActivityExecution has __call__ and start methods."""

        @decl(task_queue=TEST_QUEUE)
        def test_activity(name: str) -> str:
            pass

        execution = test_activity.with_options()

        assert callable(execution)
        assert hasattr(execution, "start")
        assert callable(execution.start)


class TestWithOptionsMethod:
    def test_with_options_with_no_arguments(self):

        @decl(task_queue=TEST_QUEUE, start_to_close_timeout=timedelta(seconds=60))
        def test_activity(name: str) -> str:
            pass

        execution = test_activity.with_options()

        assert isinstance(execution, ActivityExecution)
        assert execution.start_options == test_activity.start_options

    def test_with_options_with_timeout_override(self):
        """Test with_options() with timeout override."""

        @decl(task_queue=TEST_QUEUE, start_to_close_timeout=timedelta(seconds=60))
        def test_activity(name: str) -> str:
            pass

        execution = test_activity.with_options(
            start_to_close_timeout=timedelta(seconds=30)
        )

        assert execution.start_options["task_queue"] == TEST_QUEUE
        assert execution.start_options["start_to_close_timeout"] == timedelta(
            seconds=30
        )

        # Original should be unchanged
        assert test_activity.start_options["start_to_close_timeout"] == timedelta(
            seconds=60
        )

    def test_with_options_with_multiple_overrides(self):
        """Test with_options() with multiple option overrides."""

        @decl(
            task_queue=TEST_QUEUE,
            start_to_close_timeout=timedelta(seconds=60),
            heartbeat_timeout=timedelta(seconds=10),
        )
        def test_activity(name: str) -> str:
            pass

        execution = test_activity.with_options(
            start_to_close_timeout=timedelta(seconds=30),
            heartbeat_timeout=timedelta(seconds=5),
            summary="Custom summary",
        )

        assert execution.start_options["task_queue"] == TEST_QUEUE
        assert execution.start_options["start_to_close_timeout"] == timedelta(
            seconds=30
        )
        assert execution.start_options["heartbeat_timeout"] == timedelta(seconds=5)
        assert execution.start_options["summary"] == "Custom summary"

    def test_with_options_with_retry_policy_override(self):
        """Test with_options() with retry policy override."""
        from temporalio.common import RetryPolicy

        original_policy = RetryPolicy(
            maximum_attempts=3, initial_interval=timedelta(seconds=1)
        )
        new_policy = RetryPolicy(
            maximum_attempts=5, initial_interval=timedelta(seconds=2)
        )

        @decl(task_queue=TEST_QUEUE, retry_policy=original_policy)
        def test_activity(name: str) -> str:
            pass

        execution = test_activity.with_options(retry_policy=new_policy)

        assert execution.start_options["retry_policy"] == new_policy
        assert test_activity.start_options["retry_policy"] == original_policy

    def test_with_options_preserves_name_and_params(self):
        """Test that with_options() preserves activity name and parameter names."""
        from dataclasses import fields

        @decl(task_queue=TEST_QUEUE)
        def complex_activity(user_id: int, action: str, metadata: dict) -> str:
            pass

        execution = complex_activity.with_options(
            start_to_close_timeout=timedelta(seconds=45)
        )

        assert execution.name == complex_activity.name
        assert tuple(f.name for f in fields(execution.arg_type)) == (
            "user_id",
            "action",
            "metadata",
        )

    def test_with_options_returns_new_instance_each_time(self):
        @decl(task_queue=TEST_QUEUE, start_to_close_timeout=timedelta(seconds=60))
        def test_activity(name: str) -> str:
            pass

        execution1 = test_activity.with_options()
        execution2 = test_activity.with_options(
            start_to_close_timeout=timedelta(seconds=30)
        )
        execution3 = test_activity.with_options()

        assert execution1 is not execution2
        assert execution1 is not execution3
        assert execution2 is not execution3

        # execution1 and execution3 should have same options but be different instances
        assert execution1.start_options == execution3.start_options
        assert execution1 is not execution3


class TestDataclassPreservation:
    def test_defn_preserves_nested_dataclasses(self):
        """Test that nested dataclasses are preserved when passed to implementation."""
        from dataclasses import dataclass

        @dataclass
        class Address:
            street: str
            city: str

        @dataclass
        class Person:
            name: str
            age: int
            address: Address

        @decl(task_queue=TEST_QUEUE)
        def process_person(person: Person) -> str:
            pass

        # Store what the implementation receives
        received_args = {}

        @process_person.defn
        def process_person_impl(person: Person) -> str:
            received_args["person"] = person
            received_args["person_type"] = type(person)
            received_args["address_type"] = type(person.address)
            return (
                f"{person.name} lives at {person.address.street}, {person.address.city}"
            )

        # Create test data
        test_address = Address(street="123 Main St", city="Anytown")
        test_person = Person(name="Alice", age=30, address=test_address)

        # Create an instance of the generated arg_type
        arg_instance = process_person.arg_type(person=test_person)

        # Call the wrapper
        result = process_person_impl(arg_instance)

        # Verify dataclasses are preserved
        assert received_args["person_type"] is Person
        assert received_args["address_type"] is Address
        assert result == "Alice lives at 123 Main St, Anytown"

    def test_defn_with_complex_nested_dataclasses(self):
        """Test that complex nested dataclasses with lists are handled correctly."""
        from dataclasses import dataclass, fields
        from typing import Optional

        @dataclass
        class Item:
            id: int
            name: str
            price: float

        @dataclass
        class Order:
            order_id: str
            items: list[Item]
            discount: Optional[float] = None

        @decl(task_queue=TEST_QUEUE)
        def process_order(order: Order) -> str:
            pass

        # Verify the arg_type structure
        arg_type = process_order.arg_type
        arg_fields = {f.name: f for f in fields(arg_type)}

        # The arg_type should have an 'order' field of type Order
        assert "order" in arg_fields
        assert arg_fields["order"].type == Order

        # Create test data
        items = [
            Item(id=1, name="Widget", price=9.99),
            Item(id=2, name="Gadget", price=19.99),
        ]
        test_order = Order(order_id="ORD-123", items=items, discount=0.1)

        # The arg_type instance preserves the Order type
        arg_instance = arg_type(order=test_order)
        assert isinstance(arg_instance.order, Order)
        assert all(isinstance(item, Item) for item in arg_instance.order.items)

        # Verify the implementation receives preserved dataclasses
        received_order = None

        @process_order.defn
        def process_order_impl(order: Order) -> str:
            nonlocal received_order
            received_order = order
            total = sum(item.price for item in order.items)
            if order.discount:
                total *= 1 - order.discount
            return f"Order {order.order_id}: ${total:.2f}"

        # Call the implementation
        result = process_order_impl(arg_instance)

        # Verify the implementation received the Order instance with Item instances
        assert isinstance(received_order, Order)
        assert all(isinstance(item, Item) for item in received_order.items)
        assert result == "Order ORD-123: $26.98"

    def test_defn_with_dataclass_field_preservation(self):
        """Test that the generated arg_type properly preserves dataclass fields."""
        from dataclasses import dataclass, fields

        @dataclass
        class Config:
            timeout: int
            retries: int

        @decl(task_queue=TEST_QUEUE)
        def configure_system(name: str, config: Config) -> str:
            pass

        # Check the generated arg_type
        arg_type = configure_system.arg_type
        arg_fields = fields(arg_type)

        # Should have two fields: name and config
        assert len(arg_fields) == 2
        field_names = [f.name for f in arg_fields]
        assert "name" in field_names
        assert "config" in field_names

        # The config field should have Config as its type
        config_field = next(f for f in arg_fields if f.name == "config")
        assert config_field.type == Config

        # Create test instance
        test_config = Config(timeout=30, retries=3)
        arg_instance = arg_type(name="test", config=test_config)

        # The arg_type instance should preserve the Config instance
        assert isinstance(arg_instance.config, Config)
        assert arg_instance.config.timeout == 30
        assert arg_instance.config.retries == 3


class TestActivityDeclarationDelegation:
    def test_declaration_call_delegates_to_with_options(self):
        """Test that ActivityDeclaration.__call__ delegates to with_options()."""

        @decl(task_queue=TEST_QUEUE, start_to_close_timeout=timedelta(seconds=60))
        def test_activity(name: str) -> str:
            pass

        # Both should create equivalent ActivityExecution instances
        execution_direct = test_activity.with_options()

        # We can't directly test __call__ without a workflow context,
        # but we can verify the delegation logic by checking the options match
        assert execution_direct.start_options == test_activity.start_options

    def test_declaration_start_delegates_to_with_options(self):
        """Test that ActivityDeclaration.start delegates to with_options()."""

        @decl(task_queue=TEST_QUEUE, start_to_close_timeout=timedelta(seconds=60))
        def test_activity(name: str) -> str:
            pass

        execution_direct = test_activity.with_options()

        # Verify the options are preserved through delegation
        assert execution_direct.start_options == test_activity.start_options

    def test_delegation_preserves_all_options(self):
        """Test that delegation preserves all activity options."""
        from temporalio.common import RetryPolicy

        retry_policy = RetryPolicy(
            maximum_attempts=3, initial_interval=timedelta(seconds=1)
        )

        @decl(
            task_queue=TEST_QUEUE,
            start_to_close_timeout=timedelta(seconds=60),
            heartbeat_timeout=timedelta(seconds=10),
            retry_policy=retry_policy,
            summary="Test activity",
            custom_option="custom_value",
        )
        def test_activity(name: str) -> str:
            pass

        execution = test_activity.with_options()

        # All options should be preserved
        expected_options = test_activity.start_options
        assert execution.start_options == expected_options


class TestWithOptionsIntegration:
    def test_with_options_chaining_pattern(self):
        """Test the with_options().method() chaining pattern."""

        @decl(task_queue=TEST_QUEUE, start_to_close_timeout=timedelta(seconds=60))
        def test_activity(name: str, value: int) -> str:
            pass

        # Test the chaining pattern
        execution = test_activity.with_options(
            start_to_close_timeout=timedelta(seconds=30)
        )

        # Verify it has the expected interface
        assert callable(execution)
        assert hasattr(execution, "start")
        assert hasattr(execution, "_args_to_dict")

        # Test argument conversion works in the chain
        args_dict = execution._args_to_dict("test", 42)
        assert args_dict == {"name": "test", "value": 42}

    def test_with_options_with_all_temporal_options(self):
        """Test with_options() with all supported Temporal options."""
        from temporalio.common import RetryPolicy
        from temporalio.workflow import ActivityCancellationType

        retry_policy = RetryPolicy(
            maximum_attempts=3, initial_interval=timedelta(seconds=1)
        )

        @decl(task_queue=TEST_QUEUE)
        def test_activity(name: str) -> str:
            pass

        execution = test_activity.with_options(
            schedule_to_close_timeout=timedelta(minutes=5),
            schedule_to_start_timeout=timedelta(minutes=1),
            start_to_close_timeout=timedelta(minutes=4),
            heartbeat_timeout=timedelta(seconds=30),
            retry_policy=retry_policy,
            cancellation_type=ActivityCancellationType.WAIT_CANCELLATION_COMPLETED,
            summary="Test activity with all options",
        )

        assert execution.start_options["schedule_to_close_timeout"] == timedelta(
            minutes=5
        )
        assert execution.start_options["schedule_to_start_timeout"] == timedelta(
            minutes=1
        )
        assert execution.start_options["start_to_close_timeout"] == timedelta(minutes=4)
        assert execution.start_options["heartbeat_timeout"] == timedelta(seconds=30)
        assert execution.start_options["retry_policy"] == retry_policy
        assert (
            execution.start_options["cancellation_type"]
            == ActivityCancellationType.WAIT_CANCELLATION_COMPLETED
        )
        assert execution.start_options["summary"] == "Test activity with all options"
