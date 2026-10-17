"""Shared pytest fixtures for the offline unit test suite."""

import pytest

from src.json_file_parsing import FunctionDefinition, FunctionParameter


def _function(
    name: str,
    description: str,
    parameters: dict[str, str],
    returns: str,
) -> FunctionDefinition:
    """Build a FunctionDefinition from a compact parameters mapping."""
    return FunctionDefinition(
        name=name,
        description=description,
        parameters={
            parameter_name: FunctionParameter(type=parameter_type)
            for parameter_name, parameter_type in parameters.items()
        },
        returns={"type": returns},
    )


@pytest.fixture
def functions() -> list[FunctionDefinition]:
    """Return the reference function set used across the tests."""
    return [
        _function(
            "fn_add_numbers",
            "Add two numbers together and return their sum.",
            {"a": "number", "b": "number"},
            "number",
        ),
        _function(
            "fn_greet",
            "Generate a greeting message for a person by name.",
            {"name": "string"},
            "string",
        ),
        _function(
            "fn_get_square_root",
            "Calculate the square root of a number.",
            {"a": "number"},
            "number",
        ),
        _function(
            "fn_substitute_string_with_regex",
            "Replace all occurrences matching a regex pattern in a string.",
            {
                "source_string": "string",
                "regex": "string",
                "replacement": "string",
            },
            "string",
        ),
    ]
