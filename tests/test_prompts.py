"""Unit tests for prompt construction."""

from src.json_file_parsing import FunctionDefinition
from src.prompts import (
    build_function_selection_prompt,
    build_value_generation_prompt,
)


def test_function_selection_prompt_lists_the_candidates(
    functions: list[FunctionDefinition],
) -> None:
    prompt = build_function_selection_prompt(
        "What is the sum of 2 and 3?", functions
    )

    assert "Available functions:" in prompt
    assert "fn_add_numbers" in prompt
    assert "Add two numbers together and return their sum." in prompt
    assert "parameters: a, b" in prompt
    assert 'User request: "What is the sum of 2 and 3?"' in prompt
    assert prompt.endswith("Function to call: ")


def test_value_prompt_mentions_the_selected_function_and_values(
    functions: list[FunctionDefinition],
) -> None:
    function = functions[0]

    prompt = build_value_generation_prompt(
        "What is the sum of 2 and 3?",
        function,
        {"a": "2"},
        "b",
    )

    assert 'User request: "What is the sum of 2 and 3?"' in prompt
    assert "Function selected: fn_add_numbers" in prompt
    assert "Values generated so far: a=2" in prompt
    assert "parameter 'b'" in prompt
    assert "type: number" in prompt


def test_value_prompt_handles_the_first_parameter(
    functions: list[FunctionDefinition],
) -> None:
    prompt = build_value_generation_prompt(
        "Greet shrek",
        functions[1],
        {},
        "name",
    )

    assert "Values generated so far: (none yet)" in prompt
