"""Unit tests for the constrained-decoding state machine."""

from src.json_file_parsing import FunctionDefinition
from src.state_machine import JSONStateMachine


def test_function_token_accepts_a_valid_prefix(
    functions: list[FunctionDefinition],
) -> None:
    state = JSONStateMachine()
    state.start_function()

    assert state.valid_function_token("fn_a", functions) is True
    assert state.current_text == "fn_a"
    assert state.current_function is None
    assert state.function_complete is False


def test_function_token_completes_an_unique_name(
    functions: list[FunctionDefinition],
) -> None:
    state = JSONStateMachine()
    state.start_function()

    assert state.valid_function_token("fn_add_numbers", functions) is True
    assert state.function_complete is True
    assert state.current_function is not None
    assert state.current_function.fn_name == "fn_add_numbers"


def test_function_token_rejects_an_unknown_prefix(
    functions: list[FunctionDefinition],
) -> None:
    state = JSONStateMachine()
    state.start_function()

    assert state.valid_function_token("fn_unknown", functions) is False
    assert state.current_text == ""


def test_function_stays_ambiguous_while_a_prefix_matches_many(
    functions: list[FunctionDefinition],
) -> None:
    state = JSONStateMachine()
    state.start_function()

    assert state.valid_function_token("fn_g", functions) is True
    assert state.function_complete is False

    assert state.valid_function_token("reet", functions) is True
    assert state.function_complete is True
    assert state.current_function is not None
    assert state.current_function.fn_name == "fn_greet"


def test_function_rejects_tokens_after_completion(
    functions: list[FunctionDefinition],
) -> None:
    state = JSONStateMachine()
    state.start_function()
    assert state.valid_function_token("fn_greet", functions) is True

    assert state.valid_function_token("_x", functions) is False


def test_string_value_requires_a_leading_quote() -> None:
    state = JSONStateMachine()
    state.start_value("string")

    assert state.valid_value_token("hello") is False
    assert state.value_started is False


def test_string_value_is_built_until_the_closing_quote() -> None:
    state = JSONStateMachine()
    state.start_value("string")

    assert state.valid_value_token('"hello') is True
    assert state.current_text == "hello"
    assert state.value_complete is False

    assert state.valid_value_token(" world") is True
    assert state.current_text == "hello world"

    assert state.valid_value_token('"') is True
    assert state.value_complete is True
    assert state.current_text == "hello world"

    assert state.valid_value_token("ignored") is False


def test_string_value_can_open_and_close_in_one_token() -> None:
    state = JSONStateMachine()
    state.start_value("string")

    assert state.valid_value_token(' "done"') is True
    assert state.current_text == "done"
    assert state.value_complete is True


def test_number_value_accepts_a_numeric_prefix() -> None:
    state = JSONStateMachine()
    state.start_value("number")

    assert state.valid_value_token("-") is True
    assert state.valid_value_token("3") is True
    assert state.valid_value_token(".") is True
    assert state.valid_value_token("14") is True
    assert state.current_text == "-3.14"
    assert state.value_started is True


def test_number_value_rejects_non_numeric_tokens() -> None:
    state = JSONStateMachine()
    state.start_value("number")

    assert state.valid_value_token("abc") is False
    assert state.valid_value_token("") is False


def test_number_value_completes_on_a_delimiter() -> None:
    state = JSONStateMachine()
    state.start_value("number")
    assert state.valid_value_token("4") is True
    assert state.valid_value_token("2") is True

    assert state.valid_value_token(" ") is True
    assert state.value_complete is True


def test_number_value_is_forced_complete_without_a_delimiter() -> None:
    state = JSONStateMachine()
    state.start_value("number")
    assert state.valid_value_token("4") is True
    assert state.valid_value_token("2") is True

    assert state.is_forced_complete() is True

    state.start_value("number")
    assert state.valid_value_token("4") is True
    assert state.valid_value_token(".") is True
    assert state.is_forced_complete() is False


def test_boolean_value_accepts_only_prefixes_of_true_or_false() -> None:
    state = JSONStateMachine()
    state.start_value("boolean")

    assert state.valid_value_token("t") is True
    assert state.valid_value_token("ru") is True
    assert state.valid_value_token("e") is True
    assert state.current_text == "true"
    assert state.value_complete is True

    assert state.valid_value_token("x") is False


def test_boolean_value_rejects_an_invalid_first_token() -> None:
    state = JSONStateMachine()
    state.start_value("boolean")

    assert state.valid_value_token("yes") is False
    assert state.value_started is False


def test_boolean_value_is_forced_complete_on_an_exact_match() -> None:
    state = JSONStateMachine()
    state.start_value("boolean")
    assert state.valid_value_token("false") is True

    assert state.is_forced_complete() is True


def test_string_value_is_never_forced_complete() -> None:
    state = JSONStateMachine()
    state.start_value("string")
    assert state.valid_value_token('"hello"') is True

    assert state.is_forced_complete() is False
