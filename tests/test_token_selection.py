"""Unit tests for constrained token selection."""

import pytest

from src.json_file_parsing import FunctionDefinition
from src.state_machine import JSONStateMachine
from src.token_selection import (
    NoValidTokenError,
    get_next_token,
    prepare_vocabulary,
    should_stop_early,
)


def test_prepare_vocabulary_inverts_and_replaces_markers() -> None:
    vocabulary = prepare_vocabulary({"Ġhello": 3, "world": 4})

    assert vocabulary == {3: " hello", 4: "world"}


def test_get_next_token_picks_the_highest_scoring_valid_token(
    functions: list[FunctionDefinition],
) -> None:
    # Token 98 scores higher but is not a valid function prefix.
    vocabulary = {99: "fn_add_numbers", 98: "zzz"}
    logits = [0.0] * 100
    logits[98] = 5.0
    logits[99] = 1.0

    state = JSONStateMachine()
    state.start_function()

    token, token_id = get_next_token(
        logits, vocabulary, state, functions, mode="function"
    )

    assert (token, token_id) == ("fn_add_numbers", 99)
    assert state.function_complete is True
    assert state.current_function is not None
    assert state.current_function.fn_name == "fn_add_numbers"


def test_get_next_token_persists_accepted_state() -> None:
    vocabulary = {0: '"hi"', 1: "nope"}
    logits = [1.0, 10.0]

    state = JSONStateMachine()
    state.start_value("string")

    token, token_id = get_next_token(
        logits, vocabulary, state, [], mode="value"
    )

    assert (token, token_id) == ('"hi"', 0)
    assert state.current_text == "hi"
    assert state.value_complete is True


def test_get_next_token_skips_ids_missing_from_the_vocabulary(
    functions: list[FunctionDefinition],
) -> None:
    vocabulary = {0: "fn_add_numbers"}
    logits = [1.0, 2.0, 3.0]

    state = JSONStateMachine()
    state.start_function()

    token, token_id = get_next_token(
        logits, vocabulary, state, functions, mode="function"
    )

    assert (token, token_id) == ("fn_add_numbers", 0)


def test_get_next_token_raises_when_nothing_is_valid(
    functions: list[FunctionDefinition],
) -> None:
    state = JSONStateMachine()
    state.start_function()

    with pytest.raises(NoValidTokenError):
        get_next_token([1.0], {0: "zzz"}, state, functions, mode="function")


def test_get_next_token_rejects_an_unknown_mode() -> None:
    state = JSONStateMachine()
    state.start_value("string")

    with pytest.raises(ValueError):
        get_next_token([1.0], {0: "x"}, state, [], mode="bogus")


def test_should_stop_early_when_the_model_wants_to_stop() -> None:
    state = JSONStateMachine()
    state.start_value("number")
    assert state.valid_value_token("4") is True
    assert state.valid_value_token("2") is True

    # The unconstrained top token ("abc") cannot extend the number.
    vocabulary = {0: "abc", 1: "5"}
    logits = [10.0, 0.0]

    assert should_stop_early(logits, vocabulary, state) is True


def test_should_not_stop_early_while_a_digit_can_extend_the_number() -> None:
    state = JSONStateMachine()
    state.start_value("number")
    assert state.valid_value_token("4") is True
    assert state.valid_value_token("2") is True

    vocabulary = {0: "abc", 1: "5"}
    logits = [0.0, 10.0]

    assert should_stop_early(logits, vocabulary, state) is False


def test_should_stop_early_when_the_top_id_has_no_token() -> None:
    state = JSONStateMachine()
    state.start_value("number")
    assert state.valid_value_token("4") is True
    assert state.valid_value_token("2") is True

    # Top token id (0) is not present in this vocabulary.
    vocabulary = {1: "5"}
    logits = [10.0, 0.0]

    assert should_stop_early(logits, vocabulary, state) is True


def test_should_not_stop_early_for_an_incomplete_value() -> None:
    state = JSONStateMachine()
    state.start_value("number")
    assert state.valid_value_token("4") is True
    assert state.valid_value_token(".") is True

    vocabulary = {0: "abc"}
    logits = [10.0]

    assert should_stop_early(logits, vocabulary, state) is False


def test_should_not_stop_early_for_non_numeric_types() -> None:
    state = JSONStateMachine()
    state.start_value("string")
    assert state.valid_value_token('"hello"') is True

    vocabulary = {0: "abc"}
    logits = [10.0]

    assert should_stop_early(logits, vocabulary, state) is False
