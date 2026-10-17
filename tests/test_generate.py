"""Unit tests for raw-value to Python-type conversion."""

import pytest

from src.generate import ValueGenerationError, convert_value


def test_convert_integer_number() -> None:
    value = convert_value("42", "number")

    assert value == 42
    assert isinstance(value, int)


def test_convert_float_number() -> None:
    value = convert_value("-3.14", "number")

    assert value == -3.14
    assert isinstance(value, float)


def test_convert_boolean() -> None:
    assert convert_value("true", "boolean") is True
    assert convert_value("false", "boolean") is False


def test_convert_string_is_identity() -> None:
    assert convert_value("hello world", "string") == "hello world"


def test_convert_invalid_number_raises() -> None:
    with pytest.raises(ValueGenerationError):
        convert_value("not-a-number", "number")
