"""Unit tests for JSON fixture parsing and validation."""

import json
from pathlib import Path

import pytest

from src.json_file_parsing import (
    FunctionDefinition,
    parse_function_calling_tests,
    parse_functions,
)

FUNCTIONS_PAYLOAD = [
    {
        "name": "fn_add_numbers",
        "description": "Add two numbers together and return their sum.",
        "parameters": {"a": {"type": "number"}, "b": {"type": "number"}},
        "returns": {"type": "number"},
    }
]


def _write_json(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_parse_functions_returns_validated_models(tmp_path: Path) -> None:
    path = _write_json(tmp_path / "functions.json", FUNCTIONS_PAYLOAD)

    functions = parse_functions(str(path))

    assert len(functions) == 1
    assert isinstance(functions[0], FunctionDefinition)
    assert functions[0].fn_name == "fn_add_numbers"
    assert functions[0].args_names == ["a", "b"]
    assert functions[0].args_types == {"a": "number", "b": "number"}


def test_parse_function_calling_tests(tmp_path: Path) -> None:
    path = _write_json(
        tmp_path / "tests.json",
        [{"prompt": "What is the sum of 2 and 3?"}],
    )

    tests = parse_function_calling_tests(str(path))

    assert [test.prompt for test in tests] == ["What is the sum of 2 and 3?"]


def test_missing_file_returns_an_empty_list(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    functions = parse_functions(str(tmp_path / "missing.json"))

    assert functions == []
    assert "file not found" in capsys.readouterr().out


def test_invalid_json_returns_an_empty_list(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "broken.json"
    path.write_text("{not json", encoding="utf-8")

    assert parse_functions(str(path)) == []
    assert "invalid JSON" in capsys.readouterr().out


def test_json_object_instead_of_array_returns_an_empty_list(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = _write_json(tmp_path / "object.json", {"name": "fn_greet"})

    assert parse_functions(str(path)) == []
    assert "must contain a JSON array" in capsys.readouterr().out


def test_invalid_schema_returns_an_empty_list(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = _write_json(
        tmp_path / "invalid.json",
        [{"name": "fn_greet"}],
    )

    assert parse_functions(str(path)) == []
    assert "invalid data" in capsys.readouterr().out
