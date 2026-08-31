from pydantic import BaseModel, ValidationError
from json import load, JSONDecodeError


class FunctionCallingTest(BaseModel):
    prompt: str


def parse_function_calling_tests(file_path: str) -> list[FunctionCallingTest]:
    """Parse the function calling tests from the input JSON file."""
    try:
        with open(file_path, "r") as f:
            tests = load(f)
    except (FileNotFoundError, JSONDecodeError) as e:
        print(f"Error loading function calling tests: {e}")
        return []
    if not isinstance(tests, list):
        return []
    valid_tests = []
    for test in tests:
        if not isinstance(test, dict):
            continue
        try:
            valid_tests.append(FunctionCallingTest(**test))
        except (TypeError, ValidationError):
            continue
    return valid_tests
