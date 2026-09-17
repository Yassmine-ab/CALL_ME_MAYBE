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


class FunctionDefinition(BaseModel):
    fn_name: str
    args_names: list[str]
    args_types: dict[str, str]
    return_type: str


TYPE_MAPPING = {
    "string": "str",
    "number": "float",
    "boolean": "bool",
}


def _map_type(raw_type: str, default: str = "str") -> str:
    if not isinstance(raw_type, str):
        return default
    return TYPE_MAPPING.get(raw_type, raw_type)


def _parse_function_definition(function: dict) -> FunctionDefinition | None:
    name = function.get("name")
    parameters = function.get("parameters")
    returns = function.get("returns")
    if not isinstance(name, str) or not isinstance(parameters, dict):
        return None

    args_names: list[str] = []
    args_types: dict[str, str] = {}
    for arg_name, arg_def in parameters.items():
        if not isinstance(arg_name, str) or not isinstance(arg_def, dict):
            return None
        arg_type = arg_def.get("type")
        if not isinstance(arg_type, str):
            return None
        args_names.append(arg_name)
        args_types[arg_name] = _map_type(arg_type)

    return_type = _map_type(returns.get("type")) if isinstance(returns, dict) else "str"
    return FunctionDefinition(
        fn_name=name,
        args_names=args_names,
        args_types=args_types,
        return_type=return_type,
    )


def parse_functions(file_path: str) -> list[FunctionDefinition]:
    """Parse the functions definitions from the input JSON file."""
    try:
        with open(file_path, "r") as f:
            functions = load(f)
    except (FileNotFoundError, JSONDecodeError) as e:
        print(f"Error loading functions definitions: {e}")
        return []
    if not isinstance(functions, list):
        return []
    valid_functions = []
    for function in functions:
        if not isinstance(function, dict):
            continue
        try:
            parsed = _parse_function_definition(function)
            if parsed is not None:
                valid_functions.append(parsed)
        except (TypeError, ValidationError):
            continue
    return valid_functions
