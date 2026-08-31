from pydantic import BaseModel, ValidationError
from json import load, JSONDecodeError


class FunctionDefinition(BaseModel):
    fn_name: str
    args_names: list[str]
    args_types: dict[str, str]
    return_type: str

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
            valid_functions.append(FunctionDefinition(**function))
        except (TypeError, ValidationError):
            continue
    return valid_functions
