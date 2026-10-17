from json import JSONDecodeError, load
from typing import TypeVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError

ModelT = TypeVar("ModelT", bound=BaseModel)


class FunctionParameter(BaseModel):
    """A single function parameter definition."""
    type: str


class FunctionDefinition(BaseModel):
    """Normalized function metadata loaded from the JSON fixtures."""
    model_config = ConfigDict(populate_by_name=True)

    fn_name: str = Field(alias="name")
    description: str
    parameters: dict[str, FunctionParameter]
    returns: dict[str, str]

    @property
    def args_names(self) -> list[str]:
        return list(self.parameters.keys())

    @property
    def args_types(self) -> dict[str, str]:
        return {
            name: parameter.type for name, parameter in self.parameters.items()
            }


class FunctionCallingTest(BaseModel):
    """A single prompt used to exercise function calling."""

    prompt: str


def parse_json_list(
        file_path: str, model_cls: type[ModelT]
        ) -> list[ModelT]:
    """Load a JSON array from file_path and validate each item as model_cls."""
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            data = load(file)

        if not isinstance(data, list):
            raise ValueError(f"{file_path} must contain a JSON array.")

        return [model_cls.model_validate(item) for item in data]

    except FileNotFoundError:
        print(f"Error: file not found: {file_path}")
    except JSONDecodeError as error:
        print(f"Error: invalid JSON in {file_path}: {error}")
    except ValidationError as error:
        print(f"Error: invalid data in {file_path}: {error}")
    except ValueError as error:
        print(f"Error: {error}")

    return []


def parse_functions(file_path: str) -> list[FunctionDefinition]:
    """Load and validate function definitions from a JSON file."""
    return parse_json_list(file_path, FunctionDefinition)


def parse_function_calling_tests(file_path: str) -> list[FunctionCallingTest]:
    """Load and validate function-calling tests."""
    return parse_json_list(file_path, FunctionCallingTest)
