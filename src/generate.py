from llm_sdk import Small_LLM_Model

from src.json_file_parsing import FunctionDefinition
from src.prompts import (
    build_function_selection_prompt,
    build_value_generation_prompt
    )
from src.state_machine import JSONStateMachine
from src.token_selection import (
    NoValidTokenError,
    get_next_token,
    should_stop_early
    )

MAX_FUNCTION_TOKENS = 30
MAX_VALUE_TOKENS = 60


class FunctionSelectionError(Exception):
    """Raised when the LLM cannot select a valid function for a prompt."""


class ValueGenerationError(Exception):
    """Raised when the LLM cannot generate a valid value for a parameter."""


def _encode(model: Small_LLM_Model, text: str) -> list[int]:
    """Tokenize text into a flat list of token ids."""
    tensor = model.encode(text)
    return [int(token_id) for token_id in tensor[0].tolist()]


def select_function(
    model: Small_LLM_Model,
    vocabulary: dict[int, str],
    functions: list[FunctionDefinition],
    user_prompt: str,
) -> FunctionDefinition:
    """Generate the function name chosen by the LLM for this prompt."""
    prompt_text = build_function_selection_prompt(user_prompt, functions)
    input_ids = _encode(model, prompt_text)

    state_machine = JSONStateMachine()
    state_machine.start_function()

    for _ in range(MAX_FUNCTION_TOKENS):
        logits = model.get_logits_from_input_ids(input_ids)

        try:
            token, token_id = get_next_token(
                logits, vocabulary, state_machine, functions, mode="function"
            )
        except NoValidTokenError as error:
            raise FunctionSelectionError(
                f"Could not select a function for prompt: {user_prompt!r}"
            ) from error

        input_ids.append(token_id)

        if state_machine.function_complete:
            assert state_machine.current_function is not None
            return state_machine.current_function
    else:
        raise FunctionSelectionError(
            f"Function name did not terminate for prompt: {user_prompt!r}"
        )


def generate_value(
    model: Small_LLM_Model,
    vocabulary: dict[int, str],
    user_prompt: str,
    function: FunctionDefinition,
    generated_values: dict[str, str],
    parameter_name: str,
) -> str:
    """Generate the raw text value for one parameter."""
    param_type = function.args_types[parameter_name]
    prompt_text = build_value_generation_prompt(
        user_prompt, function, generated_values, parameter_name
    )
    input_ids = _encode(model, prompt_text)

    state_machine = JSONStateMachine()
    state_machine.start_value(param_type)

    for _ in range(MAX_VALUE_TOKENS):
        logits = model.get_logits_from_input_ids(input_ids)

        if should_stop_early(logits, vocabulary, state_machine):
            state_machine.value_complete = True
            break

        try:
            token, token_id = get_next_token(
                logits, vocabulary, state_machine, [], mode="value"
            )
        except NoValidTokenError as error:
            if state_machine.is_forced_complete():
                state_machine.value_complete = True
                break
            raise ValueGenerationError(
                f"Could not generate value for '{parameter_name}' "
                f"(prompt: {user_prompt!r})"
            ) from error

        input_ids.append(token_id)

        if state_machine.value_complete:
            break
    else:
        raise ValueGenerationError(
            f"Value generation did not terminate for '{parameter_name}' "
            f"(prompt: {user_prompt!r})"
        )

    return state_machine.current_text


def convert_value(raw_value: str, param_type: str) -> object:
    """Convert the raw generated text into the correct Python type."""
    try:
        if param_type == "number":
            return float(raw_value) if "." in raw_value else int(raw_value)
        if param_type == "boolean":
            return raw_value == "true"
        return raw_value
    except ValueError as error:
        raise ValueGenerationError(
            f"Generated value {raw_value!r} is not a valid {param_type}"
        ) from error


def generate_function_call(
    model: Small_LLM_Model,
    vocabulary: dict[int, str],
    functions: list[FunctionDefinition],
    user_prompt: str,
) -> dict[str, object]:
    """Generate a full function call
    (name + typed parameters) for one prompt."""
    function = select_function(model, vocabulary, functions, user_prompt)

    raw_values: dict[str, str] = {}
    parameters: dict[str, object] = {}

    for parameter_name in function.args_names:
        raw_value = generate_value(
            model, vocabulary, user_prompt,
            function, raw_values, parameter_name
        )
        raw_values[parameter_name] = raw_value
        parameters[parameter_name] = convert_value(
            raw_value, function.args_types[parameter_name]
        )

    return {
        "prompt": user_prompt,
        "name": function.fn_name,
        "parameters": parameters,
    }
