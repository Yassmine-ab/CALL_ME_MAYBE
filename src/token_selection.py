from json import load
import token
from llm_sdk import Small_LLM_Model
from src.state_machine import JSONState, JSONStateMachine
from src.tests_parsing import FunctionCallingTest
from src.functions_parsing import FunctionDefinition
from copy import deepcopy


def is_token_valid(
    token_text: str,
    state_machine: JSONStateMachine,
    test: FunctionCallingTest,
    functions: list[FunctionDefinition],
) -> bool:
    """Check whether a token is valid in the current state."""
    machine = deepcopy(state_machine)
    for char in token_text:
        machine.current_text += char
        if machine.state == JSONState.START:
            if machine.current_text != "{":
                return False
            machine.current_text = ""
            machine.state = JSONState.PROMPT_KEY
        elif machine.state == JSONState.PROMPT_KEY:
            expected = '"prompt"'
            if not expected.startswith(machine.current_text):
                return False
            if machine.current_text == expected:
                machine.current_text = ""
                machine.state = JSONState.COLON
                machine.next_state = JSONState.PROMPT_VALUE

        elif machine.state == JSONState.COLON:
            if machine.current_text != ":":
                return False

            machine.current_text = ""
            machine.state = machine.next_state

        elif machine.state == JSONState.PROMPT_VALUE:
            expected = '"' + test.prompt + '"'

            if not expected.startswith(machine.current_text):
                return False

            if machine.current_text == expected:
                machine.current_text = ""
                machine.state = JSONState.COMMA
                machine.next_state = JSONState.NAME_KEY

        elif machine.state == JSONState.COMMA:
            if machine.current_text != ",":
                return False

            machine.current_text = ""
            machine.state = machine.next_state

        elif machine.state == JSONState.NAME_KEY:
            expected = '"name"'

            if not expected.startswith(machine.current_text):
                return False

            if machine.current_text == expected:
                machine.current_text = ""
                machine.state = JSONState.COLON
                machine.next_state = JSONState.NAME_VALUE

        elif machine.state == JSONState.NAME_VALUE:
            matched = False

            for function in functions:
                expected = '"' + function.fn_name + '"'

                if expected.startswith(machine.current_text):
                    matched = True

                    if machine.current_text == expected:
                        machine.current_function = function
                        machine.current_text = ""
                        machine.state = JSONState.COMMA
                        machine.next_state = JSONState.PARAMETERS_KEY

                    break

            if not matched:
                return False

        elif machine.state == JSONState.PARAMETERS_KEY:
            expected = '"parameters"'

            if not expected.startswith(machine.current_text):
                return False

            if machine.current_text == expected:
                machine.current_text = ""
                machine.state = JSONState.COLON
                machine.next_state = JSONState.PARAMETERS_VALUE

        elif machine.state == JSONState.PARAMETERS_VALUE:
            if machine.current_text != "{":
                return False

            machine.current_text = ""
            machine.state = JSONState.PARAMETER_KEY

        elif machine.state == JSONState.PARAMETER_KEY:
            if machine.current_function is None:
                return False

            matched = False

            for parameter in machine.current_function.args_names:
                expected = '"' + parameter + '"'

                if expected.startswith(machine.current_text):
                    matched = True

                    if machine.current_text == expected:
                        machine.current_parameter = parameter
                        machine.current_text = ""
                        machine.state = JSONState.PARAMETER_VALUE

                    break

            if not matched:
                return False

        elif machine.state == JSONState.PARAMETER_VALUE:
            # À compléter avec la logique correspondant
            # au type/à la valeur du paramètre.
            return False

        elif machine.state == JSONState.CLOSE_BRACE:
            if machine.current_text != "}":
                return False

            machine.current_text = ""
            machine.state = JSONState.END

        elif machine.state == JSONState.END:
            return False

    return True


def is_token_valid(
        token_text: str,
        state_machine: JSONStateMachine,
        test: FunctionCallingTest,
        functions: list[FunctionDefinition]
        ) -> bool:
    """Check if a token is valid based on the current state of the JSON state machine."""
    candidate = state_machine.current_text + token_text
    if state_machine.state == JSONState.START:    
        return candidate.startswith("{")
    if state_machine.state == JSONState.PROMPT_KEY:
        return is_prefix(candidate, "\"prompt\"")
    if state_machine.state == JSONState.PROMPT_VALUE:
        return is_prefix(candidate, ": \"" + test.prompt + "\"")
    if state_machine.state == JSONState.NAME_KEY:
        return is_prefix(candidate, "\"name\"")
    if state_machine.state == JSONState.NAME_VALUE:
        return any(is_prefix(candidate, ": \"" + function.fn_name + "\"") for function in functions)
    if state_machine.state == JSONState.PARAMETERS_KEY:
        return is_prefix(candidate, "\"parameters\"")
    if state_machine.state == JSONState.PARAMETERS_VALUE:
        return is_prefix(candidate, ": {")
    if state_machine.state == JSONState.PARAMETER_KEY:
        current_function = state_machine.current_function
        return any(is_prefix(candidate, "\"" + param + "\"") for param in current_function.args_names)
    return False


# def get_valid_tokens(model: Small_LLM_Model, state_machine: JSONStateMachine, functions: list[FunctionDefinition]) -> list[int]:
#     """Determine the valid tokens based on the JSON output syntax and the vocabulary."""
#     with open(model.get_path_to_vocabulary_json(), "r") as f:
#         vocabulary = load(f)
#         valid_tokens = []
#         for token_text, token_id in vocabulary.items():
#             if is_token_valid(token_text, state_machine):
#                 valid_tokens.append(token_id)
#     return valid_tokens
