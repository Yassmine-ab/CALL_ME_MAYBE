from enum import Enum
from src.json_file_parsing import FunctionDefinition, FunctionCallingTest


class JSONState(Enum):
    START = "start"
    PROMPT_KEY = "prompt_key"
    COLON = "colon"
    PROMPT_VALUE = "prompt_value"
    COMMA = "comma"
    NAME_KEY = "name_key"
    NAME_VALUE = "name_value"
    PARAMETERS_KEY = "parameters_key"
    PARAMETERS_VALUE = "parameters_value"
    PARAMETER_KEY = "parameter_key"
    PARAMETER_VALUE = "parameter_value"
    CLOSE_BRACE = "close_brace"
    END = "end"


def is_complete_number(text: str, arg_type: str) -> bool:
    """Check if text is a fully valid int or float."""
    try:
        if arg_type == "int":
            int(text)
        elif arg_type == "float":
            float(text)
        return True
    except ValueError:
        return False


def is_number_prefix(text: str, arg_type: str) -> bool:
    """Check if text could still become a valid int/float with more characters."""
    if text.startswith("-"):
        body = text[1:]
    else:
        body = text
    for char in body:
        if char not in set("0123456789."):
            return False
    if (arg_type == "float" and body.count(".") > 1) or (
        arg_type == "int" and "." in body
        ):
        return False
    return True


class JSONStateMachine:
    def __init__(self) -> None:
        self.state = JSONState.START
        self.next_state: JSONState = None
        self.current_text: str = ""
        self.current_function: FunctionDefinition = None
        self.current_parameter: str = None
        self.parameter_index = 0
        self.braces_to_close: int = 0

    def finish_parameter_value(self) -> None:
        """Called once a parameter's value is complete: decide what comes next."""
        self.parameter_index += 1
        self.current_text = ""
        if self.current_function.args_names[self.parameter_index:]:
            self.state = JSONState.COMMA
            self.next_state = JSONState.PARAMETER_KEY
        else:
            self.state = JSONState.CLOSE_BRACE

    def transition(self, token: str, test: FunctionCallingTest, functions: list[FunctionDefinition]) -> bool:
        i = 0
        while i < len(token):
            char = token[i]

            if self.state == JSONState.PARAMETER_VALUE and self.current_function.args_types[self.current_parameter] in ("int", "float"):
                arg_type = self.current_function.args_types[self.current_parameter]
                candidate = self.current_text + char
                if is_number_prefix(candidate, arg_type):
                    self.current_text = candidate
                    i += 1
                    continue
                if not is_complete_number(self.current_text, arg_type):
                    return False
                self.finish_parameter_value()
                continue

            self.current_text += char

            if self.state == JSONState.START:
                if self.current_text == "{":
                    self.current_text = ""
                    self.braces_to_close += 1
                    self.state = JSONState.PROMPT_KEY
                else:
                    return False

            elif self.state == JSONState.PROMPT_KEY:
                if self.current_text == "\"prompt\"":
                    self.current_text = ""
                    self.state = JSONState.COLON
                    self.next_state = JSONState.PROMPT_VALUE
                elif not "\"prompt\"".startswith(self.current_text):
                    return False

            elif self.state == JSONState.PROMPT_VALUE:
                if self.current_text == "\"" + test.prompt + "\"":
                    self.current_text = ""
                    self.state = JSONState.COMMA
                    self.next_state = JSONState.NAME_KEY
                elif not ("\"" + test.prompt + "\"").startswith(self.current_text):
                    return False

            elif self.state == JSONState.NAME_KEY:
                if self.current_text == "\"name\"":
                    self.current_text = ""
                    self.state = JSONState.COLON
                    self.next_state = JSONState.NAME_VALUE
                elif not "\"name\"".startswith(self.current_text):
                    return False

            elif self.state == JSONState.NAME_VALUE:
                valid_prefix = False
                for function in functions:
                    target = "\"" + function.fn_name + "\""
                    if target.startswith(self.current_text):
                        valid_prefix = True
                        if self.current_text == target:
                            self.current_function = function
                            self.current_text = ""
                            self.state = JSONState.COMMA
                            self.next_state = JSONState.PARAMETERS_KEY
                            break
                if not valid_prefix:
                    return False

            elif self.state == JSONState.PARAMETERS_KEY:
                if self.current_text == "\"parameters\"":
                    self.current_text = ""
                    self.state = JSONState.COLON
                    self.next_state = JSONState.PARAMETERS_VALUE
                elif not "\"parameters\"".startswith(self.current_text):
                    return False

            elif self.state == JSONState.PARAMETERS_VALUE:
                if self.current_text == "{":
                    self.current_text = ""
                    self.braces_to_close += 1
                    self.state = JSONState.PARAMETER_KEY
                else:
                    return False

            elif self.state == JSONState.PARAMETER_KEY:
                valid_prefix = False
                for arg_name in self.current_function.args_names[self.parameter_index:]:
                    target = "\"" + arg_name + "\""
                    if target.startswith(self.current_text):
                        valid_prefix = True
                        if self.current_text == target:
                            self.current_parameter = arg_name
                            self.current_text = ""
                            self.state = JSONState.COLON
                            self.next_state = JSONState.PARAMETER_VALUE
                            break
                if not valid_prefix:
                    return False

            elif self.state == JSONState.PARAMETER_VALUE:
                arg_type = self.current_function.args_types[self.current_parameter]
                if arg_type == "str":
                    if not self.current_text.startswith("\""):
                        return False
                    if len(self.current_text) > 1 and self.current_text.endswith("\""):
                        self.finish_parameter_value()
                elif arg_type == "bool":
                    lowered = self.current_text.lower()
                    if lowered in ("true", "false"):
                        self.finish_parameter_value()
                    elif not ("true".startswith(lowered) or "false".startswith(lowered)):
                        return False

            elif self.state == JSONState.CLOSE_BRACE:
                if self.current_text == "}":
                    self.current_text = ""
                    self.braces_to_close -= 1
                    if self.braces_to_close == 0:
                        self.state = JSONState.END
                    else:
                        self.state = JSONState.CLOSE_BRACE
                else:
                    return False

            elif self.state == JSONState.COLON:
                if self.current_text == ":":
                    self.current_text = ""
                    self.state = self.next_state
                else:
                    return False

            elif self.state == JSONState.COMMA:
                if self.current_text == ",":
                    self.current_text = ""
                    self.state = self.next_state
                else:
                    return False

            i += 1
        return True
