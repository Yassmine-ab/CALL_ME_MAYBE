from enum import Enum
from src.functions_parsing import FunctionDefinition
from src.tests_parsing import FunctionCallingTest


class JSONState(Enum):
    START = "start"
    PROMPT_KEY = "prompt_key"
    COLON = "colon"
    PROMPT_VALUE = "prompt_value"
    COMMA = "comma"
    NAME_KEY = "name_key"
    NAME_VALUE = "name_value"
    PARAMETERS_KEY = "parameters_key"
    OPEN_BRACE = "open_brace"
    PARAMETERS_VALUE = "parameters_value"
    PARAMETER_KEY = "parameter_key"
    PARAMETER_VALUE = "parameter_value"
    CLOSE_BRACE = "close_brace"
    END = "end"


class JSONStateMachine:
    def __init__(self) -> None:
        self.state = JSONState.START
        self.next_state: JSONState = None
        self.current_text: str = ""
        self.current_function: FunctionDefinition = None
        self.current_parameter: str = None

    def transition(self, token: str, test: FunctionCallingTest, functions: list[FunctionDefinition]) -> bool:
        for char in token:
            self.current_text += char
            if self.state == JSONState.START:
                if self.current_text == "{":
                    self.current_text = ""
                    self.state = JSONState.PROMPT_KEY
                elif not "{".startswith(self.current_text):
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
                    if "\"" + function.fn_name + "\"".startswith(self.current_text):
                        valid_prefix = True
                        if self.current_text == "\"" + function.fn_name + "\"":
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
                    self.state = JSONState.PARAMETER_KEY
                elif not "{".startswith(self.current_text):
                    return False
            elif self.state == JSONState.COLON:
                if self.current_text == ": ":
                    self.current_text = ""
                    self.state = self.next_state
            elif self.state == JSONState.COMMA:
                if self.current_text == ", ":
                    self.current_text = ""
                    self.state = self.next_state
        return True