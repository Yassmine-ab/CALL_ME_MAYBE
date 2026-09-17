from enum import Enum
from src.json_file_parsing import FunctionDefinition


class JSONState(Enum):
    NAME_VALUE = "name_value"
    NAME_CHOSEN = "name_chosen"
    PARAMETER_VALUE = "parameter_value"
    VALUE_DONE = "value_done"


def is_complete_number(text: str, arg_type: str) -> bool:
    try:
        if arg_type == "int":
            int(text)
        elif arg_type == "float":
            float(text)
        return True
    except ValueError:
        return False


def is_number_prefix(text: str, arg_type: str) -> bool:
    body = text[1:] if text.startswith("-") else text
    if any(c not in "0123456789." for c in body):
        return False
    if arg_type == "float" and body.count(".") > 1:
        return False
    if arg_type == "int" and "." in body:
        return False
    return True


class JSONStateMachine:
    """Ne contraint que les deux vraies décisions : le nom de la fonction
    et la valeur de chaque paramètre. Toute la ponctuation JSON fixe est
    injectée séparément par le générateur (main.py), pas par cette classe."""

    def __init__(self) -> None:
        self.state: JSONState = None
        self.current_text = ""
        self.current_function: FunctionDefinition = None
        self.current_parameter: str = None

    def start_name(self) -> None:
        self.state = JSONState.NAME_VALUE
        self.current_text = ""

    def start_parameter(self, function: FunctionDefinition, arg_name: str) -> None:
        self.current_function = function
        self.current_parameter = arg_name
        self.state = JSONState.PARAMETER_VALUE
        self.current_text = ""

    def transition(self, token: str, functions: list[FunctionDefinition]) -> bool:
        if self.state in (JSONState.NAME_CHOSEN, JSONState.VALUE_DONE):
            return len(token) == 0  # déjà terminé : aucun caractère supplémentaire n'est valide

        i = 0
        while i < len(token):
            char = token[i]

            if self.state == JSONState.PARAMETER_VALUE:
                arg_type = self.current_function.args_types[self.current_parameter]

                if arg_type in ("int", "float"):
                    candidate = self.current_text + char
                    if is_number_prefix(candidate, arg_type):
                        self.current_text = candidate
                        i += 1
                        continue
                    if not is_complete_number(self.current_text, arg_type):
                        return False
                    self.state = JSONState.VALUE_DONE
                    continue

                self.current_text += char
                if arg_type == "str":
                    if not self.current_text.startswith("\""):
                        return False
                    if len(self.current_text) > 1 and self.current_text.endswith("\""):
                        self.state = JSONState.VALUE_DONE
                elif arg_type == "bool":
                    lowered = self.current_text.lower()
                    if lowered in ("true", "false"):
                        self.state = JSONState.VALUE_DONE
                    elif not ("true".startswith(lowered) or "false".startswith(lowered)):
                        return False

            elif self.state == JSONState.NAME_VALUE:
                self.current_text += char
                valid_prefix = False
                for function in functions:
                    target = function.fn_name
                    if target.startswith(self.current_text):
                        valid_prefix = True
                        if self.current_text == target:
                            self.current_function = function
                            self.state = JSONState.NAME_CHOSEN
                            break
                if not valid_prefix:
                    return False

            i += 1
            if self.state in (JSONState.NAME_CHOSEN, JSONState.VALUE_DONE) and i < len(token):
                return False
        return True
