import re

from pydantic import BaseModel

from src.json_file_parsing import FunctionDefinition

NUMBER_PREFIX_PATTERN = re.compile(r"^-?\d*\.?\d*$")
NUMBER_COMPLETE_PATTERN = re.compile(r"^-?\d+(\.\d+)?$")


class JSONStateMachine(BaseModel):
    """Keep track of the current constrained generation."""

    current_function: FunctionDefinition | None = None
    current_param_type: str | None = None
    current_text: str = ""
    function_complete: bool = False
    value_started: bool = False
    value_complete: bool = False

    def start_function(self) -> None:
        """Reset the state machine to start selecting a function."""
        self.current_function = None
        self.current_text = ""
        self.function_complete = False

    def start_value(self, param_type: str) -> None:
        """Prepare the state machine to generate one parameter value."""
        self.current_param_type = param_type
        self.current_text = ""
        self.value_started = False
        self.value_complete = False

    def valid_function_token(
        self,
        token: str,
        functions: list[FunctionDefinition],
    ) -> bool:
        """Check whether a token can continue/complete a function name."""
        if self.function_complete:
            return False

        candidate = self.current_text + token
        matching = [f for f in functions if f.fn_name.startswith(candidate)]

        if not matching:
            return False

        self.current_text = candidate
        self.current_function = next(
            (f for f in matching if f.fn_name == candidate), None
        )
        if self.current_function is not None and len(matching) == 1:
            self.function_complete = True

        return True

    def valid_value_token(self, token: str) -> bool:
        """Check whether a token can continue/close the current value."""
        if self.value_complete:
            return False
        if self.current_param_type == "string":
            return self._valid_string_token(token)
        if self.current_param_type == "number":
            return self._valid_number_token(token)
        if self.current_param_type == "boolean":
            return self._valid_boolean_token(token)
        return False

    def _valid_string_token(self, token: str) -> bool:
        if not self.value_started:
            quote_index = token.find('"')
            if quote_index == -1:
                return False

            self.value_started = True
            self.current_text = token[quote_index + 1:]

            closing_index = self.current_text.find('"')
            if closing_index != -1:
                self.current_text = self.current_text[:closing_index]
                self.value_complete = True

            return True

        quote_index = token.find('"')
        if quote_index != -1:
            self.current_text += token[:quote_index]
            self.value_complete = True
            return True

        self.current_text += token
        return True

    def _valid_number_token(self, token: str) -> bool:
        candidate = self.current_text + token
        if NUMBER_PREFIX_PATTERN.match(candidate) and candidate != "":
            self.current_text = candidate
            self.value_started = True
            return True

        if (
            self.value_started
            and NUMBER_COMPLETE_PATTERN.match(self.current_text)
            and token.strip() == ""
        ):
            self.value_complete = True
            return True

        return False

    def _valid_boolean_token(self, token: str) -> bool:
        candidate = self.current_text + token
        for word in ("true", "false"):
            if word.startswith(candidate):
                self.current_text = candidate
                self.value_started = True
                if candidate == word:
                    self.value_complete = True
                return True
        return False

    def is_forced_complete(self) -> bool:
        """Whether the current value is already complete
        without an explicit closing token."""
        if self.current_param_type == "number":
            return bool(NUMBER_COMPLETE_PATTERN.match(self.current_text))
        if self.current_param_type == "boolean":
            return self.current_text in ("true", "false")
        return False
