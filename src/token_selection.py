from copy import copy
from src.state_machine import JSONStateMachine
from src.json_file_parsing import FunctionCallingTest, FunctionDefinition


def get_valid_tokens(
        vocabulary: dict[str, int],
        state_machine: JSONStateMachine,
        test: FunctionCallingTest,
        functions: list[FunctionDefinition]
        ) -> list[int]:
    """Determine the valid tokens based on the JSON output syntax and the vocabulary."""
    valid_tokens = []
    for token_text, token_id in vocabulary.items():
        machine = copy(state_machine)
        if machine.transition(token_text, test, functions) is True:
            valid_tokens.append((token_text, token_id))
    return valid_tokens


def get_next_token(logits: list[float], valid_tokens: list[tuple[str, int]]) -> tuple[str, int]:
    """
    Return the (text, id) of the token with the highest logit among valid tokens.
    """
    return ((max(valid_tokens, key= lambda x: logits[x[1]])))
