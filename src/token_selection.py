from copy import copy
import numpy as np
from src.state_machine import JSONStateMachine
from src.json_file_parsing import FunctionDefinition


def prepare_vocabulary(vocabulary: dict[str, int]) -> dict[int, str]:
    """Decode token strings once and index them by token id for fast lookup."""
    return {token_id: token_text.replace("Ġ", " ") for token_text, token_id in vocabulary.items()}


def get_valid_tokens(
        vocabulary: dict[int, str],
        state_machine: JSONStateMachine,
        functions: list[FunctionDefinition],
        ) -> list[tuple[str, int]]:
    """Determine the valid tokens based on the JSON output syntax and the vocabulary."""
    valid_tokens = []
    for token_id, token_text in vocabulary.items():
        machine = copy(state_machine)
        if machine.transition(token_text, functions) is True:
            valid_tokens.append((token_text, token_id))
    return valid_tokens


def get_next_token(
        logits: list[float],
        vocabulary: dict[int, str],
        state_machine: JSONStateMachine,
        functions: list[FunctionDefinition],
        initial_top_k: int = 256,
        ) -> tuple[str, int]:
    """Return the highest-logit valid token using a progressive top-k search."""
    if not vocabulary:
        raise ValueError("No vocabulary available")

    scores = np.asarray(logits, dtype=np.float32)
    vocab_size = scores.shape[0]
    if vocab_size == 0:
        raise ValueError("No logits available")

    k = min(max(1, initial_top_k), vocab_size)
    tested_ids: set[int] = set()

    while True:
        candidate_ids = np.argpartition(-scores, k - 1)[:k]
        candidate_ids = candidate_ids[np.argsort(-scores[candidate_ids])]

        for idx in candidate_ids.tolist():
            token_id = int(idx)
            if token_id in tested_ids:
                continue
            tested_ids.add(token_id)
            token_text = vocabulary.get(token_id)
            if token_text is None:
                continue
            machine = copy(state_machine)
            if machine.transition(token_text, functions):
                return token_text, token_id

        if k == vocab_size:
            break
        k = min(vocab_size, k * 2)

    raise ValueError("No valid tokens available for the current state")
