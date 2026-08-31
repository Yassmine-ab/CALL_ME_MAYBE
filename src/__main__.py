import argparse
from json import load
from llm_sdk import Small_LLM_Model
from src.tests_parsing import parse_function_calling_tests
from src.functions_parsing import parse_functions
from src.state_machine import JSONState, JSONStateMachine
from src.token_selection import get_valid_tokens, is_token_valid


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--functions_definition",
        default="data/input/functions_definition.json",
    )
    parser.add_argument(
        "--input",
        default="data/input/function_calling_tests.json",
    )
    parser.add_argument(
        "--output",
        default="data/output/function_calling_results.json",
    )
    args = parser.parse_args()
    tests = parse_function_calling_tests(args.input)
    functions = parse_functions(args.functions_definition)
    model = Small_LLM_Model()
    for test in tests:
        state_machine = JSONStateMachine()
        input_tokens = model._encode(test.prompt)[0].tolist()
        with open(model.get_path_to_vocabulary_json(), "r") as f:
            vocabulary = load(f)
        while state_machine.state != JSONState.END:
            logits = model.get_logits_from_input_ids(input_tokens)
            # valid_tokens = get_valid_tokens(state_machine, functions)
            valid_tokens = []
            for token_text, token_id in vocabulary.items():
                if is_token_valid(token_text, state_machine, test, functions):
                        valid_tokens.append(token_id)
            masked_logits = mask_logits(logits, valid_tokens)
            next_token = get_next_token(masked_logits)
            input_tokens.append(next_token)
            state_machine.transition(next_token)


if __name__ == "__main__":
    main()
