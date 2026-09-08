import argparse
from json import load, loads, dump
from llm_sdk import Small_LLM_Model
from src.json_file_parsing import parse_function_calling_tests, parse_functions
from src.state_machine import JSONState, JSONStateMachine
from src.token_selection import get_valid_tokens, get_next_token


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
    with open(model.get_path_to_vocabulary_json(), "r") as f:
        vocabulary = load(f)
    output = []
    for test in tests:
        state_machine = JSONStateMachine()
        input_tokens_ids = model._encode(test.prompt)[0].tolist()
        generated_text = ""
        while state_machine.state != JSONState.END:
            logits = model.get_logits_from_input_ids(input_tokens_ids)
            valid_tokens = get_valid_tokens(vocabulary, state_machine, test, functions)
            next_token = get_next_token(logits, valid_tokens)
            input_tokens_ids.append(next_token[1])
            generated_text += next_token[0]
            state_machine.transition(next_token[0], test, functions)
        output.append(generated_text)
    with open(args.output, "w") as f:
        dump(output)


if __name__ == "__main__":
    main()
