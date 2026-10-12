import argparse
import json
import os
import sys

from llm_sdk import Small_LLM_Model
from src.generate import (
    FunctionSelectionError,
    ValueGenerationError,
    generate_function_call,
)
from src.json_file_parsing import parse_function_calling_tests, parse_functions
from src.token_selection import prepare_vocabulary


def main() -> None:
    """Run the function-calling program."""
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
    if not tests or not functions:
        sys.stderr.write("Error: no tests or no functions to process.\n")
        return

    model = Small_LLM_Model()
    try:
        vocabulary_path = model.get_path_to_vocab_file()
        with open(vocabulary_path, "r", encoding="utf-8") as file:
            vocabulary = prepare_vocabulary(json.load(file))
    except (OSError, json.JSONDecodeError) as error:
        sys.stderr.write(f"Error: could not load the vocab file: {error}\n")
        return

    results = []

    for test in tests:
        print(f"Processing: {test.prompt}")

        try:
            result = generate_function_call(
                model, vocabulary, functions, test.prompt
                )
            results.append(result)
        except (FunctionSelectionError, ValueGenerationError) as error:
            sys.stderr.write(f"Skipping prompt {test.prompt!r}: {error}\n")
            continue
        except Exception as error:
            sys.stderr.write(
                f"Skipping prompt {test.prompt!r}: unexpected error: {error}\n"
            )
            continue

    directory = os.path.dirname(args.output)
    if directory:
        os.makedirs(directory, exist_ok=True)

    try:
        with open(args.output, "w", encoding="utf-8") as file:
            json.dump(results, file, indent=2, ensure_ascii=False)
    except OSError as error:
        print(f"Error: could not write output file: {error}", file=sys.stderr)
        return

    print(f"Done: {len(results)}/{len(tests)} prompts processed successfully.")


if __name__ == "__main__":
    main()
