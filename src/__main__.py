import argparse
import os
from json import dump, load

from llm_sdk import Small_LLM_Model
from src.json_file_parsing import parse_function_calling_tests, parse_functions
from src.state_machine import JSONState, JSONStateMachine
from src.token_selection import get_next_token, prepare_vocabulary


def emit_literal(model: Small_LLM_Model, input_ids: list[int], text: str) -> None:
    """Texte 100% déterministe (ponctuation JSON fixe) : pas de logits,
    pas de contrainte, juste encodé et ajouté au contexte."""
    input_ids.extend(model.encode(text)[0].tolist())


def build_prompt_context(test, functions: list) -> str:
    """Provide the model with the available function catalog before decoding."""
    lines = ["Available functions:"]
    for function in functions:
        args = ", ".join(f"{arg_name}:{function.args_types[arg_name]}" for arg_name in function.args_names)
        lines.append(f"- {function.fn_name}({args})")
    lines.append(f"User prompt: {test.prompt}")
    lines.append("Return only the function call JSON." )
    return "\n".join(lines)


def generate_constrained(
    model: Small_LLM_Model,
    vocabulary: dict[int, str],
    input_ids: list[int],
    state_machine: JSONStateMachine,
    stop_state: JSONState,
    functions: list,
) -> str:
    """Boucle logits -> masque -> échantillonnage, uniquement pendant
    qu'une vraie décision reste à prendre (choix de fonction ou valeur
    de paramètre)."""
    generated_text = ""
    while state_machine.state != stop_state:
        logits = model.get_logits_from_input_ids(input_ids)
        token_text, token_id = get_next_token(logits, vocabulary, state_machine, functions)
        input_ids.append(token_id)
        generated_text += token_text
        if not state_machine.transition(token_text, functions):
            raise ValueError(f"Invalid token generated: {token_text!r}")
    return generated_text


def parse_typed_value(raw_text: str, arg_type: str):
    raw_text = raw_text.strip()
    raw_text = raw_text.rstrip(",} ")
    if arg_type == "str":
        return raw_text.strip('"')
    if arg_type == "int":
        return int(raw_text)
    if arg_type == "float":
        return float(raw_text)
    if arg_type == "bool":
        return raw_text.lower() == "true"
    raise ValueError(f"Unknown argument type: {arg_type}")


def generate_function_call(model: Small_LLM_Model, vocabulary: dict, test, functions: list):
    input_ids = model.encode(build_prompt_context(test, functions))[0].tolist()

    emit_literal(model, input_ids, '{"name": "')
    sm = JSONStateMachine()
    sm.start_name()
    generate_constrained(model, vocabulary, input_ids, sm, JSONState.NAME_CHOSEN, functions)
    function = sm.current_function

    emit_literal(model, input_ids, '", "parameters": {')
    parameters = {}
    for i, arg_name in enumerate(function.args_names):
        emit_literal(model, input_ids, f'"{arg_name}": ')
        sm = JSONStateMachine()
        sm.start_parameter(function, arg_name)
        raw_value = generate_constrained(model, vocabulary, input_ids, sm, JSONState.VALUE_DONE, functions)
        parameters[arg_name] = parse_typed_value(raw_value, function.args_types[arg_name])
        if i < len(function.args_names) - 1:
            emit_literal(model, input_ids, ", ")

    emit_literal(model, input_ids, "}}")
    return function.fn_name, parameters


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--functions_definition", default="data/input/functions_definition.json")
    parser.add_argument("--input", default="data/input/function_calling_tests.json")
    parser.add_argument("--output", default="data/output/function_calling_results.json")
    args = parser.parse_args()

    tests = parse_function_calling_tests(args.input)
    functions = parse_functions(args.functions_definition)
    model = Small_LLM_Model()
    with open(model.get_path_to_vocab_file(), "r") as f:
        vocabulary = prepare_vocabulary(load(f))

    output = []
    for test in tests:
        try:
            fn_name, parameters = generate_function_call(model, vocabulary, test, functions)
        except ValueError as e:
            print(f"Skipping prompt {test.prompt!r}: {e}")
            continue
        output.append({"prompt": test.prompt, "name": fn_name, "parameters": parameters})

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        dump(output, f, indent=2)


if __name__ == "__main__":
    main()