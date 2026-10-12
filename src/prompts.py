from src.json_file_parsing import FunctionDefinition


def build_function_selection_prompt(
    user_prompt: str,
    functions: list[FunctionDefinition],
) -> str:
    """Build the input text used to let the LLM select a function."""
    functions_text = "\n".join(
        f"- {f.fn_name}: {f.description} "
        f"(parameters: {', '.join(f.args_names)})"
        for f in functions
    )

    return (
        "Available functions:\n"
        f"{functions_text}\n\n"
        f'User request: "{user_prompt}"\n\n'
        "Function to call: "
    )


def build_value_generation_prompt(
    user_prompt: str,
    function: FunctionDefinition,
    generated_values: dict[str, str],
    current_parameter: str,
) -> str:
    """Build the input text used to let the LLM
    generate one parameter value."""
    values_text = ", ".join(
        f"{name}={value}" for name, value in generated_values.items()
    ) or "(none yet)"

    param_type = function.args_types[current_parameter]

    return (
        f'User request: "{user_prompt}"\n\n'
        f"Function selected: {function.fn_name} ({function.description})\n"
        f"Values generated so far: {values_text}\n\n"
        f"Generate the value for parameter '{current_parameter}' "
        f"(type: {param_type}): "
    )
