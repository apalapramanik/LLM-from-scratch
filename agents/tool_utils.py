"""
Shared helpers for tool calling.

validate_args: check the model's arguments against the tool's own schema
(the same `tools = [...]` list we send to the model) and return a clear,
fixable message instead of a cryptic KeyError.
"""

TYPE_MAP = {"string": str, "number": (int, float), "integer": int, "boolean": bool}


def validate_args(tools: list, name: str, inputs: dict):
    """Return None if OK, else an error string the model can act on."""
    schema = next((t["function"] for t in tools if t["function"]["name"] == name), None)
    if schema is None:
        known = [t["function"]["name"] for t in tools]
        return f"Unknown tool '{name}'. Available tools: {known}"

    params = schema["parameters"]
    props = params.get("properties", {})

    for key in params.get("required", []):
        if key not in inputs:
            expected = {k: v.get("type") for k, v in props.items()}
            return f"Missing required argument '{key}' for {name}. Expected: {expected}"

    for key, value in inputs.items():
        if key in props:
            want = TYPE_MAP.get(props[key].get("type"))
            if want and not isinstance(value, want):
                return f"Argument '{key}' should be {props[key]['type']}, got {type(value).__name__}"
    return None


if __name__ == "__main__":
    demo = [{"function": {"name": "get_weather", "parameters": {
        "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
    print(validate_args(demo, "get_weather", {"city": "Tokyo"}))   # None = OK
    print(validate_args(demo, "get_weather", {"cty": "Tokyo"}))    # missing
    print(validate_args(demo, "get_weather", {"city": 5}))         # wrong type
    print(validate_args(demo, "fly_to_moon", {}))                  # unknown tool
