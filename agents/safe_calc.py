"""
Safe calculator — a whitelist-based replacement for eval().

Idea: parse the text into a tree (this does NOT run it), then only compute
pieces we explicitly allow. Everything else raises an error.
"""
import ast
import math
import operator

# The whitelist: only these operators...
OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,   # negative numbers, e.g. -5
}

# ...and only these functions / constants
FUNCTIONS = {
    "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "log": math.log, "log10": math.log10, "exp": math.exp,
    "abs": abs, "round": round,
}
CONSTANTS = {"pi": math.pi, "e": math.e}


def _eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value                                   # a plain number
    if isinstance(node, ast.BinOp) and type(node.op) in OPERATORS:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 1000:
            raise ValueError("Exponent too large")          # stop 9**9**9 hangs
        return OPERATORS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in OPERATORS:
        return OPERATORS[type(node.op)](_eval(node.operand))
    if isinstance(node, ast.Name) and node.id in CONSTANTS:
        return CONSTANTS[node.id]
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id in FUNCTIONS and not node.keywords):
        return FUNCTIONS[node.func.id](*[_eval(a) for a in node.args])
    raise ValueError(f"Not allowed: {ast.dump(node)[:60]}")  # not on the list


def safe_calc(expression: str):
    expression = expression.replace("^", "**")  # model sometimes writes ^ for power
    tree = ast.parse(expression, mode="eval")   # parse only — nothing runs
    return _eval(tree.body)


if __name__ == "__main__":
    for expr in ["2 + 2", "sqrt(144)", "2^10", "-5 * 3", "pi * 2",
                 "__import__('os').system('ls')", "9**9**9", "open('/etc/passwd')"]:
        try:
            print(f"{expr:35} → {safe_calc(expr)}")
        except Exception as ex:
            print(f"{expr:35} → REJECTED ({type(ex).__name__}: {ex})")
