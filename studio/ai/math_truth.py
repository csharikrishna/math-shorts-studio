"""Single mathematical source of truth: AST-safe SymPy + NumPy verification & derivation."""

from __future__ import annotations

import ast
import sympy as sp

from studio.ai.expr import compile_expr
from studio.qa.engine import safe_sympy_from_expr


def ast_to_sympy(source: str, var: str = "x") -> sp.Expr:
    """Convert restricted math expression (including finite sum(k, a, b, step, body)) to SymPy Expr."""
    s = source.strip()
    if "sum(" not in s:
        return safe_sympy_from_expr(s, (var,))

    tree = ast.parse(s, mode="eval")
    sym_x = sp.Symbol(var, real=True)

    def _eval_node(node: ast.AST, env: dict[str, sp.Expr]) -> sp.Expr:
        if isinstance(node, ast.Expression):
            return _eval_node(node.body, env)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return sp.nsimplify(node.value) if isinstance(node.value, int) else sp.Float(node.value)
        if isinstance(node, ast.Name):
            if node.id in env:
                return env[node.id]
            if node.id == var:
                return sym_x
            if node.id == "pi":
                return sp.pi
            if node.id == "e":
                return sp.E
            raise ValueError(f"Disallowed symbol: {node.id!r}")
        if isinstance(node, ast.UnaryOp):
            v = _eval_node(node.operand, env)
            return v if isinstance(node.op, ast.UAdd) else -v
        if isinstance(node, ast.BinOp):
            l, r = _eval_node(node.left, env), _eval_node(node.right, env)
            if isinstance(node.op, ast.Add):
                return l + r
            if isinstance(node.op, ast.Sub):
                return l - r
            if isinstance(node.op, ast.Mult):
                return l * r
            if isinstance(node.op, ast.Div):
                return l / r
            if isinstance(node.op, ast.Pow):
                return l**r
            raise ValueError(f"Disallowed op: {type(node.op).__name__}")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            fname = node.func.id
            if fname == "sum" and len(node.args) == 5 and isinstance(node.args[0], ast.Name):
                k_name = node.args[0].id
                start = int(sp.Integer(_eval_node(node.args[1], env)))
                stop = int(sp.Integer(_eval_node(node.args[2], env)))
                step = int(sp.Integer(_eval_node(node.args[3], env)))
                if step <= 0 or (stop - start) // step > 64:
                    raise ValueError("Sum range exceeds safety limit.")
                total = sp.Integer(0)
                for k_val in range(start, stop + 1, step):
                    total += _eval_node(node.args[4], {**env, k_name: sp.Integer(k_val)})
                return total
            if fname == "factorial" and len(node.args) == 1:
                return sp.factorial(_eval_node(node.args[0], env))
            fn_map = {"sin": sp.sin, "cos": sp.cos, "tan": sp.tan, "exp": sp.exp, "log": sp.log, "sqrt": sp.sqrt, "abs": sp.Abs}
            if fname in fn_map and len(node.args) == 1:
                return fn_map[fname](_eval_node(node.args[0], env))
            raise ValueError(f"Disallowed call: {fname!r}")
        raise ValueError(f"Disallowed AST node: {type(node).__name__}")

    return _eval_node(tree, {})


def verify_polynomial_degree(expr_str: str, expected_degree: int, var: str = "x") -> bool:
    """Verify exact degree of a polynomial expression via SymPy."""
    sym = sp.Symbol(var, real=True)
    poly = sp.Poly(sp.expand(ast_to_sympy(expr_str, var)), sym)
    return int(poly.degree()) == expected_degree


def derive_latex(expr_str: str, var: str = "x") -> str:
    """Derive canonical LaTeX directly from validated expression."""
    return sp.latex(ast_to_sympy(expr_str, var))


def extract_taylor_coefficients(fn_str: str, max_degree: int, x0: float = 0.0, var: str = "x") -> list[float]:
    """Compute exact Taylor coefficients c_k at x0 via SymPy series."""
    sym = sp.Symbol(var, real=True)
    expr = ast_to_sympy(fn_str, var)
    ser = sp.series(expr, sym, x0, max_degree + 1).removeO()
    poly = sp.Poly(ser, sym)
    return [float(poly.coeff_monomial(sym**k)) for k in range(max_degree + 1)]
