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
    if poly.is_zero:
        return False
    return int(poly.degree()) == expected_degree


def derive_latex(expr_str: str, var: str = "x") -> str:
    """Derive canonical LaTeX directly from validated expression."""
    return sp.latex(ast_to_sympy(expr_str, var))


def extract_taylor_coefficients(fn_str: str, max_degree: int, x0: float = 0.0, var: str = "x") -> list[float]:
    """Compute exact Taylor coefficients c_k of (x - x0)^k at x0 via SymPy series."""
    sym = sp.Symbol(var, real=True)
    u = sp.Symbol("_u_shift", real=True)
    expr = ast_to_sympy(fn_str, var)
    ser = sp.series(expr, sym, x0, max_degree + 1).removeO()
    shifted = sp.expand(ser.subs(sym, u + sp.nsimplify(x0)))
    poly = sp.Poly(shifted, u)
    return [float(poly.coeff_monomial(u**k)) for k in range(max_degree + 1)]


def eval_function(expr_str: str, x_val: float, var: str = "x") -> float:
    """Safely evaluate f(x_val) using compiled AST expression."""
    fn = compile_expr(expr_str, (var,))
    return float(fn(**{var: float(x_val)}))


def symbolic_derivative(expr_str: str, var: str = "x", order: int = 1) -> str:
    """Compute symbolic derivative d^order/dvar^order as a clean expression string."""
    sym = sp.Symbol(var, real=True)
    d = sp.diff(ast_to_sympy(expr_str, var), sym, order)
    return str(sp.simplify(d))


def eval_derivative(expr_str: str, x_val: float, var: str = "x", order: int = 1) -> float:
    """Evaluate symbolic derivative at x_val."""
    sym = sp.Symbol(var, real=True)
    d = sp.diff(ast_to_sympy(expr_str, var), sym, order)
    return float(d.subs(sym, float(x_val)).evalf())


def is_stationary_point(expr_str: str, x_val: float, tol: float = 1e-4, var: str = "x") -> bool:
    """Return True if |f'(x_val)| <= tol."""
    return abs(eval_derivative(expr_str, x_val, var=var, order=1)) <= tol


def is_local_minimum(expr_str: str, x_val: float, tol: float = 1e-4, var: str = "x") -> bool:
    """Return True if |f'(x_val)| <= tol and f''(x_val) > tol."""
    d1 = eval_derivative(expr_str, x_val, var=var, order=1)
    d2 = eval_derivative(expr_str, x_val, var=var, order=2)
    return abs(d1) <= tol and d2 > tol


def functions_equivalent(expr1: str, expr2: str, var: str = "x", tol: float = 1e-6) -> bool:
    """Verify whether two expressions are mathematically equivalent."""
    s1 = ast_to_sympy(expr1, var)
    s2 = ast_to_sympy(expr2, var)
    diff = sp.simplify(s1 - s2)
    if diff == 0:
        return True
    fn1, fn2 = compile_expr(expr1, (var,)), compile_expr(expr2, (var,))
    for xv in (-1.7, -0.4, 0.3, 1.1, 2.3):
        try:
            if abs(float(fn1(**{var: xv})) - float(fn2(**{var: xv}))) > tol:
                return False
        except Exception:
            return False
    return True


def gradient_descent_trajectory(
    expr_str: str,
    x0: float,
    lr: float,
    steps: int,
    var: str = "x",
) -> list[tuple[float, float, float]]:
    """Compute exact gradient descent iterations [(x_k, f(x_k), f'(x_k)), ...]."""
    if lr <= 0 or steps < 1:
        raise ValueError("Learning rate must be positive and steps >= 1.")
    traj: list[tuple[float, float, float]] = []
    x = float(x0)
    for _ in range(steps + 1):
        y = eval_function(expr_str, x, var=var)
        g = eval_derivative(expr_str, x, var=var, order=1)
        traj.append((x, y, g))
        x = x - float(lr) * g
    return traj


def discrete_convolution(base_probs: list[float], n_trials: int, offset_per_trial: int = 1) -> tuple[list[int], list[float]]:
    """Compute exact n-fold discrete convolution (e.g. sum of n fair dice)."""
    import numpy as np

    if n_trials < 1 or not base_probs:
        raise ValueError("n_trials >= 1 and non-empty base_probs required.")
    dist = np.asarray(base_probs, dtype=float)
    dist = dist / dist.sum()
    acc = dist.copy()
    for _ in range(n_trials - 1):
        acc = np.convolve(acc, dist)
    support = [n_trials * offset_per_trial + i for i in range(len(acc))]
    return support, [float(p) for p in acc]


def distribution_stats(support: list[float], probs: list[float]) -> tuple[float, float]:
    """Compute exact (mean, std) of a discrete probability distribution."""
    import math

    total = sum(probs)
    if abs(total - 1.0) > 1e-4:
        raise ValueError(f"Probabilities must sum to 1.0 (got {total}).")
    mu = sum(x * p for x, p in zip(support, probs))
    var = sum(((x - mu) ** 2) * p for x, p in zip(support, probs))
    return float(mu), float(math.sqrt(max(0.0, var)))


def bayes_posterior(prior: float, p_e_given_h: float, p_e_given_not_h: float) -> dict[str, float]:
    """Compute exact Bayes posterior and verify non-independence when claimed."""
    for name, v in (("prior", prior), ("p_e_given_h", p_e_given_h), ("p_e_given_not_h", p_e_given_not_h)):
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"{name} must be in [0, 1], got {v}.")
    tp = prior * p_e_given_h
    fp = (1.0 - prior) * p_e_given_not_h
    p_e = tp + fp
    if p_e <= 0.0:
        raise ValueError("P(E) must be positive.")
    posterior = tp / p_e
    return {
        "prior": float(prior),
        "tp": float(tp),
        "fp": float(fp),
        "p_e": float(p_e),
        "posterior": float(posterior),
        "delta": float(posterior - prior),
    }
