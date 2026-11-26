"""Safe, vectorized math-expression compiler for AI-authored plans.

Gemini writes formulas as plain math strings ("sin(x) + x**3/6", "1 - 10**(-n)").
They are parsed with `ast`, checked against a strict whitelist (no attributes,
subscripts, lambdas, comprehensions or dunder access), and evaluated with NumPy.
Nothing an LLM writes is ever passed to `eval`.

Finite sums use a dedicated form: ``sum(k, start, stop, step, <expr in k>)`` with an
inclusive `stop`, e.g. the Fourier square wave ``4/pi*sum(k, 1, 19, 2, sin(k*x)/k)``.
"""

from __future__ import annotations

import ast
import math
from typing import Callable

import numpy as np

_FUNCS: dict[str, Callable] = {
    "sin": np.sin, "cos": np.cos, "tan": np.tan,
    "arcsin": np.arcsin, "arccos": np.arccos, "arctan": np.arctan, "arctan2": np.arctan2,
    "asin": np.arcsin, "acos": np.arccos, "atan": np.arctan, "atan2": np.arctan2,
    "sinh": np.sinh, "cosh": np.cosh, "tanh": np.tanh,
    "exp": np.exp, "log": np.log, "ln": np.log, "log10": np.log10, "log2": np.log2,
    "sqrt": np.sqrt, "abs": np.abs, "sign": np.sign, "floor": np.floor, "ceil": np.ceil,
    "round": np.round, "minimum": np.minimum, "maximum": np.maximum, "min": np.minimum, "max": np.maximum,
    "mod": np.mod, "heaviside": lambda x: np.heaviside(x, 0.5),
    "factorial": np.vectorize(lambda k: math.gamma(float(k) + 1.0), otypes=[float]),
    "gamma": np.vectorize(lambda k: math.gamma(float(k)), otypes=[float]),
}
_CONSTS = {"pi": np.pi, "e": np.e, "tau": 2.0 * np.pi, "phi": (1.0 + 5**0.5) / 2.0, "inf": np.inf}
_BINOPS = {
    ast.Add: np.add, ast.Sub: np.subtract, ast.Mult: np.multiply, ast.Div: np.true_divide,
    ast.Pow: np.power, ast.Mod: np.mod, ast.FloorDiv: np.floor_divide,
}
_CMP = {ast.Lt: np.less, ast.LtE: np.less_equal, ast.Gt: np.greater, ast.GtE: np.greater_equal}
MAX_SUM_TERMS = 400


class ExprError(ValueError):
    """Raised for any expression outside the safe grammar."""


class MathExpr:
    """A compiled, whitelisted expression over named variables (e.g. x, y, t, n)."""

    def __init__(self, source: str, variables: tuple[str, ...]) -> None:
        self.source = source.strip().replace("^", "**")
        self.variables = variables
        try:
            self.tree = ast.parse(self.source, mode="eval")
        except SyntaxError as exc:
            raise ExprError(f"cannot parse {source!r}: {exc.msg}") from exc
        self._check(self.tree.body, set(variables))

    def _check(self, node: ast.AST, scope: set[str]) -> None:
        if isinstance(node, ast.Constant):
            if not isinstance(node.value, (int, float)) or isinstance(node.value, bool):
                raise ExprError(f"only numeric constants are allowed in {self.source!r}")
        elif isinstance(node, ast.Name):
            if node.id not in scope and node.id not in _CONSTS:
                raise ExprError(f"unknown name {node.id!r} in {self.source!r} (allowed: {sorted(scope)})")
        elif isinstance(node, ast.BinOp):
            if type(node.op) not in _BINOPS:
                raise ExprError(f"operator not allowed in {self.source!r}")
            self._check(node.left, scope)
            self._check(node.right, scope)
        elif isinstance(node, ast.UnaryOp):
            if not isinstance(node.op, (ast.USub, ast.UAdd)):
                raise ExprError(f"unary operator not allowed in {self.source!r}")
            self._check(node.operand, scope)
        elif isinstance(node, ast.Compare):
            if len(node.ops) != 1 or type(node.ops[0]) not in _CMP:
                raise ExprError(f"only single <, <=, >, >= comparisons allowed in {self.source!r}")
            self._check(node.left, scope)
            self._check(node.comparators[0], scope)
        elif isinstance(node, ast.IfExp):
            for part in (node.test, node.body, node.orelse):
                self._check(part, scope)
        elif isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.keywords:
                raise ExprError(f"only plain function calls allowed in {self.source!r}")
            name = node.func.id
            if name == "sum":
                if len(node.args) != 5 or not isinstance(node.args[0], ast.Name):
                    raise ExprError("sum takes (k, start, stop, step, expr) with a bare index name")
                for bound in node.args[1:4]:
                    self._check(bound, set())
                self._check(node.args[4], scope | {node.args[0].id})
                return
            if name not in _FUNCS:
                raise ExprError(f"unknown function {name!r} in {self.source!r}")
            for arg in node.args:
                self._check(arg, scope)
        else:
            raise ExprError(f"syntax {type(node).__name__} not allowed in {self.source!r}")

    def _eval(self, node: ast.AST, env: dict[str, object]):
        if isinstance(node, ast.Constant):
            return float(node.value)
        if isinstance(node, ast.Name):
            return env[node.id] if node.id in env else _CONSTS[node.id]
        if isinstance(node, ast.BinOp):
            return _BINOPS[type(node.op)](self._eval(node.left, env), self._eval(node.right, env))
        if isinstance(node, ast.UnaryOp):
            val = self._eval(node.operand, env)
            return -val if isinstance(node.op, ast.USub) else val
        if isinstance(node, ast.Compare):
            return _CMP[type(node.ops[0])](self._eval(node.left, env), self._eval(node.comparators[0], env)).astype(float)
        if isinstance(node, ast.IfExp):
            return np.where(self._eval(node.test, env), self._eval(node.body, env), self._eval(node.orelse, env))
        if isinstance(node, ast.Call):
            name = node.func.id  # type: ignore[attr-defined]
            if name == "sum":
                idx = node.args[0].id  # type: ignore[attr-defined]
                start, stop, step = (float(self._eval(a, {})) for a in node.args[1:4])
                if step == 0 or (stop - start) / step > MAX_SUM_TERMS:
                    raise ExprError(f"sum range too long in {self.source!r} (max {MAX_SUM_TERMS} terms)")
                total = 0.0
                k = start
                while (step > 0 and k <= stop + 1e-9) or (step < 0 and k >= stop - 1e-9):
                    total = total + self._eval(node.args[4], {**env, idx: k})
                    k += step
                return total
            return _FUNCS[name](*(self._eval(a, env) for a in node.args))
        raise ExprError(f"unsupported node {type(node).__name__}")

    def __call__(self, **values) -> np.ndarray:
        env = {k: np.asarray(v, dtype=float) for k, v in values.items()}
        with np.errstate(all="ignore"):
            out = self._eval(self.tree.body, env)
        shape = np.broadcast(*env.values()).shape if env else ()
        return np.broadcast_to(np.asarray(out, dtype=float), shape).copy()


def compile_expr(source: str, variables: tuple[str, ...]) -> MathExpr:
    return MathExpr(source, variables)


def finite_fraction(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    return float(np.mean(np.isfinite(values))) if values.size else 0.0
