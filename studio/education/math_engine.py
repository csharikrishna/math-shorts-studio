"""Deterministic school-level mathematics engine using exact rational arithmetic and SymPy."""

from __future__ import annotations

from fractions import Fraction
import math
from typing import Any
from pydantic import BaseModel, Field
import sympy as sp

from studio.ai.expr import compile_expr


class WorkedExample(BaseModel):
    problem: str
    knowns: list[str] = Field(default_factory=list)
    goal: str
    steps: list[str] = Field(default_factory=list)
    final_answer: str
    common_mistake: str = ""
    verified: bool = True


class PracticeQuestion(BaseModel):
    question: str
    choices: list[str] = Field(default_factory=list)
    correct_index: int = 0
    correct_answer: str
    explanation: str
    grade: int = 5
    verified: bool = True


def exact_fraction_add(a_num: int, a_den: int, b_num: int, b_den: int) -> dict[str, Any]:
    """Add two fractions using exact rational arithmetic with LCD and verified equivalence."""
    if a_den == 0 or b_den == 0:
        raise ValueError("Denominator cannot be zero.")
    fa = Fraction(a_num, a_den)
    fb = Fraction(b_num, b_den)
    lcd = math.lcm(fa.denominator, fb.denominator)
    eq_a_num = fa.numerator * (lcd // fa.denominator)
    eq_b_num = fb.numerator * (lcd // fb.denominator)
    sum_raw_num = eq_a_num + eq_b_num
    exact_sum = fa + fb
    assert Fraction(eq_a_num, lcd) == fa
    assert Fraction(eq_b_num, lcd) == fb
    assert Fraction(sum_raw_num, lcd) == exact_sum
    return {
        "a": f"{fa.numerator}/{fa.denominator}",
        "b": f"{fb.numerator}/{fb.denominator}",
        "lcd": lcd,
        "eq_a": f"{eq_a_num}/{lcd}",
        "eq_b": f"{eq_b_num}/{lcd}",
        "sum_unsimplified": f"{sum_raw_num}/{lcd}",
        "sum_simplified": f"{exact_sum.numerator}/{exact_sum.denominator}" if exact_sum.denominator != 1 else str(exact_sum.numerator),
        "steps": [
            f"Find the least common denominator of {fa.denominator} and {fb.denominator}, which is {lcd}.",
            f"Convert {fa.numerator}/{fa.denominator} to {eq_a_num}/{lcd} and {fb.numerator}/{fb.denominator} to {eq_b_num}/{lcd}.",
            f"Add the numerators: ({eq_a_num} + {eq_b_num})/{lcd} = {sum_raw_num}/{lcd}.",
        ]
        + ([f"Simplify {sum_raw_num}/{lcd} = {exact_sum.numerator}/{exact_sum.denominator}."] if exact_sum.denominator != lcd else []),
    }


def exact_fraction_equivalent(num: int, den: int, factor: int) -> dict[str, Any]:
    if den == 0 or factor == 0:
        raise ValueError("Denominator and scale factor must be nonzero.")
    f = Fraction(num, den)
    eq_n, eq_d = num * factor, den * factor
    assert Fraction(eq_n, eq_d) == f
    return {"base": f"{num}/{den}", "factor": factor, "equivalent": f"{eq_n}/{eq_d}", "reduced": f"{f.numerator}/{f.denominator}"}


def solve_linear_equation_steps(a: int, b: int, c: int, d: int) -> dict[str, Any]:
    """Solve a*x + b = c*x + d exactly using Fraction arithmetic and verified balance steps."""
    if a == c:
        raise ValueError("Linear coefficient (a - c) must be nonzero for a unique solution.")
    x_exact = Fraction(d - b, a - c)
    assert a * x_exact + b == c * x_exact + d
    rhs_const = d - b
    lhs_coeff = a - c
    steps = [
        f"Start with the balanced equation: {a}x + {b} = {c}x + {d}.",
        f"Subtract {c}x from both sides: {lhs_coeff}x + {b} = {d}.",
        f"Subtract {b} from both sides: {lhs_coeff}x = {rhs_const}.",
        f"Divide both sides by {lhs_coeff}: x = {x_exact}.",
    ]
    return {"equation": f"{a}x + {b} = {c}x + {d}", "solution": str(x_exact), "solution_float": float(x_exact), "steps": steps}


def solve_quadratic_steps(a: int | float, b: int | float, c: int | float) -> dict[str, Any]:
    if a == 0:
        raise ValueError("Quadratic leading coefficient a must be nonzero.")
    x = sp.Symbol("x", real=True)
    disc = sp.simplify(b * b - 4 * a * c)
    roots = sp.solve(sp.Eq(a * x**2 + b * x + c, 0), x)
    return {
        "equation": f"{a}x^2 + {b}x + {c} = 0",
        "discriminant": str(disc),
        "roots": [str(r) for r in roots],
        "vertex": [float(-b / (2 * a)), float(c - b * b / (4 * a))],
    }


def array_multiplication_facts(rows: int, cols: int) -> dict[str, Any]:
    if rows <= 0 or cols <= 0:
        raise ValueError("Array dimensions must be positive.")
    prod = rows * cols
    return {
        "rows": rows,
        "cols": cols,
        "product": prod,
        "repeated_addition": " + ".join([str(cols)] * rows) + f" = {prod}",
        "commutative_form": f"{rows} x {cols} = {cols} x {rows} = {prod}",
    }


def place_value_decompose(n: int) -> dict[str, Any]:
    if n < 0 or n > 9999:
        raise ValueError("Supported range is 0..9999.")
    thousands = n // 1000
    hundreds = (n % 1000) // 100
    tens = (n % 100) // 10
    ones = n % 10
    parts = []
    if thousands:
        parts.append(f"{thousands * 1000}")
    if hundreds:
        parts.append(f"{hundreds * 100}")
    if tens:
        parts.append(f"{tens * 10}")
    parts.append(f"{ones}")
    return {"number": n, "thousands": thousands, "hundreds": hundreds, "tens": tens, "ones": ones, "expanded": " + ".join(parts)}


def integer_number_line_op(start: int, delta: int) -> dict[str, Any]:
    result = start + delta
    direction = "right" if delta >= 0 else "left"
    return {
        "start": start,
        "delta": delta,
        "result": result,
        "direction": direction,
        "steps": [
            f"Start at {start} on the number line.",
            f"Move {abs(delta)} units to the {direction}.",
            f"Land on {result}.",
        ],
    }


def ratio_and_percentage_steps(part: int, whole: int) -> dict[str, Any]:
    if whole <= 0:
        raise ValueError("Whole must be positive.")
    f = Fraction(part, whole)
    pct = Fraction(part * 100, whole)
    return {
        "ratio": f"{f.numerator}:{f.denominator}",
        "fraction": f"{f.numerator}/{f.denominator}",
        "decimal": round(float(f), 4),
        "percent": f"{float(pct):g}%",
    }


def right_triangle_trig(opp: float, adj: float) -> dict[str, Any]:
    if opp <= 0 or adj <= 0:
        raise ValueError("Triangle legs must be positive.")
    hyp = math.hypot(opp, adj)
    return {
        "opposite": opp,
        "adjacent": adj,
        "hypotenuse": round(hyp, 6),
        "sin": round(opp / hyp, 6),
        "cos": round(adj / hyp, 6),
        "tan": round(opp / adj, 6),
    }


def calculus_secant_to_tangent(expr_str: str, x0: float, h_values: list[float] | None = None) -> dict[str, Any]:
    """Compute exact derivative at x0 and secant slopes for shrinking h."""
    hs = h_values or [1.0, 0.5, 0.2, 0.05]
    x = sp.Symbol("x", real=True)
    sym_expr = sp.sympify(expr_str.replace("^", "**"), locals={"x": x})
    deriv_expr = sp.diff(sym_expr, x)
    exact_slope = float(deriv_expr.subs(x, x0))
    fn = compile_expr(expr_str, ("x",))
    y0 = float(fn(x=x0))
    secants = []
    for h in hs:
        if abs(h) < 1e-12:
            continue
        y1 = float(fn(x=x0 + h))
        secants.append({"h": h, "secant_slope": round((y1 - y0) / h, 6)})
    return {"expr": expr_str, "derivative_expr": str(deriv_expr), "x0": x0, "exact_slope": round(exact_slope, 6), "secants": secants}


def generate_worked_example(kind: str, **kwargs: Any) -> WorkedExample:
    if kind == "adding_fractions":
        r = exact_fraction_add(kwargs.get("a_num", 1), kwargs.get("a_den", 2), kwargs.get("b_num", 1), kwargs.get("b_den", 3))
        return WorkedExample(
            problem=f"Add {r['a']} + {r['b']}.",
            knowns=[f"First fraction: {r['a']}", f"Second fraction: {r['b']}"],
            goal="Find the exact sum as a simplified fraction.",
            steps=r["steps"],
            final_answer=r["sum_simplified"],
            common_mistake="Adding numerators and denominators directly (e.g., 1/2 + 1/3 != 2/5).",
        )
    if kind == "linear_equation":
        r = solve_linear_equation_steps(kwargs.get("a", 3), kwargs.get("b", 4), kwargs.get("c", 1), kwargs.get("d", 10))
        return WorkedExample(
            problem=f"Solve {r['equation']} for x.",
            knowns=[r["equation"]],
            goal="Find the value of x that keeps both sides equal.",
            steps=r["steps"],
            final_answer=f"x = {r['solution']}",
            common_mistake="Applying an operation to only one side of the equation.",
        )
    if kind == "multiplication_array":
        r = array_multiplication_facts(kwargs.get("rows", 3), kwargs.get("cols", 4))
        return WorkedExample(
            problem=f"Find {r['rows']} x {r['cols']} using an array.",
            knowns=[f"{r['rows']} equal rows", f"{r['cols']} items in each row"],
            goal="Count the total number of items.",
            steps=[f"Arrange {r['rows']} rows of {r['cols']} tiles.", f"Repeated addition: {r['repeated_addition']}."],
            final_answer=str(r["product"]),
            common_mistake="Adding rows and columns (3 + 4 = 7) instead of multiplying.",
        )
    if kind == "integers":
        r = integer_number_line_op(kwargs.get("start", -3), kwargs.get("delta", 5))
        return WorkedExample(
            problem=f"Compute ({r['start']}) + ({r['delta']}) on a number line.",
            knowns=[f"Starting position: {r['start']}", f"Directed step: {r['delta']:+d}"],
            goal="Find the final position on the number line.",
            steps=r["steps"],
            final_answer=str(r["result"]),
            common_mistake="Moving in the wrong direction when adding a negative or positive integer.",
        )
    r = calculus_secant_to_tangent(kwargs.get("expr", "x^2"), kwargs.get("x0", 1.0))
    return WorkedExample(
        problem=f"Find the instantaneous slope of f(x) = {r['expr']} at x = {r['x0']}.",
        knowns=[f"f(x) = {r['expr']}", f"f'(x) = {r['derivative_expr']}"],
        goal=f"Evaluate f'({r['x0']}).",
        steps=[f"Compute secant slopes as h shrinks: {r['secants']}", f"Take the limit h -> 0 using f'(x) = {r['derivative_expr']}."],
        final_answer=str(r["exact_slope"]),
        common_mistake="Using f(x0) instead of the slope f'(x0).",
    )


def generate_practice_question(kind: str, grade: int = 5, **kwargs: Any) -> PracticeQuestion:
    we = generate_worked_example(kind, **kwargs)
    ans = we.final_answer
    if kind == "adding_fractions":
        a_n, a_d = kwargs.get("a_num", 1), kwargs.get("a_den", 2)
        b_n, b_d = kwargs.get("b_num", 1), kwargs.get("b_den", 3)
        wrong_naive = f"{a_n + b_n}/{a_d + b_d}"
        cands = [ans, wrong_naive, f"{a_n + b_n}/{max(a_d, b_d)}", "1/1"]
    elif kind == "linear_equation":
        cands = [ans, "x = 0", "x = 1", "x = -1"]
    elif kind == "multiplication_array":
        rows, cols = kwargs.get("rows", 3), kwargs.get("cols", 4)
        cands = [ans, str(rows + cols), str(rows * cols + 2), str(max(1, rows * cols - 1))]
    else:
        cands = [ans, "0", "1", "-2"]
    uniq: list[str] = []
    for c in cands:
        if c not in uniq:
            uniq.append(c)
    while len(uniq) < 4:
        uniq.append(str(len(uniq) + 7))
    return PracticeQuestion(
        question=we.problem,
        choices=uniq[:4],
        correct_index=0,
        correct_answer=ans,
        explanation=" ".join(we.steps) + f" Therefore, {ans}.",
        grade=grade,
    )
