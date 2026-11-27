"""Single mathematical source of truth: AST-safe SymPy + NumPy verification & derivation."""

from __future__ import annotations

import ast
import re
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


def definite_integral(
    expr_str: str,
    a: float,
    b: float,
    n: int = 12,
    method: str = "midpoint",
    var: str = "x",
) -> dict[str, object]:
    """Compute definite integral and Riemann rectangle partition [(x0, w, y), ...] over [a, b]."""
    import math

    a_f, b_f = float(a), float(b)
    if n < 1 or b_f <= a_f:
        raise ValueError(f"Expected n >= 1 and b > a (got n={n}, [{a}, {b}]).")
    dx = (b_f - a_f) / n
    boxes: list[tuple[float, float, float]] = []
    r_sum = 0.0
    for i in range(n):
        x0 = a_f + i * dx
        if method == "left":
            xs = x0
        elif method == "right":
            xs = x0 + dx
        else:
            xs = x0 + 0.5 * dx
        y = eval_function(expr_str, xs, var=var)
        if not math.isfinite(y):
            raise ValueError(f"Non-finite sample y={y} at {var}={xs}.")
        boxes.append((float(x0), float(dx), float(y)))
        r_sum += y * dx

    exact_val: float
    try:
        sym = sp.Symbol(var, real=True)
        res = sp.integrate(ast_to_sympy(expr_str, var), (sym, a_f, b_f))
        exact_val = float(sp.N(res))
        if not math.isfinite(exact_val):
            raise ValueError("Non-finite symbolic integral")
    except Exception:
        # High-resolution composite Simpson fallback
        m_steps = 200
        h = (b_f - a_f) / m_steps
        s = eval_function(expr_str, a_f, var=var) + eval_function(expr_str, b_f, var=var)
        for k in range(1, m_steps):
            s += (4.0 if k % 2 else 2.0) * eval_function(expr_str, a_f + k * h, var=var)
        exact_val = float(s * h / 3.0)

    return {"integral_exact": exact_val, "riemann_sum": float(r_sum), "boxes": boxes}


def rk4_ode_trajectory(
    dx_expr: str,
    dy_expr: str | None = None,
    x0: float = 1.0,
    y0: float = 0.0,
    t0: float = 0.0,
    dt: float = 0.05,
    steps: int = 80,
) -> list[tuple[float, float, float]]:
    """Integrate 1D/2D ODE dx/dt=f(x,y,t), dy/dt=g(x,y,t) via 4th-order Runge-Kutta."""
    import math

    if dt <= 0 or steps < 1:
        raise ValueError("dt must be positive and steps >= 1.")
    fx = compile_expr(dx_expr, ("x", "y", "t"))
    fy = compile_expr(dy_expr, ("x", "y", "t")) if dy_expr else None

    def _rhs(t_v: float, x_v: float, y_v: float) -> tuple[float, float]:
        vx = float(fx(x=x_v, y=y_v, t=t_v))
        vy = float(fy(x=x_v, y=y_v, t=t_v)) if fy is not None else 0.0
        if not (math.isfinite(vx) and math.isfinite(vy)):
            raise ValueError(f"Non-finite ODE RHS at t={t_v}, x={x_v}, y={y_v}.")
        return vx, vy

    t_c, x_c, y_c = float(t0), float(x0), float(y0)
    out: list[tuple[float, float, float]] = [(t_c, x_c, y_c)]
    for _ in range(steps):
        k1x, k1y = _rhs(t_c, x_c, y_c)
        k2x, k2y = _rhs(t_c + 0.5 * dt, x_c + 0.5 * dt * k1x, y_c + 0.5 * dt * k1y)
        k3x, k3y = _rhs(t_c + 0.5 * dt, x_c + 0.5 * dt * k2x, y_c + 0.5 * dt * k2y)
        k4x, k4y = _rhs(t_c + dt, x_c + dt * k3x, y_c + dt * k3y)
        x_c += (dt / 6.0) * (k1x + 2.0 * k2x + 2.0 * k3x + k4x)
        y_c += (dt / 6.0) * (k1y + 2.0 * k2y + 2.0 * k3y + k4y)
        t_c += dt
        if not (math.isfinite(x_c) and math.isfinite(y_c)):
            raise ValueError("ODE integration diverged to non-finite state.")
        out.append((round(t_c, 6), float(x_c), float(y_c)))
    return out


def matrix_transform_and_eigen(matrix_2x2: list[list[float]]) -> dict[str, object]:
    """Compute exact 2x2 determinant, trace, classification, eigenvalues, and eigenvectors."""
    import numpy as np

    if len(matrix_2x2) != 2 or any(len(r) != 2 for r in matrix_2x2):
        raise ValueError("matrix_2x2 must be 2x2.")
    A = np.asarray(matrix_2x2, dtype=float)
    det = float(np.linalg.det(A))
    tr = float(np.trace(A))
    disc = tr * tr - 4.0 * det
    vals, vecs = np.linalg.eig(A)
    is_complex = bool(np.any(np.abs(vals.imag) > 1e-7))

    real_eigvals: list[float] = []
    eig_pairs: list[tuple[float, float]] = []
    real_eigvecs: list[list[float]] = []
    for i in range(2):
        eig_pairs.append((float(vals[i].real), float(vals[i].imag)))
        if abs(float(vals[i].imag)) <= 1e-7:
            real_eigvals.append(float(vals[i].real))
            vec = vecs[:, i].real
            norm = float(np.linalg.norm(vec)) or 1.0
            real_eigvecs.append([float(vec[0] / norm), float(vec[1] / norm)])

    if is_complex:
        re_p = eig_pairs[0][0]
        regime = "center" if abs(re_p) < 1e-5 else ("stable_spiral" if re_p < 0 else "unstable_spiral")
    elif abs(det) < 1e-7:
        regime = "degenerate"
    elif det < 0:
        regime = "saddle"
    elif tr < 0:
        regime = "stable_node"
    elif tr > 0:
        regime = "unstable_node"
    else:
        regime = "degenerate"

    return {
        "matrix": [[float(A[0, 0]), float(A[0, 1])], [float(A[1, 0]), float(A[1, 1])]],
        "determinant": det,
        "trace": tr,
        "discriminant": float(disc),
        "is_complex": is_complex,
        "eigenvalues_ri": eig_pairs,
        "real_eigenvalues": real_eigvals,
        "real_eigenvectors": real_eigvecs,
        "regime": regime,
    }


def verify_eigenvalue_stability_regime(
    matrix_2x2: list[list[float]],
    *,
    expected_regime: str | None = None,
    statement: str = "",
) -> tuple[bool, str]:
    """Deterministically verify 2D linear stability definitions & distinguish saddle vs unstable source vs sink."""
    info = matrix_transform_and_eigen(matrix_2x2)
    regime = str(info["regime"])
    eigs: list[tuple[float, float]] = list(info["eigenvalues_ri"])  # type: ignore[arg-type]
    re_parts = [re_v for re_v, _ in eigs]

    if expected_regime and expected_regime != regime:
        return False, f"expected_regime={expected_regime!r} disagrees with computed regime={regime!r} (eigs={eigs})"

    low = (statement or "").lower()
    all_neg = all(r < -1e-6 for r in re_parts)
    all_pos = all(r > 1e-6 for r in re_parts)
    any_pos = any(r > 1e-6 for r in re_parts)
    any_zero = any(abs(r) <= 1e-6 for r in re_parts)

    has_stable_phrase = bool(
        re.search(r"\b(?:stable\s+sink|asymptotically\s+stable|stable\s+node|stable\s+spiral|attracting\s+equilibrium)\b", low)
    )
    has_unstable_word = bool(re.search(r"\bunstable\b", low))
    has_not_stable = bool(re.search(r"\bnot\s+stable\b", low))

    if has_stable_phrase and not all_neg:
        return False, f"claimed asymptotically stable regime ({statement!r}), but eigenvalues {eigs} do not satisfy all Re(lambda) < 0"
    if re.search(r"\b(?:unstable\s+source|source)\b", low) and not re.search(r"\bsaddle\b", low) and not all_pos:
        return False, f"claimed unstable source ({statement!r}), but regime is {regime!r} (all eigenvalues must have Re(lambda) > 0)"
    if re.search(r"\bsaddle\b", low) and regime != "saddle":
        return False, f"claimed saddle point ({statement!r}), but computed regime is {regime!r}"
    if re.search(r"\bcenter\b", low) and regime != "center":
        return False, f"claimed center ({statement!r}), but computed regime is {regime!r}"
    if "re(lambda) > 0" in low or r"re(\lambda) > 0" in low or "positive real part" in low:
        if re.search(r"\bstable\b", low) and not has_unstable_word and not has_not_stable:
            return False, f"statement mixes Re(lambda) > 0 with stable behavior ({statement!r})"
    if (re.search(r"\bstable\b", low) and not has_unstable_word and not has_not_stable) and any_pos:
        return False, f"statement claims stability despite positive real-part eigenvalue {eigs}"
    if (re.search(r"\b(?:linearization|nonlinear)\b", low)) and any_zero and ("non-hyperbolic" not in low and "inconclusive" not in low and "center" not in low):
        return False, f"linearization at Re(lambda)=0 is non-hyperbolic; note non-hyperbolic caveat ({eigs})"

    return True, f"verified {regime} (Re={[round(r, 4) for r in re_parts]})"


def transform_points_2x2(points: list[list[float]], matrix_2x2: list[list[float]]) -> list[list[float]]:
    """Transform 2D coordinates by linear matrix A."""
    a, b = float(matrix_2x2[0][0]), float(matrix_2x2[0][1])
    c, d = float(matrix_2x2[1][0]), float(matrix_2x2[1][1])
    return [[a * float(p[0]) + b * float(p[1]), c * float(p[0]) + d * float(p[1])] for p in points]


def discrete_pmf_convolution(
    support: list[float],
    probs: list[float],
    n_trials: int = 1,
) -> tuple[list[float], list[float], float, float]:
    """Compute exact n-fold sum distribution for ANY arbitrary discrete PMF, plus mean & std."""
    if not support or len(support) != len(probs) or n_trials < 1:
        raise ValueError("support and probs must be non-empty of equal length with n_trials >= 1.")
    total = sum(probs)
    if total <= 0:
        raise ValueError("Sum of probs must be positive.")
    base_norm = [(round(float(x), 8), float(p) / total) for x, p in zip(support, probs) if p > 0]
    acc: dict[float, float] = {x: p for x, p in base_norm}
    for _ in range(n_trials - 1):
        nxt: dict[float, float] = {}
        for x1, p1 in acc.items():
            for x2, p2 in base_norm:
                s = round(x1 + x2, 6)
                nxt[s] = nxt.get(s, 0.0) + p1 * p2
        acc = nxt
    xs = sorted(acc.keys())
    ps = [float(acc[x]) for x in xs]
    mu, sig = distribution_stats(xs, ps)
    return xs, ps, mu, sig


def fourier_partial_sum_terms(
    expr_str: str,
    period: float = 6.283185307179586,
    n_harmonics: int = 5,
    var: str = "x",
) -> dict[str, object]:
    """Project arbitrary periodic function onto trig basis over [-L, L] via inner products."""
    import math
    import numpy as np

    L = 0.5 * float(period)
    if L <= 0 or n_harmonics < 1:
        raise ValueError("Period must be positive and n_harmonics >= 1.")
    fn = compile_expr(expr_str, (var,))
    n_grid = 1024
    dx = (2.0 * L) / n_grid
    xs = -L + (np.arange(n_grid, dtype=float) + 0.5) * dx
    ys = np.array([float(fn(**{var: float(xv)})) for xv in xs], dtype=float)
    a0 = float((1.0 / L) * float(np.sum(ys)) * dx)
    terms: list[str] = [f"{a0 * 0.5:.6g}"] if abs(a0) > 1e-6 else []
    harmonics: list[tuple[int, float, float]] = []
    for n in range(1, n_harmonics + 1):
        omega = (n * math.pi) / L
        an = float((1.0 / L) * float(np.sum(ys * np.cos(omega * xs))) * dx)
        bn = float((1.0 / L) * float(np.sum(ys * np.sin(omega * xs))) * dx)
        harmonics.append((n, an, bn))
        if abs(an) > 1e-5:
            terms.append(f"({an:.6g})*cos({omega:.6g}*{var})")
        if abs(bn) > 1e-5:
            terms.append(f"({bn:.6g})*sin({omega:.6g}*{var})")
    partial_expr = " + ".join(terms) if terms else "0"
    return {"a0": a0, "harmonics": harmonics, "expr": partial_expr}


def newton_raphson_trajectory(
    expr_str: str,
    x0: float,
    steps: int = 5,
    var: str = "x",
) -> list[tuple[float, float, float]]:
    """Compute exact Newton root-finding iterations [(x_k, f(x_k), f'(x_k)), ...]."""
    x = float(x0)
    out: list[tuple[float, float, float]] = []
    for _ in range(max(1, steps) + 1):
        y = eval_function(expr_str, x, var=var)
        d = eval_derivative(expr_str, x, var=var, order=1)
        out.append((x, y, d))
        if abs(d) < 1e-9:
            break
        x = x - y / d
    return out


def solve_equation_roots(
    lhs_expr: str,
    rhs_expr: str = "0",
    x_min: float = -6.0,
    x_max: float = 6.0,
    var: str = "x",
) -> list[float]:
    """Find real roots of lhs_expr = rhs_expr on [x_min, x_max] via bracketing + SymPy nsolve."""
    sym = sp.Symbol(var, real=True)
    diff_sym = sp.simplify(ast_to_sympy(lhs_expr, var) - ast_to_sympy(rhs_expr, var))
    f_l = compile_expr(lhs_expr, (var,))
    f_r = compile_expr(rhs_expr, (var,))
    roots: list[float] = []
    n_scan = 200
    xs = [x_min + (x_max - x_min) * i / n_scan for i in range(n_scan + 1)]
    vals: list[float | None] = []
    for xv in xs:
        try:
            vals.append(float(f_l(**{var: xv})) - float(f_r(**{var: xv})))
        except Exception:
            vals.append(None)
    for i in range(n_scan):
        v0, v1 = vals[i], vals[i + 1]
        if v0 is None or v1 is None:
            continue
        if abs(v0) < 1e-8 or v0 * v1 <= 0:
            guess = 0.5 * (xs[i] + xs[i + 1])
            try:
                r = float(sp.re(sp.nsolve(diff_sym, sym, guess)))
                if x_min - 1e-4 <= r <= x_max + 1e-4 and all(abs(r - prev) > 1e-3 for prev in roots):
                    roots.append(round(r, 6))
            except Exception:
                pass
    return roots


def compose_expressions(outer_expr: str, inner_expr: str, var: str = "x") -> str:
    """Return simplified symbolic composition outer(inner(var))."""
    sym = sp.Symbol(var, real=True)
    outer = ast_to_sympy(outer_expr, var)
    inner = ast_to_sympy(inner_expr, var)
    return str(sp.simplify(outer.subs(sym, inner)))


def markov_chain_trajectory(
    transition_matrix: list[list[float]],
    initial_state: list[float],
    steps: int = 8,
) -> dict[str, object]:
    """Propagate state distribution p_{k+1} = p_k @ P for an arbitrary square row-stochastic Markov chain."""
    import numpy as np

    P = np.asarray(transition_matrix, dtype=float)
    p = np.asarray(initial_state, dtype=float)
    if P.ndim != 2 or P.shape[0] != P.shape[1] or p.shape != (P.shape[0],):
        raise ValueError("Invalid Markov dimensions.")
    row_sums = P.sum(axis=1)
    col_sums = P.sum(axis=0)
    if np.any(P < -1e-9) or np.any(np.abs(row_sums - 1.0) > 1e-3):
        hint = " (matrix appears column-stochastic; transpose rows/columns so EACH ROW sums to 1.0)" if (np.all(P >= -1e-9) and np.all(np.abs(col_sums - 1.0) <= 1e-3)) else ""
        raise ValueError(f"Transition matrix rows must be non-negative and sum to 1 (got row sums={[round(float(x), 4) for x in row_sums]}){hint}.")
    if np.any(p < -1e-9) or abs(float(p.sum()) - 1.0) > 1e-3:
        raise ValueError(f"Initial distribution must be non-negative and sum to 1 (got sum={float(p.sum()):.4f}).")
    p = p / p.sum()
    states: list[list[float]] = [[float(v) for v in p]]
    cur = p.copy()
    for _ in range(max(1, steps)):
        cur = cur @ P
        states.append([float(v) for v in cur])
    # Exact left eigenvector pi @ P = pi
    vals, vecs = np.linalg.eig(P.T)
    idx = int(np.argmin(np.abs(vals - 1.0)))
    pi_vec = np.abs(vecs[:, idx].real)
    pi_vec = pi_vec / (pi_vec.sum() or 1.0)
    return {"states": states, "stationary": [float(v) for v in pi_vec]}
