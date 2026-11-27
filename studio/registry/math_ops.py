"""Generalized mathematical and scientific computational operations with typed contracts."""

from __future__ import annotations

import math
from typing import Any, Literal
from pydantic import BaseModel, Field

from studio.ai import math_truth, world_model
from studio.registry.base import ComponentRegistry, ComponentSpec


# ---------------------------------------------------------------------------
# Typed input / output schema models for math operations
# ---------------------------------------------------------------------------
class ExprEvalInput(BaseModel):
    expr_str: str
    x_val: float
    var: str = "x"


class ExprEvalOutput(BaseModel):
    value: float


class IntegralInput(BaseModel):
    expr_str: str
    a: float
    b: float
    n: int = 12
    method: Literal["left", "right", "midpoint"] = "midpoint"


class IntegralOutput(BaseModel):
    integral_exact: float
    riemann_sum: float
    boxes: list[tuple[float, float, float]] = Field(default_factory=list)


class OptimizationInput(BaseModel):
    expr_str: str
    x0: float
    lr: float = 0.2
    steps: int = 5


class TrajectoryOutput(BaseModel):
    points: list[list[float]] = Field(default_factory=list)


class ODEInput(BaseModel):
    dx_expr: str
    dy_expr: str
    x0: float
    y0: float
    dt: float = 0.08
    steps: int = 40


class MatrixEigenInput(BaseModel):
    matrix_2x2: list[list[float]]


class MatrixEigenOutput(BaseModel):
    determinant: float
    trace: float
    regime: str
    real_eigenvalues: list[float] = Field(default_factory=list)


class ProbInput(BaseModel):
    prior: float
    p_e_given_h: float
    p_e_given_not_h: float


class ProbOutput(BaseModel):
    posterior: float
    delta: float


class FourierInput(BaseModel):
    expr_str: str
    period: float = 6.283185307179586
    n_harmonics: int = 5


class FourierOutput(BaseModel):
    a0: float
    harmonics: list[tuple[int, float, float]] = Field(default_factory=list)
    expr: str


class GeometryInput(BaseModel):
    a: float
    b: float
    origin: tuple[float, float] = (0.0, 0.0)


class GeometryOutput(BaseModel):
    vertices: list[list[float]] = Field(default_factory=list)


class SolveInput(BaseModel):
    lhs_expr: str
    rhs_expr: str = "0"
    x_min: float = -6.0
    x_max: float = 6.0


class SolveOutput(BaseModel):
    roots: list[float] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Validation & behavioral contract self-checks
# ---------------------------------------------------------------------------
def _val_expr(data: dict[str, Any]) -> list[str]:
    errs: list[str] = []
    if not str(data.get("expr_str") or data.get("expr") or "").strip():
        errs.append("expr_str is required")
    xv = data.get("x_val")
    if xv is not None and not math.isfinite(float(xv)):
        errs.append("x_val must be finite")
    return errs


def _val_integral(data: dict[str, Any]) -> list[str]:
    errs: list[str] = []
    if float(data.get("b", 0.0)) <= float(data.get("a", 0.0)):
        errs.append("integral requires b > a")
    if int(data.get("n", 1)) < 1:
        errs.append("n must be >= 1")
    return errs


def _val_gd(data: dict[str, Any]) -> list[str]:
    errs: list[str] = []
    if float(data.get("lr", 0.0)) <= 0.0:
        errs.append("learning rate lr must be > 0")
    if int(data.get("steps", 0)) < 1:
        errs.append("steps must be >= 1")
    return errs


def _val_ode(data: dict[str, Any]) -> list[str]:
    errs: list[str] = []
    if not str(data.get("dx_expr", "")).strip() or not str(data.get("dy_expr", "")).strip():
        errs.append("dx_expr and dy_expr required")
    if float(data.get("dt", 0.0)) <= 0.0 or int(data.get("steps", 0)) < 1:
        errs.append("dt > 0 and steps >= 1 required")
    return errs


def _val_eigen(data: dict[str, Any]) -> list[str]:
    mat = data.get("matrix_2x2")
    if not isinstance(mat, list) or len(mat) != 2 or any(not isinstance(r, list) or len(r) != 2 for r in mat):
        return ["matrix_2x2 must be 2x2 numeric matrix"]
    return []


def _val_prob(data: dict[str, Any]) -> list[str]:
    errs: list[str] = []
    for k in ("prior", "p_e_given_h", "p_e_given_not_h"):
        if k in data and not (0.0 <= float(data[k]) <= 1.0):
            errs.append(f"{k} must be in [0, 1]")
    return errs


def _val_fourier(data: dict[str, Any]) -> list[str]:
    errs: list[str] = []
    if float(data.get("period", 0.0)) <= 0.0:
        errs.append("period must be positive")
    if int(data.get("n_harmonics", 0)) < 1:
        errs.append("n_harmonics must be >= 1")
    return errs


def _val_geom(data: dict[str, Any]) -> list[str]:
    if float(data.get("a", 0.0)) <= 0.0 or float(data.get("b", 0.0)) <= 0.0:
        return ["triangle leg lengths a and b must be positive"]
    return []


def _val_solve(data: dict[str, Any]) -> list[str]:
    if float(data.get("x_max", 0.0)) <= float(data.get("x_min", 0.0)):
        return ["x_max must exceed x_min"]
    return []


def _test_symbolic_eval() -> bool:
    return abs(math_truth.eval_function("x**2", 3.0) - 9.0) < 1e-6 and bool(_val_expr({"expr_str": ""}))


def _test_integral() -> bool:
    res = math_truth.definite_integral("x", 0.0, 2.0, n=4, method="midpoint")
    return abs(float(res["integral_exact"]) - 2.0) < 1e-4 and bool(_val_integral({"a": 2.0, "b": 1.0}))


def _test_gd() -> bool:
    traj = math_truth.gradient_descent_trajectory("x**2", x0=2.0, lr=0.25, steps=3)
    return len(traj) == 4 and float(traj[-1][0]) < 1.0 and bool(_val_gd({"lr": -0.1, "steps": 0}))


def _test_ode() -> bool:
    pts = math_truth.rk4_ode_trajectory("-y", "x", 1.0, 0.0, dt=0.1, steps=5)
    return len(pts) == 6 and abs((pts[-1][1] ** 2 + pts[-1][2] ** 2) - 1.0) < 1e-3 and bool(_val_ode({"dx_expr": ""}))


def _test_eigen() -> bool:
    info = math_truth.matrix_transform_and_eigen([[-1.0, 0.0], [0.0, -2.0]])
    return abs(float(info["determinant"]) - 2.0) < 1e-6 and info["regime"] == "stable_node" and bool(_val_eigen({"matrix_2x2": [[1]]}))


def _test_prob() -> bool:
    post = math_truth.bayes_posterior(0.05, 0.9, 0.1)
    mk = math_truth.markov_chain_trajectory([[0.8, 0.2], [0.4, 0.6]], [1.0, 0.0], steps=12)
    return float(post["posterior"]) > 0.30 and abs(sum(mk["stationary"]) - 1.0) < 1e-5 and bool(_val_prob({"prior": 1.5}))  # type: ignore[arg-type]


def _test_fourier() -> bool:
    res = math_truth.fourier_partial_sum_terms("x", period=6.283185307179586, n_harmonics=3)
    return len(res["harmonics"]) == 3 and bool(_val_fourier({"period": -1.0, "n_harmonics": 0}))


def _test_geom() -> bool:
    tri = world_model.right_triangle(3.0, 4.0)
    return world_model.is_right_triangle(tri) and bool(_val_geom({"a": -1.0, "b": 2.0}))


def _test_solve() -> bool:
    roots = math_truth.solve_equation_roots("x**2 - 4", "0", x_min=-3.0, x_max=3.0)
    return len(roots) >= 2 and bool(_val_solve({"x_min": 2.0, "x_max": 1.0}))


MATH_OPERATION_SPECS: list[ComponentSpec] = [
    ComponentSpec(
        name="symbolic_eval_and_derivative",
        category="math_operation",
        description="Safely evaluate f(x), compute symbolic derivatives d^n/dx^n, stationary points, and Taylor expansions via SymPy.",
        input_model="ExprEvalInput",
        output_model="ExprEvalOutput",
        input_schema_model=ExprEvalInput,
        output_schema_model=ExprEvalOutput,
        inputs_schema=["expr_str: str", "x_val: float", "var: str"],
        outputs=["value: float"],
        render_implementations={"manimgx": "materialize_constructions+V3Scene", "hyperframes": "materialize_constructions+svg_math"},
        supported_ops=["create", "draw", "transform", "animate_param"],
        layout_requirements={"requires_axes": False},
        resource_constraints={"max_ast_nodes": 128},
        math_grounding="SymPy exact AST symbolic algebra + finite float check",
        example_use_case="Secant-to-tangent slope limits, curvature, and Taylor approximations of any smooth function.",
        compute_fn=math_truth.eval_function,
        validate_fn=_val_expr,
        test_fn=_test_symbolic_eval,
    ),
    ComponentSpec(
        name="definite_integral_and_riemann",
        category="math_operation",
        description="Compute definite integrals numerically/symbolically and partition interval [a,b] into Riemann rectangles.",
        input_model="IntegralInput",
        output_model="IntegralOutput",
        input_schema_model=IntegralInput,
        output_schema_model=IntegralOutput,
        inputs_schema=["expr_str: str", "a: float", "b: float", "n: int", "method: left|right|midpoint"],
        outputs=["integral_exact: float", "riemann_sum: float", "boxes: list[x,w,h]"],
        render_implementations={"manimgx": "world_model.materialize_constructions+polygon", "hyperframes": "world_model.materialize_constructions+svg"},
        supported_ops=["create", "draw", "sequence", "iterate_states"],
        layout_requirements={"requires_axes": True},
        resource_constraints={"max_rects": 64},
        math_grounding="SymPy integrate + deterministic Riemann sum accumulation",
        example_use_case="Visualizing area under curves and convergence as dx shrinks for arbitrary integrable f(x).",
        compute_fn=math_truth.definite_integral,
        validate_fn=_val_integral,
        test_fn=_test_integral,
    ),
    ComponentSpec(
        name="numerical_optimization_trajectory",
        category="math_operation",
        description="Run deterministic gradient descent, momentum, or Newton iteration on any differentiable scalar objective.",
        input_model="OptimizationInput",
        output_model="TrajectoryOutput",
        input_schema_model=OptimizationInput,
        output_schema_model=TrajectoryOutput,
        inputs_schema=["expr_str: str", "x0: float", "lr: float", "steps: int"],
        outputs=["points: list[(x_k, f(x_k), f'(x_k))]"],
        render_implementations={"manimgx": "world_model.materialize_constructions+data_series", "hyperframes": "world_model.materialize_constructions+svg"},
        supported_ops=["create", "trace", "iterate_states", "animate_param"],
        layout_requirements={"requires_axes": True},
        resource_constraints={"max_steps": 120},
        math_grounding="Exact symbolic gradient evaluated step-by-step via math_truth",
        example_use_case="Optimization on non-convex landscapes, learning-rate instability, and saddle points.",
        compute_fn=math_truth.gradient_descent_trajectory,
        validate_fn=_val_gd,
        test_fn=_test_gd,
    ),
    ComponentSpec(
        name="ode_rk4_trajectory",
        category="math_operation",
        description="Integrate general 1D or 2D autonomous ODE systems dx/dt=f(x,y,t), dy/dt=g(x,y,t) via 4th-order Runge-Kutta.",
        input_model="ODEInput",
        output_model="TrajectoryOutput",
        input_schema_model=ODEInput,
        output_schema_model=TrajectoryOutput,
        inputs_schema=["dx_expr: str", "dy_expr: str", "x0: float", "y0: float", "dt: float", "steps: int"],
        outputs=["points: list[(t, x, y)]"],
        render_implementations={"manimgx": "world_model.materialize_constructions+data_series", "hyperframes": "world_model.materialize_constructions+svg"},
        supported_ops=["create", "trace", "animate_param", "draw"],
        layout_requirements={"requires_axes": True},
        resource_constraints={"max_steps": 400},
        math_grounding="Deterministic RK4 numerical integrator over AST-verified vector fields",
        example_use_case="Predator-prey phase portraits, harmonic oscillators, eigenvalue stability spirals, and orbits.",
        compute_fn=math_truth.rk4_ode_trajectory,
        validate_fn=_val_ode,
        test_fn=_test_ode,
    ),
    ComponentSpec(
        name="linear_transform_eigen_2x2",
        category="math_operation",
        description="Apply any 2x2 matrix A to basis vectors/polygons, compute det(A), trace, and real/complex eigenpairs.",
        input_model="MatrixEigenInput",
        output_model="MatrixEigenOutput",
        input_schema_model=MatrixEigenInput,
        output_schema_model=MatrixEigenOutput,
        inputs_schema=["matrix_2x2: [[a,b],[c,d]]"],
        outputs=["determinant: float", "trace: float", "regime", "real_eigenvalues"],
        render_implementations={"manimgx": "world_model.materialize_constructions+matrix+vector", "hyperframes": "world_model.materialize_constructions+html_matrix"},
        supported_ops=["create", "transform", "morph", "scale", "rotate"],
        layout_requirements={"requires_axes": False},
        resource_constraints={"max_dim": 4},
        math_grounding="NumPy/SymPy exact determinant and eigendecomposition",
        example_use_case="Linear transformations, shear/rotation decomposition, PCA covariance, and Markov transition dynamics.",
        compute_fn=math_truth.matrix_transform_and_eigen,
        validate_fn=_val_eigen,
        test_fn=_test_eigen,
    ),
    ComponentSpec(
        name="discrete_probability_and_convolution",
        category="math_operation",
        description="Normalize any discrete PMF, compute exact mean/variance, n-fold convolution, Markov powers, and Bayes posterior odds.",
        input_model="ProbInput",
        output_model="ProbOutput",
        input_schema_model=ProbInput,
        output_schema_model=ProbOutput,
        inputs_schema=["prior: float", "p_e_given_h: float", "p_e_given_not_h: float"],
        outputs=["posterior: float", "delta: float"],
        render_implementations={"manimgx": "world_model.materialize_constructions+bar_chart", "hyperframes": "world_model.materialize_constructions+svg_bars"},
        supported_ops=["create", "transform", "iterate_states", "highlight"],
        layout_requirements={"requires_axes": False},
        resource_constraints={"max_states": 128},
        math_grounding="Exact discrete convolution, Markov row-stochastic power iteration, & Kolmogorov posterior",
        example_use_case="Central Limit Theorem convergence, Markov chains, Bayesian evidence updates.",
        compute_fn=math_truth.bayes_posterior,
        validate_fn=_val_prob,
        test_fn=_test_prob,
    ),
    ComponentSpec(
        name="fourier_partial_sum_synthesis",
        category="math_operation",
        description="Compute Fourier cosine/sine coefficients (a_n, b_n) numerically for any periodic function and build N-term waveforms.",
        input_model="FourierInput",
        output_model="FourierOutput",
        input_schema_model=FourierInput,
        output_schema_model=FourierOutput,
        inputs_schema=["expr_str: str", "period: float", "n_harmonics: int"],
        outputs=["a0: float", "harmonics: list[(n, a_n, b_n)]", "expr: str"],
        render_implementations={"manimgx": "world_model.materialize_constructions+function_plot", "hyperframes": "world_model.materialize_constructions+svg_math"},
        supported_ops=["create", "draw", "transform", "iterate_states"],
        layout_requirements={"requires_axes": True},
        resource_constraints={"max_harmonics": 32},
        math_grounding="Deterministic trapezoidal inner-product projection onto orthogonal trigonometric basis",
        example_use_case="Decomposing square waves, sawtooths, or custom signals into harmonics.",
        compute_fn=math_truth.fourier_partial_sum_terms,
        validate_fn=_val_fourier,
        test_fn=_test_fourier,
    ),
    ComponentSpec(
        name="euclidean_constraint_geometry",
        category="math_operation",
        description="Construct right triangles, squares on legs/hypotenuse, circumcircles, projections, and non-overlapping dissections.",
        input_model="GeometryInput",
        output_model="GeometryOutput",
        input_schema_model=GeometryInput,
        output_schema_model=GeometryOutput,
        inputs_schema=["a: float", "b: float", "origin: (x, y)"],
        outputs=["vertices: list[[x, y]]"],
        render_implementations={"manimgx": "world_model.materialize_constructions+polygon", "hyperframes": "world_model.materialize_constructions+svg_poly"},
        supported_ops=["create", "duplicate", "move", "rotate", "group"],
        layout_requirements={"equal_aspect": True},
        resource_constraints={"max_polygons": 32},
        math_grounding="Deterministic Euclidean invariants, orientation check, and polygon area conservation",
        example_use_case="Pythagorean tilings, Thales theorem, vector projections, and geometric inequalities.",
        compute_fn=world_model.right_triangle,
        validate_fn=_val_geom,
        test_fn=_test_geom,
    ),
    ComponentSpec(
        name="equation_solve_and_compose",
        category="math_operation",
        description="Solve algebraic/transcendental equations over intervals and evaluate multi-step function compositions g(f(x)).",
        input_model="SolveInput",
        output_model="SolveOutput",
        input_schema_model=SolveInput,
        output_schema_model=SolveOutput,
        inputs_schema=["lhs_expr: str", "rhs_expr: str", "x_min: float", "x_max: float"],
        outputs=["roots: list[float]"],
        render_implementations={"manimgx": "world_model.materialize_constructions+point", "hyperframes": "world_model.materialize_constructions+svg_dot"},
        supported_ops=["create", "highlight", "focus"],
        layout_requirements={"requires_axes": False},
        resource_constraints={"scan_points": 200},
        math_grounding="SymPy nsolve/solve bracketed root search and symbolic substitution",
        example_use_case="Fixed points f(x)=x, equilibrium intersections, and chain-rule compositions.",
        compute_fn=math_truth.solve_equation_roots,
        validate_fn=_val_solve,
        test_fn=_test_solve,
    ),
]


def register_math_ops(registry: ComponentRegistry) -> None:
    registry.register_many(MATH_OPERATION_SPECS)
