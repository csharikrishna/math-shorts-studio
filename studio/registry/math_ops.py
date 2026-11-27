"""Generalized mathematical and scientific computational operations."""

from __future__ import annotations

from studio.ai import math_truth, world_model
from studio.registry.base import ComponentRegistry, ComponentSpec


def _test_symbolic_eval() -> bool:
    return abs(math_truth.eval_function("x**2", 3.0) - 9.0) < 1e-6


def _test_integral() -> bool:
    res = math_truth.definite_integral("x", 0.0, 2.0, n=4, method="midpoint")
    return abs(float(res["integral_exact"]) - 2.0) < 1e-4


def _test_gd() -> bool:
    traj = math_truth.gradient_descent_trajectory("x**2", x0=2.0, lr=0.25, steps=3)
    return len(traj) == 4 and float(traj[-1][0]) < 1.0


def _test_ode() -> bool:
    pts = math_truth.rk4_ode_trajectory("-y", "x", 1.0, 0.0, dt=0.1, steps=5)
    return len(pts) == 6 and abs((pts[-1][1] ** 2 + pts[-1][2] ** 2) - 1.0) < 1e-3


def _test_eigen() -> bool:
    info = math_truth.matrix_transform_and_eigen([[-1.0, 0.0], [0.0, -2.0]])
    return abs(float(info["determinant"]) - 2.0) < 1e-6 and info["regime"] == "stable_node"


def _test_prob() -> bool:
    post = math_truth.bayes_posterior(0.05, 0.9, 0.1)
    return float(post["posterior"]) > 0.30


def _test_fourier() -> bool:
    res = math_truth.fourier_partial_sum_terms("x", period=6.283185307179586, n_harmonics=3)
    return len(res["harmonics"]) == 3


def _test_geom() -> bool:
    tri = world_model.right_triangle(3.0, 4.0)
    return world_model.is_right_triangle(tri)


def _test_solve() -> bool:
    roots = math_truth.solve_equation_roots("x**2 - 4", "0", x_min=-3.0, x_max=3.0)
    return len(roots) >= 2


MATH_OPERATION_SPECS: list[ComponentSpec] = [
    ComponentSpec(
        name="symbolic_eval_and_derivative",
        category="math_operation",
        description="Safely evaluate f(x), compute symbolic derivatives d^n/dx^n, stationary points, and Taylor expansions via SymPy.",
        input_model="ExprEvalRequest(expr, x_val, order, var)",
        output_model="dict(y, derivative_expr, taylor_coeffs)",
        inputs_schema=["expr: str", "x_val: float", "order: int", "var: str"],
        outputs=["y: float", "derivative_expr: str", "taylor_coeffs: list[float]"],
        supported_ops=["create", "draw", "transform", "animate_param"],
        math_grounding="SymPy exact AST symbolic algebra + finite float check",
        example_use_case="Secant-to-tangent slope limits, curvature, and Taylor approximations of any smooth function.",
        compute_fn=math_truth.eval_function,
        test_fn=_test_symbolic_eval,
    ),
    ComponentSpec(
        name="definite_integral_and_riemann",
        category="math_operation",
        description="Compute definite integrals numerically/symbolically and partition interval [a,b] into Riemann rectangles.",
        input_model="RiemannRequest(expr, a, b, n_rects, method)",
        output_model="dict(exact, riemann_sum, rects)",
        inputs_schema=["expr: str", "a: float", "b: float", "n: int", "method: left|right|midpoint"],
        outputs=["integral_exact: float", "riemann_sum: float", "rect_boxes: list[x,w,h]"],
        supported_ops=["create", "draw", "sequence", "iterate_states"],
        math_grounding="SymPy integrate + deterministic Riemann sum accumulation",
        example_use_case="Visualizing area under curves and convergence as dx shrinks for arbitrary integrable f(x).",
        compute_fn=math_truth.definite_integral,
        test_fn=_test_integral,
    ),
    ComponentSpec(
        name="numerical_optimization_trajectory",
        category="math_operation",
        description="Run deterministic gradient descent, momentum, or Newton iteration on any differentiable scalar objective.",
        input_model="OptimizationRequest(expr, x0, lr, steps)",
        output_model="dict(points, trace)",
        inputs_schema=["expr: str", "x0: float", "lr: float", "steps: int", "method: gd|newton"],
        outputs=["trajectory: list[(x_k, f(x_k), f'(x_k))]", "stationary_check: bool"],
        supported_ops=["create", "trace", "iterate_states", "animate_param"],
        math_grounding="Exact symbolic gradient evaluated step-by-step via math_truth",
        example_use_case="Optimization on non-convex landscapes, learning-rate instability, and saddle points.",
        compute_fn=math_truth.gradient_descent_trajectory,
        test_fn=_test_gd,
    ),
    ComponentSpec(
        name="ode_rk4_trajectory",
        category="math_operation",
        description="Integrate general 1D or 2D autonomous ODE systems dx/dt=f(x,y,t), dy/dt=g(x,y,t) via 4th-order Runge-Kutta.",
        input_model="ODERequest(dx_expr, dy_expr, x0, y0, dt, steps)",
        output_model="list[[x, y]]",
        inputs_schema=["dx_expr: str", "dy_expr: str", "x0: float", "y0: float", "dt: float", "steps: int"],
        outputs=["states: list[(t, x, y)]", "vector_field_samples"],
        supported_ops=["create", "trace", "animate_param", "draw"],
        math_grounding="Deterministic RK4 numerical integrator over AST-verified vector fields",
        example_use_case="Predator-prey phase portraits, harmonic oscillators, eigenvalue stability spirals, and orbits.",
        compute_fn=math_truth.rk4_ode_trajectory,
        test_fn=_test_ode,
    ),
    ComponentSpec(
        name="linear_transform_eigen_2x2",
        category="math_operation",
        description="Apply any 2x2 matrix A to basis vectors/polygons, compute det(A), trace, and real/complex eigenpairs.",
        input_model="Matrix2x2Request(matrix_2x2)",
        output_model="dict(det, trace, eigenvalues, eigenvectors, classification)",
        inputs_schema=["matrix_2x2: [[a,b],[c,d]]", "vectors_or_vertices"],
        outputs=["transformed_coords", "determinant: float", "eigenvalues", "eigenvectors"],
        supported_ops=["create", "transform", "morph", "scale", "rotate"],
        math_grounding="NumPy/SymPy exact determinant and eigendecomposition",
        example_use_case="Linear transformations, shear/rotation decomposition, PCA covariance, and Markov transition dynamics.",
        compute_fn=math_truth.matrix_transform_and_eigen,
        test_fn=_test_eigen,
    ),
    ComponentSpec(
        name="discrete_probability_and_convolution",
        category="math_operation",
        description="Normalize any discrete PMF, compute exact mean/variance, n-fold convolution, and Bayes posterior odds.",
        input_model="ProbRequest(prior, p_e_given_h, p_e_given_not_h)",
        output_model="dict(posterior, marginal_evidence, joint)",
        inputs_schema=["support: list[float]", "probs: list[float]", "n_trials: int", "prior: float", "likelihoods"],
        outputs=["mean: float", "std: float", "convolved_pmf", "posterior: float"],
        supported_ops=["create", "transform", "iterate_states", "highlight"],
        math_grounding="Exact discrete convolution & Kolmogorov posterior normalization",
        example_use_case="Central Limit Theorem convergence, weighted dice, Monte Carlo frequencies, Bayesian evidence updates.",
        compute_fn=math_truth.bayes_posterior,
        test_fn=_test_prob,
    ),
    ComponentSpec(
        name="fourier_partial_sum_synthesis",
        category="math_operation",
        description="Compute Fourier cosine/sine coefficients (a_n, b_n) numerically for any periodic function and build N-term waveforms.",
        input_model="FourierRequest(expr, period, n_terms)",
        output_model="dict(a0, terms, partial_expr)",
        inputs_schema=["expr: str", "period: float", "n_harmonics: int", "samples: int"],
        outputs=["a0: float", "harmonics: list[(n, a_n, b_n)]", "partial_sum_curve"],
        supported_ops=["create", "draw", "transform", "iterate_states"],
        math_grounding="Deterministic trapezoidal inner-product projection onto orthogonal trigonometric basis",
        example_use_case="Decomposing square waves, sawtooths, or custom signals into rotating epicycles/harmonics.",
        compute_fn=math_truth.fourier_partial_sum_terms,
        test_fn=_test_fourier,
    ),
    ComponentSpec(
        name="euclidean_constraint_geometry",
        category="math_operation",
        description="Construct right triangles, squares on legs/hypotenuse, circumcircles, projections, and non-overlapping dissections.",
        input_model="ConstructionParams(a, b, origin)",
        output_model="list[[x, y]]",
        inputs_schema=["op: ConstructionKind", "params: ConstructionParams"],
        outputs=["exact_vertices", "verified_side_lengths", "area_conservation_proof"],
        supported_ops=["create", "duplicate", "move", "rotate", "group"],
        math_grounding="Deterministic Euclidean invariants, orientation check, and polygon area conservation",
        example_use_case="Pythagorean tilings, Thales theorem, vector projections, and geometric inequalities.",
        compute_fn=world_model.right_triangle,
        test_fn=_test_geom,
    ),
    ComponentSpec(
        name="equation_solve_and_compose",
        category="math_operation",
        description="Solve algebraic/transcendental equations over intervals and evaluate multi-step function compositions g(f(x)).",
        input_model="SolveRequest(expr, rhs_expr, x_range)",
        output_model="list[float]",
        inputs_schema=["f_expr: str", "g_expr: str", "rhs_expr: str", "x_range: [a,b]"],
        outputs=["roots: list[float]", "composed_expr: str"],
        supported_ops=["create", "highlight", "focus"],
        math_grounding="SymPy nsolve/solve bracketed root search and symbolic substitution",
        example_use_case="Fixed points f(x)=x, equilibrium intersections, and chain-rule compositions.",
        compute_fn=math_truth.solve_equation_roots,
        test_fn=_test_solve,
    ),
]


def register_math_ops(registry: ComponentRegistry) -> None:
    registry.register_many(MATH_OPERATION_SPECS)
