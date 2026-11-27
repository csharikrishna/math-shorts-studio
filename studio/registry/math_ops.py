"""Generalized mathematical and scientific computational operations."""

from __future__ import annotations

from studio.registry.base import ComponentRegistry, ComponentSpec


MATH_OPERATION_SPECS: list[ComponentSpec] = [
    ComponentSpec(
        name="symbolic_eval_and_derivative",
        category="math_operation",
        description="Safely evaluate f(x), compute symbolic derivatives d^n/dx^n, stationary points, and Taylor expansions via SymPy.",
        inputs_schema=["expr: str", "x_val: float", "order: int", "var: str"],
        outputs=["y: float", "derivative_expr: str", "taylor_coeffs: list[float]"],
        math_grounding="SymPy exact AST symbolic algebra + finite float check",
        example_use_case="Secant-to-tangent slope limits, curvature, and Taylor approximations of any smooth function.",
    ),
    ComponentSpec(
        name="definite_integral_and_riemann",
        category="math_operation",
        description="Compute definite integrals numerically/symbolically and partition interval [a,b] into Riemann rectangles.",
        inputs_schema=["expr: str", "a: float", "b: float", "n: int", "method: left|right|midpoint"],
        outputs=["integral_exact: float", "riemann_sum: float", "rect_boxes: list[x,w,h]"],
        math_grounding="SymPy integrate + deterministic Riemann sum accumulation",
        example_use_case="Visualizing area under curves and convergence as dx shrinks for arbitrary integrable f(x).",
    ),
    ComponentSpec(
        name="numerical_optimization_trajectory",
        category="math_operation",
        description="Run deterministic gradient descent, momentum, or Newton iteration on any differentiable scalar objective.",
        inputs_schema=["expr: str", "x0: float", "lr: float", "steps: int", "method: gd|newton"],
        outputs=["trajectory: list[(x_k, f(x_k), f'(x_k))]", "stationary_check: bool"],
        math_grounding="Exact symbolic gradient evaluated step-by-step via math_truth",
        example_use_case="Optimization on non-convex landscapes, learning-rate instability, and saddle points.",
    ),
    ComponentSpec(
        name="ode_rk4_trajectory",
        category="math_operation",
        description="Integrate general 1D or 2D autonomous ODE systems dx/dt=f(x,y,t), dy/dt=g(x,y,t) via 4th-order Runge-Kutta.",
        inputs_schema=["dx_expr: str", "dy_expr: str", "x0: float", "y0: float", "dt: float", "steps: int"],
        outputs=["states: list[(t, x, y)]", "vector_field_samples"],
        math_grounding="Deterministic RK4 numerical integrator over AST-verified vector fields",
        example_use_case="Predator-prey phase portraits, harmonic oscillators, eigenvalue stability spirals, and orbits.",
    ),
    ComponentSpec(
        name="linear_transform_eigen_2x2",
        category="math_operation",
        description="Apply any 2x2 matrix A to basis vectors/polygons, compute det(A), trace, and real/complex eigenpairs.",
        inputs_schema=["matrix_2x2: [[a,b],[c,d]]", "vectors_or_vertices"],
        outputs=["transformed_coords", "determinant: float", "eigenvalues", "eigenvectors"],
        math_grounding="NumPy/SymPy exact determinant and eigendecomposition",
        example_use_case="Linear transformations, shear/rotation decomposition, PCA covariance, and Markov transition dynamics.",
    ),
    ComponentSpec(
        name="discrete_probability_and_convolution",
        category="math_operation",
        description="Normalize any discrete PMF, compute exact mean/variance, n-fold convolution, and Bayes posterior odds.",
        inputs_schema=["support: list[float]", "probs: list[float]", "n_trials: int", "prior: float", "likelihoods"],
        outputs=["mean: float", "std: float", "convolved_pmf", "posterior: float"],
        math_grounding="Exact discrete convolution & Kolmogorov posterior normalization",
        example_use_case="Central Limit Theorem convergence, weighted dice, Monte Carlo frequencies, Bayesian evidence updates.",
    ),
    ComponentSpec(
        name="fourier_partial_sum_synthesis",
        category="math_operation",
        description="Compute Fourier cosine/sine coefficients (a_n, b_n) numerically for any periodic function and build N-term waveforms.",
        inputs_schema=["expr: str", "period: float", "n_harmonics: int", "samples: int"],
        outputs=["a0: float", "harmonics: list[(n, a_n, b_n)]", "partial_sum_curve"],
        math_grounding="Deterministic trapezoidal inner-product projection onto orthogonal trigonometric basis",
        example_use_case="Decomposing square waves, sawtooths, or custom signals into rotating epicycles/harmonics.",
    ),
    ComponentSpec(
        name="euclidean_constraint_geometry",
        category="math_operation",
        description="Construct right triangles, squares on legs/hypotenuse, circumcircles, projections, and non-overlapping dissections.",
        inputs_schema=["op: ConstructionKind", "params: ConstructionParams"],
        outputs=["exact_vertices", "verified_side_lengths", "area_conservation_proof"],
        math_grounding="Deterministic Euclidean invariants, orientation check, and polygon area conservation",
        example_use_case="Pythagorean tilings, Thales theorem, vector projections, and geometric inequalities.",
    ),
    ComponentSpec(
        name="equation_solve_and_compose",
        category="math_operation",
        description="Solve algebraic/transcendental equations over intervals and evaluate multi-step function compositions g(f(x)).",
        inputs_schema=["f_expr: str", "g_expr: str", "rhs_expr: str", "x_range: [a,b]"],
        outputs=["roots: list[float]", "composed_expr: str"],
        math_grounding="SymPy nsolve/solve bracketed root search and symbolic substitution",
        example_use_case="Fixed points f(x)=x, equilibrium intersections, and chain-rule compositions.",
    ),
]


def register_math_ops(registry: ComponentRegistry) -> None:
    registry.register_many(MATH_OPERATION_SPECS)
