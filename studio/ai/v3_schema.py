"""Composable Object-and-Event Scene Language (VideoPlan V3)."""

from __future__ import annotations

from typing import Annotated, Any, Literal, Optional
from pydantic import BaseModel, Field, field_validator
from pydantic.json_schema import SkipJsonSchema

from studio.ai.linked_models import LinkedMathModelSpec
from studio.ai.schema import ColorRole, SfxCue

ObjectKind = Literal[
    "axes",
    "function_plot",
    "parametric_curve",
    "bar_chart",
    "equation",
    "point",
    "line",
    "vector",
    "polygon",
    "circle",
    "number_line",
    "annotation",
    "angle_arc",
    "projection",
    "matrix",
    "table",
    "data_series",
    "graph_network",
    "probability_region",
    "particle_system",
    "waveform",
    "mechanical_body",
    "vector_field",
]

EventOp = Literal[
    "create",
    "draw",
    "transform",
    "highlight",
    "move",
    "scale",
    "rotate",
    "remove",
    "focus",
    "animate_param",
    "set_param",
    "reveal",
    "morph",
    "duplicate",
    "group",
    "zoom",
    "trace",
    "iterate_states",
    "interpolate_values",
    "sequence",
    "parallel",
]

CompositionKind = Literal[
    "auto",
    "centered_focus",
    "diagram_with_text",
    "side_by_side_comparison",
    "stacked_comparison",
    "main_with_equation_panel",
    "geometric_proof",
    "math_simulation",
    "annotated_chart",
    "matrix_transform",
    "data_visualization",
    "chapter_intro",
    "concept_summary",
    "step_by_step_sequence",
    "grid_of_multiple_examples",
    "spotlight_zoom",
    "annotated_overlay",
    "timeline_progression",
    "time_frequency_split",
    "model_with_readout",
    "before_after_simulation",
    "multi_view_dashboard",
    "progressive_derivation",
    "physical_system_with_displacement",
    "physical_system_with_phase_space",
    "dual_time_series",
    "parameter_sweep_with_response",
    "vector_field_with_trajectory",
    "state_with_conserved_chart",
]


class SweepKeyframe(BaseModel):
    """Synchronized keyframe for animated parameter sweeps and linked traces."""

    param: float
    point: list[float] = Field(default_factory=list)
    vector_start: list[float] = Field(default_factory=list)
    vector_end: list[float] = Field(default_factory=list)
    spec_point: list[float] = Field(default_factory=list)
    body_pos: list[float] = Field(default_factory=list)
    curve_pts: list[list[float]] = Field(default_factory=list)
    bar_values: list[float] = Field(default_factory=list)
    label: str = ""
    u: Optional[float] = None


class SceneObject(BaseModel):
    id: str = Field(description="Unique snake_case object identifier.")
    kind: ObjectKind
    persist: bool = True
    color_role: ColorRole = "primary"
    # Shared linked computational model binding (optional)
    model_id: Optional[str] = None
    model_output: Optional[str] = None
    # Math / geometry fields (kind-dependent)
    expr: Optional[str] = None
    t_range: Optional[list[float]] = None
    x_range: Optional[list[float]] = None
    y_range: Optional[list[float]] = None
    tex: Optional[str] = None
    text: Optional[str] = None
    values: list[float] = Field(default_factory=list)
    bar_labels: list[str] = Field(default_factory=list)
    log_scale: bool = False
    coords: Optional[list[float]] = None
    end_coords: Optional[list[float]] = None
    vertices: list[list[float]] = Field(default_factory=list)
    radius: Optional[float] = None
    axes_id: Optional[str] = None
    equal_aspect: bool = False
    param_name: Optional[str] = None
    param_init: float = 0.0
    label: str = ""
    x_label: str = ""
    y_label: str = ""
    legend_items: list[list[str]] = Field(default_factory=list)
    sweep_keyframes: Annotated[list[SweepKeyframe], SkipJsonSchema()] = Field(default_factory=list)
    card: bool = False
    matrix_rows: list[list[str]] = Field(default_factory=list)
    table_headers: list[str] = Field(default_factory=list)
    table_rows: list[list[str]] = Field(default_factory=list)
    points_list: list[list[float]] = Field(default_factory=list)
    edge_pairs: list[list[str]] = Field(default_factory=list)
    node_labels: list[str] = Field(default_factory=list)
    # Parameterized mechanical / field component extensions
    connector_kind: Literal["none", "spring", "rod", "damper", "spring_damper"] = "none"
    anchor_coords: Optional[list[float]] = None
    dx_expr: Optional[str] = None
    dy_expr: Optional[str] = None
    var_names: list[str] = Field(default_factory=list)
    body_mass: Optional[float] = None
    damping_coeff: Optional[float] = None
    panel_slot: Literal["auto", "left", "right", "top", "bottom", "center"] = "auto"


ConstructionKind = Literal[
    "right_triangle",
    "square_on_segment",
    "pythagorean_dissection",
    "translate_polygon",
    "rotate_polygon",
    "gradient_descent_path",
    "bayes_area_model",
    "dice_sum_distribution",
    "gaussian_approx",
    "taylor_approximation",
    "fourier_partial_sum",
    "ode_phase_trajectory",
    "linear_transform_2x2",
    "eigen_decomposition_2x2",
    "riemann_integral",
    "discrete_pmf_convolution",
    "fourier_general_synthesis",
    "newton_root_path",
    "markov_chain_evolution",
    "winding_complex_curve",
    "spectral_transform_from_signal",
    "filtered_signal_reconstruction",
]


class ConstructionParams(BaseModel):
    """Typed parameters for deterministic mathematical construction operations."""

    a: Optional[float] = None
    b: Optional[float] = None
    origin: Optional[list[float]] = None
    edge_index: Optional[int] = None
    edge_role: Optional[Literal["leg_a", "leg_b", "hypotenuse"]] = None
    outward: Optional[bool] = None
    dx: Optional[float] = None
    dy: Optional[float] = None
    angle_deg: Optional[float] = None
    center: Optional[list[float]] = None
    p1: Optional[list[float]] = None
    p2: Optional[list[float]] = None
    # Computational & statistical construction params
    expr: Optional[str] = None
    axes_id: Optional[str] = None
    x0: Optional[float] = None
    lr: Optional[float] = None
    steps: Optional[int] = None
    degree: Optional[int] = None
    n_terms: Optional[int] = None
    prior: Optional[float] = None
    p_e_given_h: Optional[float] = None
    p_e_given_not_h: Optional[float] = None
    n_dice: Optional[int] = None
    mu: Optional[float] = None
    sigma: Optional[float] = None
    # General ODE, Linear Algebra, Integral, Arbitrary PMF, Markov params
    dx_expr: Optional[str] = None
    dy_expr: Optional[str] = None
    y0: Optional[float] = None
    dt: Optional[float] = None
    matrix_2x2: Optional[list[list[float]]] = None
    vertices_list: Optional[list[list[float]]] = None
    a_bound: Optional[float] = None
    b_bound: Optional[float] = None
    n_rects: Optional[int] = None
    riemann_method: Optional[Literal["left", "right", "midpoint"]] = None
    support_vals: Optional[list[float]] = None
    prob_vals: Optional[list[float]] = None
    n_trials: Optional[int] = None
    period: Optional[float] = None
    transition_matrix: Optional[list[list[float]]] = None
    initial_state: Optional[list[float]] = None
    # Linked signal & spectral parameters
    selected_freq: Optional[float] = None
    cutoff_freq: Optional[float] = None
    high_cutoff_freq: Optional[float] = None
    filter_kind: Optional[Literal["lowpass", "bandpass", "highpass"]] = None
    freq_range: Optional[list[float]] = None
    t_range: Optional[list[float]] = None
    angular_units: Optional[Literal["hz", "rad"]] = None


class ConstructionOp(BaseModel):
    """Typed deterministic mathematical construction operation."""

    op: ConstructionKind
    output_ids: list[str] = Field(default_factory=list)
    source_id: Optional[str] = None
    params: ConstructionParams = Field(default_factory=ConstructionParams)
    color_role: ColorRole = "primary"
    persist: bool = True
    label: str = ""


ClaimRuleKind = Literal[
    "right_triangle",
    "square",
    "congruent_triangles",
    "area_conservation",
    "area_equality",
    "partition_of",
    "nonzero_area",
    "point_on_function",
    "stationary_point",
    "local_minimum",
    "derivative_equality",
    "function_equivalence",
    "polynomial_degree",
    "coefficient_correctness",
    "probability_normalization",
    "conditional_probability",
    "distribution_stats",
    "convergence_assumptions",
    "definite_integral_value",
    "eigenvalue_determinant",
    "eigenvalue_stability_regime",
    "ode_equilibrium",
    "markov_stochastic_invariant",
    "fourier_winding_centroid",
    "spectral_derivation_consistency",
    "signal_filtering_equivalence",
    "parseval_energy_conservation",
    "simulation_trajectory_bounds",
    "frequency_response_peak",
    "state_bounds",
    "positivity",
    "conservation_law",
    "energy_dissipation",
    "periodicity",
    "numerical_accuracy",
    "parameter_validity",
    "physical_units",
]


class ClaimLedgerEntry(BaseModel):
    """Claim ledger tracking exact spoken text, formal claim, assumptions, and verification status."""

    claim_id: str
    exact_narration_span: str
    proposition: str
    assumptions: str = ""
    model_id: Optional[str] = None
    object_ids: list[str] = Field(default_factory=list)
    status: Literal[
        "deterministically_verified",
        "numerically_tested",
        "source_supported",
        "ai_reviewed",
        "unverified",
        "failed",
    ] = "deterministically_verified"
    evidence: str = ""


class AudioDemoSpec(BaseModel):
    """Optional synthesized educational sound demonstration tied to a LinkedMathModelSpec."""

    model_id: str
    variant: Literal["raw", "filtered", "raw_vs_filtered"] = "raw_vs_filtered"
    base_pitch_hz: float = 220.0
    duration_s: float = 2.4
    anchor_mark: Optional[str] = None


class StructuredClaim(BaseModel):
    """Explicit mathematical constraint verified against live objects and symbolic/numeric truth."""

    kind: ClaimRuleKind
    model_id: Optional[str] = None
    object_ids: list[str] = Field(default_factory=list)
    container_id: Optional[str] = None
    statement: str = ""
    expr: Optional[str] = None
    expected_expr: Optional[str] = None
    x_val: Optional[float] = None
    expected_value: Optional[float] = None
    expected_degree: Optional[int] = None
    prior: Optional[float] = None
    likelihood: Optional[float] = None
    false_positive: Optional[float] = None
    mu: Optional[float] = None
    sigma: Optional[float] = None
    a_bound: Optional[float] = None
    b_bound: Optional[float] = None
    dx_expr: Optional[str] = None
    dy_expr: Optional[str] = None
    matrix_2x2: Optional[list[list[float]]] = None
    expected_regime: Optional[
        Literal["stable_node", "unstable_node", "saddle", "stable_spiral", "unstable_spiral", "center", "degenerate"]
    ] = None
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None
    state_var: Optional[str] = None
    units: dict[str, str] = Field(default_factory=dict)
    assumptions: str = ""
    verification_level: Literal[
        "deterministically_verified",
        "numerically_tested",
        "source_supported",
        "ai_reviewed",
    ] = "deterministically_verified"
    tolerance: float = 1e-3


class AnimationEvent(BaseModel):
    op: EventOp
    target_id: str
    source_id: Optional[str] = None
    anchor_mark: Optional[str] = None
    shift: Optional[list[float]] = None
    scale_factor: Optional[float] = None
    angle_deg: Optional[float] = None
    param_range: Optional[list[float]] = None
    param_values: list[float] = Field(default_factory=list)
    target_param: Optional[float] = None
    duration_s: Optional[float] = None


class SceneV3(BaseModel):
    scene_id: str
    purpose: str
    narration: str
    speaker: str = "narrator"
    composition: CompositionKind = "auto"
    color_role: ColorRole = "primary"
    sfx_cue: SfxCue = "pop_marimba"

    @field_validator("sfx_cue", mode="before")
    @classmethod
    def _coerce_sfx_cue(cls, v: Any) -> str:
        s = str(v or "pop_marimba").strip().lower()
        if s in ("pop_marimba", "whoosh_soft", "tick_soft", "chime_reveal", "chord_epiphany"):
            return s
        if "chime" in s or "success" in s:
            return "chime_reveal"
        if "whoosh" in s or "sweep" in s or "transform" in s:
            return "whoosh_soft"
        if "tick" in s:
            return "tick_soft"
        if "chord" in s or "epiphany" in s:
            return "chord_epiphany"
        return "pop_marimba"
    constructions: list[ConstructionOp] = Field(default_factory=list)
    claims: list[StructuredClaim] = Field(default_factory=list)
    claim_ledger: list[ClaimLedgerEntry] = Field(default_factory=list)
    demo_audio: Optional[AudioDemoSpec] = None
    active_model_params: Annotated[dict[str, float], SkipJsonSchema()] = Field(default_factory=dict)
    objects: list[SceneObject] = Field(default_factory=list)
    events: list[AnimationEvent] = Field(default_factory=list)


class VideoPlanV3(BaseModel):
    schema_version: str = "3.0"
    topic: str = "Mathematical Visualization"
    title: str = ""
    audience: str = "general"
    strategy: str = ""
    target_duration_s: float = 55.0
    math_models: list[LinkedMathModelSpec] = Field(default_factory=list)
    scenes: list[SceneV3]
