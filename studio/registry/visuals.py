"""Reusable visual component catalog with typed contracts across ManimGX & HyperFrames."""

from __future__ import annotations

import math
from typing import Any
from pydantic import BaseModel, Field

from studio.registry.base import ComponentRegistry, ComponentSpec


class VisualObjectInput(BaseModel):
    id: str
    kind: str
    coords: list[float] | None = None
    vertices: list[list[float]] = Field(default_factory=list)
    values: list[float] = Field(default_factory=list)


class VisualObjectOutput(BaseModel):
    rendered_id: str
    bounds_px: tuple[float, float, float, float] | None = None


def _finite_seq(seq: Any) -> bool:
    return isinstance(seq, (list, tuple)) and all(math.isfinite(float(x)) for x in seq)


def _val_visual(data: dict[str, Any]) -> list[str]:
    errs: list[str] = []
    kind = str(data.get("kind", "")).strip()
    oid = str(data.get("id", "")).strip()
    if not oid:
        errs.append("visual object id required")
    if kind == "polygon":
        verts = data.get("vertices") or []
        if len(verts) < 3:
            errs.append("polygon requires >= 3 vertices")
    elif kind == "bar_chart":
        vals = data.get("values") or []
        if not vals or not _finite_seq(vals):
            errs.append("bar_chart requires non-empty finite values")
    elif kind == "matrix":
        rows = data.get("matrix_rows") or data.get("rows") or []
        if not rows:
            errs.append("matrix requires non-empty rows")
    return errs


def _test_visual_kind(kind: str) -> bool:
    if kind == "polygon":
        return len(_val_visual({"id": "p", "kind": "polygon", "vertices": [[0, 0], [1, 0], [0, 1]]})) == 0 and bool(_val_visual({"id": "p", "kind": "polygon", "vertices": [[0, 0]]}))
    if kind == "bar_chart":
        return len(_val_visual({"id": "b", "kind": "bar_chart", "values": [0.5, 0.5]})) == 0 and bool(_val_visual({"id": "b", "kind": "bar_chart", "values": []}))
    if kind == "matrix":
        return len(_val_visual({"id": "m", "kind": "matrix", "matrix_rows": [["1", "0"]]})) == 0 and bool(_val_visual({"id": "m", "kind": "matrix", "matrix_rows": []}))
    return len(_val_visual({"id": f"ok_{kind}", "kind": kind})) == 0 and bool(_val_visual({"id": "", "kind": kind}))


_DEFS: list[tuple[str, str, list[str], str, str, dict[str, Any], list[str]]] = [
    ("axes", "2D Cartesian or phase-plane coordinate grid with deterministic scale mapping.", ["object_id", "x_range", "y_range"], "Uniform linear mapping", "Calculus graphs, ODE phase portraits, Bayes likelihood planes.", {"min_stage_w_ratio": 0.30, "equal_aspect_optional": True}, ["create", "zoom", "focus"]),
    ("function_plot", "Smooth discontinuity-aware curve y=f(x) bound to a parent axes system.", ["object_id", "expr", "axes_id"], "Finite jump-clipped segments via plot_clipped_segments", "Taylor polynomials, density curves, activation functions.", {"requires_axes": True}, ["create", "draw", "transform", "morph", "highlight"]),
    ("parametric_curve", "Parametric trajectory (x(t), y(t)) including circles, spirals, epicycles, and orbits.", ["object_id", "expr", "t_range", "axes_id"], "Bounded parametric sweep clipped to axes", "Euler unit circle, phase orbits, cycloids.", {"requires_axes": True}, ["create", "draw", "trace", "animate_param"]),
    ("bar_chart", "Ordered discrete column distribution or frequency histogram with ticks.", ["object_id", "values", "bar_labels"], "Non-negative proportional bar normalization", "PMF distributions, convolution sums, singular value spectra.", {"max_bars": 40}, ["create", "reveal", "iterate_states", "highlight"]),
    ("equation", "Inline/callout LaTeX mathematical expression synchronized with visual state.", ["object_id", "tex", "card"], "Validated TeX expression inside reserved equation strip", "Formula derivations, term-by-term MatchingTex transforms.", {"band": "equation_box"}, ["create", "reveal", "transform", "morph", "highlight"]),
    ("vector", "Directed arrow representing displacement, gradient, eigenvector, or velocity.", ["object_id", "coords", "end_coords", "axes_id"], "Euclidean vector norm & direction preserving origin", "Eigenvector invariance A*v=lambda*v, gradient steps.", {"requires_axes_optional": True}, ["create", "draw", "move", "scale", "rotate", "highlight"]),
    ("polygon", "Closed non-self-intersecting polygonal region with verified area and congruence.", ["object_id", "vertices", "axes_id"], "Shoelace area & orientation invariant", "Pythagorean tilings, transformed unit squares, Riemann bars.", {"equal_aspect": True}, ["create", "draw", "move", "rotate", "scale", "duplicate", "transform"]),
    ("matrix", "Structured bracketed 2D numeric/symbolic matrix array.", ["object_id", "matrix_rows", "panel_slot"], "Exact 2D grid spacing without bracket overlap", "Markov transition matrices, Jacobian matrices, linear operators.", {"min_gap_px": 20}, ["create", "reveal", "highlight", "iterate_states"]),
    ("table", "Multi-column comparison or iteration table with structured row layout.", ["object_id", "table_headers", "table_rows"], "Row/column bounded tabular alignment", "Step-by-step iterations, truth tables, error comparisons.", {"max_rows": 12}, ["create", "reveal", "highlight", "sequence"]),
    ("data_series", "Connected or scatter trajectory plot for empirical runs and convergence.", ["object_id", "points_list", "axes_id"], "Finite polyline clipping across axes bounds", "Loss curves, random walks, RK4 phase orbits.", {"requires_axes_optional": True}, ["create", "draw", "trace", "highlight"]),
    ("graph_network", "Node-and-directed/weighted-edge diagram for Markov chains and graphs.", ["object_id", "points_list", "edge_pairs", "node_labels"], "Node/edge index bounds verification", "Markov state graphs, Huffman trees, Bayes networks.", {"max_nodes": 16}, ["create", "reveal", "highlight", "sequence"]),
    ("probability_region", "Proportional area tile encoding prior/conditional/posterior probability mass.", ["object_id", "vertices", "axes_id"], "Area ratios equal conditional probabilities", "Bayes partitions, joint probability geometry.", {"equal_aspect": True}, ["create", "highlight", "transform"]),
    ("particle_system", "Deterministic sample ensemble (Monte Carlo drops, phase-space walkers).", ["object_id", "points_list", "axes_id"], "Bounded finite sample coordinates", "Sampling convergence, phase-space flow ensembles.", {"max_particles": 300}, ["create", "animate_param", "iterate_states"]),
    ("waveform", "Time- or frequency-domain oscillating curve supporting harmonic sums.", ["object_id", "expr", "x_range", "axes_id"], "Trigonometric superpositions & Nyquist-safe sampling", "Fourier partial sums, damped resonance, signal filtering.", {"requires_axes": True}, ["create", "draw", "transform", "morph"]),
    ("point", "Tracked mathematical point (extremum, fixed point, eigenstate, probe dot).", ["object_id", "coords", "axes_id"], "Parametric updater or stationary coordinate anchor", "Local minima, moving phase dots, fixed points.", {"requires_axes_optional": True}, ["create", "move", "trace", "animate_param", "highlight"]),
    ("line", "Geometric segment, tangent line, asymptote, or eigenvector span ray.", ["object_id", "coords", "end_coords", "axes_id"], "Euclidean chord / tangent construction", "Eigenvector spans, tangent/secant lines.", {"requires_axes_optional": True}, ["create", "draw", "move", "rotate", "highlight"]),
    ("circle", "Circle, unit circle, or orbital locus bound cleanly to axes space.", ["object_id", "coords", "radius", "axes_id"], "Positive finite radius mapped via axes unit scale", "Complex unit circle, Gershgorin disks.", {"equal_aspect": True}, ["create", "draw", "scale", "highlight"]),
    ("number_line", "1D oriented number axis with tick marks and numeric labels.", ["object_id", "x_range"], "Uniform tick intervals across domain", "1D eigenvalues, random walks, root brackets.", {"orientation": "horizontal"}, ["create", "highlight"]),
    ("annotation", "Concise callout badge placed cleanly outside obstacle bounding boxes.", ["object_id", "text", "coords"], "Collision-free SafeZone obstacle resolution", "Calling out eigenvalues, saddle points, stability regimes.", {"gap_px": 20}, ["create", "reveal", "highlight", "remove"]),
    ("angle_arc", "Measured angular wedge tracking continuous rotation or phase angle.", ["object_id", "coords", "radius", "param_name"], "Positive arc radius + continuous ValueTracker sweep", "Phase angle tracking, complex multiplication rotation.", {"manimgx_specialized": True}, ["create", "animate_param", "highlight"]),
    ("projection", "Dashed orthogonal drop-line projecting an active point onto an axis.", ["object_id", "expr", "param_name", "axes_id"], "Exact cos/sin projection coupling", "Real vs imaginary components, subspace projections.", {"manimgx_specialized": True}, ["create", "animate_param", "parallel"]),
    ("mechanical_body", "Parameterized moving mass, bob, slider, or coupled block connected via spring, dashpot damper, pendulum rod, or free anchor driven deterministically by simulation keyframes.", ["object_id", "connector_kind", "anchor_coords", "coords", "axes_id"], "Coupled position & connector kinematics synced with ODE/simulation state", "Mass-spring-damper systems, pendulums, coupled resonators, force balance.", {"requires_axes_optional": True}, ["create", "draw", "animate_param", "set_param", "focus"]),
    ("vector_field", "Parameterized 2D tangent/force arrow field (dx_expr, dy_expr) rendered cleanly over axes space.", ["object_id", "dx_expr", "dy_expr", "axes_id"], "Finite uniform grid direction/magnitude normalization", "Phase-space flows, oscillator spirals, electric & gradient fields.", {"requires_axes": True}, ["create", "draw", "highlight"]),
    ("fraction_bar", "Partitioned rectangular bar representing numerator/denominator with optional equivalent subdivisions.", ["object_id", "numerator", "denominator"], "Exact rational part-whole ratio", "Equivalent fractions, adding unlike denominators, percentages.", {"min_stage_w_ratio": 0.25}, ["create", "draw", "highlight", "transform"]),
    ("fraction_circle", "Partitioned circular sector diagram representing numerator/denominator.", ["object_id", "numerator", "denominator"], "Equal angular sectors summing to 2*pi", "Part-whole intuition for halves, quarters, thirds, and equivalent sectors.", {"equal_aspect": True}, ["create", "draw", "highlight", "transform"]),
    ("array_grid", "Parameterized rows x cols rectangular tile array for multiplication, area, and decimal/percent grids.", ["object_id", "rows", "cols"], "Exact rows*cols tile count", "Multiplication arrays, area models, 10x10 percentage grids.", {"min_gap_px": 16}, ["create", "draw", "highlight", "group"]),
    ("counting_group", "Concrete grouped objects (dots, blocks, tens-bundles) for counting, place value, and arithmetic.", ["object_id", "count", "group_size"], "Exact integer object count", "Class 1 addition, Class 2 tens-and-ones grouping, division sharing.", {"min_gap_px": 16}, ["create", "reveal", "group", "highlight"]),
    ("equation_balance", "Visual balance scale displaying left and right algebraic expressions.", ["object_id", "lhs_text", "rhs_text"], "Balanced equality invariant", "Linear equation solving by applying equal operations to both sides.", {"min_stage_w_ratio": 0.30}, ["create", "reveal", "transform", "highlight"]),
    ("concept_card", "Structured educational panel for definitions, worked-example steps, practice questions, and recaps.", ["object_id", "label", "steps_list"], "Readable bounded educational typography", "Step-by-step worked examples, practice prompts, lesson recaps.", {"min_gap_px": 18}, ["create", "reveal", "sequence", "highlight"]),
]


VISUAL_OBJECT_SPECS: list[ComponentSpec] = [
    ComponentSpec(
        name=name,
        category="visual_object",
        description=desc,
        input_model="VisualObjectInput",
        output_model="VisualObjectOutput",
        input_schema_model=VisualObjectInput,
        output_schema_model=VisualObjectOutput,
        inputs_schema=inputs,
        outputs=[f"{name}_render_node"],
        render_implementations={
            "manimgx": f"scenes.v3_scene.V3Scene._build_object[{name}]",
            "hyperframes": f"studio.renderers.hyperframes.components.compile_scene_object_element[{name}]",
        },
        supported_ops=ops,
        layout_requirements=layout_req,
        resource_constraints={"max_points": 600, "fps_safe": True},
        math_grounding=ground,
        example_use_case=eg,
        validate_fn=_val_visual,
        test_fn=(lambda k=name: _test_visual_kind(k)),
    )
    for name, desc, inputs, ground, eg, layout_req, ops in _DEFS
]


def register_visuals(registry: ComponentRegistry) -> None:
    registry.register_many(VISUAL_OBJECT_SPECS)
