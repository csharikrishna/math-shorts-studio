"""Adaptive composition templates supporting 16:9 widescreen and recomposed 9:16 portrait."""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel

from studio.registry.base import ComponentRegistry, ComponentSpec


class CompositionInput(BaseModel):
    composition: str
    aspect_label: str = "16:9"


class CompositionOutput(BaseModel):
    resolved_composition: str


def _val_comp(data: dict[str, Any]) -> list[str]:
    comp = str(data.get("composition", "")).strip()
    if not comp:
        return ["composition name required"]
    return []


def _test_comp(comp_name: str, aspect: str) -> bool:
    adapted = adapt_composition_for_aspect(comp_name, aspect)
    return bool(adapted) and len(_val_comp({"composition": comp_name, "aspect_label": aspect})) == 0


_COMP_DEFS: list[tuple[str, str, list[str], list[str], str, list[str]]] = [
    ("centered_focus", "Single dominant geometric construction, phase plane, or network occupying center stage cleanly.", ["primary_object_ids"], ["16:9", "9:16"], "Opening intuition hook or climactic visual proof.", ["centered_focus"]),
    ("side_by_side_comparison", "Dual horizontal comparison (before vs after, wrong vs right intuition, cause vs effect).", ["left_ids", "right_ids"], ["16:9"], "16:9 landscape comparison of eigenvector vs non-eigenvector or prior vs posterior.", ["side_by_side_comparison"]),
    ("stacked_comparison", "Portrait-native vertical split placing reference view above and transformed/resulting view below.", ["top_ids", "bottom_ids"], ["9:16"], "Recomposed 9:16 portrait Short equivalent of side_by_side_comparison.", ["stacked_comparison"]),
    ("main_with_equation_panel", "Primary visual apparatus alongside a synchronized symbolic/equation strip.", ["main_visual_ids", "equation_ids"], ["16:9", "9:16"], "Connecting geometric matrix stretch to det(A - lambda*I) = 0.", ["main_with_equation_panel"]),
    ("step_by_step_sequence", "Guided progressive derivation where each beat updates one focused mathematical step in place.", ["sequence_event_ids"], ["16:9", "9:16"], "Algorithm step-through, Bayesian update, Riemann refinement.", ["step_by_step_sequence"]),
    ("grid_of_multiple_examples", "Compact comparison grid illustrating contrasting regimes (e.g. stable, saddle, spiral).", ["cell_groups"], ["16:9", "9:16"], "Classification of 2x2 dynamical fixed points or distributions.", ["grid_of_multiple_examples"]),
    ("spotlight_zoom", "Zoomed camera push magnifying local structure while keeping context.", ["focus_target_id", "zoom_factor"], ["16:9", "9:16"], "Local linearity of smooth manifolds, narrow posterior peak.", ["spotlight_zoom"]),
    ("annotated_overlay", "Persistent mathematical state enriched with concise directional labels and projections.", ["persistent_ids", "annotation_ids"], ["16:9", "9:16"], "Labeling eigenvalues and invariant lines over an existing construction.", ["annotated_overlay"]),
    ("timeline_progression", "Chronological progression across iterations combining discrete table/bar updates and phase state.", ["iteration_visual_id"], ["16:9", "9:16"], "Markov convergence A^k * p_0 -> pi* or Euler integration steps.", ["timeline_progression"]),
    ("time_frequency_split", "Synchronized dual-domain layout linking time/spatial input with frequency or spectral view.", ["time_ids", "freq_ids"], ["16:9", "9:16"], "Time signal alongside Fourier spectrum or damped response curve.", ["time_frequency_split"]),
    ("model_with_readout", "Dominant geometric plot paired with live numeric/symbolic telemetry readout panel.", ["main_ids", "readout_ids"], ["16:9", "9:16"], "Winding trajectory with live C(w) center-of-mass coordinates.", ["model_with_readout"]),
    ("before_after_simulation", "Causal before/after transformation view (raw vs filtered or prior vs posterior).", ["before_ids", "after_ids"], ["16:9", "9:16"], "Raw vs low-pass reconstructed waveform preserving exact time axis.", ["before_after_simulation"]),
    ("multi_view_dashboard", "Coordinated 3-panel analytical view linking state, curve, and derived invariant.", ["panel_groups"], ["16:9", "9:16"], "Linked signal, complex winding plane, and magnitude spectrum.", ["multi_view_dashboard"]),
    ("progressive_derivation", "Stepwise symbolic/visual build preserving context while introducing one new step.", ["step_ids"], ["16:9", "9:16"], "Progressive derivation of integral transforms or eigenvalue stability.", ["progressive_derivation"]),
    ("physical_system_with_displacement", "Synchronized physical apparatus (spring/damper/pendulum body) alongside displacement-vs-time curve.", ["physical_ids", "displacement_ids"], ["16:9", "9:16"], "Physical spring-mass-damper coupled directly to x(t) time series.", ["physical_system_with_displacement"]),
    ("physical_system_with_phase_space", "Physical dynamical body synchronized directly with state-space velocity/position phase portrait.", ["physical_ids", "phase_ids"], ["16:9", "9:16"], "Physical oscillator bob paired with spiraling (x, v) phase trajectory.", ["physical_system_with_phase_space"]),
    ("dual_time_series", "Synchronized multi-variable time-series plot or stacked trajectories sharing common time horizon.", ["series_ids"], ["16:9", "9:16"], "Coupled two-population cycles x(t), y(t) with distinct semantic colors.", ["dual_time_series"]),
    ("parameter_sweep_with_response", "Live parameter sweep paired side-by-side with resulting frequency/objective response curve.", ["system_ids", "response_ids"], ["16:9", "9:16"], "Varying damping or frequency with moving resonance peak marker.", ["parameter_sweep_with_response"]),
    ("vector_field_with_trajectory", "2D slope/velocity vector field carrying an animated initial-value trajectory and equilibrium marker.", ["field_ids", "trajectory_ids"], ["16:9", "9:16"], "Phase-plane vector field guiding closed orbits or sink spirals.", ["vector_field_with_trajectory"]),
    ("state_with_conserved_chart", "Dynamical state trajectory linked directly to its conserved invariant or energy bar/time chart.", ["state_ids", "invariant_ids"], ["16:9", "9:16"], "Kinetic/potential exchange vs total energy or Hamiltonian drift check.", ["state_with_conserved_chart"]),
]


COMPOSITION_SPECS: list[ComponentSpec] = [
    ComponentSpec(
        name=name,
        category="composition_template",
        description=desc,
        input_model="CompositionInput",
        output_model="CompositionOutput",
        input_schema_model=CompositionInput,
        output_schema_model=CompositionOutput,
        inputs_schema=inputs,
        outputs=["safe_frame_layout"],
        aspect_support=aspects,
        render_implementations={
            "manimgx": "studio.layout.solver.resolve_composition_layout",
            "hyperframes": "studio.layout.solver.resolve_composition_layout",
        },
        supported_ops=ops,
        layout_requirements={"aspect_support": aspects},
        example_use_case=eg,
        validate_fn=_val_comp,
        test_fn=(lambda n=name, a=aspects[0]: _test_comp(n, a)),
    )
    for name, desc, inputs, aspects, eg, ops in _COMP_DEFS
]


def register_compositions(registry: ComponentRegistry) -> None:
    registry.register_many(COMPOSITION_SPECS)


def adapt_composition_for_aspect(composition: str, aspect_label: str) -> str:
    """Recompose incompatible horizontal/vertical layouts instead of squashing frames."""
    if aspect_label == "9:16" and composition == "side_by_side_comparison":
        return "stacked_comparison"
    if aspect_label == "16:9" and composition == "stacked_comparison":
        return "side_by_side_comparison"
    return composition
