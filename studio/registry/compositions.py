"""Adaptive composition templates supporting 16:9 widescreen and recomposed 9:16 portrait."""

from __future__ import annotations

from studio.registry.base import ComponentRegistry, ComponentSpec


COMPOSITION_SPECS: list[ComponentSpec] = [
    ComponentSpec(
        name="centered_focus",
        category="composition_template",
        description="Single dominant geometric construction, phase plane, or network occupying center stage with clean margins.",
        inputs_schema=["primary_object_ids", "optional_title"],
        outputs=["safe_frame_layout"],
        aspect_support=["16:9", "9:16"],
        example_use_case="Opening intuition hook or climactic visual proof.",
    ),
    ComponentSpec(
        name="side_by_side_comparison",
        category="composition_template",
        description="Dual horizontal comparison (before vs after, wrong vs right intuition, cause vs effect).",
        inputs_schema=["left_ids", "right_ids"],
        outputs=["dual_column_layout"],
        aspect_support=["16:9"],
        example_use_case="16:9 landscape comparison of eigenvector vs non-eigenvector or prior vs posterior.",
    ),
    ComponentSpec(
        name="stacked_comparison",
        category="composition_template",
        description="Portrait-native vertical split placing reference view above and transformed/resulting view below.",
        inputs_schema=["top_ids", "bottom_ids"],
        outputs=["dual_row_layout"],
        aspect_support=["9:16"],
        example_use_case="Recomposed 9:16 portrait Short equivalent of side_by_side_comparison (never squashed horizontally).",
    ),
    ComponentSpec(
        name="main_with_equation_panel",
        category="composition_template",
        description="Primary visual apparatus alongside a synchronized symbolic/equation strip (side strip in 16:9, upper-right/lower-stage in 9:16).",
        inputs_schema=["main_visual_ids", "equation_ids"],
        outputs=["split_visual_symbolic_layout"],
        aspect_support=["16:9", "9:16"],
        example_use_case="Connecting geometric matrix stretch to det(A - lambda*I) = 0.",
    ),
    ComponentSpec(
        name="step_by_step_sequence",
        category="composition_template",
        description="Guided progressive derivation where each beat adds or updates one focused mathematical step in place.",
        inputs_schema=["sequence_event_ids"],
        outputs=["progressive_derivation_layout"],
        aspect_support=["16:9", "9:16"],
        example_use_case="Algorithm step-through, Bayesian evidence update, Riemann refinement.",
    ),
    ComponentSpec(
        name="grid_of_multiple_examples",
        category="composition_template",
        description="2x2 or 1x3 compact comparison grid illustrating contrasting regimes (e.g. stable, saddle, spiral).",
        inputs_schema=["cell_groups"],
        outputs=["grid_layout"],
        aspect_support=["16:9", "9:16"],
        example_use_case="Classification of 2x2 dynamical fixed points or distributions.",
    ),
    ComponentSpec(
        name="spotlight_zoom",
        category="composition_template",
        description="Zoomed inset or camera push magnifying local structure while keeping global context cue.",
        inputs_schema=["focus_target_id", "zoom_factor"],
        outputs=["zoomed_layout"],
        aspect_support=["16:9", "9:16"],
        example_use_case="Local linearity of smooth manifolds, narrow posterior peak.",
    ),
    ComponentSpec(
        name="annotated_overlay",
        category="composition_template",
        description="Existing persistent mathematical state enriched with concise directional labels and projections.",
        inputs_schema=["persistent_ids", "annotation_ids"],
        outputs=["annotated_layout"],
        aspect_support=["16:9", "9:16"],
        example_use_case="Labeling eigenvalues and invariant lines over an already constructed transformation.",
    ),
    ComponentSpec(
        name="timeline_progression",
        category="composition_template",
        description="Chronological progression across iterations t=0..N combining discrete table/bar updates and phase state.",
        inputs_schema=["iteration_visual_id", "timeline_axis_id"],
        outputs=["temporal_layout"],
        aspect_support=["16:9", "9:16"],
        example_use_case="Markov convergence A^k * p_0 -> pi* or Euler integration steps.",
    ),
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
