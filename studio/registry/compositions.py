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
