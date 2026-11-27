"""Reusable animation mechanisms, audio cues, and caption styles."""

from __future__ import annotations

from studio.registry.base import ComponentRegistry, ComponentSpec


ANIMATION_AND_MEDIA_SPECS: list[ComponentSpec] = [
    ComponentSpec(
        name="create",
        category="animation_mechanism",
        description="Introduce an axes, geometric region, matrix, table, or diagram cleanly into view.",
        inputs_schema=["target_id", "at: hook|bookmark", "duration"],
        outputs=["state: visible"],
    ),
    ComponentSpec(
        name="draw",
        category="animation_mechanism",
        description="Stroke-draw a function curve, parametric path, waveform, or polygon perimeter progressively.",
        inputs_schema=["target_id", "at", "duration"],
        outputs=["state: visible"],
    ),
    ComponentSpec(
        name="reveal",
        category="animation_mechanism",
        description="Progressive reveal (staggered rows, terms, equation segments, or table columns) synchronized to beat phrasing.",
        inputs_schema=["target_id", "at", "duration"],
        outputs=["state: revealed"],
    ),
    ComponentSpec(
        name="transform",
        category="animation_mechanism",
        description="Continuously morph source geometry, equation terms, or matrix into target form preserving identity.",
        inputs_schema=["target_id", "to_target_id", "expr_or_text", "to_coords"],
        outputs=["state: transformed"],
    ),
    ComponentSpec(
        name="morph",
        category="animation_mechanism",
        description="Smooth symbolic/geometric morph between comparable structures (Taylor degree N->N+1, stretched unit square).",
        inputs_schema=["target_id", "to_target_id/expr_or_text"],
        outputs=["state: morphed"],
    ),
    ComponentSpec(
        name="highlight",
        category="animation_mechanism",
        description="Call immediate visual attention (circumscribe/flash/accent shift) to active variable, node, or region.",
        inputs_schema=["target_id", "color", "duration"],
        outputs=["state: emphasized"],
    ),
    ComponentSpec(
        name="move",
        category="animation_mechanism",
        description="Translate an existing object smoothly in math or screen coordinates while respecting persistent identity.",
        inputs_schema=["target_id", "to_coords"],
        outputs=["state: moved"],
    ),
    ComponentSpec(
        name="scale",
        category="animation_mechanism",
        description="Proportionally scale an element (e.g., stretching an eigenvector by eigenvalue lambda).",
        inputs_schema=["target_id", "scale_factor"],
        outputs=["state: scaled"],
    ),
    ComponentSpec(
        name="rotate",
        category="animation_mechanism",
        description="Rigidly rotate geometry or complex vectors around origin/pivot by exact radians.",
        inputs_schema=["target_id", "angle_rad"],
        outputs=["state: rotated"],
    ),
    ComponentSpec(
        name="duplicate",
        category="animation_mechanism",
        description="Spawn a visual copy of an object into a comparison panel or rearrangement zone while keeping original.",
        inputs_schema=["target_id", "to_target_id", "to_coords"],
        outputs=["copied_object"],
    ),
    ComponentSpec(
        name="group",
        category="animation_mechanism",
        description="Coalesce related objects into a single logical assembly for joint transforms or camera framing.",
        inputs_schema=["target_id", "param_values/member_ids"],
        outputs=["logical_group"],
    ),
    ComponentSpec(
        name="zoom",
        category="animation_mechanism",
        description="Tighten framing onto a sub-region (singularity, fixed point, local linearity) and pull back cleanly.",
        inputs_schema=["target_id", "scale_factor"],
        outputs=["camera_zoomed"],
    ),
    ComponentSpec(
        name="focus",
        category="animation_mechanism",
        description="Dim secondary visuals slightly and spotlight target element(s) to guide viewer eye.",
        inputs_schema=["target_id"],
        outputs=["spotlight_focus"],
    ),
    ComponentSpec(
        name="trace",
        category="animation_mechanism",
        description="Move an active probe along a parameterized trajectory while leaving a visible locus trail.",
        inputs_schema=["target_id", "param_values", "expr_or_text"],
        outputs=["traced_locus"],
    ),
    ComponentSpec(
        name="iterate_states",
        category="animation_mechanism",
        description="Step deterministically through computed discrete iterations (GD steps, Markov power A^k*v, Newton steps).",
        inputs_schema=["target_id", "param_values: list[float]"],
        outputs=["stepped_trajectory"],
    ),
    ComponentSpec(
        name="interpolate_values",
        category="animation_mechanism",
        description="Smoothly sweep continuous parameter (matrix blend I -> A, angle 0 -> pi, sample size n -> N).",
        inputs_schema=["target_id", "param_values: [start, mid, ..., end]"],
        outputs=["continuous_sweep"],
    ),
    ComponentSpec(
        name="sequence",
        category="animation_mechanism",
        description="Choreograph child events sequentially with measured stagger timing within a beat.",
        inputs_schema=["target_id", "param_values/stagger"],
        outputs=["staggered_sequence"],
    ),
    ComponentSpec(
        name="parallel",
        category="animation_mechanism",
        description="Synchronize simultaneous cause-and-effect updates (curve + tangent slope + live readout together).",
        inputs_schema=["target_id", "to_target_id"],
        outputs=["coupled_motion"],
    ),
    ComponentSpec(
        name="animate_param",
        category="animation_mechanism",
        description="Drive parameter-coupled updaters cleanly across discrete or continuous parameter milestones.",
        inputs_schema=["target_id", "param_values"],
        outputs=["updated_visuals"],
    ),
    ComponentSpec(
        name="remove",
        category="animation_mechanism",
        description="Retire obsolete objects cleanly and purge them from active registry state to prevent screen clutter.",
        inputs_schema=["target_id"],
        outputs=["state: removed"],
    ),
    # Caption styles
    ComponentSpec(
        name="cartesian_active_word",
        category="caption_style",
        description="Bottom-band phrase captions with active spoken-word color highlight derived from exact forced/TTS alignment.",
        inputs_schema=["words_with_timestamps", "accent_color", "bottom_band_px"],
        outputs=["ass_or_html_captions"],
    ),
    ComponentSpec(
        name="dialogue_speaker_badge",
        category="caption_style",
        description="Color-coded speaker tag pill paired with synced phrase captions for multi-speaker Socratic/podcast modes.",
        inputs_schema=["speaker", "speaker_color", "caption_text"],
        outputs=["speaker_caption_track"],
    ),
    # Audio cues
    ComponentSpec(
        name="pop_marimba",
        category="audio_cue",
        description="Clean soft percussion cue synchronized to visual emergence or reveal.",
        inputs_schema=["timestamp_s"],
        outputs=["sfx_mix_event"],
    ),
    ComponentSpec(
        name="chime_success",
        category="audio_cue",
        description="Warm harmonic chime on mathematical insight, proof completion, or convergence milestone.",
        inputs_schema=["timestamp_s"],
        outputs=["sfx_mix_event"],
    ),
    ComponentSpec(
        name="whoosh_transform",
        category="audio_cue",
        description="Subtle air-swipe cue supporting geometric rearrangement, transform, or chapter transition.",
        inputs_schema=["timestamp_s"],
        outputs=["sfx_mix_event"],
    ),
]


def register_animations(registry: ComponentRegistry) -> None:
    registry.register_many(ANIMATION_AND_MEDIA_SPECS)
