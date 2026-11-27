"""Reusable animation mechanisms, audio cues, and caption styles with enforceable contracts."""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field

from studio.registry.base import ComponentRegistry, ComponentSpec


class AnimationEventInput(BaseModel):
    op: str
    target_id: str
    source_id: str | None = None


class AnimationEventOutput(BaseModel):
    state_transition: str


def _val_anim(data: dict[str, Any]) -> list[str]:
    errs: list[str] = []
    if not str(data.get("target_id", "")).strip():
        errs.append("animation event target_id is required")
    if str(data.get("op", "")) in ("transform", "morph", "duplicate") and not str(data.get("source_id", "") or "").strip():
        errs.append(f"{data.get('op')} requires source_id")
    return errs


def _test_anim(op_name: str) -> bool:
    if op_name in ("transform", "morph", "duplicate"):
        ok = len(_val_anim({"op": op_name, "target_id": "dst", "source_id": "src"})) == 0
        fail = bool(_val_anim({"op": op_name, "target_id": "dst"}))
        return ok and fail
    return len(_val_anim({"op": op_name, "target_id": "obj_1"})) == 0 and bool(_val_anim({"op": op_name, "target_id": ""}))


def _val_caption(data: dict[str, Any]) -> list[str]:
    if not data.get("words_with_timestamps") and not data.get("caption_text"):
        return ["caption component requires words_with_timestamps or caption_text"]
    return []


def _test_caption() -> bool:
    return len(_val_caption({"caption_text": "hello"})) == 0 and bool(_val_caption({}))


def _val_audio(data: dict[str, Any]) -> list[str]:
    if float(data.get("timestamp_s", 0.0)) < 0.0:
        return ["timestamp_s must be >= 0"]
    return []


def _test_audio() -> bool:
    return len(_val_audio({"timestamp_s": 1.2})) == 0 and bool(_val_audio({"timestamp_s": -1.0}))


_ANIM_DEFS: list[tuple[str, str, list[str], str]] = [
    ("create", "Introduce an axes, geometric region, matrix, table, or diagram cleanly into view.", ["target_id", "at", "duration"], "state: visible"),
    ("draw", "Stroke-draw a function curve, parametric path, waveform, or polygon perimeter progressively.", ["target_id", "at", "duration"], "state: visible"),
    ("reveal", "Progressive reveal (staggered rows, terms, equation segments, or table columns) synchronized to beat phrasing.", ["target_id", "at", "duration"], "state: revealed"),
    ("transform", "Continuously morph source geometry, equation terms (TransformMatchingTex), or matrix into target form preserving identity.", ["target_id", "source_id"], "state: transformed"),
    ("morph", "Smooth symbolic/geometric morph between comparable structures (Taylor degree N->N+1, stretched unit square).", ["target_id", "source_id"], "state: morphed"),
    ("highlight", "Call immediate visual attention (circumscribe/flash/accent shift) to active variable, node, or region.", ["target_id", "duration"], "state: emphasized"),
    ("move", "Translate an existing object smoothly in math or screen coordinates while respecting persistent identity.", ["target_id", "shift"], "state: moved"),
    ("scale", "Proportionally scale an element (e.g., stretching an eigenvector by eigenvalue lambda).", ["target_id", "scale_factor"], "state: scaled"),
    ("rotate", "Rigidly rotate geometry or complex vectors around origin/pivot by exact degrees.", ["target_id", "angle_deg"], "state: rotated"),
    ("duplicate", "Spawn a visual copy of an object into a comparison panel or rearrangement zone while keeping original.", ["target_id", "source_id", "shift"], "copied_object"),
    ("group", "Coalesce related objects into a single logical assembly for joint transforms or camera framing.", ["target_id", "source_id"], "logical_group"),
    ("zoom", "Camera reframe tightening onto an active stage focal target and restoring cleanly.", ["target_id", "scale_factor"], "camera_zoomed"),
    ("focus", "Dim secondary visuals slightly and spotlight target element(s) to guide viewer eye.", ["target_id"], "spotlight_focus"),
    ("trace", "Move an active probe along a parameterized trajectory while leaving a visible locus trail.", ["target_id", "param_values"], "traced_locus"),
    ("iterate_states", "Step deterministically through computed discrete iterations (GD steps, Markov power A^k*v, Riemann n).", ["target_id", "param_values"], "stepped_trajectory"),
    ("interpolate_values", "Smoothly sweep continuous parameter (matrix blend I -> A, angle 0 -> pi, sample size n -> N).", ["target_id", "param_values"], "continuous_sweep"),
    ("sequence", "Choreograph child events sequentially with measured stagger timing within a beat.", ["target_id"], "staggered_sequence"),
    ("parallel", "Synchronize simultaneous cause-and-effect updates (curve + tangent slope + live readout together).", ["target_id", "source_id"], "coupled_motion"),
    ("animate_param", "Drive parameter-coupled updaters cleanly across discrete or continuous parameter milestones.", ["target_id", "param_range"], "updated_visuals"),
    ("remove", "Retire obsolete objects cleanly and purge them from active registry state to prevent screen clutter.", ["target_id"], "state: removed"),
]


ANIMATION_AND_MEDIA_SPECS: list[ComponentSpec] = [
    *[
        ComponentSpec(
            name=name,
            category="animation_mechanism",
            description=desc,
            input_model="AnimationEventInput",
            output_model="AnimationEventOutput",
            input_schema_model=AnimationEventInput,
            output_schema_model=AnimationEventOutput,
            inputs_schema=inputs,
            outputs=[out],
            render_implementations={
                "manimgx": f"scenes.v3_scene.V3Scene._compile_event[{name}]",
                "hyperframes": f"studio.renderers.hyperframes.runtime.build_runtime_js[{name}]",
            },
            supported_ops=[name],
            layout_requirements={"preserves_safe_zones": True},
            resource_constraints={"bookmark_sync_supported": True},
            validate_fn=_val_anim,
            test_fn=(lambda op=name: _test_anim(op)),
        )
        for name, desc, inputs, out in _ANIM_DEFS
    ],
    ComponentSpec(
        name="cartesian_active_word",
        category="caption_style",
        description="Bottom-band phrase captions with active spoken-word color highlight derived from exact CaptionIR alignment.",
        input_model="AnimationEventInput",
        output_model="AnimationEventOutput",
        input_schema_model=AnimationEventInput,
        output_schema_model=AnimationEventOutput,
        inputs_schema=["words_with_timestamps", "accent_color", "bottom_band_px"],
        outputs=["ass_or_html_captions"],
        render_implementations={
            "manimgx": "studio.layout.components.CaptionCard",
            "hyperframes": "studio.renderers.hyperframes.captions.build_hyperframes_captions",
        },
        validate_fn=_val_caption,
        test_fn=_test_caption,
    ),
    ComponentSpec(
        name="dialogue_speaker_badge",
        category="caption_style",
        description="Color-coded speaker tag pill paired with synced phrase captions for multi-speaker Socratic/podcast modes.",
        input_model="AnimationEventInput",
        output_model="AnimationEventOutput",
        input_schema_model=AnimationEventInput,
        output_schema_model=AnimationEventOutput,
        inputs_schema=["speaker", "speaker_color", "caption_text"],
        outputs=["speaker_caption_track"],
        render_implementations={
            "manimgx": "studio.layout.components.CaptionCard._speaker_badge",
            "hyperframes": "studio.renderers.hyperframes.compiler#speaker-badge",
        },
        validate_fn=_val_caption,
        test_fn=_test_caption,
    ),
    ComponentSpec(
        name="pop_marimba",
        category="audio_cue",
        description="Warm wooden felt marimba tap synchronized to visual emergence or reveal.",
        input_model="AnimationEventInput",
        output_model="AnimationEventOutput",
        input_schema_model=AnimationEventInput,
        output_schema_model=AnimationEventOutput,
        inputs_schema=["timestamp_s"],
        outputs=["sfx_mix_event"],
        render_implementations={"manimgx": "studio.audio.sfx.SfxLibrary[pop_marimba]", "hyperframes": "studio.audio.master.master_video"},
        validate_fn=_val_audio,
        test_fn=_test_audio,
    ),
    ComponentSpec(
        name="chime_success",
        category="audio_cue",
        description="Soft resonant bell chime on mathematical insight, proof completion, or convergence milestone.",
        input_model="AnimationEventInput",
        output_model="AnimationEventOutput",
        input_schema_model=AnimationEventInput,
        output_schema_model=AnimationEventOutput,
        inputs_schema=["timestamp_s"],
        outputs=["sfx_mix_event"],
        render_implementations={"manimgx": "studio.audio.sfx.SfxLibrary[chime_reveal]", "hyperframes": "studio.audio.master.master_video"},
        validate_fn=_val_audio,
        test_fn=_test_audio,
    ),
    ComponentSpec(
        name="whoosh_transform",
        category="audio_cue",
        description="Smooth brushed-pad transition cue supporting geometric rearrangement, transform, or chapter change.",
        input_model="AnimationEventInput",
        output_model="AnimationEventOutput",
        input_schema_model=AnimationEventInput,
        output_schema_model=AnimationEventOutput,
        inputs_schema=["timestamp_s"],
        outputs=["sfx_mix_event"],
        render_implementations={"manimgx": "studio.audio.sfx.SfxLibrary[whoosh_transition]", "hyperframes": "studio.audio.master.master_video"},
        validate_fn=_val_audio,
        test_fn=_test_audio,
    ),
]


def register_animations(registry: ComponentRegistry) -> None:
    registry.register_many(ANIMATION_AND_MEDIA_SPECS)
