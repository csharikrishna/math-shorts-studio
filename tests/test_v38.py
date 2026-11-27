"""Comprehensive V3.8 verification suite covering script-first ordering, enforceable registry,
SceneRuntimeState cross-renderer parity, canonical CaptionIR provenance, strict publication gates,
scene-aware frame QA, Markov stochastic math verification, dependency caching, and scene preview."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from urllib.request import urlopen

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from studio.ai.math_truth import (
    is_row_stochastic_matrix,
    verify_graph_stochastic_weights,
    verify_stationary_distribution,
)
from studio.ai.v3_schema import (
    AnimationEvent,
    ConstructionOp,
    ConstructionParams,
    SceneObject,
    SceneV3,
    StructuredClaim,
    VideoPlanV3,
)
from studio.ai.validate_v3 import validate_v3_plan
from studio.audio.tts import SyncedBeat, find_ffmpeg
from studio.captions import build_caption_ir, export_ass
from studio.creative.dialogue import detect_narrative_anti_patterns, detect_unpronounced_identifiers
from studio.creative.script_ir import (
    NarrationBeatIR,
    ScriptIR,
    approve_script_ir,
    commit_script_revision,
    lock_script_to_plan,
    revise_beat_text,
    split_anchors_and_spoken_text,
    validate_script_ir,
)
from studio.design.spec import default_design_spec
from studio.preview import (
    build_scene_preview_bundle,
    slice_plan_for_scene,
    start_preview_server,
)
from studio.qa.engine import audit_educational_quality, verify_caption_synchronization
from studio.qa.frames import inspect_video_scene_aware
from studio.registry import get_default_registry, verify_registry_contracts
from studio.registry.base import ComponentSpec
from studio.renderers.hyperframes.compiler import compile_plan_to_html
from studio.renderers.scene_state import advance_scene_state, evaluate_plan_states, materialize_carry_objects
from studio.storage.cache import compute_dependency_fingerprints, validate_sidecar_bundle
from studio.timeline import AudioTimeline, MasterTimeline


class TestV38Suite(unittest.TestCase):
    def test_script_first_revisions_immutability_and_no_positional_fallback(self) -> None:
        beat1 = NarrationBeatIR(
            beat_id="ch_1__scene_1",
            chapter_id="ch_1",
            scene_id="scene_1",
            speaker="narrator",
            spoken_text="Imagine a <bookmark mark='b1'/>steady probability vector sliding along a line.",
            educational_purpose="Introduce equilibrium intuitively",
            visual_intent="point along line",
            anchor_marks=["b1"],
        )
        clean, offsets = split_anchors_and_spoken_text(beat1.spoken_text)
        self.assertNotIn("bookmark", clean)
        self.assertIn("b1", offsets)

        ir = ScriptIR(
            topic="Eigenvalues",
            title="Eigenvalues",
            narration_mode="single_educator",
            beats=[beat1],
        )
        approve_script_ir(ir, notes="Reviewed & verified")
        self.assertTrue(ir.locked)
        self.assertEqual(ir.approved_revision, 1)
        self.assertEqual(len(ir.revisions), 1)
        self.assertEqual(validate_script_ir(ir), [])

        # Modifying beat spoken text MUST create a new unapproved revision requiring re-review
        rev2 = revise_beat_text(ir, "ch_1__scene_1", "What happens when x_dot hits zero?")
        self.assertFalse(rev2.approved)
        self.assertFalse(ir.locked)
        issues = validate_script_ir(ir)
        self.assertTrue(any("unpronounced code identifier" in m for m in issues), f"Expected identifier catch, got {issues}")

        # Fix spoken text (preserving anchor tag b1 cleanly), approve rev 3, then test strict_ids rejection
        rev3 = revise_beat_text(ir, "ch_1__scene_1", "When velocity <bookmark mark='b1'/>drops to zero, the trajectory settles cleanly.")
        self.assertFalse(rev3.approved)
        approve_script_ir(ir, notes="Re-approved clean narration")
        self.assertTrue(ir.locked)

        mismatched_plan = VideoPlanV3(
            schema_version="3.0",
            topic="Eigenvalues",
            title="Eigenvalues",
            scenes=[
                SceneV3(
                    scene_id="wrong_id",
                    purpose="Wrong",
                    narration="Different text",
                    objects=[SceneObject(id="pt", kind="point", coords=[0.0, 0.0])],
                )
            ],
        )
        with self.assertRaises(ValueError):
            lock_script_to_plan(mismatched_plan, ir, strict_ids=True)

        # Narrative anti-pattern detectors
        self.assertIn("P_steady", detect_unpronounced_identifiers("The state reaches P_steady after ten steps."))
        self.assertTrue(
            detect_narrative_anti_patterns(
                ["Without needing to solve it explicitly, watch.", "Without needing to solve it explicitly, stop."]
            )
        )

    def test_enforceable_component_registry_contracts_and_certification(self) -> None:
        reg = get_default_registry()
        self.assertEqual(verify_registry_contracts(reg), [])

        uncertified = ComponentSpec(
            name="broken_widget",
            category="visual_object",
            description="Missing test_fn and handlers",
            supported_backends="both",
        )
        errs = uncertified.certification_errors()
        self.assertTrue(any("missing required executable test_fn" in e for e in errs))
        self.assertTrue(any("missing validate_fn" in e for e in errs))
        self.assertTrue(any("render_implementations" in e for e in errs))

        reg_copy = reg.clone() if hasattr(reg, "clone") else get_default_registry()
        reg_copy.register(uncertified)
        try:
            self.assertTrue(any("broken_widget" in e for e in verify_registry_contracts(reg_copy)))
            self.assertNotIn("broken_widget", reg_copy.capabilities_prompt_summary())
        finally:
            reg_copy._items.pop(("visual_object", "broken_widget"), None)

    def test_scene_runtime_state_persistence_and_hyperframes_transforms(self) -> None:
        plan = VideoPlanV3(
            schema_version="3.0",
            topic="Transforms",
            title="Transforms",
            scenes=[
                SceneV3(
                    scene_id="s1",
                    purpose="Move and remove",
                    narration="A square moves and a temporary point is removed.",
                    objects=[
                        SceneObject(id="sq", kind="polygon", vertices=[[0, 0], [1, 0], [1, 1], [0, 1]], persist=True),
                        SceneObject(id="temp_pt", kind="point", coords=[0.5, 0.5], persist=True),
                    ],
                    events=[
                        AnimationEvent(op="move", target_id="sq", shift=[2.0, 1.0], duration_s=0.8),
                        AnimationEvent(op="rotate", target_id="sq", angle_deg=90.0, duration_s=0.6),
                        AnimationEvent(op="remove", target_id="temp_pt", duration_s=0.4),
                    ],
                ),
                SceneV3(
                    scene_id="s2",
                    purpose="Continue carried geometry",
                    narration="Only the transformed square carries into scene two.",
                    objects=[SceneObject(id="badge", kind="equation", tex="R(90^\\circ)", card=False)],
                ),
            ],
        )
        states = evaluate_plan_states(plan)
        s2_in = states[0]
        self.assertIn("temp_pt", s2_in.removed_ids)
        self.assertIn("sq", s2_in.objects)
        sq_state = s2_in.objects["sq"]
        self.assertEqual(sq_state.rotation_deg, 90.0)
        xs = [round(v[0], 2) for v in sq_state.vertices]
        ys = [round(v[1], 2) for v in sq_state.vertices]
        self.assertAlmostEqual(sum(xs) / 4.0, 2.5, places=2)
        self.assertAlmostEqual(sum(ys) / 4.0, 1.5, places=2)

        carried = materialize_carry_objects(s2_in, {"badge", "sq", "temp_pt"})
        carried_ids = {o.id for o in carried}
        self.assertIn("sq", carried_ids)
        self.assertNotIn("temp_pt", carried_ids)

        # HyperFrames compiler preserves keyframed transforms on SVG elements
        design = default_design_spec(format_name="widescreen_hd", look_name="light")
        html_doc, dur_s = compile_plan_to_html(plan, design)
        self.assertGreater(dur_s, 3.0)
        self.assertIn('"transforms"', html_doc)
        self.assertIn('"angle_deg": 90.0', html_doc)

    def test_canonical_caption_ir_shared_provenance_and_sync_qa(self) -> None:
        beat = SyncedBeat(
            id="b1",
            narration="Watch probabilities settle onto equilibrium.",
            caption="Equilibrium",
            audio_path="",
            start_time=0.0,
            end_time=4.0,
            speech_duration=3.6,
            total_duration=4.0,
            words=[
                ["Watch", 0.0, 0.5, True],
                ["probabilities", 0.5, 1.3, True],
                ["settle", 1.3, 2.0, True],
                ["onto", 2.0, 2.5, True],
                ["equilibrium.", 2.5, 3.6, True],
            ],
            word_provenance=["forced_aligned", "forced_aligned", "forced_aligned", "forced_aligned", "forced_aligned"],
            speaker="narrator",
        )
        tl = AudioTimeline(
            beats=[beat],
            voice="kokoro",
            rate="0%",
            master_voice_wav="",
            total_duration=4.0,
        )
        cap_ir = build_caption_ir(tl.beats, style="editorial", width_px=1920, height_px=1080)
        self.assertEqual(cap_ir.timing_provenance_counts.get("forced_aligned"), 5)

        with tempfile.TemporaryDirectory() as tmp:
            tdir = Path(tmp)
            mtl = MasterTimeline(audio=tl)
            outs = mtl.export_subtitles(tdir, "out", caption_style="editorial", format_name="widescreen_hd", look_name="light")
            srt_p = Path(outs["srt"])
            ass_p = Path(outs["ass"])
            cap_p = tdir / "caption_ir.json"

            srt_txt = srt_p.read_text(encoding="utf-8")
            ass_txt = ass_p.read_text(encoding="utf-8")
            for cue in cap_ir.cues:
                self.assertIn(cue.text, srt_txt)
                self.assertIn(cue.text, ass_txt)

            chk = verify_caption_synchronization(caption_ir_path=cap_p, srt_path=srt_p, caption_zone_ok=True)
            self.assertEqual(chk.status, "passed")

            bad_chk = verify_caption_synchronization(caption_ir_path=cap_p, srt_path=srt_p, caption_zone_ok=False)
            self.assertEqual(bad_chk.status, "failed")

    def test_markov_stochastic_validation_and_strict_publication_gates(self) -> None:
        self.assertTrue(is_row_stochastic_matrix([[0.7, 0.3], [0.2, 0.8]])[0])
        self.assertFalse(is_row_stochastic_matrix([[0.7, 0.4], [0.2, 0.8]])[0])
        self.assertTrue(verify_stationary_distribution([[0.7, 0.3], [0.2, 0.8]], [0.4, 0.6])[0])
        self.assertFalse(verify_stationary_distribution([[0.7, 0.3], [0.2, 0.8]], [0.5, 0.5])[0])
        self.assertTrue(verify_graph_stochastic_weights(2, [["0", "0", "0.7"], ["0", "1", "0.3"], ["1", "0", "0.2"], ["1", "1", "0.8"]])[0])
        self.assertFalse(verify_graph_stochastic_weights(2, [["0", "0", "0.7"], ["0", "1", "0.5"], ["1", "0", "1.0"]])[0])

        # Valid Markov plan with an ungrounded narration claim MUST NEVER be publication_approved
        unsupported_plan = VideoPlanV3(
            schema_version="3.0",
            topic="Markov Test",
            title="Markov Test",
            scenes=[
                SceneV3(
                    scene_id="s1",
                    purpose="Stochastic check",
                    narration="Watch the Markov matrix converge to steady state, and remember the determinant stays positive!",
                    constructions=[
                        ConstructionOp(
                            op="markov_chain_evolution",
                            output_ids=["mk_bars"],
                            params=ConstructionParams(transition_matrix=[[0.7, 0.3], [0.2, 0.8]], n_trials=8, initial_state=[1.0, 0.0]),
                        )
                    ],
                    objects=[
                        SceneObject(
                            id="mat_p",
                            kind="matrix",
                            matrix_rows=[["0.7", "0.3"], ["0.2", "0.8"]],
                        ),
                        SceneObject(
                            id="mk_bars",
                            kind="bar_chart",
                            bar_labels=["S0", "S1"],
                            values=[0.4, 0.6],
                        ),
                    ],
                    claims=[
                        StructuredClaim(
                            kind="markov_stochastic_invariant",
                            object_ids=["mat_p", "mk_bars"],
                            statement="Row-stochastic matrix reaches stationary [0.4, 0.6]",
                        ),
                        StructuredClaim(
                            kind="eigenvalue_determinant",
                            object_ids=["mat_p"],
                            statement="Secondary matrix invariant requiring human/symbolic check",
                            verification_level="source_supported",
                        ),
                    ],
                )
            ],
        )
        rev = audit_educational_quality(
            topic="Markov Test",
            title="Markov Test",
            plan=unsupported_plan,
            format_name="widescreen_hd",
            look_name="light",
        )
        self.assertFalse(rev.approved)
        self.assertIn(rev.production_state, ("review_required", "rejected"))
        self.assertGreater(rev.verification_summary.unsupported + rev.verification_summary.human_review_required, 0)

        # Bad stationary bar values MUST be rejected by validate_v3_plan
        bad_markov = unsupported_plan.model_copy(deep=True)
        bad_markov.scenes[0].objects[1].values = [0.9, 0.1]
        bad_markov.scenes[0].claims = [bad_markov.scenes[0].claims[0]]
        issues = validate_v3_plan(bad_markov, require_claims=True)
        self.assertTrue(any("markov_stochastic_invariant" in i for i in issues), f"Expected rejection, got {issues}")

    def test_scene_aware_frame_qa_catches_blank_midpoint_scene(self) -> None:
        from PIL import Image, ImageDraw

        with tempfile.TemporaryDirectory() as tmp:
            tdir = Path(tmp)
            ff = find_ffmpeg()
            # Build 3x 3-second segments (9s total): s1 rich, s2 EMPTY solid dark, s3 rich
            frames_dir = tdir / "f"
            frames_dir.mkdir()
            for idx in range(9):
                img = Image.new("RGB", (640, 360), (247, 245, 240))
                dr = ImageDraw.Draw(img)
                if idx < 3 or idx >= 6:
                    dr.rectangle([120, 60, 520, 260], outline=(15, 76, 129), width=4)
                    dr.text((220, 150), f"MATH SCENE FRAME {idx}", fill=(17, 24, 39))
                img.save(frames_dir / f"img_{idx:03d}.png")

            mp4 = tdir / "test_seq.mp4"
            subprocess.run(
                [ff, "-y", "-framerate", "1", "-i", str(frames_dir / "img_%03d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p", str(mp4)],
                check=True,
                capture_output=True,
            )
            plan = VideoPlanV3(
                schema_version="3.0",
                topic="Frame Check",
                title="Frame Check",
                scenes=[
                    SceneV3(scene_id="s1", purpose="Opening", narration="One two three four five six."),
                    SceneV3(scene_id="s2_blank", purpose="Blank", narration="One two three four five six."),
                    SceneV3(scene_id="s3", purpose="Ending", narration="One two three four five six."),
                ],
            )
            tl = AudioTimeline(
                beats=[
                    SyncedBeat(id="s1", narration="a", caption="a", audio_path="", start_time=0.0, end_time=3.0, speech_duration=2.8, total_duration=3.0),
                    SyncedBeat(id="s2_blank", narration="b", caption="b", audio_path="", start_time=3.0, end_time=6.0, speech_duration=2.8, total_duration=3.0),
                    SyncedBeat(id="s3", narration="c", caption="c", audio_path="", start_time=6.0, end_time=9.0, speech_duration=2.8, total_duration=3.0),
                ],
                voice="kokoro",
                rate="0%",
                master_voice_wav="",
                total_duration=9.0,
            )
            sheet_p = tdir / "sheet.png"
            rep_p = tdir / "frames.json"
            res = inspect_video_scene_aware(mp4, look_name="light", plan=plan, timeline=tl, contact_sheet_out=sheet_p, report_out=rep_p)
            self.assertFalse(res.passed)
            self.assertTrue(
                any("s2_blank" in entry for entry in res.empty_midpoint_scenes),
                f"Expected s2_blank flagged in {res.empty_midpoint_scenes}",
            )
            self.assertTrue(sheet_p.exists())
            self.assertTrue(rep_p.exists())

    def test_dependency_fingerprints_and_local_scene_preview(self) -> None:
        fp = compute_dependency_fingerprints()
        for k in ("script_gen", "math_compute", "math_validate", "captions", "tts_voice", "design_tokens", "fonts", "components", "manimgx_backend", "hyperframes_backend", "layout_solver", "audio_master"):
            self.assertIn(k, fp)
            self.assertEqual(len(fp[k]), 12)

        with tempfile.TemporaryDirectory() as tmp:
            tdir = Path(tmp)
            ok_json = tdir / "a.json"
            ok_json.write_text('{"ok": true}', encoding="utf-8")
            self.assertTrue(validate_sidecar_bundle([ok_json])[0])
            bad_json = tdir / "bad.json"
            bad_json.write_text("{broken", encoding="utf-8")
            self.assertFalse(validate_sidecar_bundle([bad_json])[0])

            plan = VideoPlanV3(
                schema_version="3.0",
                topic="Preview Flow",
                title="Preview Flow",
                scenes=[
                    SceneV3(
                        scene_id="s1",
                        purpose="Setup axes",
                        narration="Draw the coordinate plane and keep the curve.",
                        objects=[
                            SceneObject(id="ax", kind="axes", x_range=[-3, 3], y_range=[-2, 2], persist=True),
                            SceneObject(id="fn", kind="function_plot", axes_id="ax", expr="0.3*x**2", x_range=[-3, 3], persist=True),
                        ],
                    ),
                    SceneV3(
                        scene_id="s2",
                        purpose="Inspect carried curve",
                        narration="Notice how the carried parabola behaves cleanly.",
                        objects=[SceneObject(id="eq", kind="equation", tex="y = 0.3 x^2", card=False)],
                    ),
                ],
            )
            sliced, st_info = slice_plan_for_scene(plan, "s2")
            self.assertEqual(len(sliced.scenes), 1)
            self.assertIn("ax", st_info["carried_object_ids"])
            self.assertIn("fn", st_info["carried_object_ids"])

            plan_p = tdir / "plan.json"
            plan_p.write_text(plan.model_dump_json(indent=2), encoding="utf-8")
            idx_p = build_scene_preview_bundle(plan_p, out_dir=tdir / "prev", look_name="light")
            self.assertTrue(idx_p.exists())
            self.assertTrue((tdir / "prev" / "preview_manifest.json").exists())

            srv, url = start_preview_server(idx_p.parent, port=0)
            try:
                self.assertTrue(url.startswith("http://127.0.0.1:"))
                with urlopen(url, timeout=3) as resp:
                    self.assertEqual(resp.status, 200)
                    body = resp.read().decode("utf-8")
                    self.assertIn("Storyboard Scenes", body)
            finally:
                srv.shutdown()
                srv.server_close()


if __name__ == "__main__":
    unittest.main()
