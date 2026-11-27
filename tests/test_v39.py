"""V3.9 regression test suite covering all 15 correctness, QA, provenance, KaTeX, audio & linked-model checks."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from studio.ai.linked_models import LinkedMathModelSpec, evaluate_signal_spectral_pipeline
from studio.ai.v3_schema import (
    AnimationEvent,
    ClaimLedgerEntry,
    SceneObject,
    SceneV3,
    StructuredClaim,
    VideoPlanV3,
)
from studio.ai.validate_v3 import validate_v3_plan, verify_v3_plan_detailed
from studio.ai.world_model import materialize_constructions
from studio.audio.demo_signals import synthesize_math_audio_demo
from studio.audio.tts import SyncedBeat
from studio.captions import build_caption_ir, merge_chapter_caption_irs
from studio.captions.layout import extract_semantic_emphasis_tokens
from studio.captions.model import CaptionCue, CaptionIR, CaptionWord
from studio.design.spec import default_design_spec
from studio.provenance import build_and_write_manifest, verify_manifest_matches_mp4
from studio.qa.engine import audit_educational_quality, verify_caption_synchronization
from studio.qa.frames import FrameAuditResult, plan_scene_aware_timestamps
from studio.renderers.scene_state import SceneRuntimeState, advance_scene_state
from studio.registry import get_default_registry, verify_registry_contracts
from studio.renderers.hyperframes.compiler import compile_plan_to_html
from studio.renderers.hyperframes.typography import verify_vendored_katex


class TestV39Regressions(unittest.TestCase):
    def _linked_fourier_plan(self) -> VideoPlanV3:
        model = LinkedMathModelSpec(
            model_id="sig_mix",
            kind="signal_spectral_pipeline",
            expr="0.9*cos(2*pi*2*t) + 0.6*cos(2*pi*5*t)",
            t_range=[0.0, 2.0],
            freq_range=[0.5, 6.5],
            selected_freq=2.0,
            cutoff_freq=3.5,
            filter_kind="lowpass",
            angular_units="hz",
        )
        return VideoPlanV3(
            topic="Fourier Linked Signal Analysis",
            title="Linked Winding & Filtering",
            math_models=[model],
            scenes=[
                SceneV3(
                    scene_id="s1_winding",
                    purpose="Wind signal at 2 Hz and track trapezoidal centroid",
                    narration="Wrapping the signal at two hertz pulls its center of mass away from zero.",
                    composition="multi_view_dashboard",
                    objects=[
                        SceneObject(id="ax_w", kind="axes", x_range=[-1.6, 1.6], y_range=[-1.6, 1.6], equal_aspect=True, panel_slot="left", x_label="Re z", y_label="Im z"),
                        SceneObject(id="w_curve", kind="data_series", model_id="sig_mix", model_output="winding_curve", axes_id="ax_w"),
                        SceneObject(id="w_ctr", kind="point", model_id="sig_mix", model_output="centroid_point", axes_id="ax_w", color_role="highlight"),
                        SceneObject(id="ax_sp", kind="axes", x_range=[0.5, 6.5], y_range=[0.0, 0.65], panel_slot="right", x_label="Freq (Hz)", y_label="|C(f)|"),
                        SceneObject(id="sp_curve", kind="data_series", model_id="sig_mix", model_output="spectrum", axes_id="ax_sp", color_role="secondary"),
                        SceneObject(id="flt_sig", kind="data_series", model_id="sig_mix", model_output="filtered_signal", axes_id="ax_sp", color_role="tertiary"),
                    ],
                    claims=[
                        StructuredClaim(kind="fourier_winding_centroid", model_id="sig_mix", object_ids=["w_curve", "w_ctr"], statement="Centroid C(w) equals trapezoidal integral of x(t)e^{-iwt}"),
                        StructuredClaim(kind="spectral_derivation_consistency", model_id="sig_mix", object_ids=["sp_curve"], statement="Spectrum peak derives from same x(t)"),
                        StructuredClaim(kind="signal_filtering_equivalence", model_id="sig_mix", object_ids=["flt_sig"], statement="Low-pass removes 5 Hz energy"),
                        StructuredClaim(kind="parseval_energy_conservation", model_id="sig_mix", object_ids=["w_curve"], statement="Time and DFT energy match"),
                    ],
                    claim_ledger=[
                        ClaimLedgerEntry(
                            claim_id="cl_1",
                            exact_narration_span="Wrapping the signal at two hertz pulls its center of mass",
                            proposition="|C(2 Hz)| > 0",
                            assumptions="T=2.0s observation window",
                            model_id="sig_mix",
                            object_ids=["w_curve", "w_ctr"],
                            status="deterministically_verified",
                        )
                    ],
                    events=[
                        AnimationEvent(op="draw", target_id="w_curve"),
                        AnimationEvent(op="animate_param", target_id="w_curve", param_range=[0.5, 6.5]),
                    ],
                )
            ],
        )

    def test_01_cartesian_fake_winding_rejected(self) -> None:
        bad = VideoPlanV3(
            topic="Fourier Winding",
            title="Fake Winding",
            scenes=[
                SceneV3(
                    scene_id="s1",
                    purpose="Fake winding",
                    narration="Winding the signal around the origin reveals its center of mass at a Fourier frequency.",
                    objects=[
                        SceneObject(id="ax", kind="axes", x_range=[-3, 3], y_range=[-2, 2]),
                        SceneObject(id="fake_w", kind="function_plot", expr="cos(x)", axes_id="ax"),
                    ],
                    events=[AnimationEvent(op="draw", target_id="fake_w")],
                )
            ],
        )
        issues = validate_v3_plan(bad)
        self.assertTrue(any("1d cos/sin" in i.lower() or "winding_complex_curve" in i.lower() for i in issues), issues)

    def test_02_disconnected_data_and_point_mismatch_rejected(self) -> None:
        plan = self._linked_fourier_plan()
        mat = materialize_constructions(plan)
        # Tamper centroid coordinates so visual point diverges from numerical integral
        mat.scenes[0].objects[2].coords = [0.99, -0.99]
        mat.scenes[0].objects[2].model_output = None
        det = verify_v3_plan_detailed(mat, require_claims=True)
        self.assertTrue(det["failed"], f"Expected centroid mismatch failure, got {det}")

    def test_03_trapezoidal_centroid_and_spectral_peak_accuracy(self) -> None:
        spec = LinkedMathModelSpec(
            model_id="sig_2_5",
            kind="signal_spectral_pipeline",
            expr="0.9*cos(2*pi*2*t) + 0.5*cos(2*pi*5*t)",
            t_range=[0.0, 2.0],
            freq_range=[0.5, 6.5],
            selected_freq=2.0,
            cutoff_freq=3.5,
            filter_kind="lowpass",
            angular_units="hz",
        )
        res = evaluate_signal_spectral_pipeline(spec)
        # Amplitude 0.9 real cosine gives |C(2 Hz)| ~= 0.45
        self.assertAlmostEqual(float(res["centroid_point"][0]), 0.45, delta=0.03)
        self.assertAlmostEqual(float(res["spectrum_peak"][0]), 2.0, delta=0.15)
        self.assertLess(float(res["measurements"]["parseval_rel_error"]), 1e-4)

    def test_04_unverified_filtering_and_spectral_mismatch_caught(self) -> None:
        plan = self._linked_fourier_plan()
        # Filter cutoff above all frequencies (cutoff=10 Hz) leaves energy unchanged -> filtering equivalence should fail!
        plan.math_models[0].cutoff_freq = 10.0
        det = verify_v3_plan_detailed(plan, require_claims=True)
        self.assertTrue(any("filtering" in f.lower() for f in det["failed"]), det)

    def test_05_cue_overlap_catches_prev_end_bug(self) -> None:
        c1 = CaptionCue(cue_id="c1", beat_id="b1", start_s=0.0, end_s=2.4, lines=["First cue line"], words=[CaptionWord(text="First", start_s=0.0, end_s=1.0)])
        # c2 starts AFTER c1.start_s (1.5 > 0.0) but BEFORE c1.end_s (1.5 < 2.4)! Old check missed this!
        c2 = CaptionCue(cue_id="c2", beat_id="b2", start_s=1.5, end_s=3.2, lines=["Second cue line"], words=[CaptionWord(text="Second", start_s=1.5, end_s=2.5)])
        ir = CaptionIR(style="editorial", aspect_label="16:9", width_px=1920, height_px=1080, cues=[c1, c2])
        res = verify_caption_synchronization(caption_ir=ir)
        self.assertFalse(res.passed)
        self.assertIn("overlap", res.detail.lower())

    def test_06_incomplete_doc_caption_ir_caught_and_global_merge_works(self) -> None:
        b1 = SyncedBeat(id="c1__s1", narration="One two", caption="One", audio_path="", start_time=0.0, end_time=1.0, speech_duration=0.9, total_duration=1.0, words=[["One", 0.0, 0.4, False], ["two", 0.4, 0.9, False]], marks={})
        b2 = SyncedBeat(id="c2__s1", narration="Three four", caption="Three", audio_path="", start_time=0.0, end_time=1.0, speech_duration=0.9, total_duration=1.0, words=[["Three", 0.0, 0.4, False], ["four", 0.4, 0.9, False]], marks={})
        ir1 = build_caption_ir([b1])
        ir2 = build_caption_ir([b2])
        # Incomplete documentary check fails when passing only 1 chapter of 1s for a 10s multi-beat documentary
        bad_chk = verify_caption_synchronization(caption_ir=ir1, expected_duration_s=10.0, expected_beat_count=2)
        self.assertFalse(bad_chk.passed)
        merged = merge_chapter_caption_irs([("c1", 0.0, ir1), ("c2", 1.2, ir2)])
        self.assertEqual(len(merged.cues), 2)
        self.assertGreater(merged.cues[1].start_s, merged.cues[0].end_s)
        ok_chk = verify_caption_synchronization(caption_ir=merged, expected_duration_s=2.2, expected_beat_count=2)
        self.assertTrue(ok_chk.passed, ok_chk.detail)

    def test_07_stale_manifest_and_mp4_sha_mismatch_detected(self) -> None:
        plan = self._linked_fourier_plan()
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            mp4 = tdp / "demo.mp4"
            mp4.write_bytes(b"FAKE_MP4_VERSION_1")
            _manifest, mpath = build_and_write_manifest(
                slug="demo",
                topic=plan.topic,
                title=plan.title,
                mp4_path=mp4,
                plan=plan,
                backend="hyperframes",
                format_name="widescreen_hd",
                look_name="editorial_light",
                caption_style="editorial",
                build_dir=tdp,
            )
            ok1, errs1 = verify_manifest_matches_mp4(mpath, mp4)
            self.assertTrue(ok1, errs1)
            # Overwrite MP4 silently (simulating the V3.8 stale telemetry discrepancy)
            mp4.write_bytes(b"OVERWRITTEN_DIFFERENT_MP4")
            ok2, mismatches = verify_manifest_matches_mp4(mpath, mp4)
            self.assertFalse(ok2)
            self.assertTrue(any("sha" in m.lower() or "mismatch" in m.lower() for m in mismatches), mismatches)

    def test_08_missing_scene_qa_coverage_and_empty_metadata_caught(self) -> None:
        # Build a 10-scene plan and verify plan_scene_aware_timestamps covers all 10 scenes without truncation
        scenes = [
            SceneV3(
                scene_id=f"c{idx // 3 + 1}__scene_{idx}",
                purpose=f"Beat {idx}",
                narration=f"Step {idx} demonstrates linked math.",
                objects=[SceneObject(id=f"ax_{idx}", kind="axes", x_range=[-2, 2], y_range=[-2, 2]), SceneObject(id=f"pt_{idx}", kind="point", coords=[0.0, 0.0], axes_id=f"ax_{idx}")],
                events=[AnimationEvent(op="draw", target_id=f"pt_{idx}")],
            )
            for idx in range(1, 11)
        ]
        big_plan = VideoPlanV3(topic="Coverage Test", title="Coverage Test", scenes=scenes)
        targets = plan_scene_aware_timestamps(45.0, plan=big_plan, timeline=None)
        covered = {t.scene_id for t in targets if t.scene_id and t.scene_id.startswith("c")}
        self.assertEqual(len(covered), 10)
        # Also verify audit_educational_quality refuses unverified plan without publication metadata
        rep = audit_educational_quality(topic="T", title="T", plan=big_plan, format_name="widescreen_hd")
        self.assertFalse(rep.approved)  # type: ignore[attr-defined]
        self.assertNotEqual(rep.production_state, "publication_approved")  # type: ignore[attr-defined]

    def test_09_self_declared_source_supported_requires_review_and_ledger_span_checked(self) -> None:
        plan = self._linked_fourier_plan()
        plan.scenes[0].claims.append(
            StructuredClaim(kind="right_triangle", object_ids=["w_curve"], verification_level="source_supported")
        )
        plan.scenes[0].claim_ledger.append(
            ClaimLedgerEntry(claim_id="bad_span", exact_narration_span="totally invented text not in narration", proposition="X", status="source_supported")
        )
        det = verify_v3_plan_detailed(plan, require_claims=True)
        cov = det["verification_coverage"]
        self.assertEqual(cov["human_review_required"], 1)  # type: ignore[index]
        self.assertTrue(any("exact_narration_span" in str(i) for i in det["issues"]))  # type: ignore[arg-type]

    def test_10_executable_registry_outputs_and_contracts(self) -> None:
        self.assertEqual(verify_registry_contracts(), [])
        reg = get_default_registry()
        spec = reg.get("linked_signal_spectral_pipeline")
        self.assertIsNotNone(spec)
        assert spec is not None
        self.assertTrue(spec.validate_outputs({"model_id": "bad"}))  # missing required kind fails typed output validation
        self.assertFalse(spec.validate_outputs(LinkedMathModelSpec(model_id="ok", kind="signal_spectral_pipeline")))

    def test_11_offline_katex_vendored_and_hyperframes_compiles_with_metadata(self) -> None:
        self.assertEqual(verify_vendored_katex(), [])
        plan = self._linked_fourier_plan()
        plan.scenes[0].objects.append(SceneObject(id="eq_f", kind="equation", tex=r"\hat{x}(\omega)=\int x(t)e^{-i\omega t}\,dt", card=True))
        with tempfile.TemporaryDirectory() as td:
            meta_p = Path(td) / "scene_frame_metadata.json"
            html_doc, _ = compile_plan_to_html(
                plan,
                default_design_spec("widescreen_hd", "editorial_light"),
                caption_style="editorial",
                meta_out_path=meta_p,
            )
            self.assertIn("/vendor/katex/katex.min.js", html_doc)
            self.assertIn("sweep_frames_px", html_doc)
            self.assertTrue(meta_p.exists())
            meta = json.loads(meta_p.read_text(encoding="utf-8"))
            self.assertGreater(len(meta["s1_winding"]), 0)

    def test_12_semantic_caption_emphasis_never_highlights_stopwords(self) -> None:
        text = "Every signal can be written by combining Fourier frequency components with a clear centroid."
        emph = extract_semantic_emphasis_tokens(text)
        for sw in ("every", "can", "be", "written", "by", "with", "a"):
            self.assertNotIn(sw, emph)
        for kw in ("fourier", "frequency", "centroid"):
            self.assertIn(kw, emph)

    def test_13_synthesized_audio_demo_wav_nonempty(self) -> None:
        spec = LinkedMathModelSpec(model_id="aud", kind="signal_spectral_pipeline", expr="cos(2*pi*2*t)+0.7*cos(2*pi*6*t)", cutoff_freq=3.5)
        with tempfile.TemporaryDirectory() as td:
            wav_p = synthesize_math_audio_demo(spec, Path(td) / "demo.wav", variant="raw_vs_filtered", duration_s=1.8)
            self.assertTrue(wav_p.exists())
            with wave.open(str(wav_p), "rb") as rf:
                self.assertEqual(rf.getframerate(), 48000)
                self.assertEqual(rf.getnchannels(), 2)
                self.assertGreater(rf.getnframes(), 48000)

    def test_14_no_ext_post_lock_bypass_in_longform(self) -> None:
        lf_src = (ROOT / "studio" / "longform.py").read_text(encoding="utf-8")
        self.assertNotIn("doc_script.locked = False", lf_src)

    def test_15_duplicate_scene_ids_across_plan_rejected(self) -> None:
        dup = VideoPlanV3(
            topic="Dup IDs",
            title="Dup IDs",
            scenes=[
                SceneV3(scene_id="ch1__s1", purpose="A", narration="First scene.", objects=[SceneObject(id="eq1", kind="equation", tex="x")]),
                SceneV3(scene_id="ch1__s1", purpose="B", narration="Second scene.", objects=[SceneObject(id="eq2", kind="equation", tex="y")]),
            ],
        )
        issues = validate_v3_plan(dup)
        self.assertTrue(any("unique" in i.lower() or "scene_id" in i.lower() for i in issues), issues)

    def test_16_scene_runtime_state_parity_across_renderer_boundaries(self) -> None:
        st = SceneRuntimeState()
        sc1 = SceneV3(
            scene_id="ch1__s1",
            purpose="Draw vector",
            narration="Vector v stands here.",
            objects=[
                SceneObject(id="ax1", kind="axes", x_range=[-2, 2], y_range=[-2, 2], persist=True),
                SceneObject(id="vec1", kind="vector", coords=[0.0, 0.0], end_coords=[1.0, 0.0], axes_id="ax1", persist=True),
                SceneObject(id="tmp_note", kind="annotation", text="Temp", persist=False),
            ],
            events=[
                AnimationEvent(op="draw", target_id="vec1"),
                AnimationEvent(op="zoom", target_id="ax1", scale_factor=1.35),
            ],
        )
        st = advance_scene_state(st, sc1)
        self.assertIn("vec1", st.objects)
        self.assertTrue(st.objects["vec1"].visible)
        self.assertAlmostEqual(st.camera_zoom, 1.35)
        self.assertFalse(st.objects["tmp_note"].visible)
        # Verify stale non-visible/removed object targeted in next scene triggers validation error
        bad_sc2 = VideoPlanV3(
            topic="State",
            title="State",
            scenes=[
                sc1,
                SceneV3(scene_id="ch1__s2", purpose="Move removed note", narration="Move temp.", events=[AnimationEvent(op="move", target_id="tmp_note", shift=[1.0, 0.0])]),
            ],
        )
        self.assertTrue(validate_v3_plan(bad_sc2))

    def test_17_reusable_non_fourier_linked_models_execute_deterministically(self) -> None:
        opt_plan = VideoPlanV3(
            topic="Gradient Optimization",
            title="Optimization Linked Model",
            math_models=[LinkedMathModelSpec(model_id="opt1", kind="optimization_trajectory", expr="0.25*x**4 - x**2", x0=2.1, lr=0.12, steps=8)],
            scenes=[
                SceneV3(
                    scene_id="s1",
                    purpose="Plot curve and optimization steps",
                    narration="Each gradient step lowers the objective along the slope.",
                    objects=[
                        SceneObject(id="ax", kind="axes", x_range=[-3, 3], y_range=[-2, 4]),
                        SceneObject(id="obj_c", kind="data_series", model_id="opt1", model_output="curve", axes_id="ax"),
                        SceneObject(id="traj_c", kind="data_series", model_id="opt1", model_output="trajectory", axes_id="ax", color_role="highlight"),
                    ],
                    events=[AnimationEvent(op="draw", target_id="obj_c"), AnimationEvent(op="draw", target_id="traj_c")],
                )
            ],
        )
        mat = materialize_constructions(opt_plan)
        self.assertGreater(len(mat.scenes[0].objects[1].points_list), 50)
        self.assertEqual(len(mat.scenes[0].objects[2].points_list), 9)
        self.assertEqual(validate_v3_plan(mat), [])

    def test_18_gemini_json_mode_cache_and_hybrid_speed_paths(self) -> None:
        import inspect
        from types import SimpleNamespace
        from studio.providers.gemini import GeminiProvider, _COMPACT_SCHEMA_CACHE, _JSON_MODE_ONLY_SCHEMAS
        from studio.renderers.hyperframes.assembly import HyperFramesBackend

        self.assertIn("VideoPlanV3", _JSON_MODE_ONLY_SCHEMAS)
        sig_hf = inspect.signature(HyperFramesBackend.render)
        self.assertIn("video_only", sig_hf.parameters)

        # Verify GeminiProvider skips constrained-schema roundtrip for VideoPlanV3 in a single call
        calls: list[dict] = []
        valid_plan = VideoPlanV3(
            topic="T",
            title="T",
            scenes=[SceneV3(scene_id="s1", purpose="P", narration="Explain the ratio cleanly.")],
        ).model_dump_json()

        class _FakeModels:
            def generate_content(self, *, model, contents, config):
                calls.append({"model": model, "has_response_schema": bool(getattr(config, "response_schema", None) or getattr(config, "response_json_schema", None))})
                return SimpleNamespace(text=valid_plan, usage_metadata=SimpleNamespace(prompt_token_count=10, candidates_token_count=20))

        prov = GeminiProvider(api_key="test-key")
        prov._client = SimpleNamespace(models=_FakeModels())
        parsed, _resp = prov.generate_structured("Make plan", VideoPlanV3, system="SYS")
        self.assertEqual(len(calls), 1)
        self.assertFalse(calls[0]["has_response_schema"])
        self.assertIn("VideoPlanV3", _COMPACT_SCHEMA_CACHE)
        self.assertEqual(parsed.topic, "T")


if __name__ == "__main__":
    unittest.main()
