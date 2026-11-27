"""V4.0 simulation-driven filmmaking, linked parameter sync, spectral accuracy, caption & manifest test suite."""

from __future__ import annotations

import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from studio.ai.linked_models import LinkedMathModelSpec, evaluate_linked_math_model, find_spectral_peaks
from studio.ai.simulation import compute_driven_frequency_response, integrate_ode_system
from studio.ai.v3_schema import AnimationEvent, SceneObject, SceneV3, StructuredClaim, VideoPlanV3
from studio.ai.validate_v3 import validate_v3_plan
from studio.audio.tts import SyncedBeat, align_words_with_provenance, find_ffmpeg, parse_script, sanitize_word_boundaries
from studio.captions import build_caption_ir
from studio.captions.layout import chunk_caption_cues, extract_semantic_emphasis_tokens
from studio.design.spec import default_design_spec
from studio.provenance import (
    build_and_write_manifest,
    finalize_manifest_qa,
    inspect_manifest_verification,
    load_font_versions,
    probe_mp4_streams,
    verify_manifest_matches_mp4,
)
from studio.qa.engine import audit_educational_quality, verify_caption_synchronization
from studio.registry import verify_registry_contracts
from studio.renderers.hyperframes.compiler import compile_plan_to_html
from studio.renderers.scene_state import evaluate_plan_states


class TestV40Regressions(unittest.TestCase):
    def test_01_kokoro_trailing_bookmark_punctuation_and_raw_word_repair(self) -> None:
        clean, tokens, marks = parse_script("They eventually <bookmark mark='stop'/>.")
        self.assertEqual(clean, "They eventually.")
        self.assertEqual([t.text for t in tokens], ["They", "eventually."])
        self.assertGreaterEqual(marks["stop"], 1)

        # Simulate Kokoro emitting an inverted interval on trimmed trailing punctuation
        raw = [["They", 0.0, 0.35], ["eventually", 0.36, 4.75], [".", 5.1835, 4.8029]]
        repaired, repairs = sanitize_word_boundaries(raw, duration_s=4.8029, beat_id="c2__cue_0003")
        self.assertEqual(len(repaired), 2)
        self.assertEqual(repaired[1][0], "eventually.")
        self.assertLessEqual(repaired[1][1], repaired[1][2])
        self.assertLessEqual(repaired[1][2], 4.8029)
        self.assertTrue(repairs)

        aligned, prov = align_words_with_provenance(clean, tokens, raw, speech_dur=4.8029)
        self.assertEqual(len(aligned), 2)
        for s_t, e_t in aligned:
            self.assertGreaterEqual(s_t, 0.0)
            self.assertLessEqual(s_t, e_t)
            self.assertLessEqual(e_t, 4.803)
        self.assertEqual(len(prov), 2)

        beat = SyncedBeat(
            id="c2__cue_0003",
            narration="They eventually.",
            caption="Damping settles",
            audio_path=Path("c2__cue_0003.wav"),
            start_time=0.0,
            end_time=4.803,
            speech_duration=4.803,
            total_duration=4.9,
            words=[["They", aligned[0][0], aligned[0][1]], ["eventually.", aligned[1][0], aligned[1][1]]],
            raw_word_boundaries=raw,
            timing_repairs=repairs,
            word_provenance=prov,
        )
        ir = build_caption_ir([beat])
        chk = verify_caption_synchronization(caption_ir=ir)
        self.assertTrue(chk.passed, chk.detail)

    def test_02_strict_semantic_captions_exclude_ordinary_prose(self) -> None:
        prose_text = "We *capture* the *rest* and *repeatedly* test a *strong* *reveal* unless we *refuse* *resonance* in *Fourier* space e^(i*theta)."
        emph = extract_semantic_emphasis_tokens(prose_text)
        for bad in ("capture", "rest", "repeatedly", "strong", "reveal", "refuse"):
            self.assertNotIn(bad, emph, f"Ordinary prose word {bad!r} must NOT receive caption pill styling")
        self.assertTrue({"resonance", "fourier"} & emph, f"Expected domain terms in {emph}")

        beat = SyncedBeat(
            id="b1",
            narration=prose_text,
            caption="Concept check",
            audio_path=Path("b1.wav"),
            start_time=0.0,
            end_time=4.0,
            speech_duration=3.8,
            total_duration=4.0,
            words=[[w, i * 0.2, i * 0.2 + 0.18] for i, w in enumerate(prose_text.replace("*", "").split())],
        )
        cues = chunk_caption_cues([beat])
        emph_words = {w.text.lower().strip(".,") for c in cues for w in c.words if w.emphasis}
        for bad in ("capture", "rest", "repeatedly", "strong", "reveal", "refuse"):
            self.assertNotIn(bad, emph_words)

    def test_03_manifest_completeness_fonts_streams_and_certification(self) -> None:
        fonts = load_font_versions()
        self.assertGreater(len(fonts), 0, "Expected pinned font hashes in FONT_MANIFEST.json")
        self.assertTrue(any("katex" in k.lower() for k in fonts), "Expected vendored KaTeX asset hashes")

        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            mp4 = tdp / "tiny_real.mp4"
            ff = find_ffmpeg()
            subprocess.run(
                [
                    ff,
                    "-y",
                    "-f",
                    "lavfi",
                    "-i",
                    "color=c=#111827:s=640x360:r=60:d=1.2",
                    "-f",
                    "lavfi",
                    "-i",
                    "sine=frequency=440:sample_rate=48000:duration=1.2",
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-c:a",
                    "aac",
                    "-b:a",
                    "128k",
                    "-shortest",
                    str(mp4),
                ],
                check=True,
                capture_output=True,
            )
            streams = probe_mp4_streams(mp4)
            self.assertEqual(streams.get("width"), 640)
            self.assertEqual(streams.get("height"), 360)
            self.assertEqual(streams.get("sample_rate"), 48000)
            self.assertGreater(float(streams.get("v_dur") or 0.0), 1.0)

            plan = VideoPlanV3(
                topic="Mini Test",
                title="Mini",
                target_duration_s=1.2,
                scenes=[
                    SceneV3(
                        scene_id="s1",
                        purpose="Check manifest",
                        narration="Testing live MP4 stream verification.",
                        objects=[SceneObject(id="title_t", kind="annotation", text="Mini")],
                        events=[AnimationEvent(op="reveal", target_id="title_t")],
                    )
                ],
            )
            mf, mf_p = build_and_write_manifest(
                slug="mini",
                topic="Mini Test",
                title="Mini",
                mp4_path=mp4,
                plan=plan,
                backend="hyperframes",
                format_name="widescreen_hd",
                look_name="midnight_studio",
                caption_style="editorial",
                build_dir=tdp,
                telemetry_summary={"cold_or_warm": "cold", "total_wall_s": 1.5},
            )
            self.assertTrue(mf.identity_verified)
            self.assertFalse(mf.production_certified)
            ok_un, _ = verify_manifest_matches_mp4(mf_p, mp4, require_certification=False)
            self.assertTrue(ok_un)
            ok_cert_fail, cert_errs = verify_manifest_matches_mp4(mf_p, mp4, require_certification=True)
            self.assertFalse(ok_cert_fail)
            self.assertTrue(cert_errs)

            class _FakeRev:
                production_state = "publication_approved"
                overall_score = 92.5
                approved = True

            rev_p = tdp / "REVIEW.md"
            rev_p.write_text("# Approved Review\n", encoding="utf-8")
            mf_final = finalize_manifest_qa(mf_p, _FakeRev(), mp4_path=mp4, review_path=rev_p, build_dir=tdp)
            self.assertTrue(mf_final.production_certified)
            info = inspect_manifest_verification(mf_p, mp4, require_certification=True)
            self.assertTrue(info["production_certified"], info)

    def _sweep_reset_plan(self, include_reset: bool = True) -> VideoPlanV3:
        model = LinkedMathModelSpec(
            model_id="sig_mix",
            kind="signal_spectral_pipeline",
            expr="0.85*cos(2*pi*2*t) + 0.55*cos(2*pi*5*t)",
            t_range=[0.0, 2.0],
            freq_range=[0.5, 6.8],
            selected_freq=2.0,
            cutoff_freq=3.5,
            filter_kind="lowpass",
            angular_units="hz",
            n_samples=220,
        )
        s3_events = [AnimationEvent(op="create", target_id="ctr_vec")]
        s3_active = {"sig_mix": 2.0} if include_reset else {}
        if include_reset:
            s3_events.append(AnimationEvent(op="set_param", target_id="w_curve", target_param=2.0))
        return VideoPlanV3(
            topic="Fourier Linked Sync",
            title="Sync Check",
            target_duration_s=18.0,
            math_models=[model],
            scenes=[
                SceneV3(
                    scene_id="s1",
                    purpose="Signal",
                    narration="Look at the sum of two frequencies.",
                    objects=[
                        SceneObject(id="ax", kind="axes", x_range=[0, 2, 0.5], y_range=[-2, 2, 1]),
                        SceneObject(id="sig_c", kind="function_plot", axes_id="ax", expr="cos(x)"),
                    ],
                    events=[
                        AnimationEvent(op="create", target_id="ax"),
                        AnimationEvent(op="create", target_id="sig_c"),
                    ],
                ),
                SceneV3(
                    scene_id="s2_sweep",
                    purpose="Sweep up to 6.8 Hz",
                    narration="Sweeping across frequencies from 0.5 to 6.8 Hz.",
                    objects=[
                        SceneObject(id="w_ax", kind="axes", x_range=[-1.5, 1.5, 0.5], y_range=[-1.5, 1.5, 0.5], persist=True),
                        SceneObject(id="w_curve", kind="parametric_curve", axes_id="w_ax", model_id="sig_mix", model_output="winding_curve", persist=True),
                    ],
                    events=[
                        AnimationEvent(op="create", target_id="w_ax"),
                        AnimationEvent(op="create", target_id="w_curve"),
                        AnimationEvent(op="animate_param", target_id="w_curve", duration_s=3.0),
                    ],
                ),
                SceneV3(
                    scene_id="s3_centroid",
                    purpose="Lock 2 Hz match",
                    narration="At 2 Hz the centroid pulls right.",
                    active_model_params=s3_active,
                    objects=[
                        SceneObject(id="ctr_vec", kind="vector", axes_id="w_ax", model_id="sig_mix", model_output="centroid_vector", persist=True),
                    ],
                    events=s3_events,
                ),
            ],
        )

    def test_04_fourier_sweep_parameter_reset_and_continuity_both_renderers(self) -> None:
        good_plan = self._sweep_reset_plan(include_reset=True)
        model_res = evaluate_linked_math_model(good_plan.math_models[0])
        frames = model_res["sweep_frames"]
        sel_u = float(model_res["measurements"]["selected_sweep_u"])
        freqs = [float(f["param"]) for f in frames]
        self.assertIn(2.0, freqs, "Exact selected_freq=2.0 Hz must appear in sweep_frames")
        idx_2 = freqs.index(2.0)
        self.assertAlmostEqual(float(frames[idx_2]["u"]), sel_u, places=5)

        states = evaluate_plan_states(good_plan)
        self.assertEqual(states[-1].model_params.get("sig_mix"), 2.0)

        html, _ = compile_plan_to_html(good_plan, default_design_spec(look_name="midnight_studio"))
        self.assertIn("sweep_segments", html)
        self.assertIn("sweepUFor", html)
        import re as _re

        m_data = _re.search(r"const DATA = (\{.*?\});\n", html)
        self.assertIsNotNone(m_data)
        data_obj = json.loads(m_data.group(1))
        w_el = next(e for e in data_obj["elements"] if e["id"] == "w_curve")
        segs = w_el.get("sweep_segments") or []
        self.assertGreaterEqual(len(segs), 2, segs)
        self.assertAlmostEqual(float(segs[-1]["u1"]), sel_u, places=4)

        # Negative test: omitting parameter reset after sweeping to 6.8 Hz must be flagged
        bad_plan = self._sweep_reset_plan(include_reset=False)
        issues = validate_v3_plan(bad_plan, require_claims=False)
        self.assertTrue(any("selected_freq" in m or "active_model_params" in m or "6.8" in m for m in issues), issues)

    def test_05_dft_endpoint_exclusive_spectral_peaks_and_filtering(self) -> None:
        spec = LinkedMathModelSpec(
            model_id="sig",
            kind="signal_spectral_pipeline",
            expr="0.85*cos(2*pi*2*t) + 0.55*cos(2*pi*5*t)",
            t_range=[0.0, 2.0],
            freq_range=[0.5, 6.8],
            selected_freq=2.0,
            cutoff_freq=3.5,
            filter_kind="lowpass",
            angular_units="hz",
            n_samples=200,
        )
        res = evaluate_linked_math_model(spec)
        peaks = res["detected_peaks"]
        peak_freqs = [p["freq"] for p in peaks]
        self.assertTrue(any(abs(f - 2.0) < 0.05 for f in peak_freqs), peaks)
        self.assertTrue(any(abs(f - 5.0) < 0.05 for f in peak_freqs), peaks)

        # Spectrum bars MUST include the 2.0 Hz and 5.0 Hz bins, not miss them on a coarse grid
        labels = [float(str(lbl).rstrip("Hz*")) for lbl in res["spectrum_bars_labels"]]
        vals = res["spectrum_bars_values"]
        self.assertIn(2.0, labels)
        self.assertIn(5.0, labels)
        self.assertTrue("2*" in res["spectrum_bars_labels"] and "5*" in res["spectrum_bars_labels"], res["spectrum_bars_labels"])
        self.assertGreater(vals[labels.index(2.0)], 0.75)
        self.assertGreater(vals[labels.index(5.0)], 0.45)

        # Filtered spectrum must reflect actual reconstructed signal (< 3.5 Hz passed, 5 Hz suppressed)
        filt_bins = dict(res["dft_filtered_spectrum"])
        self.assertGreater(filt_bins.get(2.0, 0.0), 0.70)
        self.assertLess(filt_bins.get(5.0, 1.0), 0.05)

        # Nonfinite signal must fail explicitly instead of silently converting NaN/Inf to 0
        bad_spec = LinkedMathModelSpec(
            model_id="bad_sig",
            kind="signal_spectral_pipeline",
            expr="1/(t - 1.0)",
            t_range=[0.0, 2.0],
            freq_range=[0.5, 4.0],
            selected_freq=1.0,
            cutoff_freq=2.0,
            n_samples=201,
        )
        with self.assertRaises(ValueError):
            evaluate_linked_math_model(bad_spec)

    def test_06_simulation_rk45_oscillator_and_resonance_threshold(self) -> None:
        # Exact underdamped oscillator: m=1, c=0.5 (zeta=0.125), k=4 (w0=2.0), x(0)=1, v(0)=0
        sim = integrate_ode_system(
            state_vars=["x", "v"],
            rhs_exprs=["v", "-0.5*v - 4.0*x"],
            initial_state=[1.0, 0.0],
            t_range=[0.0, 6.0],
            steps=120,
            method="rk45",
        )
        w0 = 2.0
        gamma = 0.25
        wd = math.sqrt(w0 * w0 - gamma * gamma)
        max_err = 0.0
        for t_v, (x_num, _v_num) in zip(sim.t_axis, sim.states):
            x_exact = math.exp(-gamma * t_v) * (math.cos(wd * t_v) + (gamma / wd) * math.sin(wd * t_v))
            max_err = max(max_err, abs(x_num - x_exact))
        self.assertLess(max_err, 5e-5, f"RK45 numerical error too large: {max_err}")

        # Forced frequency response threshold: zeta < 1/sqrt(2) has nonzero resonance; zeta >= 1/sqrt(2) does NOT
        res_under = compute_driven_frequency_response(natural_freq=3.0, damping_ratio=0.2, drive_amp=1.0)
        self.assertTrue(res_under["has_nonzero_resonance_peak"])
        self.assertAlmostEqual(res_under["resonance_freq"], 3.0 * math.sqrt(1.0 - 2 * 0.04), places=4)

        res_high_damp = compute_driven_frequency_response(natural_freq=3.0, damping_ratio=0.8, drive_amp=1.0)
        self.assertFalse(res_high_damp["has_nonzero_resonance_peak"])
        self.assertEqual(res_high_damp["resonance_freq"], 0.0)

    def test_07_mechanical_body_and_vector_field_both_renderers(self) -> None:
        self.assertEqual(verify_registry_contracts(), [])
        model = LinkedMathModelSpec(
            model_id="osc",
            kind="dynamical_simulation",
            state_vars=["x", "v"],
            rhs_exprs=["v", "-0.4*v - 4.0*x"],
            initial_state=[1.2, 0.0],
            t_range=[0.0, 8.0],
            n_samples=120,
        )
        plan = VideoPlanV3(
            topic="Oscillator Visual Parity",
            title="Mechanical Body + Vector Field",
            target_duration_s=10.0,
            math_models=[model],
            scenes=[
                SceneV3(
                    scene_id="s1",
                    purpose="Simulate spring-damper mass and phase vector field",
                    narration="Observe the damped spring-mass motion alongside its phase-space velocity field.",
                    composition="side_by_side_comparison",
                    objects=[
                        SceneObject(id="ax_m", kind="axes", panel_slot="left", x_range=[-2, 2, 1], y_range=[-1, 1, 1], persist=True),
                        SceneObject(
                            id="mass_1",
                            kind="mechanical_body",
                            panel_slot="left",
                            axes_id="ax_m",
                            connector_kind="spring_damper",
                            anchor_coords=[-1.5, 0.0],
                            coords=[1.2, 0.0],
                            model_id="osc",
                            model_output="body_point",
                            persist=True,
                        ),
                        SceneObject(id="ax_p", kind="axes", panel_slot="right", x_range=[-2, 2, 1], y_range=[-3, 3, 1], persist=True),
                        SceneObject(
                            id="vf_1",
                            kind="vector_field",
                            panel_slot="right",
                            axes_id="ax_p",
                            dx_expr="v",
                            dy_expr="-0.4*v - 4.0*x",
                            var_names=["x", "v"],
                            x_range=[-1.5, 1.5, 0.75],
                            y_range=[-2.0, 2.0, 1.0],
                            persist=True,
                        ),
                        SceneObject(id="phase_curve", kind="parametric_curve", panel_slot="right", axes_id="ax_p", model_id="osc", model_output="phase_orbit"),
                    ],
                    events=[
                        AnimationEvent(op="create", target_id="ax_m"),
                        AnimationEvent(op="create", target_id="mass_1"),
                        AnimationEvent(op="create", target_id="ax_p"),
                        AnimationEvent(op="create", target_id="vf_1"),
                        AnimationEvent(op="trace", target_id="phase_curve"),
                        AnimationEvent(op="animate_param", target_id="mass_1", param_range=[0.0, 8.0], duration_s=3.5),
                    ],
                    claims=[
                        StructuredClaim(
                            kind="simulation_trajectory_bounds",
                            model_id="osc",
                            statement="Underdamped linear oscillator decays exponentially in phase space",
                        )
                    ],
                )
            ],
        )
        issues = validate_v3_plan(plan, require_claims=True)
        self.assertEqual(issues, [], issues)
        html, _ = compile_plan_to_html(plan, default_design_spec(look_name="editorial_light"))
        self.assertIn("mechanical_body", html)
        self.assertIn("vector_field", html)
        self.assertIn("sweep_frames", html)

    def test_08_multi_scene_dynamic_qa_gate_rejects_single_motion_doc(self) -> None:
        scenes = []
        for idx in range(6):
            ops = (
                [AnimationEvent(op="create", target_id=f"c{idx}"), AnimationEvent(op="move", target_id=f"c{idx}", to=[1, 0])]
                if idx == 0
                else [AnimationEvent(op="create", target_id=f"c{idx}"), AnimationEvent(op="reveal", target_id=f"lbl{idx}")]
            )
            scenes.append(
                SceneV3(
                    scene_id=f"s{idx}",
                    purpose=f"Oscillation step {idx}",
                    narration="The oscillator swings back and forth while damping pulls the trajectory toward rest.",
                    objects=[
                        SceneObject(id=f"ax{idx}", kind="axes", x_range=[-2, 2, 1], y_range=[-2, 2, 1]),
                        SceneObject(id=f"c{idx}", kind="function_plot", axes_id=f"ax{idx}", expr="cos(x)"),
                        SceneObject(id=f"lbl{idx}", kind="annotation", text=f"Step {idx}"),
                    ],
                    events=ops,
                )
            )
        plan = VideoPlanV3(topic="Oscillations", title="Static Motion Trap", target_duration_s=36.0, scenes=scenes)
        rep = audit_educational_quality(topic="Oscillations", title="Static Motion Trap", plan=plan, format_name="widescreen_hd")
        self.assertEqual(rep.category_statuses.get("animation_meaningfulness"), "failed", rep.category_statuses)


if __name__ == "__main__":
    unittest.main()
