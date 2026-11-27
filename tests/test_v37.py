"""Comprehensive V3.7 verification suite covering typography, light/dark WCAG contrast, CaptionIR,
adaptive layout solver, CreativeDirector script lock, executable registry, HyperFrames/Hybrid,
dynamical stability truth checks, honest 10-category QA, and cache dependency integrity."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from studio.ai.math_truth import markov_chain_trajectory, verify_eigenvalue_stability_regime
from studio.ai.v3_schema import AnimationEvent, ConstructionOp, ConstructionParams, SceneObject, SceneV3, StructuredClaim, VideoPlanV3
from studio.ai.validate_v3 import validate_v3_plan
from studio.audio.tts import AudioTimeline, SyncedBeat
from studio.captions import CAPTION_PRESETS, build_caption_ir, export_ass, normalize_caption_style
from studio.config import CAPTION_STYLE_CHOICES, THEMES
from studio.creative.dialogue import resolve_speaker_voices
from studio.creative.script_ir import NarrationBeatIR, ScriptIR, lock_script_to_plan
from studio.design.fonts import (
    MANIM_BODY_FONT,
    MANIM_DISPLAY_FONT,
    MANIM_SERIF_FONT,
    effective_px_scale,
    font_face_css,
    manimgx_font_paths,
    verify_pinned_fonts,
)
from studio.design.spec import NAMED_LOOKS, resolve_look, validate_look_contrast, wcag_contrast_ratio
from studio.layout.solver import (
    PixelBox,
    compute_adaptive_zones,
    fit_axes_pixel_box,
    resolve_non_overlapping_boxes,
)
from studio.longform import _allocate_chapter_budgets, chapter_cache_key
from studio.registry import (
    backend_capability_matrix,
    get_default_registry,
    select_scene_backend,
    verify_registry_contracts,
)
from studio.registry.base import ComponentSpec
from studio.renderers.hybrid import partition_plan_by_backend
from studio.renderers.hyperframes.typography import _pretty_math_label
from studio.qa.engine import audit_educational_quality, verify_decoded_frames
from studio.storage.cache import ArtifactCache, JobStore
from studio.storyboard.model import VisualReviewNote, storyboard_from_v3_plan


class TestV37Suite(unittest.TestCase):
    def test_pinned_fonts_sha256_and_scales(self) -> None:
        self.assertEqual(verify_pinned_fonts(), [])
        self.assertEqual(MANIM_DISPLAY_FONT, "Inter")
        self.assertEqual(MANIM_BODY_FONT, "Source Sans 3")
        self.assertEqual(MANIM_SERIF_FONT, "STIX Two Text")
        self.assertTrue(manimgx_font_paths())
        css = font_face_css(url_prefix="/fonts/", embed_base64=False)
        for fam in ("Inter", "Source Sans 3", "STIX Two Text", "STIX Two Math"):
            self.assertIn(f"font-family: '{fam}'", css)
        px_hd = effective_px_scale(1920, 1080)
        px_port = effective_px_scale(1080, 1920)
        self.assertGreaterEqual(px_hd.caption_px, 28)
        self.assertGreaterEqual(px_port.caption_px, 34)

    def test_unified_themes_wcag_contrast_all_looks(self) -> None:
        self.assertGreaterEqual(len(THEMES), 6)
        self.assertEqual(set(THEMES.keys()), set(NAMED_LOOKS.keys()))
        self.assertEqual(resolve_look("editorial_light").name, "light")
        self.assertEqual(resolve_look("cinematic_dark").name, "dark")
        self.assertTrue(resolve_look("light").is_light)
        self.assertFalse(resolve_look("dark").is_light)
        self.assertGreater(wcag_contrast_ratio("#000000", "#FFFFFF"), 20.0)
        for look_name, look_tok in NAMED_LOOKS.items():
            errs = validate_look_contrast(look_tok)
            self.assertEqual(errs, [], f"WCAG contrast regression on {look_name}: {errs}")

    def test_caption_ir_presets_orphan_avoidance_and_ass_export(self) -> None:
        beats = [
            SyncedBeat(
                id="b1",
                narration="If all eigenvalues have negative real parts and trajectories decay exponentially to zero.",
                caption="Stability",
                audio_path="",
                start_time=0.0,
                end_time=6.2,
                speech_duration=5.9,
                total_duration=6.2,
                words=[
                    [w, round(i * 0.4, 3), round((i + 1) * 0.4, 3)]
                    for i, w in enumerate(
                        ["If", "all", "eigenvalues", "have", "negative", "real", "parts", "and", "trajectories", "decay", "exponentially", "to", "zero."]
                    )
                ],
                speaker="learner",
            )
        ]
        self.assertEqual(set(CAPTION_STYLE_CHOICES), set(CAPTION_PRESETS.keys()))
        self.assertEqual(normalize_caption_style("minimal"), "minimal")

        ir_ed = build_caption_ir(beats, style="editorial", width_px=1920, height_px=1080)
        self.assertTrue(ir_ed.cues)
        for cue in ir_ed.cues[:-1]:
            first_line_tail = cue.lines[0].split()[-1].lower().strip(".,!?")
            if len(cue.lines) > 1 and len(cue.lines[0].split()) >= 3:
                self.assertNotIn(first_line_tail, {"and", "or", "to", "of"})

        ir_off = build_caption_ir(beats, style="off", width_px=1920, height_px=1080)
        self.assertLess(ir_off.reserve_band_px, ir_ed.reserve_band_px)
        with tempfile.TemporaryDirectory() as tmp:
            ass_off = Path(tmp) / "off.ass"
            export_ass(ir_off, ass_off, look_name="light")
            self.assertNotIn("Dialogue:", ass_off.read_text(encoding="utf-8"))

        with tempfile.TemporaryDirectory() as tmp:
            ass_path = Path(tmp) / "captions.ass"
            export_ass(ir_ed, ass_path, look_name="light")
            txt = ass_path.read_text(encoding="utf-8")
            self.assertIn("[V4+ Styles]", txt)
            self.assertIn("Dialogue: 0,", txt)

    def test_adaptive_layout_solver_and_equal_aspect_sizing(self) -> None:
        z_plain = compute_adaptive_zones("widescreen_hd", has_header_banner=False, has_equation_strip=False, caption_style="editorial")
        z_off = compute_adaptive_zones("widescreen_hd", has_header_banner=False, has_equation_strip=False, caption_style="off")
        z_hdr = compute_adaptive_zones("widescreen_hd", has_header_banner=True, has_equation_strip=True, caption_style="editorial")
        self.assertGreater(z_off.stage_box.h, z_plain.stage_box.h)
        self.assertGreater(z_plain.stage_box.h, z_hdr.stage_box.h)

        ax_box = fit_axes_pixel_box(
            z_plain,
            xr=[-3.0, 3.0],
            yr=[-3.0, 3.0],
            equal_aspect=True,
            composition="centered_focus",
        )
        # Must not shrink equal-aspect square plots into <25% thumbnails
        self.assertGreaterEqual(ax_box.w / z_plain.width_px, 0.30)
        self.assertAlmostEqual(ax_box.w, ax_box.h, delta=2.0)

        obs = [PixelBox(100, 100, 300, 200, role="axes")]
        placed = resolve_non_overlapping_boxes([PixelBox(120, 120, 120, 50, role="annotation")], obs, z_plain)
        self.assertFalse(placed[0].overlaps(obs[0], pad=4.0))

    def test_dialogue_speaker_isolation_and_script_lock(self) -> None:
        voices_single = resolve_speaker_voices("single_educator")
        voices_soc = resolve_speaker_voices("socratic")
        voices_doc = resolve_speaker_voices("documentary")
        self.assertEqual(voices_single["narrator"][0], "am_michael")
        self.assertEqual(voices_soc["educator"][0], "am_michael")
        self.assertEqual(voices_soc["LEARNER".lower()][0], "af_bella")
        self.assertEqual(voices_doc["narrator"][0], "bm_george")

        script = ScriptIR(
            topic="Eigenvalues",
            title="Eigenvalues Intro",
            narration_mode="socratic",
            beats=[
                NarrationBeatIR(
                    beat_id="b1",
                    scene_id="s1",
                    speaker="learner",
                    spoken_text="Why does a matrix preserve certain vectors?",
                    educational_purpose="Raise eigenvector intuition",
                    visual_intent="matrix_m:matrix",
                )
            ],
        )
        plan = VideoPlanV3(
            schema_version="3.0",
            topic="Eigenvalues",
            title="Eigenvalues Intro",
            scenes=[
                SceneV3(
                    scene_id="s1",
                    purpose="Eigenvector question",
                    speaker="narrator",
                    narration="Different text that script lock overrides.",
                    objects=[SceneObject(id="mat", kind="matrix", matrix_rows=[["2", "0"], ["0", "3"]])],
                    events=[AnimationEvent(op="create", target_id="mat")],
                )
            ],
        )
        lock_script_to_plan(plan, script, stage="editorial_repair")
        self.assertEqual(plan.scenes[0].speaker, "learner")
        self.assertEqual(plan.scenes[0].narration, "Why does a matrix preserve certain vectors?")

        tl = AudioTimeline(
            voice="am_michael",
            rate="-4%",
            master_voice_wav="",
            total_duration=4.2,
            beats=[SyncedBeat(id="s1", narration="Why?", caption="", audio_path="", start_time=0.0, end_time=4.2, speech_duration=3.9, total_duration=4.2, words=[], speaker="learner")],
        )
        sb = storyboard_from_v3_plan(plan, script=script, audio_timeline=tl)
        self.assertEqual(sb.total_duration_s, 4.2)
        self.assertEqual(sb.shots[0].duration_s, 4.2)
        self.assertEqual(sb.shots[0].speaker, "learner")

    def test_chapter_budget_allocation_varies_by_depth(self) -> None:
        class _FakeCh:
            def __init__(self, title: str, target_duration_s: float, node_ids: list[str], summary: str) -> None:
                self.title = title
                self.target_duration_s = target_duration_s
                self.node_ids = node_ids
                self.summary = summary

        chs = [
            _FakeCh("Intro Hook", 50.0, ["n1"], "Quick hook"),
            _FakeCh("Formal Eigen-Decomposition", 110.0, ["n2", "n3", "n4"], "Deep eigen-decomposition mechanism"),
        ]
        budgets = _allocate_chapter_budgets(chs, 180.0)
        self.assertAlmostEqual(sum(budgets), 180.0, places=0)
        self.assertGreater(budgets[1], budgets[0])

    def test_executable_registry_backend_matrix_and_hybrid_partition(self) -> None:
        self.assertEqual(verify_registry_contracts(), [])
        reg = get_default_registry()
        reg.register(ComponentSpec(name="unimpl_stub", category="visual_object", description="not ready", implemented=False))
        mat = backend_capability_matrix()
        self.assertIn("math_operation:ode_rk4_trajectory", mat)
        self.assertFalse(mat["visual_object:unimpl_stub"]["manimgx"])

        plan = VideoPlanV3(
            schema_version="3.0",
            topic="Hybrid Partition",
            title="Hybrid Partition",
            scenes=[
                SceneV3(
                    scene_id="s1",
                    purpose="Phase orbit",
                    narration="Continuous vector flow.",
                    objects=[
                        SceneObject(id="ax", kind="axes", x_range=[-2, 2, 1], y_range=[-2, 2, 1]),
                        SceneObject(id="v1", kind="vector", axes_id="ax", coords=[0.0, 0.0], end_coords=[1.0, 1.0]),
                    ],
                    events=[AnimationEvent(op="create", target_id="ax")],
                ),
                SceneV3(
                    scene_id="s2",
                    purpose="Classification table",
                    narration="Regime table summary.",
                    objects=[SceneObject(id="tbl", kind="table", table_headers=["A"], table_rows=[["1"]])],
                    events=[AnimationEvent(op="create", target_id="tbl")],
                ),
            ],
        )
        self.assertEqual(select_scene_backend(plan.scenes[0], "hybrid"), "manimgx")
        self.assertEqual(select_scene_backend(plan.scenes[1], "hybrid"), "hyperframes")
        tl = AudioTimeline(
            voice="am_michael",
            rate="-4%",
            master_voice_wav="",
            total_duration=4.0,
            beats=[
                SyncedBeat(id="s1", narration="a", caption="", audio_path="", start_time=0.0, end_time=2.0, speech_duration=1.8, total_duration=2.0, words=[]),
                SyncedBeat(id="s2", narration="b", caption="", audio_path="", start_time=2.0, end_time=4.0, speech_duration=1.8, total_duration=2.0, words=[]),
            ],
        )
        parts = partition_plan_by_backend(plan, tl)
        self.assertEqual([p.backend for p in parts], ["manimgx", "hyperframes"])
        self.assertIn("±", _pretty_math_label(r"\lambda = -0.35 \pm i"))

    def test_stability_regime_validation_word_boundaries_and_markov_hint(self) -> None:
        ok, _ = verify_eigenvalue_stability_regime(
            [[2.0, 1.0], [0.0, 3.0]],
            expected_regime="unstable_node",
            statement="Positive eigenvalues produce an unstable node diverging from origin.",
        )
        self.assertTrue(ok)
        ok_bad, detail = verify_eigenvalue_stability_regime(
            [[1.0, 0.0], [0.0, -2.0]],
            expected_regime="unstable_node",
            statement="Unstable source everywhere.",
        )
        self.assertFalse(ok_bad)
        self.assertIn("saddle", detail)

        # Matrix rotate & empty axes must be rejected pre-render
        bad_plan = VideoPlanV3(
            schema_version="3.0",
            topic="Bad Ops",
            title="Bad Ops",
            scenes=[
                SceneV3(
                    scene_id="s1",
                    purpose="Empty axes & matrix rotate",
                    narration="Check invalid events.",
                    objects=[
                        SceneObject(id="ax", kind="axes", x_range=[-2, 2, 1], y_range=[-2, 2, 1]),
                        SceneObject(id="mat", kind="matrix", matrix_rows=[["1", "0"], ["0", "2"]]),
                    ],
                    events=[
                        AnimationEvent(op="create", target_id="mat"),
                        AnimationEvent(op="rotate", target_id="mat", angle_deg=30.0),
                    ],
                )
            ],
        )
        issues = validate_v3_plan(bad_plan, require_claims=True)
        self.assertTrue(any("cannot rotate non-geometric" in m for m in issues))
        self.assertTrue(any("bare axes" in m for m in issues))

        with self.assertRaises(ValueError) as ctx:
            markov_chain_trajectory(
                [[0.0, 0.5, 0.5, 0.25], [0.5, 0.0, 0.5, 0.25], [0.5, 0.5, 0.0, 0.25], [0.0, 0.0, 0.0, 0.25]],
                [0.25, 0.25, 0.25, 0.25],
            )
        self.assertIn("column-stochastic", str(ctx.exception))

    def test_honest_qa_scoring_never_fakes_100_and_notes_categories(self) -> None:
        note = VisualReviewNote(shot_id="g", severity="warning", category="motion", finding="static")
        self.assertEqual(note.category, "motion")
        note2 = VisualReviewNote(shot_id="g", severity="blocker", category="contrast", finding="low")
        self.assertEqual(note2.category, "contrast")

        unclaimed = VideoPlanV3(
            schema_version="3.0",
            topic="Unverified",
            title="Unverified",
            scenes=[SceneV3(scene_id="s1", purpose="Test", narration="Just a dot.", objects=[SceneObject(id="p", kind="point", coords=[0.0, 0.0])])],
        )
        rev = audit_educational_quality(topic="Unverified", title="Unverified", plan=unclaimed, format_name="widescreen_hd", look_name="light")
        self.assertFalse(rev.approved)
        self.assertLess(rev.overall_score, 100.0)
        self.assertEqual(rev.category_statuses["audio_quality_loudness"], "not_evaluated")

        chk, _, _ = verify_decoded_frames(Path("nonexistent_probe.mp4"))
        self.assertEqual(chk.status, "not_evaluated")

    def test_artifact_cache_tamper_detection_and_theme_key_invalidation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tdir = Path(tmp)
            db_p = tdir / "jobs.db"
            ac = ArtifactCache(db_path=db_p)
            try:
                sample = tdir / "frame.mp4"
                sample.write_bytes(b"real-video-bytes-v1")
                hit, key, _, reason = ac.lookup("chapter_assembly", "ch1", {"theme": "light", "cap": "editorial"})
                self.assertFalse(hit)
                self.assertEqual(reason, "miss:not_recorded")
                ac.store_artifact("chapter_assembly", "ch1", key, sample)
                hit2, _, _, reason2 = ac.lookup("chapter_assembly", "ch1", {"theme": "light", "cap": "editorial"})
                self.assertTrue(hit2)
                self.assertEqual(reason2, "hit")

                # Mutating underlying file triggers tamper invalidation
                sample.write_bytes(b"tampered-bytes")
                hit3, _, _, reason3 = ac.lookup("chapter_assembly", "ch1", {"theme": "light", "cap": "editorial"})
                self.assertFalse(hit3)
                self.assertEqual(reason3, "miss:checksum_mismatch")

                # Switching look or caption style invalidates chapter cache key immediately
                sample_plan = VideoPlanV3(schema_version="3.0", topic="T", title="T", scenes=[])
                k_light_ed = chapter_cache_key(sample_plan, format_name="widescreen_hd", tts_engine="kokoro", voice="am_michael", rate="-4%", ui="minimal", look="cinematic", look_name="light", caption_style="editorial")
                k_dark_ed = chapter_cache_key(sample_plan, format_name="widescreen_hd", tts_engine="kokoro", voice="am_michael", rate="-4%", ui="minimal", look="cinematic", look_name="dark", caption_style="editorial")
                k_light_min = chapter_cache_key(sample_plan, format_name="widescreen_hd", tts_engine="kokoro", voice="am_michael", rate="-4%", ui="minimal", look="cinematic", look_name="light", caption_style="minimal")
                self.assertNotEqual(k_light_ed, k_dark_ed)
                self.assertNotEqual(k_light_ed, k_light_min)
                ac.store_artifact("tts", "voice_only", JobStore.content_hash({"text": "hello"}), sample)
                self.assertGreaterEqual(ac.misses, 1)
            finally:
                ac.store._conn.close()

    def test_smooth_bell_bgm_sfx_and_velvet_mix_graph(self) -> None:
        import wave
        import numpy as np
        from studio.audio.bgm import generate_ambient_bgm
        from studio.audio.master import build_audio_mix_graph
        from studio.audio.sfx import SFX_VERSION, SfxLibrary

        with tempfile.TemporaryDirectory() as tmp:
            tdir = Path(tmp)
            bgm_path = generate_ambient_bgm(tdir / "bgm.wav", duration=10.0)
            with wave.open(str(bgm_path), "rb") as wf:
                self.assertEqual((wf.getframerate(), wf.getnchannels()), (48000, 2))
                pcm = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(np.float32).reshape(-1, 2) / 32768.0
            self.assertLess(float(np.max(np.abs(pcm[:480]))), 0.01)
            self.assertLess(float(np.max(np.abs(pcm[-480:]))), 0.01)
            self.assertGreater(float(np.mean(np.abs(pcm[:, 0] - pcm[:, 1]))), 0.01)
            self.assertEqual(SFX_VERSION, "v3")
            sfx_files = SfxLibrary(tdir / "sfx").ensure_all()
            self.assertIn("chime_reveal", sfx_files)
            self.assertTrue(all(p.exists() and p.stat().st_size > 1000 for p in sfx_files.values()))

        mix = build_audio_mix_graph(1, 2, 0)
        self.assertIn("equalizer=", mix)
        self.assertIn("sidechaincompress=", mix)
        self.assertIn("ratio=2.3", mix)


if __name__ == "__main__":
    unittest.main()
