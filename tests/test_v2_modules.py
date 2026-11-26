"""Unit tests for V2 modules: providers, pronunciation, timeline, knowledge, and QA."""

import tempfile
import unittest
from pathlib import Path

from studio.ai.knowledge import KnowledgeClaim, KnowledgeNode, KnowledgeTree
from studio.audio.pronounce import normalize_for_speech
from studio.audio.tts import AudioTimeline, SyncedBeat
from studio.providers import FreeOnlyPolicyError, OpenRouterProvider, get_provider
from studio.qa.engine import verify_storyboard_image, verify_symbolic_derivative
from studio.storage.cache import JobStore
from studio.timeline import MasterTimeline


class TestV2Modules(unittest.TestCase):
    def test_openrouter_free_only_guard(self):
        p = OpenRouterProvider(api_key="dummy-key", free_only=True)
        with self.assertRaises(FreeOnlyPolicyError):
            p._validate_model("openai/gpt-4o")
        p._validate_model("google/gemma-3-27b-it:free")

    def test_provider_factory(self):
        p = get_provider("ollama", model="qwen2.5:7b")
        self.assertEqual(p.name, "ollama")
        self.assertTrue(p.free_only)

    def test_pronunciation_normalizer(self):
        self.assertIn("sine of 3x", normalize_for_speech("Look at sin(3x)"))
        self.assertIn("the sum", normalize_for_speech("Compute Σ"))
        self.assertIn("as x approaches infinity", normalize_for_speech("x → ∞"))

    def test_master_timeline_and_exports(self):
        audio = AudioTimeline(
            voice="am_michael",
            rate="-4%",
            master_voice_wav="voice.wav",
            total_duration=4.0,
            beats=[
                SyncedBeat("b1", "Hello world.", "Hello world", "b1.wav", 0.25, 2.0, 1.5, 1.75, [["Hello", 0.25, 0.8, False]], {}),
                SyncedBeat("b2", "Second beat.", "Second beat", "b2.wav", 2.0, 4.0, 1.75, 2.0, [["Second", 2.0, 2.5, True]], {}),
            ],
        )
        mt = MasterTimeline(audio=audio)
        self.assertEqual(mt.validate(), [])
        with tempfile.TemporaryDirectory() as td:
            files = mt.export_subtitles(Path(td), "test")
            self.assertTrue(files["vtt"].read_text(encoding="utf-8").startswith("WEBVTT"))
            self.assertIn("Hello world", files["srt"].read_text(encoding="utf-8"))
            self.assertIn("Second beat.", files["transcript"].read_text(encoding="utf-8"))

    def test_knowledge_tree_and_repurposing(self):
        tree = KnowledgeTree(
            topic="Fourier Analysis",
            nodes=[
                KnowledgeNode(
                    node_id="gibbs",
                    title="Gibbs Phenomenon",
                    claims=[KnowledgeClaim(statement="Overshoot approaches 8.949%", kind="identity")],
                    short_hook_idea="Why does a sharp corner always overshoot by 9%?",
                )
            ],
        )
        shorts = tree.derive_short_topics()
        self.assertEqual(len(shorts), 1)
        self.assertEqual(shorts[0]["node_id"], "gibbs")

    def test_sqlite_job_store_and_cache(self):
        with tempfile.TemporaryDirectory() as td:
            store = JobStore(Path(td) / "jobs.db")
            store.upsert_job("j1", "Fourier", "render", "done", {"fps": 60})
            job = store.get_job("j1")
            self.assertIsNotNone(job)
            self.assertEqual(job["status"], "done")
            h = store.content_hash({"a": 1})
            self.assertEqual(len(h), 16)
            store.close()

    def test_sympy_and_visual_qa(self):
        chk = verify_symbolic_derivative("sin(3*x)", "3*cos(3*x)")
        self.assertTrue(chk.passed)
        sb = Path("build/AI_explain_simply_step_by_step_how_adding_smooth_si_shorts_hd/AI_explain_simply_step_by_step_how_adding_smooth_si_raw.storyboard.png")
        if sb.exists():
            vchk = verify_storyboard_image(sb)
            self.assertTrue(vchk.passed)


if __name__ == "__main__":
    unittest.main()
