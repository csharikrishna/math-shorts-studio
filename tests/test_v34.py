"""Behavioral regression tests for Math Shorts Studio V3.4."""

from __future__ import annotations

import math
from pathlib import Path
import unittest

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
from studio.ai.world_model import (
    materialize_constructions,
    polygon_inside,
    polygons_overlap,
    verify_partition,
)
from studio.audio.pronounce import normalize_for_speech_with_map
from studio.audio.tts import AudioTimeline, SyncedBeat
from studio.qa.engine import verify_media_streams
from studio.timeline import MasterTimeline, build_semantic_captions


class TestV34Regressions(unittest.TestCase):
    def test_invalid_polygon_with_misleading_name_rejected(self) -> None:
        plan = VideoPlanV3(
            topic="Bad Square",
            title="Bad Square",
            scenes=[
                SceneV3(
                    scene_id="s1",
                    purpose="misleading",
                    narration="Look at this square.",
                    objects=[
                        SceneObject(
                            id="fake_square",
                            kind="polygon",
                            vertices=[[0, 0], [3, 0], [3, 1], [0, 1]],
                            label="square",
                        )
                    ],
                )
            ],
        )
        issues = validate_v3_plan(plan)
        self.assertTrue(any("fails geometric square check" in i for i in issues))

    def test_nonfinite_and_malformed_coords_rejected(self) -> None:
        plan = VideoPlanV3(
            topic="NaN",
            title="NaN",
            scenes=[
                SceneV3(
                    scene_id="s1",
                    purpose="nan",
                    narration="Bad coords.",
                    objects=[
                        SceneObject(
                            id="bad_poly",
                            kind="polygon",
                            vertices=[[0.0, 0.0], [float("nan"), 1.0], [1.0, 1.0]],
                        ),
                        SceneObject(
                            id="bad_pt",
                            kind="point",
                            coords=[float("inf"), 0.0],
                        ),
                    ],
                )
            ],
        )
        issues = validate_v3_plan(plan)
        self.assertTrue(any("non-finite" in i for i in issues))

    def test_noncongruent_triangles_rejected(self) -> None:
        plan = VideoPlanV3(
            topic="Congruence",
            title="Congruence",
            scenes=[
                SceneV3(
                    scene_id="s1",
                    purpose="check",
                    narration="Compare two triangles.",
                    objects=[
                        SceneObject(id="t1", kind="polygon", vertices=[[0, 0], [3, 0], [0, 4]], label="triangle"),
                        SceneObject(id="t2", kind="polygon", vertices=[[0, 0], [5, 0], [0, 4]], label="triangle"),
                    ],
                    claims=[StructuredClaim(kind="congruent_triangles", object_ids=["t1", "t2"])],
                )
            ],
        )
        issues = validate_v3_plan(plan)
        self.assertTrue(any("not congruent" in i for i in issues))

    def test_overlapping_and_invalid_partitions_rejected(self) -> None:
        outer = [[0, 0], [4, 0], [4, 4], [0, 4]]
        p1 = [[0, 0], [3, 0], [3, 4], [0, 4]]
        p2 = [[1, 0], [4, 0], [4, 4], [1, 4]]
        self.assertTrue(polygons_overlap(p1, p2))
        self.assertTrue(polygon_inside(p1, outer))
        part_issues = verify_partition([p1, p2], outer)
        self.assertTrue(any("overlap" in i for i in part_issues))
        self.assertTrue(any("area conservation" in i for i in part_issues))

    def test_area_conservation_claim_validates_polygon_areas(self) -> None:
        plan = VideoPlanV3(
            topic="Area",
            title="Area",
            scenes=[
                SceneV3(
                    scene_id="s1",
                    purpose="3-4-5 squares",
                    narration="Check area conservation.",
                    constructions=[
                        ConstructionOp(op="right_triangle", output_ids=["tri"], params=ConstructionParams(a=3.0, b=4.0), label="triangle"),
                        ConstructionOp(op="square_on_segment", output_ids=["sq_a"], source_id="tri", params=ConstructionParams(edge_index=0, outward=True), label="square"),
                        ConstructionOp(op="square_on_segment", output_ids=["sq_b"], source_id="tri", params=ConstructionParams(edge_index=2, outward=True), label="square"),
                        ConstructionOp(op="square_on_segment", output_ids=["sq_c"], source_id="tri", params=ConstructionParams(edge_index=1, outward=True), label="square"),
                    ],
                    claims=[
                        StructuredClaim(kind="area_conservation", object_ids=["sq_a", "sq_b"], container_id="sq_c"),
                    ],
                )
            ],
        )
        self.assertEqual(validate_v3_plan(plan), [])
        mat = materialize_constructions(plan)
        self.assertEqual(len(mat.scenes[0].objects), 4)

    def test_event_ordering_and_bookmark_order_enforced(self) -> None:
        plan = VideoPlanV3(
            topic="Order",
            title="Order",
            scenes=[
                SceneV3(
                    scene_id="s1",
                    purpose="bad order",
                    narration="First <bookmark mark='b1'/>one, then <bookmark mark='b2'/>two.",
                    objects=[SceneObject(id="eq1", kind="equation", tex="x=1")],
                    events=[
                        AnimationEvent(op="highlight", target_id="eq1", anchor_mark="b2"),
                        AnimationEvent(op="create", target_id="eq1", anchor_mark="b1"),
                    ],
                )
            ],
        )
        issues = validate_v3_plan(plan)
        self.assertTrue(any("before it is created" in i for i in issues))
        self.assertTrue(any("out of narration order" in i for i in issues))

    def test_semantic_captions_and_pronounce_normalization(self) -> None:
        norm = normalize_for_speech_with_map("Check <bookmark mark='go'/>sin(x) plus a^2 and 1/2.")
        self.assertIn("sine of x", str(norm["spoken"]))
        self.assertIn("a squared", str(norm["spoken"]))
        self.assertIn("one half", str(norm["spoken"]))
        self.assertEqual(norm["bookmarks"], ["go"])

        beat = SyncedBeat(
            id="b1",
            narration="Start with one triangle, then rotate three copies.",
            caption="Start",
            audio_path="dummy.wav",
            start_time=0.25,
            end_time=4.25,
            speech_duration=3.7,
            total_duration=4.0,
            words=[
                ["Start", 0.25, 0.5, False],
                ["with", 0.5, 0.7, False],
                ["one", 0.7, 1.0, False],
                ["triangle,", 1.0, 1.5, True],
                ["then", 1.5, 1.8, False],
                ["rotate", 1.8, 2.4, False],
                ["three", 2.4, 2.9, False],
                ["copies.", 2.9, 3.6, True],
            ],
            marks={},
        )
        segs = build_semantic_captions([beat])
        self.assertEqual(len(segs), 2)
        self.assertTrue(segs[0].text.endswith("triangle,"))
        self.assertTrue(segs[1].text.endswith("copies."))

        tl = AudioTimeline(voice="am_michael", rate="-4%", master_voice_wav="dummy.wav", total_duration=4.25, beats=[beat])
        mt = MasterTimeline(audio=tl)
        out = mt.export_subtitles(Path("build/test_subtitles"), "demo")
        self.assertTrue(out["srt"].exists() and out["vtt"].exists() and out["captions"].exists())

    def test_rendered_pythagorean_media_streams(self) -> None:
        mp4 = Path("output/V3_v3_pythagorean_dissection_proof_shorts_hd.mp4")
        if mp4.exists():
            chk = verify_media_streams(mp4)
            self.assertTrue(chk.passed, chk.detail)


if __name__ == "__main__":
    unittest.main()
