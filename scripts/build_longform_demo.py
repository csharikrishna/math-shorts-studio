"""Generate a genuine 5-minute (300s+) widescreen_hd educational video via studio.longform."""

from __future__ import annotations

import json
from studio.ai.knowledge import ChapterSpec, ContinuityState, KnowledgeClaim, KnowledgeNode, KnowledgeTree, LongFormPlan
from studio.ai.v3_schema import AnimationEvent, ConstructionOp, ConstructionParams, SceneObject, SceneV3, StructuredClaim, VideoPlanV3
from studio.longform import build_longform_video


def _make_demo_longform() -> tuple[LongFormPlan, list[VideoPlanV3]]:
    tree = KnowledgeTree(
        topic="From Right Triangles to Fourier Waves",
        audience="general",
        nodes=[
            KnowledgeNode(
                node_id="n1_triangle",
                title="Right Triangle Geometry",
                claims=[KnowledgeClaim(statement="a^2 + b^2 = c^2 for any right triangle", kind="identity", latex="a^2 + b^2 = c^2", verified=True)],
                short_hook_idea="Why three squares on a right triangle always balance",
            ),
            KnowledgeNode(
                node_id="n2_unit_circle",
                title="Circles & Sine Waves",
                prerequisites=["n1_triangle"],
                claims=[KnowledgeClaim(statement="cos^2(theta) + sin^2(theta) = 1", kind="identity", latex="\\cos^2\\theta + \\sin^2\\theta = 1", verified=True)],
            ),
            KnowledgeNode(
                node_id="n3_harmonics",
                title="Harmonics & Superposition",
                prerequisites=["n2_unit_circle"],
                claims=[KnowledgeClaim(statement="Adding odd harmonics sharpens a wave toward a square wave", kind="fact", verified=True)],
            ),
            KnowledgeNode(
                node_id="n4_taylor",
                title="Local Polynomial Approximations",
                prerequisites=["n2_unit_circle"],
                claims=[KnowledgeClaim(statement="Taylor polynomials match derivatives at a single point", kind="approximation", verified=True)],
            ),
            KnowledgeNode(
                node_id="n5_bayes",
                title="Updating Belief with Area",
                claims=[KnowledgeClaim(statement="Posterior probability scales with likelihood times prior", kind="identity", verified=True)],
            ),
            KnowledgeNode(
                node_id="n6_synthesis",
                title="Unified Mathematical Perspective",
                prerequisites=["n1_triangle", "n2_unit_circle", "n3_harmonics", "n4_taylor", "n5_bayes"],
                claims=[KnowledgeClaim(statement="Geometry, waves, and probability share conserved quantities", kind="fact", verified=True)],
            ),
        ],
    )

    continuity = ContinuityState(
        defined_variables={"a": "horizontal leg", "b": "vertical leg", "c": "hypotenuse / radius", "theta": "rotation angle", "x": "input coordinate"},
        introduced_terms=["right triangle", "unit circle", "harmonic", "Taylor polynomial", "posterior"],
        color_map={"primary": "blue", "secondary": "purple", "highlight": "gold", "success": "green"},
    )

    chapters = [
        ChapterSpec(chapter_id="ch1_pythagoras", title="Chapter 1: Why Right Triangles Balance", node_ids=["n1_triangle"], target_duration_s=52.0),
        ChapterSpec(chapter_id="ch2_unit_circle", title="Chapter 2: Spinning Triangles into Waves", node_ids=["n2_unit_circle"], target_duration_s=52.0),
        ChapterSpec(chapter_id="ch3_fourier", title="Chapter 3: Building Sharp Corners from Smooth Waves", node_ids=["n3_harmonics"], target_duration_s=52.0),
        ChapterSpec(chapter_id="ch4_taylor", title="Chapter 4: Predicting Curves Locally", node_ids=["n4_taylor"], target_duration_s=52.0),
        ChapterSpec(chapter_id="ch5_bayes", title="Chapter 5: Measuring Evidence as Area", node_ids=["n5_bayes"], target_duration_s=52.0),
        ChapterSpec(chapter_id="ch6_synthesis", title="Chapter 6: The Hidden Unity of Math", node_ids=["n6_synthesis"], target_duration_s=52.0),
        ChapterSpec(chapter_id="ch7_gradient_descent", title="Chapter 7: Following the Slope Downhill", node_ids=["n4_taylor"], target_duration_s=30.0),
        ChapterSpec(chapter_id="ch8_central_limit", title="Chapter 8: Order from Randomness", node_ids=["n5_bayes"], target_duration_s=30.0),
        ChapterSpec(chapter_id="ch9_euler_rotation", title="Chapter 9: Complex Exponentials Spin", node_ids=["n2_unit_circle"], target_duration_s=30.0),
        ChapterSpec(chapter_id="ch10_damped_waves", title="Chapter 10: Damped Oscillations in Nature", node_ids=["n3_harmonics"], target_duration_s=30.0),
        ChapterSpec(chapter_id="ch11_exponential_growth", title="Chapter 11: Continuous Compounding", node_ids=["n4_taylor"], target_duration_s=30.0),
        ChapterSpec(chapter_id="ch12_grand_recap", title="Chapter 12: Putting Every Piece Together", node_ids=["n6_synthesis"], target_duration_s=30.0),
        ChapterSpec(chapter_id="ch13_logistic_limits", title="Chapter 13: Saturation and Logistic Curves", node_ids=["n6_synthesis"], target_duration_s=25.0),
    ]
    lf_plan = LongFormPlan(topic=tree.topic, audience="general", knowledge_tree=tree, chapters=chapters, continuity=continuity)

    ch1 = VideoPlanV3(
        topic="Right Triangle Geometry",
        title="Chapter 1: Why Right Triangles Balance",
        scenes=[
            SceneV3(
                scene_id="c1_s1",
                purpose="Introduce squares on a right triangle",
                narration="Every wave, circle, and signal in modern physics begins with a humble right triangle. Why do the squares built on its two legs always equal the square on its *hypotenuse*?",
                constructions=[
                    ConstructionOp(op="right_triangle", output_ids=["tri0"], params=ConstructionParams(a=3.0, b=4.0, origin=[-1.5, -2.0]), color_role="highlight", persist=False, label="triangle"),
                    ConstructionOp(op="square_on_segment", output_ids=["sq_a0"], source_id="tri0", params=ConstructionParams(edge_index=0, outward=True), color_role="primary", persist=False, label="square"),
                    ConstructionOp(op="square_on_segment", output_ids=["sq_b0"], source_id="tri0", params=ConstructionParams(edge_index=2, outward=True), color_role="tertiary", persist=False, label="square"),
                    ConstructionOp(op="square_on_segment", output_ids=["sq_c0"], source_id="tri0", params=ConstructionParams(edge_index=1, outward=True), color_role="secondary", persist=False, label="square"),
                ],
                objects=[SceneObject(id="eq1", kind="equation", persist=False, color_role="highlight", tex="a^2 + b^2 = c^2", card=True)],
            ),
            SceneV3(
                scene_id="c1_s2",
                purpose="Build outer square and place four triangles",
                narration="To see why without guessing, build a larger square of side *a plus b*. Tuck four identical copies of our right triangle into its four corners, leaving a tilted gap in the middle.",
                constructions=[
                    ConstructionOp(op="pythagorean_dissection", output_ids=["outer_sq", "t1", "t2", "t3", "t4", "inner_sq"], params=ConstructionParams(a=3.0, b=4.0), persist=True)
                ],
                claims=[
                    StructuredClaim(kind="square", object_ids=["outer_sq", "inner_sq"]),
                    StructuredClaim(kind="congruent_triangles", object_ids=["t1", "t2", "t3", "t4"]),
                ],
                objects=[SceneObject(id="eq2", kind="equation", persist=True, color_role="primary", tex="(a+b)^2 = 4\\left(\\tfrac{1}{2}ab\\right) + c^2", card=True)],
            ),
            SceneV3(
                scene_id="c1_s3",
                purpose="Cancel 2ab to reveal Pythagorean identity",
                narration="Because all four triangles together have area two a b, expanding the outer square and <bookmark mark='cancel'/>canceling two a b from both sides leaves *a squared plus b squared equals c squared*.",
                objects=[SceneObject(id="eq3", kind="equation", persist=True, color_role="highlight", tex="a^2 + 2ab + b^2 = 2ab + c^2 \\Rightarrow a^2 + b^2 = c^2", card=True)],
                events=[
                    AnimationEvent(op="transform", source_id="eq2", target_id="eq3", anchor_mark="cancel"),
                    AnimationEvent(op="highlight", target_id="inner_sq", anchor_mark="cancel"),
                ],
            ),
        ],
    )

    ch2 = VideoPlanV3(
        topic="Unit Circle and Sine Waves",
        title="Chapter 2: Spinning Triangles into Waves",
        scenes=[
            SceneV3(
                scene_id="c2_s1",
                purpose="Place right triangle inside the unit circle",
                narration="Now shrink that right triangle so its hypotenuse has length one, and pin one corner at the origin. As the angle *theta* turns, the tip traces a perfect unit circle.",
                objects=[
                    SceneObject(id="uc_axes", kind="axes", persist=True, color_role="primary", x_range=[-1.6, 1.6], y_range=[-1.6, 1.6], equal_aspect=True),
                    SceneObject(id="uc_circle", kind="circle", persist=True, color_role="secondary", coords=[0.0, 0.0], radius=1.0, axes_id="uc_axes"),
                    SceneObject(id="uc_vec", kind="vector", persist=True, color_role="highlight", coords=[0.0, 0.0], param_name="theta", expr="cos(theta); sin(theta)", param_init=0.3, axes_id="uc_axes"),
                    SceneObject(id="uc_proj", kind="projection", persist=True, color_role="primary", param_name="theta", expr="cos(theta); sin(theta)", param_init=0.3, axes_id="uc_axes"),
                    SceneObject(id="uc_eq", kind="equation", persist=True, color_role="highlight", tex="\\cos^2\\theta + \\sin^2\\theta = 1", card=True),
                ],
                events=[
                    AnimationEvent(op="create", target_id="uc_axes"),
                    AnimationEvent(op="create", target_id="uc_circle"),
                    AnimationEvent(op="create", target_id="uc_vec"),
                    AnimationEvent(op="create", target_id="uc_proj"),
                    AnimationEvent(op="create", target_id="uc_eq"),
                    AnimationEvent(op="animate_param", target_id="uc_vec", param_range=[0.3, 6.28]),
                ],
            ),
            SceneV3(
                scene_id="c2_s2",
                purpose="Unroll vertical height as a sine wave",
                narration="Watch the vertical height of that spinning point. Unrolled across time, the triangle's vertical leg becomes the smooth, endless *sine wave* behind sound and light.",
                objects=[
                    SceneObject(id="sin_axes", kind="axes", persist=True, color_role="primary", x_range=[-3.5, 3.5], y_range=[-1.5, 1.5]),
                    SceneObject(id="sin_curve", kind="function_plot", persist=True, color_role="secondary", expr="sin(x)", axes_id="sin_axes"),
                    SceneObject(id="sin_eq", kind="equation", persist=True, color_role="secondary", tex="y = \\sin(x)"),
                ],
                events=[
                    AnimationEvent(op="remove", target_id="uc_proj"),
                    AnimationEvent(op="remove", target_id="uc_vec"),
                    AnimationEvent(op="remove", target_id="uc_circle"),
                    AnimationEvent(op="remove", target_id="uc_axes"),
                    AnimationEvent(op="create", target_id="sin_axes"),
                    AnimationEvent(op="draw", target_id="sin_curve"),
                    AnimationEvent(op="create", target_id="sin_eq"),
                ],
            ),
        ],
    )

    ch3 = VideoPlanV3(
        topic="Fourier Square Wave",
        title="Chapter 3: Building Sharp Corners from Smooth Waves",
        scenes=[
            SceneV3(
                scene_id="c3_s1",
                purpose="Compare single sine wave to a square target",
                narration="A single sine wave is completely round and smooth. Could adding *more* smooth sine waves ever build a razor-sharp square corner?",
                objects=[
                    SceneObject(id="f_axes", kind="axes", persist=True, color_role="primary", x_range=[-3.5, 3.5], y_range=[-1.6, 1.6]),
                    SceneObject(id="f_w1", kind="function_plot", persist=True, color_role="primary", expr="4/pi*sin(x)", axes_id="f_axes"),
                    SceneObject(id="f_eq1", kind="equation", persist=True, color_role="primary", tex="S_1 = \\tfrac{4}{\\pi}\\sin x"),
                ],
            ),
            SceneV3(
                scene_id="c3_s2",
                purpose="Add third and fifth harmonics",
                narration="Add the third harmonic at one third amplitude. The rounded peaks flatten immediately, and adding the *fifth* harmonic sharpens the vertical cliff even further.",
                objects=[
                    SceneObject(id="f_w3", kind="function_plot", persist=True, color_role="secondary", expr="4/pi*(sin(x)+sin(3*x)/3+sin(5*x)/5)", axes_id="f_axes"),
                    SceneObject(id="f_eq3", kind="equation", persist=True, color_role="secondary", tex="S_5 = \\tfrac{4}{\\pi}\\sum_{k=1,3,5}\\tfrac{\\sin kx}{k}"),
                ],
                events=[
                    AnimationEvent(op="transform", source_id="f_w1", target_id="f_w3"),
                    AnimationEvent(op="transform", source_id="f_eq1", target_id="f_eq3"),
                ],
            ),
            SceneV3(
                scene_id="c3_s3",
                purpose="Converge with 19 odd harmonics",
                narration="Stack nineteen odd harmonics together, and the smooth waves lock into crisp flat plateaus and sharp vertical *jumps*.",
                objects=[
                    SceneObject(id="f_w19", kind="function_plot", persist=True, color_role="highlight", expr="4/pi*sum(k,1,19,2,sin(k*x)/k)", axes_id="f_axes"),
                    SceneObject(id="f_eq19", kind="equation", persist=True, color_role="highlight", tex="\\tfrac{4}{\\pi}\\sum_{k\\,\\text{odd}}\\tfrac{\\sin kx}{k}", card=True),
                ],
                events=[
                    AnimationEvent(op="transform", source_id="f_w3", target_id="f_w19"),
                    AnimationEvent(op="transform", source_id="f_eq3", target_id="f_eq19"),
                ],
            ),
        ],
    )

    ch4 = VideoPlanV3(
        topic="Taylor Series Approximation",
        title="Chapter 4: Predicting Curves Locally",
        scenes=[
            SceneV3(
                scene_id="c4_s1",
                purpose="Approximate cosine with a parabola",
                narration="Instead of stacking waves, what if we imitate cosine using only simple polynomials? Near zero, *one minus x squared over two* hugs cosine tightly.",
                objects=[
                    SceneObject(id="t_axes", kind="axes", persist=True, color_role="primary", x_range=[-3.5, 3.5], y_range=[-2.0, 2.0]),
                    SceneObject(id="t_cos", kind="function_plot", persist=True, color_role="primary", expr="cos(x)", axes_id="t_axes"),
                    SceneObject(id="t_p2", kind="function_plot", persist=True, color_role="secondary", expr="1 - x**2/2", axes_id="t_axes"),
                    SceneObject(id="t_eq2", kind="equation", persist=True, color_role="secondary", tex="P_2(x) = 1 - \\tfrac{x^2}{2}"),
                ],
            ),
            SceneV3(
                scene_id="c4_s2",
                purpose="Extend to degree 6 Taylor polynomial",
                narration="Match higher derivatives at zero by adding x to the fourth and sixth terms. Now the *polynomial* tracks cosine across an entire cycle.",
                objects=[
                    SceneObject(id="t_p6", kind="function_plot", persist=True, color_role="highlight", expr="1 - x**2/2 + x**4/24 - x**6/720", axes_id="t_axes"),
                    SceneObject(id="t_eq6", kind="equation", persist=True, color_role="highlight", tex="\\cos(x) \\approx 1 - \\tfrac{x^2}{2!} + \\tfrac{x^4}{4!} - \\tfrac{x^6}{6!}", card=True),
                ],
                events=[
                    AnimationEvent(op="transform", source_id="t_p2", target_id="t_p6"),
                    AnimationEvent(op="transform", source_id="t_eq2", target_id="t_eq6"),
                ],
            ),
        ],
    )

    ch5 = VideoPlanV3(
        topic="Bayes Theorem as Area",
        title="Chapter 5: Measuring Evidence as Area",
        scenes=[
            SceneV3(
                scene_id="c5_s1",
                purpose="Visualize prior probability as a rectangle",
                narration="Probability also obeys geometry. Picture all possibilities as a unit square, split into a rare *hypothesis* strip on the left and everything else on the right.",
                objects=[
                    SceneObject(id="b_axes", kind="axes", persist=True, color_role="primary", x_range=[0.0, 1.2], y_range=[0.0, 1.2], equal_aspect=True),
                    SceneObject(id="b_prior", kind="polygon", persist=True, color_role="secondary", vertices=[[0.0, 0.0], [0.2, 0.0], [0.2, 1.0], [0.0, 1.0]], axes_id="b_axes"),
                    SceneObject(id="b_rest", kind="polygon", persist=True, color_role="primary", vertices=[[0.2, 0.0], [1.0, 0.0], [1.0, 1.0], [0.2, 1.0]], axes_id="b_axes"),
                    SceneObject(id="b_eq1", kind="equation", persist=True, color_role="secondary", tex="P(H) = 0.2"),
                ],
            ),
            SceneV3(
                scene_id="c5_s2",
                purpose="Condition on positive evidence",
                narration="When new evidence arrives, we discard possibilities that disagree. Bayes' theorem simply measures the surviving *true-positive* area divided by all surviving area.",
                objects=[
                    SceneObject(id="b_tp", kind="polygon", persist=True, color_role="highlight", vertices=[[0.0, 0.0], [0.2, 0.0], [0.2, 0.9], [0.0, 0.9]], axes_id="b_axes"),
                    SceneObject(id="b_fp", kind="polygon", persist=True, color_role="tertiary", vertices=[[0.2, 0.0], [1.0, 0.0], [1.0, 0.15], [0.2, 0.15]], axes_id="b_axes"),
                    SceneObject(id="b_eq2", kind="equation", persist=True, color_role="highlight", tex="P(H\\mid E) = \\frac{P(E\\mid H)P(H)}{P(E)}", card=True),
                ],
                events=[
                    AnimationEvent(op="create", target_id="b_tp"),
                    AnimationEvent(op="create", target_id="b_fp"),
                    AnimationEvent(op="transform", source_id="b_eq1", target_id="b_eq2"),
                ],
            ),
        ],
    )

    ch6 = VideoPlanV3(
        topic="The Hidden Unity of Math",
        title="Chapter 6: The Hidden Unity of Math",
        scenes=[
            SceneV3(
                scene_id="c6_s1",
                purpose="Connect geometry, waves, and probability",
                narration="From a tilted square inside a right triangle, to spinning circles, Fourier harmonics, and Bayesian area, every idea here rests on one principle: *invariance*.",
                objects=[
                    SceneObject(id="u_axes", kind="axes", persist=True, color_role="primary", x_range=[-3.5, 3.5], y_range=[-1.5, 1.5]),
                    SceneObject(id="u_bell", kind="function_plot", persist=True, color_role="highlight", expr="exp(-x**2/2)", axes_id="u_axes"),
                    SceneObject(id="u_eq", kind="equation", persist=True, color_role="highlight", tex="a^2 + b^2 = c^2 \\;\\longleftrightarrow\\; e^{i\\theta} = \\cos\\theta + i\\sin\\theta", card=True),
                ],
            ),
            SceneV3(
                scene_id="c6_s2",
                purpose="Final takeaway",
                narration="When you understand *why* an area or length stays conserved, seemingly separate branches of mathematics snap together into one unified picture.",
                objects=[
                    SceneObject(id="u_wave", kind="function_plot", persist=True, color_role="secondary", expr="cos(x)*exp(-x**2/4)", axes_id="u_axes"),
                ],
                events=[
                    AnimationEvent(op="draw", target_id="u_wave"),
                    AnimationEvent(op="highlight", target_id="u_eq"),
                ],
            ),
        ],
    )

    ch7 = VideoPlanV3(
        topic="Gradient Descent",
        title="Chapter 7: Following the Slope Downhill",
        scenes=[
            SceneV3(
                scene_id="c7_s1",
                purpose="Introduce loss landscape and local slope",
                narration="How does a neural network learn without seeing the whole landscape at once? It measures the local *slope* beneath its feet and takes a small step downhill.",
                objects=[
                    SceneObject(id="gd_axes", kind="axes", persist=True, color_role="primary", x_range=[-3.0, 3.0], y_range=[-0.5, 9.0]),
                    SceneObject(id="gd_curve", kind="function_plot", persist=True, color_role="secondary", expr="x**2", axes_id="gd_axes"),
                    SceneObject(id="gd_pt", kind="point", persist=True, color_role="highlight", param_name="t", expr="t; t**2", param_init=2.4, axes_id="gd_axes"),
                    SceneObject(id="gd_eq", kind="equation", persist=True, color_role="highlight", tex="x_{n+1} = x_n - \\alpha f'(x_n)", card=True),
                ],
                events=[
                    AnimationEvent(op="create", target_id="gd_axes"),
                    AnimationEvent(op="draw", target_id="gd_curve"),
                    AnimationEvent(op="create", target_id="gd_pt"),
                    AnimationEvent(op="create", target_id="gd_eq"),
                    AnimationEvent(op="animate_param", target_id="gd_pt", param_range=[2.4, 0.0]),
                ],
            ),
            SceneV3(
                scene_id="c7_s2",
                purpose="Explain step size and convergence",
                narration="When the slope flattens near the bottom of the valley, each step automatically shrinks, guiding the parameters smoothly to the *minimum* without overshooting.",
                objects=[
                    SceneObject(id="gd_min", kind="point", persist=True, color_role="success", coords=[0.0, 0.0], axes_id="gd_axes"),
                ],
                events=[
                    AnimationEvent(op="create", target_id="gd_min"),
                    AnimationEvent(op="focus", target_id="gd_min"),
                ],
            ),
        ],
    )

    ch8 = VideoPlanV3(
        topic="Central Limit Theorem",
        title="Chapter 8: Order from Randomness",
        scenes=[
            SceneV3(
                scene_id="c8_s1",
                purpose="Show uniform die rolls",
                narration="Roll one fair die and every outcome from one to six is equally likely—completely flat. Yet the moment you start *averaging* independent rolls, structure appears.",
                objects=[
                    SceneObject(id="clt_bars", kind="bar_chart", persist=False, color_role="primary", values=[1, 1, 1, 1, 1, 1], bar_labels=["1", "2", "3", "4", "5", "6"]),
                    SceneObject(id="clt_eq1", kind="equation", persist=True, color_role="primary", tex="\\text{Single Roll: Uniform}"),
                ],
            ),
            SceneV3(
                scene_id="c8_s2",
                purpose="Converge to the Gaussian bell curve",
                narration="Average dozens of independent random variables together, and no matter what shape you started with, the distribution collapses into the universal *bell curve*.",
                objects=[
                    SceneObject(id="clt_axes", kind="axes", persist=True, color_role="primary", x_range=[-3.5, 3.5], y_range=[-0.2, 1.2]),
                    SceneObject(id="clt_bell", kind="function_plot", persist=True, color_role="highlight", expr="exp(-x**2/2)", axes_id="clt_axes"),
                    SceneObject(id="clt_eq2", kind="equation", persist=True, color_role="highlight", tex="f(x) = \\tfrac{1}{\\sqrt{2\\pi}}e^{-x^2/2}", card=True),
                ],
                events=[
                    AnimationEvent(op="create", target_id="clt_axes"),
                    AnimationEvent(op="draw", target_id="clt_bell"),
                    AnimationEvent(op="transform", source_id="clt_eq1", target_id="clt_eq2"),
                ],
            ),
        ],
    )

    ch9 = VideoPlanV3(
        topic="Euler Identity",
        title="Chapter 9: Complex Exponentials Spin",
        scenes=[
            SceneV3(
                scene_id="c9_s1",
                purpose="Show complex exponential rotation",
                narration="Ordinary exponentials grow along a straight line, but multiplying the exponent by *i* turns growth sideways into pure circular rotation around the origin.",
                objects=[
                    SceneObject(id="e_axes", kind="axes", persist=True, color_role="primary", x_range=[-1.5, 1.5], y_range=[-1.5, 1.5], equal_aspect=True),
                    SceneObject(id="e_circ", kind="circle", persist=True, color_role="secondary", coords=[0.0, 0.0], radius=1.0, axes_id="e_axes"),
                    SceneObject(id="e_vec", kind="vector", persist=True, color_role="highlight", coords=[0.0, 0.0], param_name="theta", expr="cos(theta); sin(theta)", param_init=0.0, axes_id="e_axes"),
                    SceneObject(id="e_eq", kind="equation", persist=True, color_role="highlight", tex="e^{i\\pi} + 1 = 0", card=True),
                ],
                events=[
                    AnimationEvent(op="create", target_id="e_axes"),
                    AnimationEvent(op="create", target_id="e_circ"),
                    AnimationEvent(op="create", target_id="e_vec"),
                    AnimationEvent(op="create", target_id="e_eq"),
                    AnimationEvent(op="animate_param", target_id="e_vec", param_range=[0.0, 3.14159]),
                ],
            ),
            SceneV3(
                scene_id="c9_s2",
                purpose="Highlight landing at minus one",
                narration="After rotating exactly pi radians—halfway around the unit circle—the vector lands at *minus one*, uniting five fundamental constants in one equation.",
                objects=[
                    SceneObject(id="e_pt", kind="point", persist=True, color_role="success", coords=[-1.0, 0.0], axes_id="e_axes"),
                ],
                events=[
                    AnimationEvent(op="create", target_id="e_pt"),
                    AnimationEvent(op="focus", target_id="e_pt"),
                ],
            ),
        ],
    )

    ch10 = VideoPlanV3(
        topic="Damped Oscillations",
        title="Chapter 10: Damped Oscillations in Nature",
        scenes=[
            SceneV3(
                scene_id="c10_s1",
                purpose="Plot damped cosine wave",
                narration="Real-world springs, bridges, and circuits never oscillate forever. Friction multiplies a pure cosine wave by a *decaying* exponential envelope.",
                objects=[
                    SceneObject(id="d_axes", kind="axes", persist=True, color_role="primary", x_range=[0.0, 10.0], y_range=[-1.2, 1.2]),
                    SceneObject(id="d_wave", kind="function_plot", persist=True, color_role="secondary", expr="exp(-0.25*x)*cos(3*x)", axes_id="d_axes"),
                    SceneObject(id="d_eq", kind="equation", persist=True, color_role="secondary", tex="y(t) = e^{-\\gamma t}\\cos(\\omega t)"),
                ],
            ),
            SceneV3(
                scene_id="c10_s2",
                purpose="Show exponential envelope",
                narration="That upper curve acts like an invisible ceiling, squeezing each swing closer to equilibrium while preserving the exact *rhythm* of oscillation.",
                objects=[
                    SceneObject(id="d_env", kind="function_plot", persist=True, color_role="highlight", expr="exp(-0.25*x)", axes_id="d_axes"),
                ],
                events=[
                    AnimationEvent(op="draw", target_id="d_env"),
                    AnimationEvent(op="highlight", target_id="d_wave"),
                ],
            ),
        ],
    )

    ch11 = VideoPlanV3(
        topic="Continuous Compounding",
        title="Chapter 11: Continuous Compounding",
        scenes=[
            SceneV3(
                scene_id="c11_s1",
                purpose="Compare linear and exponential growth",
                narration="Linear growth adds the same constant step every second, whereas *exponential* growth adds an amount proportional to how large the quantity already is.",
                objects=[
                    SceneObject(id="ex_axes", kind="axes", persist=True, color_role="primary", x_range=[-1.0, 3.0], y_range=[-0.5, 10.0]),
                    SceneObject(id="ex_lin", kind="function_plot", persist=True, color_role="primary", expr="1 + x", axes_id="ex_axes"),
                    SceneObject(id="ex_exp", kind="function_plot", persist=True, color_role="highlight", expr="exp(x)", axes_id="ex_axes"),
                    SceneObject(id="ex_eq", kind="equation", persist=True, color_role="highlight", tex="\\frac{d}{dx}e^x = e^x", card=True),
                ],
            ),
            SceneV3(
                scene_id="c11_s2",
                purpose="Highlight self-similar slope",
                narration="That is why e to the x is unique: at every single point, its height equals its *slope*, making it the natural language of continuous change.",
                objects=[],
                events=[
                    AnimationEvent(op="highlight", target_id="ex_exp"),
                    AnimationEvent(op="focus", target_id="ex_eq"),
                ],
            ),
        ],
    )

    ch12 = VideoPlanV3(
        topic="Grand Recap",
        title="Chapter 12: Putting Every Piece Together",
        scenes=[
            SceneV3(
                scene_id="c12_s1",
                purpose="Summarize geometric, wave, and probabilistic tools",
                narration="Step back and admire the toolkit we built: right triangles gave us distance, circles gave us *waves*, harmonics built sharp edges, and area measured belief.",
                objects=[
                    SceneObject(id="r_axes", kind="axes", persist=True, color_role="primary", x_range=[-3.5, 3.5], y_range=[-1.5, 1.5]),
                    SceneObject(id="r_cos", kind="function_plot", persist=True, color_role="secondary", expr="cos(x)", axes_id="r_axes"),
                    SceneObject(id="r_sin", kind="function_plot", persist=True, color_role="tertiary", expr="sin(x)", axes_id="r_axes"),
                    SceneObject(id="r_eq", kind="equation", persist=True, color_role="highlight", tex="\\text{Geometry} \\;\\leftrightarrow\\; \\text{Waves} \\;\\leftrightarrow\\; \\text{Probability}", card=True),
                ],
            ),
            SceneV3(
                scene_id="c12_s2",
                purpose="Closing invitation",
                narration="When you look at equations as living *transformations* rather than static symbols, every formula tells a clear visual story.",
                objects=[],
                events=[
                    AnimationEvent(op="highlight", target_id="r_eq"),
                ],
            ),
        ],
    )

    ch13 = VideoPlanV3(
        topic="Logistic Growth",
        title="Chapter 13: Saturation and Logistic Curves",
        scenes=[
            SceneV3(
                scene_id="c13_s1",
                purpose="Introduce logistic saturation curve",
                narration="In the real world, exponential growth cannot run forever—resources run out, bending the curve smoothly into an *S-shaped* logistic plateau.",
                objects=[
                    SceneObject(id="lg_axes", kind="axes", persist=True, color_role="primary", x_range=[-5.0, 5.0], y_range=[-0.2, 1.2]),
                    SceneObject(id="lg_curve", kind="function_plot", persist=True, color_role="highlight", expr="1/(1+exp(-x))", axes_id="lg_axes"),
                    SceneObject(id="lg_eq", kind="equation", persist=True, color_role="highlight", tex="\\sigma(x) = \\frac{1}{1 + e^{-x}}", card=True),
                ],
            ),
            SceneV3(
                scene_id="c13_s2",
                purpose="Highlight inflection point",
                narration="Right at the midpoint, growth hits its maximum *speed* before gently leveling off—a pattern shared by populations, epidemics, and neural networks.",
                objects=[
                    SceneObject(id="lg_pt", kind="point", persist=True, color_role="success", coords=[0.0, 0.5], axes_id="lg_axes"),
                ],
                events=[
                    AnimationEvent(op="create", target_id="lg_pt"),
                    AnimationEvent(op="focus", target_id="lg_pt"),
                ],
            ),
        ],
    )

    return lf_plan, [ch1, ch2, ch3, ch4, ch5, ch6, ch7, ch8, ch9, ch10, ch11, ch12, ch13]


if __name__ == "__main__":
    lf_plan, ch_plans = _make_demo_longform()
    res = build_longform_video(
        "math_unity_5min",
        lf_plan,
        ch_plans,
        format_name="widescreen_hd",
        tts_engine="kokoro",
        ui="card",
        look="cinematic",
        on_progress=lambda msg: print(f"[longform] {msg}", flush=True),
    )
    print(json.dumps(res, indent=2))
