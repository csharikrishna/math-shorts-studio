# V3.8 Baseline Audit & Empirical Quality Findings

Evaluated at baseline commits `a10b76c` -> `a785738` (branch `main`) via direct inspection of code modules, generated plans, unit tests, rendered MP4 streams (`ffprobe`/`ffmpeg`), decoded video frames (`PNG`), and voice/script transcripts (`SCRIPT.md`, `*_transcript.txt`). **Nothing in this report relies solely on `PROJECT_STATUS.md`.**

## 1. Automated Baseline Test Verification

- Command: `.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v`
- Result: **`Ran 57 tests in 1.345s — OK`** (`55` baseline V3.7 tests + `2` follow-up tests from `6629941`/`a785738`).

## 2. Timestamped Video Frame & Audio Inspection of Retained V3.7 Deliverables

### A. Eigenvalues Documentary (`light`, `editorial`, `manimgx`)
- **Artifact:** `output/LF_eigenvalues_phase_space_and_dynamical_stability_widescreen_hd.mp4` (`20.84 MB`)
- **Stream & Audio Measurements:** `1920x1080 @ 60.00 fps`, `48000 Hz stereo`, `585.42 s` (`09:45.42`), Integrated loudness `-15.3 LUFS`, TruePeak `-1.8 dBFS`, A/V drift `0.00 s`, `0` dead-air silences (`> 1.2 s`).
- **Frame & Script Findings:**
  - **Empty stage at `t=422.8s`** (`build/baseline_v38_frames/eig_422_8.png`): Decoded frame shows only bare `L`-shaped axes with zero curves, points, vectors, or equations. Existing `verify_decoded_frames()` sampled only `9` evenly spaced frames and missed this single empty frame because whole-video `median_stage_w=40.5%` stayed above `22%`.
  - **Annotation crowding at `t=487.9s`** (`QA_CONTACT_SHEET_light.png`): Annotation `"Stably Bounded Orbit"` sits close to the y-axis horizontal center line alongside `\lambda = \pm i` and a 2x2 matrix.
  - **Transcript & Script Ordering Defect:** `SCRIPT.md` shows `Revisions Preserved: 0` because `create_longform_plan()` generated chapter `VideoPlanV3` scenes first and exported `ScriptIR` afterward (`studio/longform.py:406-435`). Beat `ch_3__scene_1` contains literal `"x_dot"` in `spoken_text`, and `"without needing to solve..."` repeats across 4 beats (`ch_2__scene_4`, `ch_3__scene_1`, `ch_3__scene_3`, `ch_4__scene_2_scaling`).

### B. Markov Chains Documentary (`dark`, `editorial`, `manimgx`)
- **Artifact:** `output/LF_markov_chains_random_walks_and_stationary_equili_widescreen_hd.mp4` (`17.50 MB`)
- **Stream & Audio Measurements:** `1920x1080 @ 60.00 fps`, `48000 Hz stereo`, `490.92 s` (`08:10.92`), Integrated loudness `-15.3 LUFS`, TruePeak `-1.7 dBFS`, A/V drift `0.00 s`, `0` dead-air silences (`> 1.2 s`).
- **Frame & Publication-Gate Findings:**
  - **Blank/low-content axes at `t=81.8s`** (`build/baseline_v38_frames/mk_81_8.png`): Dark crosshair axes with a sub-pixel marker at the origin and no visible trajectory/walker while narration states *"As a particle wanders across discrete states..."*.
  - **Lonely equation card at `t=463.6s`** (`build/baseline_v38_frames/mk_463_6.png`): Displays only `\pi P = \pi` centered in an otherwise empty dark frame with zero state diagram or distribution chart.
  - **Confirmed False Publication Approval (`REVIEW.md`):** Reported `Approved: True` (`90.0/100`) despite **`unsupported=11`** and `scientific_factual_accuracy: human_review_required`, because `audit_educational_quality()` (`studio/qa/engine.py:610-612`) capped unsupported-claim penalties at `10.0` points without gating approval.
- **Root-Cause Classification of All 11 Unsupported Claims (`logs/ch_7bfe5a69f0f155d4_c*.json`):**
  - `c1/scene_1`: Unclaimed random walk intuition scene (`number_line`, `point`) — conceptual hook lacking intuition exemption or trajectory construction.
  - `c2/scene_1_amnesia_start`, `c2/scene_3_state_evolution`, `c2/scene_4_conclusion` (`3` scenes): Matrix power / stationary equilibrium assertions omitted both `StructuredClaim` and `markov_chain_evolution` construction.
  - `c3/scene_matrix_intro`, `c3/scene_probability_vector`, `c3/scene_summary` (`3` scenes): Transition matrix & row-sum-to-one assertions (`matrix`, `equation`) lacked verified `markov_stochastic_invariant` / `probability_normalization` claims.
  - `c5/scene_1`, `c5/scene_2`, `c5/scene_4` (`3` scenes): Random-surfer graph (`graph_network`), 3x3 stochastic matrix (`[[0,0.5,0.5],[1,0,0],[0.33,0.33,0.34]]`), and eigenvector equation (`equation`) omitted stochastic verification.
  - `c6/scene_2` (`1` scene): Stationary invariant equation `\pi P = \pi` had no verified claim.
  - **Underlying Validator Defect (`studio/ai/validate_v3.py:521-526`):** `markov_stochastic_invariant` only verified `bool(cl.statement.strip())` without numerically checking matrix non-negativity, row sums = `1.0`, or invariant `\pi P = \pi`.

### C. Derived Portrait Short, Theme/Caption Samples & Hybrid Experiment
- **Portrait Short (`output/SHORT_eigenvalues_phase_space_and_dynamical_stability_shorts_hd.mp4`):** `1080x1920 @ 60.00 fps`, `55.00 s`, `-15.2 LUFS`, `-2.5 dBFS`, `0.00 s` drift, `layout: no problems in 8 samples`, two-line `kinetic` captions aligned cleanly per row with `plane_1` top and `mat_a` bottom.
- **Theme / Caption Comparison (`output/V3_eig_light_widescreen_hd.mp4`, `output/V3_eig_dark_widescreen_hd.mp4`):** Verified consistent framing, math curves, matrices, and crisp `editorial` subtitles across both looks (`eig_cap_light_5s.png`, `eig_cap_dark_5s.png`).
- **Hybrid & HyperFrames Benchmark (`output/bench_*_shorts_hd.mp4`, `build/hybrid_hyb_smoke_widescreen_fast`):** Cross-backend timing parity (`58.40 s`), clean `Filter_complex` `fps/setpts/yuv420p` mux (`hyb_smoke`), while `studio/renderers/hybrid.py:157-177` only copies initial `SceneObject` declarations across backend switches rather than evaluated coordinates, transforms, parameters, and removals.

## 3. Confirmed Architectural Gaps Addressed in V3.8

1. **Authoritative Script-First Ordering & Provenance (`studio/longform.py`, `studio/creative/script_ir.py`, `director.py`):** Move `ScriptIR` drafting, review, and immutable approval BEFORE chapter scene compilation; eliminate positional fallback (`script.beats[idx]`).
2. **Enforceable Executable Registry (`studio/registry/`):** Require runnable `test_fn` (and `compute_fn` / `validate_fn` for math ops) for every implemented component; exclude uncertified entries from planning.
3. **Evaluated Scene Runtime State (`studio/renderers/scene_state.py`, `hybrid.py`):** Carry rotated/moved/transformed coordinates, parameter values, camera focus, and `removed` tombstones across `ManimGX <-> HyperFrames`.
4. **Canonical Shared `CaptionIR` (`studio/captions/`, `studio/timeline.py`, `studio/audio/tts.py`):** Compile once from `AudioTimeline` with word-level provenance (`tts_word_boundary` vs `interpolated`) and feed identical cues to `SRT`/`WebVTT`/`ASS`/`ManimGX`/`HyperFrames`/preview.
5. **Honest Publication Gates & Scene-Aware Frame QA (`studio/qa/`):** Gate `publication_approved` on zero unsupported/failed claims, genuine animation pixel deltas, caption sync audit, and scene-start/mid/end/bookmark frame sampling.
6. **Dependency-Aware Caching & Preview (`studio/storage/cache.py`, `studio/preview.py`):** Split fingerprints across all 12 pipeline layers and add lightweight scene-level inspection/rerender workbench.
