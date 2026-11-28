"""Reusable linked-data computational mathematical models driving synchronized visual views."""

from __future__ import annotations

import math
import re
from typing import Any, Literal, Optional
import numpy as np
from pydantic import BaseModel, Field

from studio.ai.expr import compile_expr
from studio.ai.math_truth import (
    bayes_posterior,
    fourier_partial_sum_terms,
    gradient_descent_trajectory,
    matrix_transform_and_eigen,
    rk4_ode_trajectory,
    symbolic_derivative,
)

LinkedModelKind = Literal[
    "signal_spectral_pipeline",
    "optimization_trajectory",
    "bayesian_probabilistic",
    "dynamical_eigen",
    "series_approximation",
    "dynamical_simulation",
    "frequency_response",
    "recurrence_system",
    "stochastic_ensemble",
]


class LinkedMathModelSpec(BaseModel):
    """Schema-safe declaration for one shared mathematical model driving multiple visual views."""

    model_id: str
    kind: LinkedModelKind
    label: str = ""
    expr: Optional[str] = None
    t_range: Optional[list[float]] = None
    freq_range: Optional[list[float]] = None
    selected_freq: Optional[float] = None
    cutoff_freq: Optional[float] = None
    high_cutoff_freq: Optional[float] = None
    filter_kind: Literal["lowpass", "bandpass", "highpass"] = "lowpass"
    angular_units: Literal["hz", "rad"] = "hz"
    n_samples: int = 240
    # Optimization / series / Bayesian / dynamical / general simulation params
    x0: Optional[float] = None
    y0: Optional[float] = None
    lr: Optional[float] = None
    steps: Optional[int] = None
    n_terms: Optional[int] = None
    period: Optional[float] = None
    prior: Optional[float] = None
    p_e_given_h: Optional[float] = None
    p_e_given_not_h: Optional[float] = None
    matrix_2x2: Optional[list[list[float]]] = None
    dx_expr: Optional[str] = None
    dy_expr: Optional[str] = None
    state_vars: list[str] = Field(default_factory=list)
    rhs_exprs: list[str] = Field(default_factory=list)
    initial_state: list[float] = Field(default_factory=list)
    params: dict[str, float] = Field(default_factory=dict)
    derived_exprs: dict[str, str] = Field(default_factory=dict)
    force_expr: Optional[str] = None
    geometry_mode: Literal["linear_spring", "pendulum", "phase_body"] = "linear_spring"
    solver: Literal["rk4", "rk4_adaptive", "rk45"] = "rk45"
    rtol: float = 1e-5
    atol: float = 1e-7
    natural_freq: Optional[float] = None
    damping_ratio: Optional[float] = None
    drive_amp: Optional[float] = None
    mass: Optional[float] = None
    stiffness: Optional[float] = None
    damping_coeff: Optional[float] = None
    units: dict[str, str] = Field(default_factory=dict)
    assumptions: list[str] = Field(default_factory=list)
    diffusion_stds: list[float] = Field(default_factory=list)
    n_paths: Optional[int] = None
    seed: Optional[int] = None


def _eval_signal_fn(expr_str: str) -> Any:
    """Compile signal expression accepting either t or x as independent time variable."""
    norm = (expr_str or "sin(2*pi*2*t)").strip()
    try:
        fn = compile_expr(norm, ("t", "x"))
        return lambda tv: fn(t=tv, x=tv)
    except Exception:
        fn = compile_expr(norm, ("x",))
        return lambda tv: fn(x=tv)


def find_spectral_peaks(
    freqs: np.ndarray | list[float],
    mags: np.ndarray | list[float],
    *,
    min_rel_height: float = 0.12,
    min_sep_bins: int = 1,
) -> list[dict[str, float]]:
    """Detect true local spectral maxima above a relative prominence floor (never labels grid samples blindly)."""
    f_arr = np.asarray(freqs, dtype=float)
    m_arr = np.asarray(mags, dtype=float)
    if f_arr.size == 0 or m_arr.size == 0 or f_arr.size != m_arr.size:
        return []
    if not (np.all(np.isfinite(f_arr)) and np.all(np.isfinite(m_arr))):
        raise ValueError("Nonfinite frequency or magnitude values encountered during spectral peak detection.")
    mx = float(np.max(m_arr))
    if mx <= 1e-12:
        return []
    thresh = mx * float(min_rel_height)
    cands: list[tuple[int, float, float]] = []
    n = len(m_arr)
    for i in range(n):
        v = float(m_arr[i])
        if v < thresh:
            continue
        left_ok = (i == 0 and n > 1 and v > float(m_arr[1])) or (i > 0 and v >= float(m_arr[i - 1]))
        right_ok = (i == n - 1 and n > 1 and v > float(m_arr[-2])) or (i < n - 1 and v > float(m_arr[i + 1]))
        if left_ok and right_ok:
            cands.append((i, float(f_arr[i]), v))
    # Suppress adjacent duplicate peaks within min_sep_bins keeping taller peak
    cands.sort(key=lambda t: t[2], reverse=True)
    kept: list[tuple[int, float, float]] = []
    for idx_c, fv, mv in cands:
        if any(abs(idx_c - k_idx) <= min_sep_bins for k_idx, _, _ in kept):
            continue
        kept.append((idx_c, fv, mv))
    kept.sort(key=lambda t: t[1])
    return [{"freq": round(fv, 4), "amplitude": round(mv, 5), "normalized": round(mv / mx, 5)} for _, fv, mv in kept]


def evaluate_signal_spectral_pipeline(spec: LinkedMathModelSpec) -> dict[str, Any]:
    """Compute linked time signal, complex winding z(t)=x(t)e^{-iwt}, DFT bins, peaks, spectrum & filtered reconstruction."""
    tr = spec.t_range or [0.0, 2.0]
    t0, t1 = float(tr[0]), float(tr[1])
    if t1 <= t0:
        t1 = t0 + 2.0
    n_pts = max(120, min(600, int(spec.n_samples or 240)))
    if n_pts % 2 == 1:
        n_pts += 1
    span_t = max(1e-9, t1 - t0)

    # Consistent endpoint-exclusive sampling for periodic DFT/FFT & continuous quadrature without duplicate-endpoint leakage
    dt_step = span_t / float(n_pts)
    ts_dft = t0 + np.arange(n_pts, dtype=float) * dt_step
    ts_plot = np.linspace(t0, t1, n_pts, endpoint=True)
    fn = _eval_signal_fn(spec.expr or "0.85*cos(2*pi*2*t) + 0.55*cos(2*pi*5*t)")
    xs_dft = np.asarray([float(fn(float(t))) for t in ts_dft], dtype=float)
    xs_plot = np.asarray([float(fn(float(t))) for t in ts_plot], dtype=float)
    if not (np.all(np.isfinite(xs_dft)) and np.all(np.isfinite(xs_plot))):
        raise ValueError(f"Nonfinite sample(s) produced evaluating signal expression {spec.expr!r}.")

    fr = spec.freq_range or [0.0, 7.0]
    f_min, f_max = float(fr[0]), float(fr[1])
    if f_max <= f_min:
        f_max = f_min + 6.0
    sel_f = float(spec.selected_freq if spec.selected_freq is not None else 2.0)
    cut_f = float(spec.cutoff_freq if spec.cutoff_freq is not None else 3.5)
    high_cut_f = float(spec.high_cutoff_freq if spec.high_cutoff_freq is not None else f_max)
    scale_w = (2.0 * math.pi) if spec.angular_units == "hz" else 1.0

    def _wind_and_centroid(f_val: float, samples_t: np.ndarray = ts_plot, samples_x: np.ndarray = xs_plot) -> tuple[list[list[float]], list[float], float]:
        ang = scale_w * f_val * samples_t
        u = samples_x * np.cos(ang)
        v = -samples_x * np.sin(ang)
        if hasattr(np, "trapezoid"):
            cx = float(np.trapezoid(u, samples_t) / span_t)
            cy = float(np.trapezoid(v, samples_t) / span_t)
        else:
            cx = float(np.trapz(u, samples_t) / span_t)
            cy = float(np.trapz(v, samples_t) / span_t)
        pts = [[round(float(ui), 5), round(float(vi), 5)] for ui, vi in zip(u, v)]
        return pts, [round(cx, 5), round(cy, 5)], round(math.hypot(cx, cy), 5)

    wound_pts, centroid_xy, centroid_mag = _wind_and_centroid(sel_f)

    # Exact periodic DFT / single-sided amplitude spectrum on endpoint-exclusive samples
    spec_fft = np.fft.rfft(xs_dft)
    fft_freqs_hz = np.fft.rfftfreq(n_pts, d=dt_step)
    fft_axis = fft_freqs_hz if spec.angular_units == "hz" else (fft_freqs_hz * 2.0 * math.pi)
    single_sided = np.abs(spec_fft) / float(n_pts)
    if len(single_sided) > 2:
        single_sided[1:-1] *= 2.0
    elif len(single_sided) == 2:
        single_sided[1] *= 2.0

    in_band = [(i, float(fft_axis[i]), float(single_sided[i])) for i in range(len(fft_axis)) if f_min - 1e-6 <= fft_axis[i] <= f_max + 1e-6]
    dft_freqs_in = [round(f_v, 4) for _, f_v, _ in in_band]
    dft_amps_in = [round(a_v, 5) for _, _, a_v in in_band]
    dft_peaks = find_spectral_peaks(dft_freqs_in, dft_amps_in, min_rel_height=0.12, min_sep_bins=1)

    # Finite-window continuous Fourier centroid curve |C(f)| ensuring selected_freq & detected peak freqs are included
    base_freqs = np.linspace(f_min, f_max, 91)
    peak_fs = [float(pk["freq"]) for pk in dft_peaks if f_min <= float(pk["freq"]) <= f_max]
    freqs = np.array(sorted({round(float(v), 5) for v in (*base_freqs, sel_f, *peak_fs) if f_min - 1e-6 <= float(v) <= f_max + 1e-6}), dtype=float)
    spec_pts: list[list[float]] = []
    mags: list[float] = []
    for fv in freqs:
        _, _, m_val = _wind_and_centroid(float(fv))
        spec_pts.append([round(float(fv), 4), round(m_val, 5)])
        mags.append(m_val)

    max_mag = max(mags) if mags else 1.0
    norm_spec_pts = [[fp, round(mp / max(max_mag, 1e-9), 5)] for fp, mp in spec_pts]
    cont_peaks = find_spectral_peaks(freqs, mags, min_rel_height=0.15, min_sep_bins=2)
    dom_freq = float(dft_peaks[0]["freq"]) if dft_peaks else (float(freqs[int(np.argmax(mags))]) if len(freqs) else sel_f)
    if sel_f in [float(pk["freq"]) for pk in dft_peaks]:
        dom_freq = sel_f
    elif dft_peaks:
        best_pk = max(dft_peaks, key=lambda d: float(d["amplitude"]))
        dom_freq = float(best_pk["freq"])

    # Synchronized frequency sweep keyframes GUARANTEED to include exact sel_f frame for state resets
    sw_base = np.linspace(max(0.2, f_min), f_max, 15)
    sweep_freqs = sorted({round(float(fv), 4) for fv in (*sw_base, sel_f) if f_min - 1e-6 <= float(fv) <= f_max + 1e-6})
    sweep_frames: list[dict[str, Any]] = []
    unit_lbl = "Hz" if spec.angular_units == "hz" else "rad/s"
    n_sw_tot = max(1, len(sweep_freqs) - 1)
    sel_sweep_u = 0.0
    for idx_s, fv in enumerate(sweep_freqs):
        w_pts_k, c_xy_k, mag_k = _wind_and_centroid(float(fv))
        stride = max(1, len(w_pts_k) // 180)
        u_val = round(idx_s / float(n_sw_tot), 6)
        if abs(fv - sel_f) <= 1e-4:
            sel_sweep_u = u_val
        _, _, c_spec_val = _wind_and_centroid(float(fv))
        sweep_frames.append(
            {
                "param": round(float(fv), 4),
                "u": u_val,
                "curve_pts": w_pts_k[::stride],
                "point": c_xy_k,
                "vector_end": c_xy_k,
                "spec_point": [round(float(fv), 4), round(c_spec_val, 5)],
                "label": f"f = {fv:.2f} {unit_lbl}  |  |C| = {mag_k:.3f}",
            }
        )

    # Exact FFT spectral filtering & Parseval energy conservation on endpoint-exclusive grid
    if spec.filter_kind == "lowpass":
        mask = (fft_axis <= cut_f + 1e-9).astype(float)
    elif spec.filter_kind == "highpass":
        mask = (fft_axis >= cut_f - 1e-9).astype(float)
    else:
        mask = ((fft_axis >= cut_f - 1e-9) & (fft_axis <= high_cut_f + 1e-9)).astype(float)

    filtered_fft = spec_fft * mask
    xs_filt_dft = np.fft.irfft(filtered_fft, n=n_pts)
    if not np.all(np.isfinite(xs_filt_dft)):
        raise ValueError("Nonfinite values in filtered signal reconstruction.")
    # Periodic smooth evaluation at ts_plot (repeating endpoint t0+T sample cleanly)
    xs_filt_plot = np.append(xs_filt_dft, xs_filt_dft[0])[:n_pts] if len(xs_filt_dft) == n_pts else xs_filt_dft
    # Reconstruct time-aligned endpoint sample accurately by shifting phase / interpolating end
    xs_filt_plot = np.interp(ts_plot - t0, np.append(ts_dft - t0, span_t), np.append(xs_filt_dft, xs_filt_dft[0]))

    # Distinguish theoretical filter mask response H(f) from actual spectrum of reconstructed discrete filtered signal!
    theo_filter_resp: list[list[float]] = []
    filt_spec_pts: list[list[float]] = []
    for fv in freqs:
        pass_h = 1.0 if ((spec.filter_kind == "lowpass" and float(fv) <= cut_f) or (spec.filter_kind == "highpass" and float(fv) >= cut_f) or (spec.filter_kind == "bandpass" and cut_f <= float(fv) <= high_cut_f)) else 0.0
        theo_filter_resp.append([round(float(fv), 4), pass_h])
        _, _, fm_val = _wind_and_centroid(float(fv), samples_t=ts_dft, samples_x=xs_filt_dft)
        filt_spec_pts.append([round(float(fv), 4), round(fm_val, 5)])

    filt_single_sided = np.abs(np.fft.rfft(xs_filt_dft)) / float(n_pts)
    if len(filt_single_sided) > 2:
        filt_single_sided[1:-1] *= 2.0
    filt_dft_amps_in = [round(float(filt_single_sided[i]), 5) for i, _, _ in in_band]

    # Parseval identity on periodic DFT grid: sum(|x|^2) == (1/N) * (|X_0|^2 + 2 sum_{k=1}^{N/2-1} |X_k|^2 + |X_{N/2}|^2)
    e_time_raw = float(np.sum(xs_dft ** 2))
    e_freq_raw = float((abs(spec_fft[0]) ** 2 + 2.0 * np.sum(np.abs(spec_fft[1:-1]) ** 2) + abs(spec_fft[-1]) ** 2) / n_pts)
    e_time_filt = float(np.sum(xs_filt_dft ** 2))
    e_freq_filt = float((abs(filtered_fft[0]) ** 2 + 2.0 * np.sum(np.abs(filtered_fft[1:-1]) ** 2) + abs(filtered_fft[-1]) ** 2) / n_pts)
    parseval_rel = abs(e_time_raw - e_freq_raw) / max(1e-12, e_time_raw)

    # Bar summary directly from exact DFT frequency bins (preserving all detected peak bins like 2.0 Hz & 5.0 Hz!)
    non_dc = [(f_v, a_v) for f_v, a_v in zip(dft_freqs_in, dft_amps_in) if f_v > 1e-6]
    if len(non_dc) > 12:
        # Keep top peak bins + even integer bins cleanly without dropping true component bins
        peak_set = {float(pk["freq"]) for pk in dft_peaks}
        chosen = [pair for pair in non_dc if (pair[0] in peak_set or abs(pair[0] - round(pair[0])) < 1e-3)]
        if len(chosen) < 6:
            chosen = non_dc[:12]
        bar_pairs = sorted(chosen, key=lambda x: x[0])
    else:
        bar_pairs = non_dc or list(zip(dft_freqs_in, dft_amps_in))
    peak_set = {float(pk["freq"]) for pk in dft_peaks}
    bar_freqs = [round(p[0], 3) for p in bar_pairs]
    bar_vals = [round(p[1], 4) for p in bar_pairs]
    bar_labels = [f"{bf:g}*" if any(abs(bf - pkf) <= 1e-3 for pkf in peak_set) else f"{bf:g}" for bf in bar_freqs]

    time_pts = [[round(float(tv), 4), round(float(xv), 5)] for tv, xv in zip(ts_plot, xs_plot)]
    filt_time_pts = [[round(float(tv), 4), round(float(xv), 5)] for tv, xv in zip(ts_plot, xs_filt_plot)]
    phase_rad = math.atan2(centroid_xy[1], centroid_xy[0])
    sel_mag_exact = next((m for f, m in spec_pts if abs(f - sel_f) <= 1e-3), centroid_mag)

    return {
        "kind": spec.kind,
        "sampling_convention": "endpoint_exclusive_periodic_dft + composite_trapezoid_winding",
        "time_signal": time_pts,
        "winding_curve": wound_pts,
        "centroid_point": centroid_xy,
        "centroid_vector": [[0.0, 0.0], centroid_xy],
        "spectrum": spec_pts,
        "continuous_fourier_spectrum": spec_pts,
        "normalized_spectrum": norm_spec_pts,
        "dft_single_sided_spectrum": [[f, a] for f, a in zip(dft_freqs_in, dft_amps_in)],
        "dft_filtered_spectrum": [[f, a] for f, a in zip(dft_freqs_in, filt_dft_amps_in)],
        "dft_bins": {"freqs": dft_freqs_in, "single_sided_amp": dft_amps_in, "filtered_single_sided_amp": filt_dft_amps_in},
        "detected_peaks": dft_peaks,
        "continuous_peaks": cont_peaks,
        "spectrum_peak": [round(sel_f, 4), round(float(sel_mag_exact), 5)],
        "spectrum_bars_labels": bar_labels,
        "spectrum_bars_values": bar_vals,
        "theoretical_filter_response": theo_filter_resp,
        "filtered_spectrum": filt_spec_pts,
        "filtered_signal": filt_time_pts,
        "sweep_frames": sweep_frames,
        "selected_sweep_u": sel_sweep_u,
        "t_axis": [round(float(v), 5) for v in ts_dft],
        "raw_samples": [round(float(v), 6) for v in xs_dft],
        "filtered_samples": [round(float(v), 6) for v in xs_filt_dft],
        "measurements": {
            "selected_freq": sel_f,
            "selected_sweep_u": sel_sweep_u,
            "dominant_freq": round(dom_freq, 4),
            "detected_peak_freqs": [float(pk["freq"]) for pk in dft_peaks],
            "detected_peak_amps": [float(pk["amplitude"]) for pk in dft_peaks],
            "centroid_x": centroid_xy[0],
            "centroid_y": centroid_xy[1],
            "centroid_mag": centroid_mag,
            "phase_rad": round(phase_rad, 5),
            "cutoff_freq": cut_f,
            "dt_s": round(dt_step, 6),
            "df_hz": round(1.0 / span_t, 6),
            "nyquist_hz": round(0.5 / dt_step, 4),
            "energy_time_raw": round(e_time_raw, 4),
            "energy_freq_raw": round(e_freq_raw, 4),
            "energy_time_filtered": round(e_time_filt, 4),
            "energy_freq_filtered": round(e_freq_filt, 4),
            "parseval_rel_error": round(parseval_rel, 10),
        },
        "readout_text": f"f={sel_f:.2f}{unit_lbl} | C=({centroid_xy[0]:+.2f}, {centroid_xy[1]:+.2f}) | |C|={centroid_mag:.3f}",
    }


def evaluate_optimization_trajectory_model(spec: LinkedMathModelSpec) -> dict[str, Any]:
    expr = spec.expr or "0.2*x**4 - 0.8*x**2 + 0.1*x"
    x0 = float(spec.x0 if spec.x0 is not None else 2.0)
    lr = float(spec.lr if spec.lr is not None else 0.12)
    steps = int(spec.steps if spec.steps is not None else 10)
    tr = spec.t_range or [-3.0, 3.0]
    fn = compile_expr(expr, ("x",))
    d_expr = symbolic_derivative(expr, "x")
    dfn = compile_expr(d_expr, ("x",))
    xs = np.linspace(float(tr[0]), float(tr[1]), 160)
    curve = [[round(float(xv), 4), round(float(fn(x=float(xv))), 5)] for xv in xs]
    d_curve = [[round(float(xv), 4), round(float(dfn(x=float(xv))), 5)] for xv in xs]
    traj = gradient_descent_trajectory(expr, x0=x0, lr=lr, steps=steps)
    traj_pts = [[round(pt[0], 5), round(pt[1], 5)] for pt in traj]
    loss_pts = [[float(idx), round(pt[1], 5)] for idx, pt in enumerate(traj)]
    sweep_frames = [
        {
            "param": float(idx),
            "point": [round(pt[0], 5), round(pt[1], 5)],
            "spec_point": [float(idx), round(pt[1], 5)],
            "label": f"step {idx}: x={pt[0]:.3f}, f={pt[1]:.3f}",
        }
        for idx, pt in enumerate(traj)
    ]
    return {
        "kind": spec.kind,
        "curve": curve,
        "derivative_curve": d_curve,
        "trajectory": traj_pts,
        "loss_curve": loss_pts,
        "end_point": traj_pts[-1],
        "sweep_frames": sweep_frames,
        "measurements": {
            "x0": x0,
            "x_final": traj[-1][0],
            "f_final": traj[-1][1],
            "grad_final": traj[-1][2],
            "derivative_expr": d_expr,
        },
        "readout_text": f"x*={traj[-1][0]:.3f}, f(x*)={traj[-1][1]:.3f}, f'(x*)={traj[-1][2]:.3f}",
    }


def evaluate_bayesian_model(spec: LinkedMathModelSpec) -> dict[str, Any]:
    prior = float(spec.prior if spec.prior is not None else 0.05)
    p_eh = float(spec.p_e_given_h if spec.p_e_given_h is not None else 0.90)
    p_enh = float(spec.p_e_given_not_h if spec.p_e_given_not_h is not None else 0.08)
    post = bayes_posterior(prior, p_eh, p_enh)
    return {
        "kind": spec.kind,
        "bar_labels": ["Prior P(H)", "Evidence P(E)", "Posterior P(H|E)"],
        "bar_values": [round(post["prior"], 4), round(post["p_e"], 4), round(post["posterior"], 4)],
        "table_headers": ["Quantity", "Value"],
        "table_rows": [
            ["Prior P(H)", f"{post['prior']:.4f}"],
            ["Likelihood P(E|H)", f"{p_eh:.4f}"],
            ["False Positive P(E|~H)", f"{p_enh:.4f}"],
            ["Posterior P(H|E)", f"{post['posterior']:.4f}"],
        ],
        "measurements": post,
        "readout_text": f"Prior={post['prior']:.3f} -> Posterior={post['posterior']:.3f}",
    }


def evaluate_dynamical_eigen_model(spec: LinkedMathModelSpec) -> dict[str, Any]:
    mat = spec.matrix_2x2 or [[-0.35, -1.2], [1.2, -0.35]]
    x0 = float(spec.x0 if spec.x0 is not None else 1.8)
    y0 = float(spec.y0 if spec.y0 is not None else 0.0)
    steps = int(spec.steps if spec.steps is not None else 90)
    if spec.dx_expr and spec.dy_expr:
        dx_e = re.sub(r"\bv\b", "y", spec.dx_expr.strip())
        dy_e = re.sub(r"\bv\b", "y", spec.dy_expr.strip())
        eig = matrix_transform_and_eigen(mat)
        traj = rk4_ode_trajectory(dx_e, dy_e, x0=x0, y0=y0, dt=0.08, steps=steps)
        readout = f"ODE orbit dx/dt={dx_e} | end=({traj[-1][1]:.2f}, {traj[-1][2]:.2f})"
    else:
        eig = matrix_transform_and_eigen(mat)
        dx_e = f"({mat[0][0]})*x + ({mat[0][1]})*y"
        dy_e = f"({mat[1][0]})*x + ({mat[1][1]})*y"
        traj = rk4_ode_trajectory(dx_e, dy_e, x0=x0, y0=y0, dt=0.08, steps=steps)
        readout = f"regime={eig['regime']}, tr={eig['trace']:.2f}, det={eig['determinant']:.2f}"
    orbit = [[round(xt, 5), round(yt, 5)] for (_, xt, yt) in traj]
    ts_x = [[round(tv, 4), round(xt, 5)] for (tv, xt, _) in traj]
    ts_y = [[round(tv, 4), round(yt, 5)] for (tv, _, yt) in traj]
    return {
        "kind": spec.kind,
        "matrix_rows": [[f"{mat[0][0]:.2g}", f"{mat[0][1]:.2g}"], [f"{mat[1][0]:.2g}", f"{mat[1][1]:.2g}"]],
        "phase_orbit": orbit,
        "time_series_x": ts_x,
        "time_series_y": ts_y,
        "end_point": orbit[-1],
        "measurements": eig,
        "readout_text": readout,
    }


def evaluate_series_approximation_model(spec: LinkedMathModelSpec) -> dict[str, Any]:
    expr = spec.expr or "x"
    period = float(spec.period if spec.period is not None else 2.0 * math.pi)
    n_terms = int(spec.n_terms if spec.n_terms is not None else 5)
    f_res = fourier_partial_sum_terms(expr, period=period, n_harmonics=n_terms)
    fn_target = compile_expr(expr, ("x",))
    fn_approx = compile_expr(str(f_res["expr"]), ("x",))
    tr = spec.t_range or [-period * 0.5, period * 0.5]
    xs = np.linspace(float(tr[0]), float(tr[1]), 180)
    tgt_pts = [[round(float(xv), 4), round(float(fn_target(x=float(xv))), 5)] for xv in xs]
    app_pts = [[round(float(xv), 4), round(float(fn_approx(x=float(xv))), 5)] for xv in xs]
    err_pts = [[round(float(xv), 4), round(abs(float(fn_target(x=float(xv))) - float(fn_approx(x=float(xv)))), 5)] for xv in xs]
    l2_err = float(math.sqrt(np.mean([p[1] ** 2 for p in err_pts])))
    return {
        "kind": spec.kind,
        "target_curve": tgt_pts,
        "approx_curve": app_pts,
        "error_curve": err_pts,
        "approx_expr": str(f_res["expr"]),
        "measurements": {"l2_error": round(l2_err, 5), "n_terms": n_terms},
        "readout_text": f"N={n_terms} harmonics | L2 error={l2_err:.4f}",
    }


def evaluate_dynamical_simulation_model(spec: LinkedMathModelSpec) -> dict[str, Any]:
    """Evaluate explicitly specified N-variable ODE simulation with strict dimension, physical unit, and assumption checks."""
    from studio.ai.simulation import (
        build_simulation_sweep_frames,
        evaluate_derived_observables,
        integrate_ode_system,
    )

    s_vars = [str(v).strip() for v in (spec.state_vars or []) if str(v).strip()]
    if not s_vars:
        raise ValueError(f"dynamical_simulation {spec.model_id!r} requires explicit non-empty state_vars (silent fallback disabled).")

    assumptions = list(spec.assumptions or [])
    if spec.mass is not None and float(spec.mass) <= 0.0:
        raise ValueError(f"dynamical_simulation {spec.model_id!r} mass must be positive (got {spec.mass}).")
    if spec.stiffness is not None and float(spec.stiffness) < 0.0:
        raise ValueError(f"dynamical_simulation {spec.model_id!r} stiffness must be non-negative (got {spec.stiffness}).")
    if spec.damping_coeff is not None and float(spec.damping_coeff) < 0.0:
        raise ValueError(f"dynamical_simulation {spec.model_id!r} damping_coeff must be non-negative (got {spec.damping_coeff}).")

    rhs: list[str]
    explicit_mech_energy = False
    eff_mass: float | None = float(spec.mass) if spec.mass is not None else None
    eff_k: float | None = float(spec.stiffness) if spec.stiffness is not None else None

    if spec.rhs_exprs:
        if len(spec.rhs_exprs) != len(s_vars):
            raise ValueError(
                f"dynamical_simulation {spec.model_id!r}: rhs_exprs length ({len(spec.rhs_exprs)}) "
                f"must equal state_vars length ({len(s_vars)})."
            )
        rhs = [str(ex).strip() for ex in spec.rhs_exprs]
        if any(not ex for ex in rhs):
            raise ValueError(f"dynamical_simulation {spec.model_id!r}: rhs_exprs cannot contain blank expressions.")
    elif spec.dx_expr and spec.dy_expr:
        if len(s_vars) != 2:
            raise ValueError(f"dynamical_simulation {spec.model_id!r}: dx_expr/dy_expr requires exactly 2 state_vars.")
        rhs = [spec.dx_expr.strip(), spec.dy_expr.strip()]
    elif spec.natural_freq is not None and spec.damping_ratio is not None and len(s_vars) == 2:
        w0 = float(spec.natural_freq)
        zeta = float(spec.damping_ratio)
        if w0 <= 0.0 or not math.isfinite(w0):
            raise ValueError(f"natural_freq must be finite and positive (got {w0}).")
        if zeta < 0.0 or not math.isfinite(zeta):
            raise ValueError(f"damping_ratio must be finite and non-negative (got {zeta}).")
        v0, v1 = s_vars[0], s_vars[1]
        rhs = [v1, f"-({w0 * w0})*{v0} - ({2.0 * zeta * w0})*{v1}"]
        eff_mass = float(spec.mass) if spec.mass is not None else 1.0
        eff_k = float(spec.stiffness) if spec.stiffness is not None else eff_mass * (w0 * w0)
        explicit_mech_energy = True
        assumptions.append(f"linear_second_order_oscillator(w0={w0}, zeta={zeta}, m={eff_mass})")
    else:
        raise ValueError(
            f"dynamical_simulation {spec.model_id!r} requires explicit rhs_exprs matching state_vars "
            "(or dx_expr/dy_expr, or both natural_freq & damping_ratio); silent fallback disabled."
        )

    if spec.initial_state:
        if len(spec.initial_state) != len(s_vars):
            raise ValueError(
                f"dynamical_simulation {spec.model_id!r}: initial_state dimension ({len(spec.initial_state)}) "
                f"does not match state_vars ({len(s_vars)})."
            )
        init = [float(v) for v in spec.initial_state]
    elif len(s_vars) == 2 and spec.x0 is not None and spec.y0 is not None:
        init = [float(spec.x0), float(spec.y0)]
    else:
        raise ValueError(
            f"dynamical_simulation {spec.model_id!r}: explicit initial_state matching state_vars={s_vars} required (silent default disabled)."
        )

    tr = spec.t_range or [0.0, 10.0]
    steps = int(spec.steps if spec.steps is not None else 140)
    sim = integrate_ode_system(
        state_vars=s_vars,
        rhs_exprs=rhs,
        initial_state=init,
        t_range=tr,
        params=spec.params,
        steps=steps,
        method=spec.solver,
        rtol=spec.rtol,
        atol=spec.atol,
    )

    derived_spec = dict(spec.derived_exprs or {})
    if not derived_spec and explicit_mech_energy and eff_mass is not None and eff_k is not None and len(s_vars) >= 2:
        v0, v1 = s_vars[0], s_vars[1]
        derived_spec = {
            "potential_energy": f"0.5*({eff_k})*({v0})**2",
            "kinetic_energy": f"0.5*({eff_mass})*({v1})**2",
            "total_energy": f"0.5*({eff_k})*({v0})**2 + 0.5*({eff_mass})*({v1})**2",
        }
    elif not derived_spec and (spec.mass is not None and spec.stiffness is not None) and len(s_vars) >= 2:
        v0, v1 = s_vars[0], s_vars[1]
        derived_spec = {
            "potential_energy": f"0.5*({float(spec.stiffness)})*({v0})**2",
            "kinetic_energy": f"0.5*({float(spec.mass)})*({v1})**2",
            "total_energy": f"0.5*({float(spec.stiffness)})*({v0})**2 + 0.5*({float(spec.mass)})*({v1})**2",
        }
        assumptions.append(f"mechanical_energy(m={spec.mass}, k={spec.stiffness})")

    derived = evaluate_derived_observables(sim, derived_spec, params=spec.params)
    ts_x = [[tv, float(st[0])] for tv, st in zip(sim.t_axis, sim.states)]
    ts_y = [[tv, float(st[1])] for tv, st in zip(sim.t_axis, sim.states)] if len(s_vars) >= 2 else []
    phase_orbit = [[float(st[0]), float(st[1])] for st in sim.states] if len(s_vars) >= 2 else []

    # Never confuse raw acceleration with force unless explicit force_expr or explicit mass*accel is declared
    force_expr: str | None = spec.force_expr
    force_mass: float | None = eff_mass
    if force_expr is None and spec.mass is not None and len(rhs) >= 2:
        force_expr = rhs[1]
        assumptions.append(f"force_from_mass_times_accel(mass={spec.mass})")
    elif force_expr is None and explicit_mech_energy and eff_mass is not None and len(rhs) >= 2:
        force_expr = rhs[1]

    sweep_frames = build_simulation_sweep_frames(
        sim,
        derived=derived,
        force_expr=force_expr,
        mass=force_mass if force_expr == (rhs[1] if len(rhs) >= 2 else None) else None,
        params=spec.params,
        n_keyframes=24,
        geometry_mode=spec.geometry_mode,
    )
    energy_keys = [k for k in ("potential_energy", "kinetic_energy", "total_energy", "conserved_invariant") if k in derived]
    if not energy_keys and derived:
        energy_keys = list(derived.keys())[:3]
    e_labels = [k.replace("_energy", "").replace("_", " ").title() for k in energy_keys]
    e_vals0 = [round(float(derived[k][0]), 5) for k in energy_keys] if energy_keys else []
    energy_total_curve = [[tv, float(derived[energy_keys[-1]][i])] for i, tv in enumerate(sim.t_axis)] if energy_keys else []
    end_pt = phase_orbit[-1] if phase_orbit else ts_x[-1]
    meas_extra: dict[str, Any] = {}
    if "total_energy" in derived:
        meas_extra["energy_initial"] = derived["total_energy"][0]
        meas_extra["energy_final"] = derived["total_energy"][-1]
    elif "conserved_invariant" in derived:
        vals_inv = derived["conserved_invariant"]
        meas_extra["invariant_initial"] = vals_inv[0]
        meas_extra["invariant_max_drift"] = round(max(abs(v - vals_inv[0]) for v in vals_inv), 6)
    return {
        "kind": spec.kind,
        "state_vars": s_vars,
        "rhs_exprs": rhs,
        "t_axis": sim.t_axis,
        "states": sim.states,
        "time_series_x": ts_x,
        "time_series_y": ts_y,
        "phase_orbit": phase_orbit,
        "energy_total": energy_total_curve,
        "energy_bars_labels": e_labels,
        "energy_bars_values": e_vals0,
        "bar_labels": e_labels,
        "bar_values": e_vals0,
        "end_point": end_pt,
        "body_point": sweep_frames[0]["body_pos"] if sweep_frames else [init[0], 0.0],
        "centroid_point": sweep_frames[0]["point"] if sweep_frames else end_pt,
        "centroid_vector": [sweep_frames[0]["vector_start"], sweep_frames[0]["vector_end"]] if sweep_frames else [[0.0, 0.0], [0.0, 0.0]],
        "sweep_frames": sweep_frames,
        "selected_sweep_u": 1.0,
        "derived": derived,
        "units": dict(spec.units or {}),
        "assumptions": assumptions,
        "measurements": {
            "solver": sim.solver,
            "solver_algorithm": sim.summary.get("solver_algorithm", sim.solver),
            "max_error_est": sim.max_error_est,
            "rtol": spec.rtol,
            "atol": spec.atol,
            "initial_state": init,
            "end_state": sim.states[-1],
            "x_min": round(min(p[1] for p in ts_x), 5),
            "x_max": round(max(p[1] for p in ts_x), 5),
            "y_min": round(min(p[1] for p in ts_y), 5) if ts_y else None,
            "y_max": round(max(p[1] for p in ts_y), 5) if ts_y else None,
            **meas_extra,
        },
        "readout_text": sweep_frames[-1]["label"] if sweep_frames else f"end={sim.states[-1]}",
    }


def evaluate_frequency_response_model(spec: LinkedMathModelSpec) -> dict[str, Any]:
    """Evaluate exact steady-state amplitude/phase resonance curve & valid resonance peak threshold."""
    from studio.ai.simulation import compute_driven_frequency_response

    w0 = float(spec.natural_freq if spec.natural_freq is not None else 2.0)
    zeta = float(spec.damping_ratio if spec.damping_ratio is not None else 0.25)
    f0 = float(spec.drive_amp if spec.drive_amp is not None else 1.0)
    res = compute_driven_frequency_response(
        natural_freq=w0,
        damping_ratio=zeta,
        drive_amp=f0,
        freq_range=spec.freq_range,
        n_pts=int(spec.n_samples or 120),
    )
    resp_pts = res["response_curve"]
    sweep_frames = [
        {
            "param": float(pt[0]),
            "u": round(i / max(1, len(resp_pts) - 1), 6),
            "point": list(pt),
            "spec_point": list(pt),
            "label": f"w={pt[0]:.2f} | |H|={pt[1]:.3f}",
        }
        for i, pt in enumerate(resp_pts[:: max(1, len(resp_pts) // 20)])
    ]
    pk = res["peak_point"]
    if res.get("has_singularity"):
        readout_txt = f"w0={res['natural_freq']:.2f}, zeta=0.00 (undamped) | unbounded resonance singularity at w=w0"
    elif res["has_nonzero_resonance_peak"]:
        readout_txt = f"w0={res['natural_freq']:.2f}, zeta={res['damping_ratio']:.2f} ({res['damping_regime']}) | w_res={res['resonance_freq']:.2f}"
    else:
        readout_txt = f"w0={res['natural_freq']:.2f}, zeta={res['damping_ratio']:.2f} ({res['damping_regime']}) | no nonzero resonance peak (zeta >= 1/sqrt(2))"
    return {
        "kind": spec.kind,
        "response_curve": resp_pts,
        "phase_curve": res["phase_curve"],
        "spectrum": resp_pts,
        "spectrum_peak": pk,
        "end_point": pk,
        "sweep_frames": sweep_frames,
        "selected_sweep_u": 1.0,
        "measurements": {k: v for k, v in res.items() if k not in ("response_curve", "phase_curve")},
        "readout_text": readout_txt,
    }


def evaluate_recurrence_model(spec: LinkedMathModelSpec) -> dict[str, Any]:
    """Evaluate discrete recurrence evolution."""
    from studio.ai.simulation import build_simulation_sweep_frames, iterate_recurrence_system

    s_vars = list(spec.state_vars or ["x"])
    rhs = list(spec.rhs_exprs or [spec.expr or "0.85*x"])
    init = list(spec.initial_state or ([spec.x0] if spec.x0 is not None else [1.0]))
    sim = iterate_recurrence_system(state_vars=s_vars, update_exprs=rhs, initial_state=init, steps=int(spec.steps or 25), params=spec.params)
    ts_x = [[tv, float(st[0])] for tv, st in zip(sim.t_axis, sim.states)]
    sweep = build_simulation_sweep_frames(sim, params=spec.params, n_keyframes=min(24, len(ts_x)))
    return {
        "kind": spec.kind,
        "time_series_x": ts_x,
        "trajectory": ts_x,
        "end_point": ts_x[-1],
        "sweep_frames": sweep,
        "bar_labels": [str(int(tv)) for tv in sim.t_axis[:12]],
        "bar_values": [float(st[0]) for st in sim.states[:12]],
        "measurements": sim.summary,
        "readout_text": f"n={int(sim.t_axis[-1])}: state={sim.states[-1]}",
    }


def evaluate_stochastic_ensemble_model(spec: LinkedMathModelSpec) -> dict[str, Any]:
    """Evaluate seeded stochastic sample ensemble, mean trajectory, and std bounds."""
    from studio.ai.simulation import run_stochastic_ensemble

    s_vars = list(spec.state_vars or ["x"])
    drift = list(spec.rhs_exprs or [spec.expr or "-0.4*x"])
    init = list(spec.initial_state or ([spec.x0] if spec.x0 is not None else [1.5]))
    stds = list(spec.diffusion_stds or [0.35] * len(s_vars))
    res = run_stochastic_ensemble(
        state_vars=s_vars,
        drift_exprs=drift,
        diffusion_stds=stds,
        initial_state=init,
        t_range=spec.t_range or [0.0, 6.0],
        steps=int(spec.steps or 80),
        n_paths=int(spec.n_paths or 10),
        seed=int(spec.seed or 42),
        params=spec.params,
    )
    return {
        "kind": spec.kind,
        "mean_curve": res["mean_curve"],
        "time_series_x": res["mean_curve"],
        "upper_std_curve": res["upper_std_curve"],
        "lower_std_curve": res["lower_std_curve"],
        "end_point": res["mean_curve"][-1],
        "measurements": {"final_mean": res["final_mean"], "final_std": res["final_std"]},
        "readout_text": f"Ensemble mean={res['final_mean']:.3f} ± {res['final_std']:.3f}",
    }


def evaluate_linked_math_model(spec: LinkedMathModelSpec) -> dict[str, Any]:
    """Dispatch any LinkedMathModelSpec deterministically into verified visual data."""
    if spec.kind == "signal_spectral_pipeline":
        return evaluate_signal_spectral_pipeline(spec)
    if spec.kind == "optimization_trajectory":
        return evaluate_optimization_trajectory_model(spec)
    if spec.kind == "bayesian_probabilistic":
        return evaluate_bayesian_model(spec)
    if spec.kind == "dynamical_eigen":
        return evaluate_dynamical_eigen_model(spec)
    if spec.kind == "series_approximation":
        return evaluate_series_approximation_model(spec)
    if spec.kind == "dynamical_simulation":
        return evaluate_dynamical_simulation_model(spec)
    if spec.kind == "frequency_response":
        return evaluate_frequency_response_model(spec)
    if spec.kind == "recurrence_system":
        return evaluate_recurrence_model(spec)
    if spec.kind == "stochastic_ensemble":
        return evaluate_stochastic_ensemble_model(spec)
    raise ValueError(f"Unsupported LinkedMathModelSpec kind: {spec.kind!r}")
