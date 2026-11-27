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
    # Optimization / series / Bayesian / dynamical params
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


def _eval_signal_fn(expr_str: str) -> Any:
    """Compile signal expression accepting either t or x as independent time variable."""
    norm = (expr_str or "sin(2*pi*2*t)").strip()
    try:
        fn = compile_expr(norm, ("t", "x"))
        return lambda tv: fn(t=tv, x=tv)
    except Exception:
        fn = compile_expr(norm, ("x",))
        return lambda tv: fn(x=tv)


def evaluate_signal_spectral_pipeline(spec: LinkedMathModelSpec) -> dict[str, Any]:
    """Compute linked time signal, complex winding curve z(t)=x(t)e^{-iwt}, numerical centroid, spectrum & filter."""
    tr = spec.t_range or [0.0, 2.0]
    t0, t1 = float(tr[0]), float(tr[1])
    if t1 <= t0:
        t1 = t0 + 2.0
    n_pts = max(120, min(600, int(spec.n_samples or 240)))
    ts = np.linspace(t0, t1, n_pts)
    fn = _eval_signal_fn(spec.expr or "0.85*cos(2*pi*2*t) + 0.55*cos(2*pi*5*t)")
    xs = np.asarray([float(fn(float(t))) for t in ts], dtype=float)
    xs = np.nan_to_num(xs, nan=0.0, posinf=0.0, neginf=0.0)

    fr = spec.freq_range or [0.0, 7.0]
    f_min, f_max = float(fr[0]), float(fr[1])
    if f_max <= f_min:
        f_max = f_min + 6.0
    sel_f = float(spec.selected_freq if spec.selected_freq is not None else 2.0)
    cut_f = float(spec.cutoff_freq if spec.cutoff_freq is not None else 3.5)
    high_cut_f = float(spec.high_cutoff_freq if spec.high_cutoff_freq is not None else f_max)
    span_t = max(1e-9, t1 - t0)
    scale_w = (2.0 * math.pi) if spec.angular_units == "hz" else 1.0

    def _wind_and_centroid(f_val: float) -> tuple[list[list[float]], list[float], float]:
        ang = scale_w * f_val * ts
        u = xs * np.cos(ang)
        v = -xs * np.sin(ang)
        if hasattr(np, "trapezoid"):
            cx = float(np.trapezoid(u, ts) / span_t)
            cy = float(np.trapezoid(v, ts) / span_t)
        else:
            cx = float(np.trapz(u, ts) / span_t)
            cy = float(np.trapz(v, ts) / span_t)
        pts = [[round(float(ui), 5), round(float(vi), 5)] for ui, vi in zip(u, v)]
        return pts, [round(cx, 5), round(cy, 5)], round(math.hypot(cx, cy), 5)

    wound_pts, centroid_xy, centroid_mag = _wind_and_centroid(sel_f)

    # Full continuous magnitude spectrum derived via integral on SAME x(t)
    n_spec = 85
    freqs = np.linspace(f_min, f_max, n_spec)
    spec_pts: list[list[float]] = []
    mags: list[float] = []
    for fv in freqs:
        _, _, m_val = _wind_and_centroid(float(fv))
        spec_pts.append([round(float(fv), 4), round(m_val, 5)])
        mags.append(m_val)

    dom_idx = int(np.argmax(mags)) if mags else 0
    dom_freq = float(freqs[dom_idx]) if len(freqs) else sel_f

    # Keyframed frequency sweep for animated synchronized views
    n_sweep = 16
    sweep_freqs = np.linspace(max(0.2, f_min), f_max, n_sweep)
    sweep_frames: list[dict[str, Any]] = []
    unit_lbl = "Hz" if spec.angular_units == "hz" else "rad/s"
    for fv in sweep_freqs:
        w_pts_k, c_xy_k, mag_k = _wind_and_centroid(float(fv))
        # Downsample polyline cleanly to 90 points per keyframe for crisp GPU & CDP playback
        stride = max(1, len(w_pts_k) // 90)
        sweep_frames.append(
            {
                "param": round(float(fv), 3),
                "curve_pts": w_pts_k[::stride],
                "point": c_xy_k,
                "vector_end": c_xy_k,
                "spec_point": [round(float(fv), 3), mag_k],
                "label": f"f = {fv:.2f} {unit_lbl}  |  |C| = {mag_k:.3f}",
            }
        )

    # Exact FFT spectral filtering & Parseval energy conservation
    dt_step = span_t / max(1, n_pts - 1)
    spec_fft = np.fft.rfft(xs)
    fft_freqs_hz = np.fft.rfftfreq(n_pts, d=dt_step)
    fft_axis = fft_freqs_hz if spec.angular_units == "hz" else (fft_freqs_hz * 2.0 * math.pi)
    if spec.filter_kind == "lowpass":
        mask = (fft_axis <= cut_f).astype(float)
    elif spec.filter_kind == "highpass":
        mask = (fft_axis >= cut_f).astype(float)
    else:
        mask = ((fft_axis >= cut_f) & (fft_axis <= high_cut_f)).astype(float)

    filtered_fft = spec_fft * mask
    xs_filt = np.fft.irfft(filtered_fft, n=n_pts)

    # Filtered magnitude curve derived via mask * spectrum
    filt_spec_pts: list[list[float]] = []
    for fv, mv in spec_pts:
        pass_h = 1.0 if ((spec.filter_kind == "lowpass" and fv <= cut_f) or (spec.filter_kind == "highpass" and fv >= cut_f) or (spec.filter_kind == "bandpass" and cut_f <= fv <= high_cut_f)) else 0.0
        filt_spec_pts.append([fv, round(mv * pass_h, 5)])

    # Parseval identity: sum(|x|^2) == (1/N) * sum(|X_dft|^2)
    e_time_raw = float(np.sum(xs ** 2))
    e_freq_raw = float((abs(spec_fft[0]) ** 2 + 2.0 * np.sum(np.abs(spec_fft[1:-1]) ** 2) + (abs(spec_fft[-1]) ** 2 if n_pts % 2 == 0 else 2.0 * abs(spec_fft[-1]) ** 2)) / n_pts)
    e_time_filt = float(np.sum(xs_filt ** 2))
    e_freq_filt = float((abs(filtered_fft[0]) ** 2 + 2.0 * np.sum(np.abs(filtered_fft[1:-1]) ** 2) + (abs(filtered_fft[-1]) ** 2 if n_pts % 2 == 0 else 2.0 * abs(filtered_fft[-1]) ** 2)) / n_pts)
    parseval_rel = abs(e_time_raw - e_freq_raw) / max(1e-9, e_time_raw)

    # Discrete bar summary at integer harmonics
    bar_freqs = [round(f_min + k * max(0.5, (f_max - f_min) / 7.0), 1) for k in range(8)]
    bar_vals = [round(_wind_and_centroid(bf)[2], 4) for bf in bar_freqs]

    time_pts = [[round(float(tv), 4), round(float(xv), 5)] for tv, xv in zip(ts, xs)]
    filt_time_pts = [[round(float(tv), 4), round(float(xv), 5)] for tv, xv in zip(ts, xs_filt)]
    phase_rad = math.atan2(centroid_xy[1], centroid_xy[0])

    return {
        "kind": spec.kind,
        "time_signal": time_pts,
        "winding_curve": wound_pts,
        "centroid_point": centroid_xy,
        "centroid_vector": [[0.0, 0.0], centroid_xy],
        "spectrum": spec_pts,
        "spectrum_peak": [round(dom_freq, 4), round(max(mags) if mags else 0.0, 5)],
        "spectrum_bars_labels": [f"{bf:g}" for bf in bar_freqs],
        "spectrum_bars_values": bar_vals,
        "filtered_spectrum": filt_spec_pts,
        "filtered_signal": filt_time_pts,
        "sweep_frames": sweep_frames,
        "t_axis": [round(float(v), 5) for v in ts],
        "raw_samples": [round(float(v), 6) for v in xs],
        "filtered_samples": [round(float(v), 6) for v in xs_filt],
        "measurements": {
            "selected_freq": sel_f,
            "dominant_freq": round(dom_freq, 4),
            "centroid_x": centroid_xy[0],
            "centroid_y": centroid_xy[1],
            "centroid_mag": centroid_mag,
            "phase_rad": round(phase_rad, 5),
            "cutoff_freq": cut_f,
            "energy_time_raw": round(e_time_raw, 4),
            "energy_freq_raw": round(e_freq_raw, 4),
            "energy_time_filtered": round(e_time_filt, 4),
            "energy_freq_filtered": round(e_freq_filt, 4),
            "parseval_rel_error": round(parseval_rel, 8),
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
    raise ValueError(f"Unsupported LinkedMathModelSpec kind: {spec.kind!r}")
