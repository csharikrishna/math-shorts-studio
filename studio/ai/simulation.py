"""General-purpose deterministic numerical simulation engine (ODEs, frequency response, recurrences, ensembles)."""

from __future__ import annotations

import math
from typing import Any, Callable, Iterable, Sequence
import numpy as np
from pydantic import BaseModel, Field

from studio.ai.expr import compile_expr


class SimulationResult(BaseModel):
    """Structured finite-checked output of a numerical simulation."""

    solver: str
    state_vars: list[str]
    t_axis: list[float]
    states: list[list[float]]  # shape: [len(t), len(state_vars)]
    derived: dict[str, list[float]] = Field(default_factory=dict)
    max_error_est: float = 0.0
    summary: dict[str, Any] = Field(default_factory=dict)


def _assert_finite_array(arr: np.ndarray, label: str) -> None:
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"Nonfinite value encountered in simulation ({label}); never silently replacing with zero.")


def _compile_param_expr(
    expr_str: str,
    var_names: tuple[str, ...],
    params: dict[str, float] | None = None,
) -> Callable[..., float]:
    """Safely substitute scalar parameter constants via allowed names into AST-whitelisted compile_expr."""
    p_map = {str(k): float(v) for k, v in (params or {}).items() if math.isfinite(float(v))}
    all_vars = tuple(dict.fromkeys([*var_names, *p_map.keys()]))
    fn = compile_expr(expr_str.strip(), all_vars)

    def _call(**vals: float) -> float:
        merged = {**p_map, **vals}
        res = float(fn(**{k: merged[k] for k in all_vars if k in merged}))
        if not math.isfinite(res):
            raise ValueError(f"Nonfinite simulation evaluation for {expr_str!r} at {vals}")
        return res

    return _call


def integrate_ode_system(
    *,
    state_vars: Sequence[str],
    rhs_exprs: Sequence[str],
    initial_state: Sequence[float],
    t_range: Sequence[float] = (0.0, 10.0),
    params: dict[str, float] | None = None,
    steps: int = 160,
    method: str = "rk45",
    rtol: float = 1e-5,
    atol: float = 1e-7,
    max_steps: int = 2000,
    max_abs_state: float = 1e5,
) -> SimulationResult:
    """Integrate arbitrary N-variable ODE dx/dt = f(state, params, t) via Cash-Karp RK4(5), RK4 step-doubling, or RK4."""
    if method not in ("rk45", "rk4_adaptive", "rk4"):
        raise ValueError(f"Unsupported ODE integration method {method!r}; expected 'rk45', 'rk4_adaptive', or 'rk4'.")
    if rtol <= 0.0 or atol <= 0.0:
        raise ValueError(f"ODE tolerances rtol and atol must be strictly positive (got rtol={rtol}, atol={atol}).")
    s_vars = [str(v).strip() for v in state_vars if str(v).strip()]
    if not s_vars or len(s_vars) != len(rhs_exprs) or len(s_vars) != len(initial_state):
        raise ValueError("state_vars, rhs_exprs, and initial_state must have identical nonzero length.")
    if len(t_range) < 2:
        raise ValueError(f"t_range must specify [t0, t1] (got {t_range!r}).")
    t0, t1 = float(t_range[0]), float(t_range[1])
    if not (math.isfinite(t0) and math.isfinite(t1)) or t1 <= t0:
        raise ValueError(f"ODE integration requires finite t1 > t0 (got [{t0}, {t1}]).")
    n_out = max(12, min(max_steps, int(steps)))
    allowed = tuple(dict.fromkeys([*s_vars, "t"]))
    fns = [_compile_param_expr(ex, allowed, params) for ex in rhs_exprs]
    y0 = np.asarray([float(v) for v in initial_state], dtype=float)
    _assert_finite_array(y0, "initial_state")
    if np.max(np.abs(y0)) > max_abs_state:
        raise ValueError(f"Initial ODE state exceeds max_abs_state={max_abs_state}.")

    def _rhs(t_val: float, y_vec: np.ndarray) -> np.ndarray:
        kw = {nm: float(y_vec[idx]) for idx, nm in enumerate(s_vars)}
        kw["t"] = float(t_val)
        out = np.asarray([fn(**kw) for fn in fns], dtype=float)
        _assert_finite_array(out, f"rhs(t={t_val:.4f})")
        return out

    def _check_state(vec: np.ndarray, t_val: float) -> np.ndarray:
        _assert_finite_array(vec, f"state(t={t_val:.4f})")
        if np.max(np.abs(vec)) > max_abs_state:
            raise ValueError(f"Simulation state diverged beyond bound {max_abs_state} at t={t_val:.3f}")
        return vec

    def _rk4_step(t_cur: float, y_cur: np.ndarray, h: float) -> np.ndarray:
        k1 = _rhs(t_cur, y_cur)
        k2 = _rhs(t_cur + 0.5 * h, y_cur + 0.5 * h * k1)
        k3 = _rhs(t_cur + 0.5 * h, y_cur + 0.5 * h * k2)
        k4 = _rhs(t_cur + h, y_cur + h * k3)
        return _check_state(y_cur + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4), t_cur + h)

    # Genuine 6-stage Cash-Karp embedded RK4(5) Butcher tableau (order 5 advance, order 4 error estimate)
    def _cash_karp_step(t_cur: float, y_cur: np.ndarray, h: float) -> tuple[np.ndarray, np.ndarray]:
        k1 = _rhs(t_cur, y_cur)
        k2 = _rhs(t_cur + 0.2 * h, y_cur + h * (0.2 * k1))
        k3 = _rhs(t_cur + 0.3 * h, y_cur + h * (0.075 * k1 + 0.225 * k2))
        k4 = _rhs(t_cur + 0.6 * h, y_cur + h * (0.3 * k1 - 0.9 * k2 + 1.2 * k3))
        k5 = _rhs(t_cur + h, y_cur + h * ((-11.0 / 54.0) * k1 + 2.5 * k2 - (70.0 / 27.0) * k3 + (35.0 / 27.0) * k4))
        k6 = _rhs(
            t_cur + 0.875 * h,
            y_cur + h * ((1631.0 / 55296.0) * k1 + (175.0 / 512.0) * k2 + (575.0 / 13824.0) * k3 + (44275.0 / 110592.0) * k4 + (253.0 / 4096.0) * k5),
        )
        y5 = y_cur + h * ((37.0 / 378.0) * k1 + (250.0 / 621.0) * k3 + (125.0 / 594.0) * k4 + (512.0 / 1771.0) * k6)
        y4 = y_cur + h * ((2825.0 / 27648.0) * k1 + (18575.0 / 48384.0) * k3 + (13525.0 / 55296.0) * k4 + (277.0 / 14336.0) * k5 + 0.25 * k6)
        _check_state(y5, t_cur + h)
        return y5, np.abs(y5 - y4)

    ts_grid = np.linspace(t0, t1, n_out + 1, dtype=float)
    states_out: list[list[float]] = [[round(float(x), 6) for x in y0]]
    max_err_abs = 0.0
    max_err_norm = 0.0
    min_h = 1e-6

    y = y0.copy()
    sub_min = 4 if method in ("rk45", "rk4_adaptive") else 2
    for i in range(n_out):
        ta, tb = float(ts_grid[i]), float(ts_grid[i + 1])
        span = tb - ta
        if method in ("rk45", "rk4_adaptive"):
            t_cur = ta
            h = span / float(sub_min)
            sub_iters = 0
            while t_cur < tb - 1e-12:
                sub_iters += 1
                if sub_iters > 256:
                    raise ValueError(f"Adaptive ODE ({method}) exceeded 256 substeps near t={t_cur:.4f} (stiff or singular).")
                h = min(h, tb - t_cur)
                if method == "rk45":
                    y_next, err_vec = _cash_karp_step(t_cur, y, h)
                else:
                    y_full = _rk4_step(t_cur, y, h)
                    y_half = _rk4_step(t_cur + 0.5 * h, _rk4_step(t_cur, y, 0.5 * h), 0.5 * h)
                    err_vec = np.abs(y_half - y_full) / 15.0
                    y_next = _check_state(y_half + (y_half - y_full) / 15.0, t_cur + h)
                scale = atol + rtol * np.maximum(np.abs(y), np.abs(y_next))
                err_norm = float(np.max(err_vec / scale))
                if err_norm <= 1.0:
                    max_err_abs = max(max_err_abs, float(np.max(err_vec)))
                    max_err_norm = max(max_err_norm, err_norm)
                    y = y_next
                    t_cur += h
                    h = h * min(2.0, max(0.5, 0.9 * (max(err_norm, 1e-12) ** -0.2)))
                elif h <= min_h + 1e-15:
                    raise ValueError(
                        f"Adaptive ODE ({method}) failed tolerance at minimum step h={h:.2e} near t={t_cur:.4f} "
                        f"(err_norm={err_norm:.3f} > 1.0 for rtol={rtol}, atol={atol})."
                    )
                else:
                    h = max(min_h, h * max(0.25, 0.85 * (err_norm ** -0.25)))
        else:
            dt_s = span / float(sub_min)
            for k in range(sub_min):
                y = _rk4_step(ta + k * dt_s, y, dt_s)
        states_out.append([round(float(x), 6) for x in y])

    solver_desc = "cash_karp_rk45" if method == "rk45" else ("rk4_step_doubling" if method == "rk4_adaptive" else "rk4_fixed")
    return SimulationResult(
        solver=method,
        state_vars=s_vars,
        t_axis=[round(float(tv), 5) for tv in ts_grid],
        states=states_out,
        max_error_est=round(max_err_abs, 9),
        summary={
            "t0": t0,
            "t1": t1,
            "n_steps": n_out,
            "solver_algorithm": solver_desc,
            "rtol": float(rtol),
            "atol": float(atol),
            "max_err_norm": round(max_err_norm, 6),
            "end_state": states_out[-1],
        },
    )


def evaluate_derived_observables(
    sim: SimulationResult,
    derived_exprs: dict[str, str],
    params: dict[str, float] | None = None,
) -> dict[str, list[float]]:
    """Compute conserved and derived scalar quantities (energy, work, invariants) across simulation trajectory."""
    out: dict[str, list[float]] = {}
    if not derived_exprs:
        return out
    var_names = tuple(dict.fromkeys([*sim.state_vars, "t"]))
    for name, expr in derived_exprs.items():
        fn = _compile_param_expr(expr, var_names, params)
        vals: list[float] = []
        for t_v, st in zip(sim.t_axis, sim.states):
            kw = {nm: float(st[idx]) for idx, nm in enumerate(sim.state_vars)}
            kw["t"] = float(t_v)
            vals.append(round(float(fn(**kw)), 6))
        out[str(name)] = vals
    return out


def compute_driven_frequency_response(
    *,
    natural_freq: float = 2.0,
    damping_ratio: float = 0.25,
    drive_amp: float = 1.0,
    freq_range: Sequence[float] | None = None,
    n_pts: int = 120,
) -> dict[str, Any]:
    """Exact steady-state amplitude, phase, and resonance classification for 2nd-order linear forced oscillator."""
    w0 = float(natural_freq)
    zeta = float(damping_ratio)
    f0 = float(drive_amp)
    if w0 <= 0.0 or not math.isfinite(w0):
        raise ValueError(f"natural_freq must be finite and positive (got {w0}).")
    if zeta < 0.0 or not math.isfinite(zeta):
        raise ValueError(f"damping_ratio must be finite and non-negative (got {zeta}).")
    if not math.isfinite(f0):
        raise ValueError(f"drive_amp must be finite (got {f0}).")

    fr = freq_range or [0.05 * w0, 2.5 * w0]
    w_min, w_max = max(0.0, float(fr[0])), float(fr[1])
    if w_max <= w_min:
        w_max = w_min + 2.5 * w0

    if zeta == 0.0:
        regime = "undamped"
    elif zeta < 1.0 - 1e-4:
        regime = "underdamped"
    elif zeta <= 1.0 + 1e-4:
        regime = "critically_damped"
    else:
        regime = "overdamped"

    w_damped = round(w0 * math.sqrt(max(0.0, 1.0 - zeta * zeta)), 6) if zeta < 1.0 else None
    has_nonzero_res = bool(zeta < (1.0 / math.sqrt(2.0)) - 1e-6)
    w_res = round(w0 * math.sqrt(max(0.0, 1.0 - 2.0 * zeta * zeta)), 6) if has_nonzero_res else 0.0

    base_w = np.linspace(w_min, w_max, max(40, int(n_pts)))
    pole_band = 0.025 * w0 if zeta == 0.0 else 0.0
    key_ws = ([w0, w_res] if zeta > 0.0 else []) + ([w_damped] if (w_damped is not None and zeta > 0.0) else [])
    ws = np.array(
        sorted(
            {
                round(float(v), 5)
                for v in (*base_w, *key_ws)
                if w_min - 1e-6 <= float(v) <= w_max + 1e-6 and (zeta > 0.0 or abs(float(v) - w0) > pole_band)
            }
        ),
        dtype=float,
    )
    resp_pts: list[list[float]] = []
    phase_pts: list[list[float]] = []
    gains: list[float] = []
    for w in ws:
        re_d = w0 * w0 - w * w
        im_d = 2.0 * zeta * w0 * w
        denom = math.hypot(re_d, im_d)
        if denom <= 1e-9:
            # Never fabricate arbitrary finite gain at an exact undamped singularity
            continue
        amp = abs(f0) / denom
        phi = -math.atan2(im_d, re_d)
        resp_pts.append([round(float(w), 5), round(float(amp), 5)])
        phase_pts.append([round(float(w), 5), round(float(phi), 5)])
        gains.append(float(amp))

    has_singularity = bool(zeta == 0.0 and w_min <= w0 <= w_max and abs(f0) > 0.0)
    pk_idx = int(np.argmax(gains)) if gains else 0
    pk_w = float(ws[pk_idx]) if len(ws) else w_res
    pk_amp = float(gains[pk_idx]) if gains else 0.0
    static_deflection = abs(f0) / (w0 * w0)
    return {
        "natural_freq": round(w0, 5),
        "damping_ratio": round(zeta, 5),
        "damping_regime": regime,
        "damped_freq": w_damped,
        "has_nonzero_resonance_peak": has_nonzero_res,
        "has_singularity": has_singularity,
        "singularity_freq": round(w0, 5) if has_singularity else None,
        "resonance_freq": w_res,
        "measured_peak_freq": round(w0, 5) if has_singularity else round(pk_w, 5),
        "peak_gain": None if has_singularity else round(pk_amp, 5),
        "static_deflection": round(static_deflection, 5),
        "response_curve": resp_pts,
        "phase_curve": phase_pts,
        "peak_point": [round(w0 if has_singularity else pk_w, 5), round(pk_amp, 5)],
        "natural_marker": None if has_singularity else [round(w0, 5), round(abs(f0) / (2.0 * zeta * w0 * w0), 5)],
    }


def iterate_recurrence_system(
    *,
    state_vars: Sequence[str],
    update_exprs: Sequence[str],
    initial_state: Sequence[float],
    steps: int = 30,
    params: dict[str, float] | None = None,
    max_abs_state: float = 1e6,
) -> SimulationResult:
    """Iterate general discrete-time recurrence x_{n+1} = F(x_n, ..., n)."""
    s_vars = [str(v).strip() for v in state_vars if str(v).strip()]
    if not s_vars or len(s_vars) != len(update_exprs) or len(s_vars) != len(initial_state):
        raise ValueError("state_vars, update_exprs, and initial_state lengths must match cleanly.")
    n_steps = max(2, min(500, int(steps)))
    allowed = tuple(dict.fromkeys([*s_vars, "n", "t"]))
    fns = [_compile_param_expr(ex, allowed, params) for ex in update_exprs]
    cur = [float(x) for x in initial_state]
    if not all(math.isfinite(x) for x in cur):
        raise ValueError("Nonfinite initial state in recurrence system.")
    states = [[round(x, 6) for x in cur]]
    ts = [0.0]
    for step_idx in range(n_steps):
        kw = {nm: cur[idx] for idx, nm in enumerate(s_vars)}
        kw["n"] = float(step_idx)
        kw["t"] = float(step_idx)
        nxt = [float(fn(**kw)) for fn in fns]
        if not all(math.isfinite(x) for x in nxt) or max(abs(x) for x in nxt) > max_abs_state:
            raise ValueError(f"Nonfinite or diverging recurrence state at n={step_idx + 1}")
        cur = nxt
        states.append([round(x, 6) for x in cur])
        ts.append(float(step_idx + 1))
    return SimulationResult(
        solver="recurrence",
        state_vars=s_vars,
        t_axis=ts,
        states=states,
        summary={"steps": n_steps, "end_state": states[-1]},
    )


def run_stochastic_ensemble(
    *,
    state_vars: Sequence[str],
    drift_exprs: Sequence[str],
    diffusion_stds: Sequence[float],
    initial_state: Sequence[float],
    t_range: Sequence[float] = (0.0, 6.0),
    steps: int = 80,
    n_paths: int = 8,
    seed: int = 42,
    params: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Run deterministic-seed Euler-Maruyama stochastic ensemble returning sample paths, mean, and std band."""
    s_vars = [str(v).strip() for v in state_vars if str(v).strip()]
    if not s_vars or len(s_vars) != len(drift_exprs) or len(s_vars) != len(initial_state):
        raise ValueError("Stochastic ensemble requires matching state_vars, drift_exprs, and initial_state.")
    sigmas = [float(diffusion_stds[i]) if i < len(diffusion_stds) else 0.1 for i in range(len(s_vars))]
    t0, t1 = float(t_range[0]), float(t_range[1]) if len(t_range) >= 2 else (0.0, 6.0)
    n_st = max(10, min(400, int(steps)))
    n_p = max(2, min(32, int(n_paths)))
    dt = (t1 - t0) / float(n_st)
    sq_dt = math.sqrt(max(1e-9, dt))
    allowed = tuple(dict.fromkeys([*s_vars, "t"]))
    fns = [_compile_param_expr(ex, allowed, params) for ex in drift_exprs]
    rng = np.random.default_rng(int(seed))
    paths = np.zeros((n_p, n_st + 1, len(s_vars)), dtype=float)
    y0 = np.asarray([float(v) for v in initial_state], dtype=float)
    _assert_finite_array(y0, "stochastic_init")
    paths[:, 0, :] = y0
    ts = np.linspace(t0, t1, n_st + 1)
    for k in range(n_st):
        tv = float(ts[k])
        noise = rng.normal(0.0, 1.0, size=(n_p, len(s_vars)))
        for p_i in range(n_p):
            kw = {nm: float(paths[p_i, k, idx]) for idx, nm in enumerate(s_vars)}
            kw["t"] = tv
            drift = np.asarray([fn(**kw) for fn in fns], dtype=float)
            _assert_finite_array(drift, "stochastic_drift")
            nxt = paths[p_i, k, :] + drift * dt + np.asarray(sigmas, dtype=float) * sq_dt * noise[p_i]
            _assert_finite_array(nxt, "stochastic_state")
            paths[p_i, k + 1, :] = nxt

    mean_x = np.mean(paths[:, :, 0], axis=0)
    std_x = np.std(paths[:, :, 0], axis=0)
    mean_curve = [[round(float(tv), 4), round(float(mv), 5)] for tv, mv in zip(ts, mean_x)]
    upper_band = [[round(float(tv), 4), round(float(mv + sv), 5)] for tv, mv, sv in zip(ts, mean_x, std_x)]
    lower_band = [[round(float(tv), 4), round(float(mv - sv), 5)] for tv, mv, sv in zip(ts, mean_x, std_x)]
    sample_paths = [
        [[round(float(tv), 4), round(float(paths[pi, ti, 0]), 5)] for ti, tv in enumerate(ts)]
        for pi in range(min(4, n_p))
    ]
    return {
        "mean_curve": mean_curve,
        "upper_std_curve": upper_band,
        "lower_std_curve": lower_band,
        "sample_paths": sample_paths,
        "final_mean": round(float(mean_x[-1]), 5),
        "final_std": round(float(std_x[-1]), 5),
    }


def build_simulation_sweep_frames(
    sim: SimulationResult,
    *,
    derived: dict[str, list[float]] | None = None,
    force_expr: str | None = None,
    mass: float | None = None,
    params: dict[str, float] | None = None,
    n_keyframes: int = 24,
    geometry_mode: str = "linear_spring",
    anchor_x: float = 0.0,
    anchor_y: float = 0.0,
) -> list[dict[str, Any]]:
    """Build unified SweepKeyframe sequence linking physical body, phase orbit, explicit force/tangent vector, and derived bars."""
    n_pts = len(sim.t_axis)
    k_target = max(6, min(n_pts, int(n_keyframes)))
    indices = np.unique(np.round(np.linspace(0, n_pts - 1, k_target)).astype(int))
    derived = derived or {}

    f_fn: Callable[..., float] | None = None
    if force_expr:
        allowed = tuple(dict.fromkeys([*sim.state_vars, "t"]))
        f_fn = _compile_param_expr(force_expr, allowed, params)
    if mass is not None and float(mass) <= 0.0:
        raise ValueError(f"Physical mass must be positive (got {mass}).")

    energy_keys = [k for k in ("potential_energy", "kinetic_energy", "total_energy", "dissipated_energy", "conserved_invariant") if k in derived]
    if not energy_keys and derived:
        energy_keys = list(derived.keys())[:3]

    frames: list[dict[str, Any]] = []
    n_last = max(1, len(indices) - 1)
    for fi, idx in enumerate(indices):
        tv = float(sim.t_axis[idx])
        st = sim.states[idx]
        x_val = float(st[0])
        v_val = float(st[1]) if len(st) >= 2 else 0.0
        if geometry_mode == "pendulum":
            rad = 1.6
            bx = anchor_x + rad * math.sin(x_val)
            by = anchor_y - rad * math.cos(x_val)
        else:
            bx = x_val
            by = anchor_y
        if f_fn is not None:
            kw_s = {nm: float(st[si]) for si, nm in enumerate(sim.state_vars)}
            kw_s["t"] = tv
            force_val = float(f_fn(**kw_s)) * (float(mass) if mass is not None else 1.0)
            f_scale = 0.35
            vec_start = [round(bx, 5), round(by, 5)]
            vec_end = (
                [round(bx + force_val * f_scale, 5), round(by, 5)]
                if geometry_mode != "pendulum"
                else [round(bx + force_val * 0.25 * math.cos(x_val), 5), round(by + force_val * 0.25 * math.sin(x_val), 5)]
            )
        else:
            # Without explicit force_expr/mass assumption, vector represents normalized phase-space velocity/displacement indicator
            vec_start = [0.0, 0.0]
            vec_end = [round(x_val, 5), round(v_val, 5)]
        bars = [round(float(derived[ek][idx]), 5) for ek in energy_keys] if energy_keys else []
        phase_trail = [[float(sim.states[k][0]), float(sim.states[k][1]) if len(sim.states[k]) >= 2 else 0.0] for k in range(0, int(idx) + 1, max(1, int(idx) // 48 or 1))]
        if phase_trail[-1] != [round(x_val, 5), round(v_val, 5)]:
            phase_trail.append([round(x_val, 5), round(v_val, 5)])
        u_norm = round(fi / float(n_last), 6)
        v_lbl0 = sim.state_vars[0] if sim.state_vars else "x"
        v_lbl1 = sim.state_vars[1] if len(sim.state_vars) >= 2 else "v"
        frames.append(
            {
                "param": round(tv, 4),
                "u": u_norm,
                "point": [round(x_val, 5), round(v_val, 5)],
                "spec_point": [round(tv, 4), round(x_val, 5)],
                "body_pos": [round(bx, 5), round(by, 5)],
                "vector_start": vec_start,
                "vector_end": vec_end,
                "curve_pts": phase_trail,
                "bar_values": bars,
                "label": f"t={tv:.2f} | {v_lbl0}={x_val:.2f}" + (f", {v_lbl1}={v_val:.2f}" if len(st) >= 2 else ""),
            }
        )
    return frames


def infer_phase_var_names(
    dx_expr: str,
    dy_expr: str,
    var_names: Sequence[str] | None = None,
) -> tuple[str, str]:
    """Resolve horizontal and vertical variable names for arbitrary 2D vector-field expressions."""
    if var_names and len(var_names) >= 2:
        return str(var_names[0]).strip() or "x", str(var_names[1]).strip() or "y"
    import ast as _ast

    _ignore = {"sin", "cos", "tan", "exp", "log", "sqrt", "abs", "pi", "e", "min", "max", "pow", "tanh", "sinh", "cosh", "atan2", "hypot", "t"}
    seen: list[str] = []
    for ex in (dx_expr, dy_expr):
        try:
            tree = _ast.parse(ex.strip(), mode="eval")
            for node in _ast.walk(tree):
                if isinstance(node, _ast.Name) and node.id not in _ignore and node.id not in seen:
                    seen.append(node.id)
        except SyntaxError:
            pass
    sset = set(seen)
    for pair in (("x", "y"), ("x", "v"), ("theta", "omega"), ("q", "p"), ("r", "v")):
        if set(pair) <= sset:
            return pair
    if len(seen) >= 2:
        return seen[0], seen[1]
    if len(seen) == 1:
        return seen[0], ("v" if seen[0] == "x" else "y")
    return "x", "y"


def sample_vector_field_grid(
    dx_expr: str,
    dy_expr: str,
    xr: Sequence[float],
    yr: Sequence[float],
    *,
    var_names: Sequence[str] | None = None,
    nx: int = 9,
    ny: int = 7,
) -> list[tuple[list[float], list[float]]]:
    """Sample normalized directional segments [(start_xy, end_xy), ...] over a 2D vector field."""
    v0, v1 = infer_phase_var_names(dx_expr, dy_expr, var_names=var_names)
    allowed = (v0, v1) if v0 != v1 else (v0, "y")
    fx = compile_expr(dx_expr.strip(), allowed)
    fy = compile_expr(dy_expr.strip(), allowed)
    xs_g = np.linspace(float(xr[0]), float(xr[1]), nx)[1:-1]
    ys_g = np.linspace(float(yr[0]), float(yr[1]), ny)[1:-1]
    dx_step = (float(xr[1]) - float(xr[0])) / float(nx) * 0.72
    dy_step = (float(yr[1]) - float(yr[0])) / float(ny) * 0.72
    segs: list[tuple[list[float], list[float]]] = []
    for gx in xs_g:
        for gy in ys_g:
            kw = {v0: float(gx), v1: float(gy)}
            vx, vy = float(fx(**kw)), float(fy(**kw))
            if not (math.isfinite(vx) and math.isfinite(vy)):
                continue
            mag = math.hypot(vx, vy)
            if mag <= 1e-6:
                continue
            sc_u = min(1.0, mag / max(mag, 2.0)) / mag
            p0 = [float(gx), float(gy)]
            p1 = [float(gx) + vx * sc_u * dx_step, float(gy) + vy * sc_u * dy_step]
            segs.append((p0, p1))
    return segs
