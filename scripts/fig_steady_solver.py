"""Figure: the differentiable steady-state solver on the ADM1 digester.

(a) Pseudo-transient continuation (PTC) finds the steady state as an algebraic
    root, faster than long forward integration and to a tighter residual.
(b) The methanogenesis washout saddle-node. As the dilution rises (SRT falls),
    the methanogenic branch ends in a fold at a critical SRT. A direct PTC solve
    reaches only the stable branch and fails past the nose; pseudo-arclength
    continuation traverses the fold and traces the unstable branch, so the S-curve
    (two steady states over a range of loadings) is recovered.
(c) The steady-state sensitivity to the loading, evaluated exactly by the implicit
    function theorem at every branch point, diverges at the fold on both arms, at
    close to the inverse-square-root rate of a saddle-node: the solver stays
    differentiable up to the tipping point, and the divergence locates it.

What the unstable branch means (text only, no panel): at matched loadings each
unstable point is a saddle (one eigenvalue with positive real part) holding fewer
acetoclastic methanogens and more acetate than the stable state. A small nudge
toward the stable state returns there; one the other way washes the methanogens
out to a third, washed-out steady state. So the unstable branch is the threshold
between the two basins, and its gap from the stable branch the disturbance margin.

Cache keys: "steady_solver"; "steady_unstable" (the unstable-branch diagnostics,
kept apart so they recompute without re-running the timings of panel (a)).
"""
from __future__ import annotations
import os
import time
import numpy as np
import common

# ADM1 digester operating point (published BSM2 digester: 3400 m3, 35 C, 178.5 m3/d).
Q_FEED, V_LIQ = 178.4674, 3400.0
INFLUENT = {"S_aa": 0.04388, "S_IC": 0.0079326, "S_IN": 0.0019721, "S_I": 0.028067,
            "X_ch": 3.7236, "X_pr": 15.9235, "X_li": 8.047, "X_I": 17.0106, "S_an": 0.0052101}
REF_SS = {"S_su": 0.012394, "S_aa": 0.0055432, "S_fa": 0.10741, "S_va": 0.012333,
          "S_bu": 0.014003, "S_pro": 0.017584, "S_ac": 0.089315, "S_ch4": 0.05549,
          "S_IC": 0.095149, "S_IN": 0.094468, "S_I": 0.13087, "X_c": 0.10792,
          "X_ch": 0.020517, "X_pr": 0.08422, "X_li": 0.043629, "X_su": 0.31222,
          "X_aa": 0.93167, "X_fa": 0.33839, "X_c4": 0.33577, "X_pro": 0.10112,
          "X_ac": 0.67724, "X_h2": 0.28484, "X_I": 17.2162, "S_gas_ch4": 1.6535,
          "S_gas_co2": 0.01354}
D_TARGET = 8.0   # continuation target (feed-flow multiplier), past the fold
FWD_HORIZON_D = 300.0   # fixed forward-integration horizon of the (a) timing baseline
PTC_TOL = 1e-8          # PTC convergence tolerance on the scaled residual (Eq. scaled-residual)
NRUNS = 200             # timed runs per solver in (a); see the note in _compute


def _compute():
    import jax, jax.numpy as jnp
    import aquakin
    from aquakin.plant import Plant, InfluentSeries
    from aquakin.plant.digester import ADM1DigesterUnit
    from aquakin.plant.steady import arclength_continuation_solve

    adm1 = aquakin.load_model("adm1")
    si = adm1.species_index
    plant = Plant("digester")
    plant.add_unit(ADM1DigesterUnit(name="digester", model=adm1, volume=V_LIQ,
                                    conditions={"T": 308.15}, input_port_names=["feed"]))
    plant.add_influent("feed", InfluentSeries.constant(adm1, Q=Q_FEED, **INFLUENT),
                       to="digester.feed")
    plant._build_state_layout(); plant._build_parameter_layout()
    params = adm1.default_parameters()
    params_full = plant._coerce_params(params)
    C0 = np.asarray(adm1.default_concentrations()).copy()
    for k, v in REF_SS.items():
        C0[si[k]] = v
    y0 = plant.initial_state(overrides={"digester": jnp.asarray(C0)})
    sac_i, ch4_i = si["S_ac"], si["S_gas_ch4"]

    # --- (a) PTC vs forward integration (timed over NRUNS runs) --------------
    # Each timing does one warm-up call (discarded, absorbs JIT compilation), then
    # NRUNS timed runs; np.asarray on the result forces JAX to finish before the clock
    # is read. The PTC solve is sub-millisecond, so its wall time is dominated by Python
    # and JAX dispatch overhead and by background machine load; we keep the full spread
    # (median and 10th-90th percentiles) rather than a single mean. Forward integration
    # is milliseconds and stable.

    def _bench(call, state_attr):
        r = call(); np.asarray(getattr(r, state_attr))          # warm-up (compile)
        ts = []
        for _ in range(NRUNS):
            t0 = time.perf_counter()
            r = call(); np.asarray(getattr(r, state_attr))
            ts.append(time.perf_counter() - t0)
        return np.asarray(ts), r

    def _stat(ts):
        return {"mean": float(ts.mean()), "std": float(ts.std()),
                "med": float(np.median(ts)),
                "p10": float(np.percentile(ts, 10)),
                "p90": float(np.percentile(ts, 90))}

    def _ptc():
        return plant.steady_state(params=params, y0=y0, max_iter=400, tol=PTC_TOL, scale_floor=common.RESID_FLOOR)

    def _fwd():
        return plant.solve((0.0, FWD_HORIZON_D), t_eval=np.array([FWD_HORIZON_D]), params=params,
                           y0=y0, forward_fast=True, integrator=common.FAST_INTEGRATOR)

    ptc_ts, ss = _bench(_ptc, "state")
    y_star = np.asarray(ss.state)
    fwd_ts, fsol = _bench(_fwd, "final_state")
    y_fwd = np.asarray(fsol.final_state)
    ptc_t, fwd_t = _stat(ptc_ts), _stat(fwd_ts)
    mask = np.abs(y_fwd) > 1e-3
    ptc_vs_fwd = float(np.max(np.abs(y_star[mask] - y_fwd[mask]) / np.abs(y_fwd[mask])))
    # Central speedup (stable forward median over PTC median) and its run-to-run range,
    # driven by the noisy PTC solve: forward median over the PTC 10th-90th band.
    speedup = fwd_t["med"] / ptc_t["med"]
    speedup_lo = fwd_t["med"] / ptc_t["p90"]
    speedup_hi = fwd_t["med"] / ptc_t["p10"]
    y_known = jnp.asarray(ss.state)

    # Same-convergence baseline: the forward horizon at which forward integration
    # reaches the PTC convergence tolerance on the scaled residual
    # (common.scaled_residual, the quantity PTC tests), so the speedup is measured to
    # equal convergence, not to a fixed (conservatively long) 300-day horizon. Match on
    # the tolerance PTC is given, not the residual it happens to reach: its final Newton
    # step overshoots the tolerance by orders of magnitude.
    # The scan uses an accurate solve: the fast solve's intermediate output times are
    # too coarsely resolved for a residual (only its end point is accurate).
    t_samp = np.linspace(2.0, FWD_HORIZON_D, 150)
    ys_traj = np.asarray(plant.solve((0.0, FWD_HORIZON_D), t_eval=jnp.asarray(t_samp),
                                     params=params, y0=y0, forward_fast=False).state)

    def _resnorm(y):
        return common.scaled_residual(plant.derivative(jnp.asarray(y), params=params), y)

    res_traj = np.array([_resnorm(ys_traj[i]) for i in range(len(t_samp))])
    below = np.where(res_traj <= PTC_TOL)[0]
    t_match = float(t_samp[below[0]]) if below.size else FWD_HORIZON_D
    # The timed solve is the fast one, whose end point is slightly less converged than
    # the accurate trajectory at the same time: advance the horizon along the sample
    # grid until the fast solve's own end point meets the tolerance.
    for tm in t_samp[t_samp >= t_match]:
        yf = plant.solve((0.0, float(tm)), t_eval=np.array([float(tm)]), params=params,
                         y0=y0, forward_fast=True, integrator=common.FAST_INTEGRATOR).final_state
        if _resnorm(np.asarray(yf)) <= PTC_TOL:
            t_match = float(tm)
            break
    else:
        t_match = FWD_HORIZON_D

    def _fwd_matched():
        return plant.solve((0.0, t_match), t_eval=np.array([t_match]), params=params,
                           y0=y0, forward_fast=True, integrator=common.FAST_INTEGRATOR)

    fwdm_ts, fwdm_sol = _bench(_fwd_matched, "final_state")
    fwdm_t = _stat(fwdm_ts)
    # The timed (fast) solve's own end-point residual, to confirm it is converged to
    # the same tolerance as the root find it is compared with.
    fwd_matched_resid = _resnorm(np.asarray(fwdm_sol.final_state))
    speedup_matched = fwdm_t["med"] / ptc_t["med"]

    # --- IFT sensitivity accuracy check (d acetate / d k_m_ac): IFT vs FD ----
    kidx = plant.parameter_index("adm1.k_m_ac")

    def sac_of(kp):
        pp = jnp.asarray(params).at[kidx].set(kp)
        s = plant.steady_state(params=pp, y0=y_known, max_iter=400, tol=1e-8, scale_floor=common.RESID_FLOOR)
        return plant.states_by_unit(s.state)["digester"][sac_i]

    k0 = float(params[kidx])
    g_ift = float(jax.grad(sac_of)(k0))
    hh = 1e-3 * k0
    g_fd = float((sac_of(k0 + hh) - sac_of(k0 - hh)) / (2 * hh))
    # forward-mode directional derivative of the converged root (an exact reference,
    # not FD-limited): custom_root supports both AD modes, so jacfwd flows through.
    try:
        g_fwd = float(jax.jacfwd(sac_of)(k0))
        ift_vs_fwd = abs(g_ift - g_fwd) / abs(g_fwd)
    except Exception:
        g_fwd, ift_vs_fwd = float("nan"), float("nan")
    sens_rel = abs(g_ift - g_fd) / max(abs(g_fd), 1e-30)

    # --- (b) the washout fold: arclength S-curve (record the branch) --------
    def rhs(y, p):
        return plant._rhs(jnp.asarray(0.0), y, params_full,
                          design={"influent": {"feed": {"Q_scale": p[0]}}})

    r = arclength_continuation_solve(rhs, jnp.array([1.0]), y_known,
                                     jnp.array([D_TARGET]), ptc_kwargs=dict(tol=1e-6, scale_floor=common.RESID_FLOOR),
                                     max_steps=300, record_branch=True)
    s_arr, ys = np.asarray(r.branch[0]), np.asarray(r.branch[1])
    dil = 1.0 + s_arr * (D_TARGET - 1.0)
    # Past the fold the unstable arm runs back to zero feed, where the continuation
    # then parks on a degenerate zero-feed path; keep the branch only up to the first
    # point past the fold whose loading has reached zero.
    inose = int(np.argmax(dil))
    past = np.where(dil[inose:] <= 1e-3)[0]
    if past.size:
        cut = inose + int(past[0]) + 1
        s_arr, ys, dil = s_arr[:cut], ys[:cut], dil[:cut]
    # Retention time is defined only for a positive feed loading (the kept end point
    # can sit at zero feed), so guard the divide.
    srt = V_LIQ / Q_FEED / np.where(dil > 0.0, dil, np.nan)
    sac = ys[:, sac_i]
    dil_fold = float(dil[inose]); srt_fold = float(srt[inose])

    # --- (b) direct PTC along the loading: converges on the stable branch,
    #         fails past the nose ---------------------------------------------
    ptc_dil, ptc_sac, ptc_ok = [], [], []
    warm = y_known
    for dm in [1.0, 2.0, 3.0, 3.5, 4.0, 4.2, 4.4, 4.6]:
        design = {"influent": {"feed": {"Q_scale": float(dm)}}}
        pr = plant.steady_state(params=params, y0=warm, design=design, max_iter=300,
                                tol=1e-7, fallback=False, arclength=False, scale_floor=common.RESID_FLOOR)
        ok = bool(pr.converged)
        val = float(plant.states_by_unit(np.asarray(pr.state))["digester"][sac_i])
        ptc_dil.append(float(dm)); ptc_sac.append(val); ptc_ok.append(ok)
        if ok:
            warm = jnp.asarray(pr.state)
    ptc_last_ok = max((dm for dm, ok in zip(ptc_dil, ptc_ok) if ok), default=1.0)
    ptc_first_fail = min((dm for dm, ok in zip(ptc_dil, ptc_ok) if not ok), default=D_TARGET)

    # --- (c) steady-state sensitivity to the loading diverges at the fold -----
    # The loading column of Eq. (ift), d x*/d lambda = -F_x^{-1} F_lambda, evaluated
    # exactly (AD Jacobians, one linear solve) at every recorded branch point. Its sign
    # tells the arm: d acetate / d loading > 0 on the stable arm, < 0 on the unstable.
    @jax.jit
    def _ift_loading(y, lam):
        p = jnp.array([lam])
        Fx = jax.jacfwd(lambda yy: rhs(yy, p))(y)
        Fl = jax.jacfwd(lambda pp: rhs(y, pp))(p)[:, 0]
        return -jnp.linalg.solve(Fx, Fl)[sac_i]

    sens_g = np.array([float(_ift_loading(jnp.asarray(ys[k]), float(dil[k])))
                       for k in range(len(dil))])
    # The branch is sampled at the continuation's step size, so its highest point only
    # brackets the fold; refine it by the vertex of the parabola lambda(acetate)
    # through the three points around the nose (a saddle-node is locally quadratic).
    kn = [inose - 1, inose, inose + 1]
    cpar = np.polyfit(sac[kn], dil[kn], 2)
    dil_fold_ref = float(np.polyval(cpar, -cpar[1] / (2 * cpar[0])))

    return {
        "t_ptc": ptc_t["med"], "t_ptc_p10": ptc_t["p10"], "t_ptc_p90": ptc_t["p90"],
        "t_ptc_mean": ptc_t["mean"], "t_ptc_std": ptc_t["std"],
        "t_fwd": fwd_t["med"], "t_fwd_p10": fwd_t["p10"], "t_fwd_p90": fwd_t["p90"],
        "t_fwd_matched": fwdm_t["med"], "t_fwd_matched_p10": fwdm_t["p10"],
        "t_fwd_matched_p90": fwdm_t["p90"],
        "n_runs": NRUNS,
        "speedup": speedup, "speedup_lo": speedup_lo, "speedup_hi": speedup_hi,
        "ptc_vs_fwd": ptc_vs_fwd,
        "ptc_iters": int(ss.iterations), "ptc_residual": float(ss.residual),
        "srt_nominal": V_LIQ / Q_FEED,
        "t_match": t_match, "speedup_matched": speedup_matched,
        "fwd_matched_resid": fwd_matched_resid, "ptc_tol": PTC_TOL,
        "sens_g_ift": g_ift, "sens_g_fd": g_fd, "sens_rel": sens_rel,
        "sens_g_fwd": g_fwd, "ift_vs_fwd": ift_vs_fwd,
        "branch_dil": dil.tolist(), "branch_srt": srt.tolist(), "branch_sac": sac.tolist(),
        "inose": inose, "dil_fold": dil_fold, "srt_fold": srt_fold,
        "ptc_dil": ptc_dil, "ptc_sac": ptc_sac, "ptc_ok": ptc_ok,
        "ptc_last_ok": ptc_last_ok, "ptc_first_fail": ptc_first_fail,
        "sens_loading": sens_g.tolist(), "dil_fold_ref": dil_fold_ref,
    }

UNS_LAMS = (1.0, 2.0, 3.0, 4.0)   # loadings at which the two arms are compared
UNS_NUDGE_LAM = 2.0               # loading of the nudge test
UNS_NUDGE = 0.02                  # nudge, as a fraction of the unstable-to-stable gap
UNS_HORIZON_D = 400.0             # forward horizon of the nudge test


def _compute_unstable():
    import jax, jax.numpy as jnp
    import aquakin
    from aquakin.plant import Plant, InfluentSeries
    from aquakin.plant.digester import ADM1DigesterUnit
    from aquakin.plant.steady import arclength_continuation_solve

    adm1 = aquakin.load_model("adm1")
    si = adm1.species_index

    def make_plant(q):
        plant = Plant("digester")
        plant.add_unit(ADM1DigesterUnit(name="digester", model=adm1, volume=V_LIQ,
                                        conditions={"T": 308.15}, input_port_names=["feed"]))
        plant.add_influent("feed", InfluentSeries.constant(adm1, Q=q, **INFLUENT),
                           to="digester.feed")
        plant._build_state_layout(); plant._build_parameter_layout()
        return plant

    plant = make_plant(Q_FEED)
    params = adm1.default_parameters()
    params_full = plant._coerce_params(params)
    C0 = np.asarray(adm1.default_concentrations()).copy()
    for k, v in REF_SS.items():
        C0[si[k]] = v
    y0 = plant.initial_state(overrides={"digester": jnp.asarray(C0)})
    y_known = jnp.asarray(plant.steady_state(params=params, y0=y0, max_iter=400, tol=PTC_TOL,
                                             scale_floor=common.RESID_FLOOR).state)

    def rhs(y, p):
        return plant._rhs(jnp.asarray(0.0), y, params_full,
                          design={"influent": {"feed": {"Q_scale": p[0]}}})

    # The same branch as panel (b).
    r = arclength_continuation_solve(rhs, jnp.array([1.0]), y_known,
                                     jnp.array([D_TARGET]), ptc_kwargs=dict(tol=1e-6, scale_floor=common.RESID_FLOOR),
                                     max_steps=300, record_branch=True)
    s_arr, ys = np.asarray(r.branch[0]), np.asarray(r.branch[1])
    dil = 1.0 + s_arr * (D_TARGET - 1.0)
    inose = int(np.argmax(dil))
    jac = jax.jit(jax.jacfwd(rhs))
    f = jax.jit(rhs)

    def n_unstable(y, lam):
        ev = np.linalg.eigvals(np.asarray(jac(jnp.asarray(y), jnp.array([lam]))))
        return int((ev.real > 0.0).sum())

    def root_at(lam, arm):
        """The steady state on one arm at exactly ``lam``: interpolate between the
        two recorded branch points that bracket it, then Newton at fixed loading
        (Newton converges to the saddle as readily as to the stable root)."""
        idx = np.arange(inose + 1) if arm == "stable" else np.arange(inose, len(dil))
        d = dil[idx]
        j = int(np.where((d[:-1] - lam) * (d[1:] - lam) <= 0.0)[0][0])
        a, b = idx[j], idx[j + 1]
        w = (lam - dil[a]) / (dil[b] - dil[a])
        y = jnp.asarray((1.0 - w) * ys[a] + w * ys[b])
        p = jnp.array([lam])
        for _ in range(30):
            y = y - jnp.linalg.solve(jac(y, p), f(y, p))
            if common.scaled_residual(f(y, p), y) < PTC_TOL:
                break
        return y, common.scaled_residual(f(y, p), y)

    dig = lambda y: plant.states_by_unit(np.asarray(y))["digester"]
    names = ("S_ac", "X_ac", "S_IC")
    arms = {}
    for arm in ("stable", "unstable"):
        rows = {n: [] for n in names}
        rows["resid"], rows["n_unstable_eig"], rows["state"] = [], [], []
        for lam in UNS_LAMS:
            y, res = root_at(lam, arm)
            for n in names:
                rows[n].append(float(dig(y)[si[n]]))
            rows["resid"].append(res)
            rows["n_unstable_eig"].append(n_unstable(y, lam))
            rows["state"].append(np.asarray(y))
        arms[arm] = rows
    # Every recorded branch point: stable arm 0 unstable eigenvalues, unstable arm 1.
    n_eig_branch = [n_unstable(ys[i], dil[i]) for i in range(len(dil)) if dil[i] > 1e-3]

    # Nudge the unstable state toward / away from the stable one and integrate.
    k = UNS_LAMS.index(UNS_NUDGE_LAM)
    yu, yst = arms["unstable"]["state"][k], arms["stable"]["state"][k]
    nplant = make_plant(Q_FEED * UNS_NUDGE_LAM)
    nudge = {}
    for tag, eps in (("toward", UNS_NUDGE), ("away", -UNS_NUDGE)):
        sol = nplant.solve((0.0, UNS_HORIZON_D), t_eval=np.array([UNS_HORIZON_D]), params=params,
                           y0=jnp.asarray(yu + eps * (yst - yu)))
        yend = np.asarray(sol.state[-1])
        resid = common.scaled_residual(nplant._rhs(jnp.asarray(0.0), jnp.asarray(yend),
                                                   nplant._coerce_params(params)), yend)
        nudge[tag] = {"S_ac": float(dig(yend)[si["S_ac"]]), "X_ac": float(dig(yend)[si["X_ac"]]),
                      "resid": resid}
    for arm in arms.values():
        del arm["state"]
    return {"lams": list(UNS_LAMS), "arms": arms, "n_eig_branch": n_eig_branch,
            "inose": inose, "nudge_lam": UNS_NUDGE_LAM, "nudge_frac": UNS_NUDGE,
            "nudge_horizon_d": UNS_HORIZON_D, "nudge": nudge}


def main():
    d = common.load_or_compute("steady_solver", _compute)
    common.load_or_compute("steady_unstable", _compute_unstable)
    plt = common.setup_mpl()
    common.scale_fonts(plt, 12.5)
    fig, ax = plt.subplots(1, 3, figsize=(12.5, 4.4), width_ratios=[0.85, 1.3, 1.0])

    # (a) solve time (compilation excluded) to steady state: forward to a fixed 300 d,
    #     forward to the same residual as PTC, and PTC. Bars are the median over the timed
    #     runs (ms); error bars span the 10th-90th percentile. The PTC solve is
    #     sub-millisecond and dispatch-limited, so its band is wide and asymmetric.
    ms = 1e3
    med = ms * np.array([d["t_fwd"], d["t_fwd_matched"], d["t_ptc"]])
    p10 = ms * np.array([d["t_fwd_p10"], d["t_fwd_matched_p10"], d["t_ptc_p10"]])
    p90 = ms * np.array([d["t_fwd_p90"], d["t_fwd_matched_p90"], d["t_ptc_p90"]])
    ax[0].bar([0, 1, 2], med, yerr=[med - p10, p90 - med],
              capsize=3, error_kw=dict(ecolor="0.3", lw=1.0),
              color=["C2", "C7", "C0"], width=0.65)
    ax[0].set_xticks([0, 1, 2])
    ax[0].set_xticklabels(["forward\n300 d", f"forward\n{d['t_match']:.0f} d", "root\nfind"])
    ax[0].set(ylabel="solve wall time (ms)", title="(a) time to steady state")

    # (b) the fold + the solver modes
    dil = np.asarray(d["branch_dil"]); sac = np.asarray(d["branch_sac"]); ino = d["inose"]
    ax[1].plot(dil[:ino + 1], sac[:ino + 1], "-", color="C0", lw=1.8,
               label="arclength (stable)")
    ax[1].plot(dil[ino:], sac[ino:], "--", color="C0", lw=1.4, alpha=0.8,
               label="arclength (unstable)")
    ax[1].plot(d["dil_fold"], sac[ino], "o", color="C3", ms=7, zorder=5)
    ax[1].annotate(f"fold (SRT {d['srt_fold']:.1f} d)", (d["dil_fold"], sac[ino]),
                   xytext=(0.97, 0.9), textcoords="axes fraction", ha="right",
                   color="C3", arrowprops=dict(arrowstyle="->", color="C3", lw=0.9))
    pdil = np.asarray(d["ptc_dil"]); psac = np.asarray(d["ptc_sac"]); pok = np.asarray(d["ptc_ok"])
    ax[1].plot(pdil[pok], psac[pok], "s", color="C2", ms=5, label="direct root find")
    if (~pok).any():
        # A failed solve has no steady value, so mark the loading where it fails on
        # the bottom of the axis rather than at an acetate value.
        ax[1].plot(pdil[~pok][:1], [0.03], "x", color="C1", ms=9, mew=2, clip_on=False,
                   transform=ax[1].get_xaxis_transform(), label="direct root find fails")
    ax[1].set(xlabel="feed-flow multiplier", ylabel="effluent acetate (kg COD/m$^3$)",
              title="(b) washout fold")
    ax[1].legend(frameon=False, loc="lower left", bbox_to_anchor=(0.06, 0.12))

    # (c) exact IFT sensitivity on both arms against the distance to the fold
    # Shown within one loading unit of the fold: the unstable arm runs on to zero
    # feed, where the Jacobian is singular again for an unrelated reason.
    dref = d["dil_fold_ref"]
    g = np.asarray(d["sens_loading"]); dist = dref - dil
    near = (dist > 0) & (dist <= 1.0) & np.isfinite(g)
    for arm, mk, fc, lab in ((g > 0, "o", "C4", "stable arm"),
                             (g < 0, "o", "none", "unstable arm")):
        sel = arm & near
        order = np.argsort(dist[sel])
        ax[2].plot(dist[sel][order], np.abs(g[sel][order]), mk + "-", color="C4",
                   mfc=fc, ms=4, lw=1.0, label=lab)
    # Saddle-node reference slope, anchored at the stable point nearest the fold.
    ks = np.where((g > 0) & near)[0]; k0 = ks[np.argmin(dist[ks])]
    xr = np.array([dist[near].min(), dist[near].max()])
    ax[2].plot(xr, abs(g[k0]) * np.sqrt(dist[k0] / xr), ":", color="0.4", lw=1.2,
               label=r"$\propto(\alpha_f-\alpha)^{-1/2}$")
    ax[2].set(xscale="log", yscale="log",
              xlabel=r"distance to the fold, $\alpha_f-\alpha$",
              ylabel=r"$|\,\partial\,$acetate$/\partial\alpha\,|$ (kg COD/m$^3$)",
              title="(c) sensitivity at the fold")
    ax[2].legend(frameon=False, loc="upper right")

    for ext in ("png", "pdf"):    # bbox=tight avoids clipping the edge titles
        fig.savefig(os.path.join(common.FIG_DIR, f"fig_steady_solver.{ext}"),
                    bbox_inches="tight")
    print("  wrote fig_steady_solver.png / .pdf")
    print(f"  PTC iters {d['ptc_iters']} residual {d['ptc_residual']:.1e}; "
          f"vs forward {d['ptc_vs_fwd']:.1e}; speedup {d['speedup']:.1f}x")
    print(f"  IFT sens d(acetate)/d(k_m_ac): IFT {d['sens_g_ift']:.3f} vs FD {d['sens_g_fd']:.3f} "
          f"(rel {d['sens_rel']:.1e})")
    print(f"  fold at feed x{d['dil_fold']:.3f} = SRT {d['srt_fold']:.2f} d "
          f"(nominal {d['srt_nominal']:.1f} d)")
    print(f"  matched forward horizon {d['t_match']:.0f} d, end-point residual "
          f"{d['fwd_matched_resid']:.1e} (PTC tol {d['ptc_tol']:.0e})")
    print(f"  direct PTC: last converged x{d['ptc_last_ok']:.1f}, first fail x{d['ptc_first_fail']:.1f}")


if __name__ == "__main__":
    main()
