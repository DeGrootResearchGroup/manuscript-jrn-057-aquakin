"""Figure: gradient-based design of the calibration experiment.

The identifiability analysis (fig_identifiability) says which parameters a design
determines; the same exact gradients also improve the design. With the two anoxic
assays fixed, ``aquakin.optimize_experiment`` moves the sampling times of the
aerobic respirogram and its initial substrate S_S to maximise the D-criterion
(log det F) of the whole three-assay design. The criterion is differentiated with
respect to the design itself, through the Fisher information and the stiff solves.

(a) The aerobic respirogram (dissolved oxygen) at the original and the designed
    initial substrate, with the original and the designed sampling times.
(b) The Cramer-Rao relative standard deviation of each parameter for the aerobic
    assay alone, the anoxic assays alone, the original three-assay design and the
    designed one, with the realized Monte Carlo spread of blind recoveries
    (fig_calibration's scheme: no priors, random starts) on the original and the
    designed experiment.

Cache key: "design".
"""
from __future__ import annotations
import numpy as np
import common
from fig_calibration import _setup, _random_start, _calibrate, SIG, OBS, FREE, PLABEL
from fig_identifiability import _experiments, _transforms

N_MC = 24                      # blind recoveries on the designed experiment
T_AER = 0.1                    # aerobic assay duration (d), as in fig_calibration
MIN_SPACING = 1e-3             # minimum gap between samples (d), about 1.4 min
SS_BOUNDS = (20.0, 200.0)      # initial substrate range for the aerobic assay (mg/L)


def _compute():
    jnp, aquakin, asm1, si, pi, rx, p_true, specs, oidx = _setup()
    ex = _experiments(aquakin, jnp, specs)
    anox = [ex["Nlim"], ex["Slim"]]
    fkw = dict(params=jnp.asarray(p_true), transforms=_transforms(FREE))

    opt = aquakin.optimize_experiment(
        rx, ex["aer"], FREE, base=anox, time_bounds=(0.0, T_AER),
        min_spacing=MIN_SPACING, initial_bounds={"SS": SS_BOUNDS}, max_iter=100, **fkw)
    aer_opt = opt.experiments[0]

    def crb(design):
        r = aquakin.identifiability(aquakin.fisher_information(rx, design, FREE, **fkw))
        return {n: float(100.0 * r.relative_std[n]) for n in FREE}

    crb_pct = {"aer": crb([ex["aer"]]), "anox": crb(anox),
               "base": crb([ex["aer"]] + anox), "opt": crb([aer_opt] + anox)}

    # Blind recoveries on the designed experiment (same scheme and seeds as
    # fig_calibration), to check the predicted spread against the realized one.
    b_aer = next(b for b in specs if b["name"] == "aer")
    t_opt = jnp.asarray(aer_opt.t_obs); C0_opt = jnp.asarray(aer_opt.C0)
    clean_aer = np.asarray(rx.solve(C0_opt, (0.0, float(t_opt[-1])), t_eval=t_opt,
                                    params=jnp.asarray(p_true)).C)[:, oidx]
    dspecs = [dict(b_aer, C0=C0_opt, t=t_opt, clean=clean_aer)] + \
             [b for b in specs if b["name"] != "aer"]
    C0s = [b["C0"] for b in dspecs]; tobs = [b["t"] for b in dspecs]
    truth = np.array([float(p_true[pi[n]]) for n in FREE])
    fits = []
    for seed in range(N_MC):
        noise = np.random.default_rng(1000 + seed)
        obs = [b["clean"] + SIG * noise.standard_normal(b["clean"].shape) for b in dspecs]
        init = _random_start(p_true, pi, np.random.default_rng(3000 + seed))
        c = _calibrate(aquakin, jnp, rx, dspecs, C0s, obs, tobs, init)
        fits.append([float(c.params_named[n]) for n in FREE])
    fits = np.array(fits)
    emp_cv = fits.std(0) / truth * 100.0

    # Dissolved-oxygen curves for panel (a).
    tt = jnp.linspace(0.0, T_AER, 401)
    so = si["SO"]
    do_base = np.asarray(rx.solve(b_aer["C0"], (0.0, T_AER), t_eval=tt,
                                  params=jnp.asarray(p_true)).C)[:, so]
    do_opt = np.asarray(rx.solve(C0_opt, (0.0, T_AER), t_eval=tt,
                                 params=jnp.asarray(p_true)).C)[:, so]
    ss = si["SS"]
    ss_base = np.asarray(rx.solve(b_aer["C0"], (0.0, T_AER), t_eval=tt,
                                  params=jnp.asarray(p_true)).C)[:, ss]
    ss_opt = np.asarray(rx.solve(C0_opt, (0.0, T_AER), t_eval=tt,
                                 params=jnp.asarray(p_true)).C)[:, ss]
    return {
        "free": FREE, "D_base": float(opt.initial_value), "D_opt": float(opt.value),
        "success": bool(opt.success), "n_iter": int(opt.n_iter),
        "t_base": np.asarray(b_aer["t"]).tolist(), "t_opt": np.asarray(t_opt).tolist(),
        "SS0_base": float(b_aer["C0"][si["SS"]]), "SS0_opt": float(C0_opt[si["SS"]]),
        "ss_bounds": list(SS_BOUNDS), "min_spacing": MIN_SPACING, "t_aer": T_AER,
        "crb_pct": crb_pct, "mc_fits": fits.tolist(), "emp_cv_opt_pct": emp_cv.tolist(),
        "n_mc": N_MC, "t_dense": np.asarray(tt).tolist(),
        "do_base": do_base.tolist(), "do_opt": do_opt.tolist(),
        "ss_base": ss_base.tolist(), "ss_opt": ss_opt.tolist(),
    }


def main():
    d = common.load_or_compute("design", _compute)
    import fig_calibration
    cal = common.load_or_compute("calibration", fig_calibration._compute)  # Fig. 9 recoveries
    plt = common.setup_mpl()
    common.scale_fonts(plt, 11.0)
    fig, ax = plt.subplots(1, 2, figsize=(11.0, 4.3), width_ratios=[1.0, 1.15])

    # (a) the aerobic assay before and after design: the observed substrate S_S (the
    #     dissolved oxygen is exhausted within minutes at either dose, so S_S is
    #     where the design shows), with the original and designed sampling times
    td = np.asarray(d["t_dense"]) * 24 * 60          # minutes
    ax[0].plot(td, d["ss_base"], "-", color="C0", lw=1.6,
               label=f"original ($S_S^0={d['SS0_base']:.0f}$ mg/L)")
    ax[0].plot(td, d["ss_opt"], "--", color="C3", lw=1.6,
               label=f"designed ($S_S^0={d['SS0_opt']:.0f}$ mg/L)")
    top = max(max(d["ss_base"]), max(d["ss_opt"]))
    for t in d["t_base"]:
        ax[0].plot([t * 1440] * 2, [-0.07 * top, -0.02 * top], color="C0", lw=1.2)
    for t in d["t_opt"]:
        ax[0].plot([t * 1440] * 2, [1.03 * top, 1.08 * top], color="C3", lw=1.2)
    from matplotlib.lines import Line2D
    ax0r = ax[0].twinx()                              # oxygen: the fast phase the
    ax0r.plot(td, d["do_base"], ":", color="0.45", lw=1.4)   # samples cluster on
    ax0r.set_ylabel("dissolved oxygen (mg/L)", color="0.4")
    ax0r.set_ylim(-0.11 * 8.4, 1.14 * 8.4); ax0r.grid(False)
    ax0r.tick_params(axis="y", colors="0.4")
    ax[0].set(xlabel="time (min)", ylabel="substrate $S_S$ (mg COD/L)",
              ylim=(-0.11 * top, 1.14 * top), title="(a) aerobic assay")
    h, l = ax[0].get_legend_handles_labels()
    h += [Line2D([], [], color="0.45", ls=":", lw=1.4),
          Line2D([], [], color="C0", marker="|", ls="none", ms=9, mew=1.2),
          Line2D([], [], color="C3", marker="|", ls="none", ms=9, mew=1.2)]
    l += ["dissolved oxygen (right axis)", "original samples", "designed samples"]
    # centred in the gap between the original and designed substrate curves
    ax[0].legend(h, l, frameon=False, loc="center right", bbox_to_anchor=(1.0, 0.58),
                 fontsize="small")

    # (b) Cramer-Rao relative std by design, with the realized Monte Carlo spread
    x = np.arange(len(FREE)); w = 0.2
    cap = 1e4                                          # an infinite bound is drawn here
    for k, (key, lab, col) in enumerate((("aer", "aerobic only", "C2"),
                                         ("anox", "anoxic only", "C1"),
                                         ("base", "original design", "C0"),
                                         ("opt", "designed", "C3"))):
        v = np.array([d["crb_pct"][key][n] for n in FREE])
        inf = ~np.isfinite(v)
        ax[1].bar(x + (k - 1.5) * w, np.where(inf, cap, v), width=w, color=col,
                  hatch=None, label=lab)
        for xi in (x + (k - 1.5) * w)[inf]:
            ax[1].text(xi, cap * 1.1, r"$\infty$", ha="center", va="bottom", color=col)
    ax[1].plot(x - 0.5 * w + w, np.asarray(cal["emp_cv_pct"]), "k_", ms=12, mew=2,
               label="realized (Monte Carlo)")
    ax[1].plot(x + 1.5 * w, np.asarray(d["emp_cv_opt_pct"]), "k_", ms=12, mew=2)
    ax[1].set_yscale("log")
    ax[1].set_xticks(x); ax[1].set_xticklabels([PLABEL[n] for n in FREE])
    ax[1].set(ylabel="relative std (%)", ylim=(0.1, cap * 3e3),
              title="(b) predicted and realized uncertainty")
    ax[1].legend(frameon=False, loc="upper right", ncol=2, fontsize="small",
                 columnspacing=1.0, handlelength=1.4)

    common.save(fig, "fig_design")
    print(f"  D {d['D_base']:.3f} -> {d['D_opt']:.3f} (success {d['success']}, "
          f"{d['n_iter']} iterations); S_S0 {d['SS0_base']:.0f} -> {d['SS0_opt']:.0f}")
    for key in ("aer", "anox", "base", "opt"):
        print(f"  CRB %CV {key:5s}", {n: round(v, 2) for n, v in d["crb_pct"][key].items()})
    print("  realized %CV designed", dict(zip(FREE, np.round(d["emp_cv_opt_pct"], 2))))


if __name__ == "__main__":
    main()
