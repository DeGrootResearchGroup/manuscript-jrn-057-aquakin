"""Figure: multi-parameter calibration of ASM1 kinetics from noisy batch assays.

A three-assay closed-batch design of the same activated-sludge reactor (the
assays differ only in the initial state):

  A  aerobic respirogram        -- dissolved-oxygen drawdown        -> muH, Y_H, bH
  B  anoxic, nitrate-limited     -- nitrate drawdown to depletion    -> etag, kh
  C  anoxic, substrate-limited   -- substrate drawdown through K_S    -> kh, bH

Five identifiable heterotroph parameters (muH, Y_H, bH, etag, kh) are recovered
jointly from realistically noised multi-species observations with the shipped
``aquakin.calibrate`` (NLL loss, cap-free stable-adjoint gradient, Gauss-Newton
Laplace uncertainty). Every fit is maximum likelihood with the priors off, started
from its own random point (see ``_random_start``). The half-saturation constants are
held at literature values, standard ASM identifiability practice: they are not
identifiable from batch data at realistic sensor resolution. Panels (a)-(c) show one
representative fit per assay; panel (d) shows the recovered-parameter sampling
distribution over many noise realizations. The cache also keeps each fit's Laplace
std, so the predicted spread can be compared with the realized one.

Cache key: "calibration".
"""
from __future__ import annotations
import numpy as np
import common

FREE = ["muH", "Y_H", "bH", "etag", "kh"]
PLABEL = {"muH": r"$\mu_H$", "Y_H": r"$Y_H$", "bH": r"$b_H$",
          "etag": r"$\eta_g$", "kh": r"$k_h$"}
OBS = ["SS", "SO", "SNO"]
OBSLBL = {"SS": r"$S_S$", "SO": r"$S_O$", "SNO": r"$S_{NO}$"}
SIG = np.array([1.0, 0.1, 0.2])           # sensor noise (mg/L): COD, DO, nitrate
N_MC = 24                                  # noise realizations
START_SPREAD = 0.4                         # log-sd of the random start around the truth


def _setup():
    import jax.numpy as jnp
    import aquakin
    asm1 = aquakin.load_model("asm1")
    si, pi = asm1.species_index, asm1.param_index
    cond = asm1.default_conditions(1)
    rx = aquakin.BatchReactor(asm1, cond,
                              integrator=aquakin.IntegratorConfig(dtmax=1e-3))
    p_true = np.asarray(asm1.default_parameters())

    def ic(**ov):
        C = np.asarray(asm1.default_concentrations()).copy()
        base = {"SI": 30., "XI": 1000., "XS": 40., "XND": 2., "SND": 1.,
                "SALK": 7., "XB_A": 80.}
        for k, v in {**base, **ov}.items():
            C[si[k]] = v
        return jnp.asarray(C)

    specs = [
        dict(name="aer",  C0=ic(XB_H=40.,  SS=60.,  SO=8., SNH=20., SNO=0.1), T=0.1),
        dict(name="Nlim", C0=ic(XB_H=600., SS=200., SO=0., SNH=15., SNO=40.), T=0.45),
        dict(name="Slim", C0=ic(XB_H=300., SS=60.,  SO=0., SNH=15., SNO=60.), T=0.3),
    ]
    oidx = [si[s] for s in OBS]
    for b in specs:
        if b["name"] == "aer":     # front-dense sampling resolves the fast O2 drawdown
            b["t"] = jnp.concatenate([jnp.linspace(0., 0.02, 13),
                                      jnp.linspace(0.025, b["T"], 7)])
        else:
            b["t"] = jnp.linspace(0., b["T"], 20)
        b["clean"] = np.asarray(rx.solve(b["C0"], (0., b["T"]), t_eval=b["t"],
                                         params=jnp.asarray(p_true)).C)[:, oidx]
    return jnp, aquakin, asm1, si, pi, rx, p_true, specs, oidx


def _random_start(p_true, pi, rng):
    """Full parameter vector with each free parameter started off its true value by
    a log-normal factor exp(START_SPREAD * N(0,1)) (about +/-40 %); the two fractions
    are clipped into (0.05, 0.95) so the logit start is finite."""
    init = np.asarray(p_true).copy()
    for n in FREE:
        val = float(p_true[pi[n]]) * float(np.exp(START_SPREAD * rng.standard_normal()))
        if n in ("etag", "Y_H"):
            val = min(max(val, 0.05), 0.95)
        init[pi[n]] = val
    return init


def _calibrate(aquakin, jnp, rx, specs, C0s, obs, tobs, init):
    # Blind recovery: no priors (the shipped asm1 priors are centred on the true
    # values) and a random start, so nothing pulls the estimate toward the truth.
    return aquakin.calibrate(
        rx, C0s, observations=obs, t_obs=tobs,
        free_params=FREE, observed_species=OBS,
        transforms={n: ("logit" if n in ("etag", "Y_H") else "positive_log")
                    for n in FREE},
        loss="nll", sigma=[SIG] * len(specs), optimizer=aquakin.OptimizerConfig(method="lbfgsb", n_starts=1),
        diff=aquakin.DifferentiationConfig(mode="reverse", method="stable"),
        laplace=aquakin.LaplaceConfig(method="gauss_newton"),
        use_priors=False, initial_params=jnp.asarray(init))


def _compute():
    jnp, aquakin, asm1, si, pi, rx, p_true, specs, oidx = _setup()
    truth = {n: float(p_true[pi[n]]) for n in FREE}
    C0s = [b["C0"] for b in specs]
    tobs = [b["t"] for b in specs]

    # --- representative fit (seed 0): noisy data + dense fitted trajectory ----
    rng = np.random.default_rng(0)
    obs0 = [b["clean"] + SIG * rng.standard_normal(b["clean"].shape) for b in specs]
    init0 = _random_start(p_true, pi, np.random.default_rng(2000))
    calib0 = _calibrate(aquakin, jnp, rx, specs, C0s, obs0, tobs, init0)
    p_fit = np.asarray(asm1.default_parameters()).copy()
    for n in FREE:
        p_fit[pi[n]] = calib0.params_named[n]
    panels = []
    PRIMARY = {"aer": "SO", "Nlim": "SNO", "Slim": "SS"}
    for b, y in zip(specs, obs0):
        td = jnp.linspace(0., b["T"], 120)
        Cfit = np.asarray(rx.solve(b["C0"], (0., b["T"]), t_eval=td,
                                   params=jnp.asarray(p_fit)).C)[:, oidx]
        Ctru = np.asarray(rx.solve(b["C0"], (0., b["T"]), t_eval=td,
                                   params=jnp.asarray(p_true)).C)[:, oidx]
        sp = PRIMARY[b["name"]]
        panels.append(dict(name=b["name"], T=float(b["T"]),
                           t=np.asarray(b["t"]).tolist(), y=np.asarray(y).tolist(),
                           td=np.asarray(td).tolist(), fit=Cfit.tolist(),
                           truth_curve=Ctru.tolist(),
                           primary_col=OBS.index(sp), primary_sp=sp))

    # --- Monte-Carlo recovery over noise realizations ------------------------
    fits, stds, inits, losses = [], [], [], []
    for seed in range(N_MC):
        rng = np.random.default_rng(1000 + seed)
        obs = [b["clean"] + SIG * rng.standard_normal(b["clean"].shape) for b in specs]
        init = _random_start(p_true, pi, np.random.default_rng(3000 + seed))
        c = _calibrate(aquakin, jnp, rx, specs, C0s, obs, tobs, init)
        fits.append([float(c.params_named[n]) for n in FREE])
        stds.append([float(c.params_named_std[n]) for n in FREE])
        inits.append([float(init[pi[n]]) for n in FREE])
        losses.append(float(c.loss))
    fits = np.array(fits); stds = np.array(stds)
    truth_v = np.array([truth[n] for n in FREE])
    bias = (fits.mean(0) - truth_v) / truth_v * 100.0
    emp_std = fits.std(0) / truth_v * 100.0             # empirical %CV
    pred_std = stds.mean(0) / truth_v * 100.0           # GN Laplace %CV
    worst = int(np.argmax(np.abs(bias)))

    return {
        "free": FREE, "truth": truth,
        "panels": panels,
        "fit0": {n: float(calib0.params_named[n]) for n in FREE},
        "fit0_std": {n: float(calib0.params_named_std[n]) for n in FREE},
        "mc_fits": fits.tolist(), "mc_stds": stds.tolist(),
        "mc_inits": inits, "mc_loss": losses, "use_priors": False,
        "start_spread": START_SPREAD,
        "bias_pct": bias.tolist(), "emp_cv_pct": emp_std.tolist(),
        "pred_cv_pct": pred_std.tolist(),
        "worst_bias_pct": float(abs(bias[worst])), "worst_bias_name": FREE[worst],
        "n_mc": N_MC, "n_obs_total": int(sum(len(p["t"]) for p in panels) * len(OBS)),
        "sigma": SIG.tolist(),
    }


def main():
    d = common.load_or_compute("calibration", _compute)
    plt = common.setup_mpl()
    common.scale_fonts(plt, 11.0)
    # Constrained layout sets the spacing; fixed hspace/wspace on top of it left the
    # top-row axes shrunk inside large gaps.
    fig = plt.figure(figsize=(11.0, 7.4))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.0])

    titles = {"aer": "(a) aerobic", "Nlim": "(b) nitrate-limited",
              "Slim": "(c) substrate-limited"}
    for k, p in enumerate(d["panels"]):
        ax = fig.add_subplot(gs[0, k])
        t = np.asarray(p["t"]); y = np.asarray(p["y"])
        td = np.asarray(p["td"]); fit = np.asarray(p["fit"])
        tru = np.asarray(p["truth_curve"]); c = p["primary_col"]; sp = p["primary_sp"]
        ax.plot(td, tru[:, c], "-", color="0.6", lw=3.0, alpha=0.45, label="truth")
        ax.plot(td, fit[:, c], "-", color="C3", lw=1.4, label="fit")
        ax.plot(t, y[:, c], "o", color="C0", ms=5.0, alpha=0.85, label="noisy data")
        ax.set(xlabel="time (d)", title=titles[p["name"]],
               ylabel=f"{OBSLBL[sp]} (mg/L)")
        if k == 0:
            ax.legend(frameon=False, loc="upper right")

    # --- recovery panel -----------------------------------------------------
    axr = fig.add_subplot(gs[1, :])
    fits = np.asarray(d["mc_fits"]); truth = np.array([d["truth"][n] for n in FREE])
    norm = fits / truth[None, :]                        # recovered / truth
    pos = np.arange(len(FREE))
    axr.axhline(1.0, color="0.4", lw=1.0, ls="--")
    bp = axr.boxplot([norm[:, j] for j in range(len(FREE))], positions=pos,
                     widths=0.5, patch_artist=True, showfliers=False)
    for patch in bp["boxes"]:
        patch.set(facecolor="C0", alpha=0.35)
    for med in bp["medians"]:
        med.set(color="C3", lw=1.5)
    # jittered points
    rjit = np.random.default_rng(0)
    for j in range(len(FREE)):
        axr.plot(pos[j] + 0.06 * rjit.standard_normal(norm.shape[0]),
                 norm[:, j], ".", color="0.3", ms=4.0, alpha=0.5)
    # Mean bias under each parameter name, to two significant figures (a fixed
    # decimal rounds the small ones to a meaningless "-0.0%").
    axr.set_xticks(pos)
    axr.set_xticklabels([f"{PLABEL[n]}\nbias {b:+.2g}%"
                         for n, b in zip(FREE, d["bias_pct"])])
    axr.set_ylabel("recovered / true")
    axr.set_title(f"(d) recovery over {d['n_mc']} noise realizations "
                  f"(worst bias {d['worst_bias_pct']:.2g}%)")
    axr.margins(y=0.08)

    import os
    for ext in ("png", "pdf"):    # bbox_inches=tight avoids clipping the rightmost title
        fig.savefig(os.path.join(common.FIG_DIR, f"fig_calibration.{ext}"),
                    bbox_inches="tight")
    print("  wrote fig_calibration.png / .pdf")
    print(f"  representative fit: " +
          ", ".join(f"{n}={d['fit0'][n]:.3f}(±{d['fit0_std'][n]:.3f})" for n in FREE))
    print(f"  MC bias%:   " + "  ".join(f"{n}:{b:+.2f}" for n, b in zip(FREE, d["bias_pct"])))
    print(f"  worst bias {d['worst_bias_pct']:.2f}% ({d['worst_bias_name']}); "
          f"n_obs={d['n_obs_total']}")


if __name__ == "__main__":
    main()
