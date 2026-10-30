"""Figure: exact gradients versus finite differences on the calibration loss.

The same joint-batch calibration loss of Fig.~\\ref{fig:calibration}, differentiated
two ways:

(a) Gradient accuracy. The finite-difference gradient error (relative to the exact
    reverse-mode gradient) is truncation-limited at large step and floors at the
    solver-noise level at small step, so a step size must be tuned; the exact
    gradient needs no step and carries only the solver tolerance. On this
    well-resolved batch the floor is low; it rises steeply on a stiff,
    recycle-coupled plant, whose small-step error is irregular rather than flat.
    Each floor is the median error at the best step and its two neighbours
    (``fd_floor``).
(b) Gradient cost. The reverse-mode gradient costs one adjoint pass regardless of
    the number of parameters; a finite-difference gradient costs two solves per
    parameter, so its cost grows linearly. Measured wall-clock per gradient.

Cache key: "ad_vs_fd".
"""
from __future__ import annotations
import time
import numpy as np
import common

FREE = ["muH", "Y_H", "bH", "etag", "kh"]
NREP = 30   # timed trials per parameter count in (b)
LOGIT = {"Y_H", "etag"}
OBS = ["SS", "SO", "SNO"]
SIG = np.array([1.0, 0.1, 0.2])
H_GRID = [1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5,
          1e-6, 1e-7, 1e-8, 1e-9]


def fd_floor(relerr):
    """The finite-difference error floor: the median error at the best step and its
    two grid neighbours. A single minimum can be a lucky cancellation (the plant
    curve has no plateau, and dips once at h = 1e-4), so it is not used alone."""
    e = np.asarray(relerr, float)
    i = int(np.clip(np.argmin(e), 1, len(e) - 2))
    return float(np.median(e[i - 1:i + 2]))


def _compute():
    import jax
    import jax.numpy as jnp
    import aquakin
    asm1 = aquakin.load_model("asm1")
    si, pi = asm1.species_index, asm1.param_index
    cond = asm1.default_conditions(1)
    rx = aquakin.BatchReactor(asm1, cond,
                              integrator=aquakin.IntegratorConfig(dtmax=1e-3))
    p_true = jnp.asarray(np.asarray(asm1.default_parameters()))

    def ic(**ov):
        C = np.asarray(asm1.default_concentrations()).copy()
        base = {"SI": 30., "XI": 1000., "XS": 40., "XND": 2., "SND": 1.,
                "SALK": 7., "XB_A": 80.}
        for k, v in {**base, **ov}.items():
            C[si[k]] = v
        return jnp.asarray(C)

    specs = [dict(C0=ic(XB_H=40.,  SS=60.,  SO=8., SNH=20., SNO=0.1), T=0.1),
             dict(C0=ic(XB_H=600., SS=200., SO=0., SNH=15., SNO=40.), T=0.45),
             dict(C0=ic(XB_H=300., SS=60.,  SO=0., SNH=15., SNO=60.), T=0.3)]
    oidx = jnp.array([si[s] for s in OBS])
    for b in specs:
        b["t"] = jnp.linspace(0., b["T"], 20)
        clean = rx.solve(b["C0"], (0., b["T"]), t_eval=b["t"], params=p_true).C[:, oidx]
        rng = np.random.default_rng(1)
        b["y"] = clean + jnp.asarray(SIG * rng.standard_normal((20, 3)))

    def make_loss(k):
        def loss(u):
            p = p_true
            for j in range(k):
                n = FREE[j]
                val = 1. / (1. + jnp.exp(-u[j])) if n in LOGIT else jnp.exp(u[j])
                p = p.at[pi[n]].set(val)
            tot = 0.0
            for b in specs:
                pred = rx.solve(b["C0"], (0., b["T"]), t_eval=b["t"], params=p).C[:, oidx]
                tot = tot + 0.5 * jnp.sum(((pred - b["y"]) / jnp.asarray(SIG)) ** 2)
            return tot
        return loss

    u_true = jnp.array([np.log(0.67 / 0.33) if n == "Y_H"
                        else np.log(0.8 / 0.2) if n == "etag"
                        else float(np.log(np.asarray(asm1.default_parameters())[pi[n]]))
                        for n in FREE])
    u_eval = u_true + 0.25            # off-optimum point where |grad| is sizeable

    # --- (a) gradient-accuracy V-curve --------------------------------------
    loss5 = make_loss(5)
    gfun = jax.jit(jax.grad(loss5)); lfun = jax.jit(loss5)
    g_ad = np.asarray(gfun(u_eval))
    relerr = []
    for h in H_GRID:
        g = np.zeros(5)
        for j in range(5):
            up = u_eval.at[j].add(h); um = u_eval.at[j].add(-h)
            g[j] = float((lfun(up) - lfun(um)) / (2 * h))
        relerr.append(float(np.linalg.norm(g - g_ad) / np.linalg.norm(g_ad)))
    floor = fd_floor(relerr)

    # --- (b) cost scaling ---------------------------------------------------
    # Each trial times one AD gradient and one central-FD gradient at every parameter
    # count; the trials loop outermost so slow machine drift spreads over all counts
    # instead of landing on one. Reported: the median over NREP trials, with the
    # 10th-90th percentile band (compilation excluded by a warm-up call).
    kk = list(range(1, 6))
    fns = {}
    for k in kk:
        lf = jax.jit(make_loss(k)); gf = jax.jit(jax.grad(make_loss(k)))
        ue = u_eval[:k]
        gf(ue).block_until_ready(); lf(ue).block_until_ready()   # warm compile
        fns[k] = (lf, gf, ue)
    raw_ad = {k: [] for k in kk}; raw_fd = {k: [] for k in kk}
    for _ in range(NREP):
        for k in kk:
            lf, gf, ue = fns[k]
            t0 = time.perf_counter()
            gf(ue).block_until_ready()
            raw_ad[k].append((time.perf_counter() - t0) * 1e3)
            t0 = time.perf_counter()
            for j in range(k):
                lf(ue.at[j].add(1e-3)).block_until_ready()
                lf(ue.at[j].add(-1e-3)).block_until_ready()
            raw_fd[k].append((time.perf_counter() - t0) * 1e3)
    ks = kk
    t_ad = [float(np.median(raw_ad[k])) for k in kk]
    t_fd = [float(np.median(raw_fd[k])) for k in kk]
    pct = lambda r, q: [float(np.percentile(r[k], q)) for k in kk]
    t_ad_p10, t_ad_p90 = pct(raw_ad, 10), pct(raw_ad, 90)
    t_fd_p10, t_fd_p90 = pct(raw_fd, 10), pct(raw_fd, 90)

    ratio5 = t_fd[-1] / t_ad[-1]
    # linear-fit the FD cost (through origin-ish) to project to many parameters
    per_param = float(np.polyfit(ks, t_fd, 1)[0])
    ad_flat = float(np.mean(t_ad))
    proj20 = (per_param * 20) / ad_flat

    # --- the same accuracy V-curve on the stiff, recycle-coupled BSM1 plant --
    # A calibration-style loss on the effluent-ammonia trajectory of the full
    # open-loop plant, at the benchmark 15 C operating point, differentiated
    # w.r.t. three ASM1 rates. The plant solve is longer, stiffer and
    # recycle-coupled than the batch, so the finite-difference floor is orders
    # higher; the exact adjoint carries only the solver tolerance either way.
    from aquakin.plant.bsm import build_bsm1, bsm1_warm_start
    from aquakin.plant import InfluentSeries
    # The validated plant of Figs. 6-7 (ten-layer Takacs settler, soluble holdup).
    plant = build_bsm1(model=asm1, use_takacs=True, settler_soluble_holdup=True,
                       settler_composition_mode="lumped_tss")
    plant.add_influent("feed", InfluentSeries.constant(
        asm1, Q=18446.0, SS=69.5, XS=202.32, XB_H=28.17, SNH=31.56, SND=6.95,
        XND=10.59, SI=30.0, XI=51.2, SALK=7.0))
    pparams = jnp.asarray(np.asarray(common.bsm1_params(plant)))
    py0 = jnp.asarray(bsm1_warm_start(plant))
    PFREE = ["asm1.muA", "asm1.muH", "asm1.bH"]
    ppidx = [plant.parameter_index(n) for n in PFREE]
    pth0 = jnp.array([float(pparams[i]) for i in ppidx])
    pteval = jnp.linspace(1.0, 5.0, 9)
    snh = si["SNH"]

    def ptraj(theta):
        p = pparams
        for k, i in enumerate(ppidx):
            p = p.at[i].set(theta[k])
        sol = plant.solve((0.0, 5.0), t_eval=pteval, params=p, y0=py0,
                          rtol=1e-6, forward_fast=False)
        return sol.unit_state("tank5")[:, snh]

    pref = jnp.asarray(np.asarray(ptraj(pth0)))

    def ploss(theta):
        return 0.5 * jnp.sum(((ptraj(theta) - pref) / 0.2) ** 2)

    pth_eval = pth0 * 1.15                      # perturbed point, |grad| nonzero
    pg_ad = np.asarray(jax.grad(ploss)(pth_eval))
    plj = jax.jit(ploss); plj(pth_eval).block_until_ready()
    p_relerr = []
    for h in H_GRID:
        g = np.zeros(len(PFREE))
        for j in range(len(PFREE)):
            hp = float(pth_eval[j]) * h
            up = pth_eval.at[j].add(hp); um = pth_eval.at[j].add(-hp)
            g[j] = float((plj(up) - plj(um)) / (2 * hp))
        p_relerr.append(float(np.linalg.norm(g - pg_ad) / np.linalg.norm(pg_ad)))
    p_floor = fd_floor(p_relerr)

    return {
        "h": H_GRID, "relerr": relerr, "floor": floor,
        "g_ad": g_ad.tolist(),
        "plant_relerr": p_relerr, "plant_floor": p_floor,
        "floor_ratio": float(p_floor / floor),
        "k": ks, "t_ad": t_ad, "t_fd": t_fd, "n_rep": NREP,
        "t_ad_p10": t_ad_p10, "t_ad_p90": t_ad_p90,
        "t_fd_p10": t_fd_p10, "t_fd_p90": t_fd_p90,
        "ratio5": float(ratio5), "ad_flat_ms": ad_flat,
        "fd_per_param_ms": per_param, "proj20x": float(proj20),
        "n_free": len(FREE),
    }


def _pow10(x):
    """A small number as LaTeX mantissa x 10^exp, e.g. 2.0\times10^{-8}."""
    m, e = f"{x:.1e}".split("e")
    return rf"{m}\times10^{{{int(e)}}}"


def main():
    d = common.load_or_compute("ad_vs_fd", _compute)
    plt = common.setup_mpl()
    common.scale_fonts(plt, 11.0)
    fig, ax = plt.subplots(1, 2, figsize=(11.0, 4.3))

    h = np.asarray(d["h"]); rel = np.asarray(d["relerr"])
    prel = np.asarray(d["plant_relerr"])
    ax[0].loglog(h, rel, "o-", color="C0", ms=4, label="batch")
    ax[0].loglog(h, prel, "s-", color="C1", ms=4, label="BSM1 plant")
    # The figure is drawn 11 in wide and shrunk to the text width, so the floor lines
    # need a heavy weight to stay visible on the page.
    for fl, col in ((d["floor"], "C0"), (d["plant_floor"], "C1")):
        ax[0].axhline(fl, color=col, ls=(0, (1.2, 1.2)), lw=2.4, zorder=1)
    # the batch floor label goes below its line (room made by lowering the y-limit),
    # clear of the batch curve, which sits on the floor at small steps
    ax[0].annotate(f"floor ${_pow10(d['floor'])}$", (h[0], d["floor"]),
                   textcoords="offset points", xytext=(0, -4), va="top", color="C0")
    ax[0].set_ylim(bottom=d["floor"] / 8)
    # the plant floor label goes at the small-step (right) end, below its line, where
    # the plant curve is far above it and the batch curve still well below
    ax[0].annotate(f"floor ${_pow10(d['plant_floor'])}$", (h[-1], d["plant_floor"]),
                   textcoords="offset points", xytext=(-2, -4), va="top", ha="right",
                   color="C1")
    ax[0].text(0.97, 0.95, "exact gradient (AD):\nno step to tune",
               transform=ax[0].transAxes, color="C3", ha="right", va="top")
    ax[0].set(xlabel="relative finite-difference step $h$",
              ylabel="FD gradient error (rel. to exact)",
              title="(a) finite-difference accuracy")
    # the legend sits in the gap between the batch floor and the plant curve
    ax[0].legend(frameon=False, loc="lower center", bbox_to_anchor=(0.62, 0.13))
    ax[0].invert_xaxis()

    k = np.asarray(d["k"]); tad = np.asarray(d["t_ad"]); tfd = np.asarray(d["t_fd"])
    ax[1].errorbar(k, tad, yerr=[tad - np.asarray(d["t_ad_p10"]), np.asarray(d["t_ad_p90"]) - tad],
                   fmt="s-", color="C3", ms=5, capsize=3, label="exact gradient (AD)")
    ax[1].errorbar(k, tfd, yerr=[tfd - np.asarray(d["t_fd_p10"]), np.asarray(d["t_fd_p90"]) - tfd],
                   fmt="o-", color="C0", ms=5, capsize=3, label="finite difference")
    kx = np.linspace(1, d["n_free"], 10)
    ax[1].plot(kx, d["fd_per_param_ms"] * kx, "--", color="C0", lw=0.9, alpha=0.6)
    ax[1].set(xlabel="number of calibrated parameters",
              ylabel="wall-clock per gradient (ms)",
              title=f"(b) gradient cost ({d['ratio5']:.1f}$\\times$ at "
                    f"{d['n_free']} params)")
    ax[1].legend(frameon=False, loc="upper left")
    ax[1].set_ylim(bottom=0)

    common.save(fig, "fig_ad_vs_fd")
    print(f"  batch FD floor {d['floor']:.1e}; plant FD floor {d['plant_floor']:.1e} "
          f"({d['floor_ratio']:.0f}x higher)")
    print(f"  AD flat {d['ad_flat_ms']:.0f} ms; FD {d['fd_per_param_ms']:.0f} ms/param; "
          f"cost {d['ratio5']:.1f}x at {d['n_free']} params, proj {d['proj20x']:.0f}x at 20")


if __name__ == "__main__":
    main()
