"""Figure: practical identifiability from the exact Fisher information.

The engine's exact gradients give the Fisher information of a design, F = S^T
Sigma^-1 S, before any data are collected (``aquakin.fisher_information``, at the
true parameter values), and ``aquakin.identifiability`` reads it: the eigen-
directions with their loadings, the Cramer-Rao standard deviations (infinite on an
exactly null direction), and the correlations.

We fix the five heterotroph parameters (muH, Y_H, bH, etag, kh) and vary the
*design*, using the data likelihood alone (no priors):

  full    aerobic respirogram + two anoxic assays
  anoxic  the two anoxic assays only (the aerobic respirogram removed)
  aerobic the aerobic respirogram alone (for the additivity check)

In anoxic conditions heterotrophic growth carries the factor ``etag*muH`` as a
product, so anoxic data alone cannot separate the two -- only the aerobic assay,
where growth runs at ``muH`` without ``etag``, breaks the degeneracy. The
information of independent assays adds (F_full = F_aerobic + F_anoxic), so the
aerobic respirogram adds information exactly along the anoxic design's null
direction; ``aquakin.rank_experiments`` confirms that it is the only addition to
the anoxic pair that makes every parameter identifiable.

(a) The Fisher eigenvalue spectrum. Dropping the aerobic assay turns the weakest
    direction from weak (practical) into null (structural).
(b) The loadings of that weakest direction in both designs: the same combination
    of log muH and logit etag, the one that keeps etag*muH fixed.
(c) Monte-Carlo recovery from perturbed starts. Under the anoxic-only design muH and
    etag slide along the product-preserving ridge (muH*etag fixed) while their
    individual values are lost -- the degradation the Fisher information predicts.
(d) The profile likelihood of muH, which checks the conclusion beyond the local
    quadratic approximation.

Cache key: "identifiability".
"""
from __future__ import annotations
import numpy as np
import common
from fig_calibration import _setup, SIG, OBS, FREE, PLABEL

PL = dict(PLABEL)
N_MC = 16                                   # recovery realizations per design


def _transforms(names):
    return {n: ("logit" if n in ("etag", "Y_H") else "positive_log") for n in names}


def _fit(aquakin, rx, C0s, obs, tobs, initial=None):
    return aquakin.calibrate(
        rx, C0s, observations=obs, t_obs=tobs, free_params=FREE,
        observed_species=OBS, transforms=_transforms(FREE),
        loss="nll", sigma=[SIG] * len(C0s), optimizer=aquakin.OptimizerConfig(method="lbfgsb", n_starts=1),
        diff=aquakin.DifferentiationConfig(mode="reverse", method="stable"),
        laplace=aquakin.LaplaceConfig(method="gauss_newton"),
        use_priors=False, initial_params=initial)   # data-only Fisher


def _experiments(aquakin, jnp, specs):
    """The assays as ``aquakin.Experiment`` designs, keyed by name."""
    return {b["name"]: aquakin.Experiment(b["C0"], t_obs=b["t"], sigma=jnp.asarray(SIG),
                                          observed_species=OBS) for b in specs}


def _report(aquakin, fim):
    """The identifiability read-out of one design: descending eigenvalues, the
    weakest direction (relative eigenvalue and loadings), the Cramer-Rao relative
    std, the muH-etag correlation and the D-criterion."""
    r = aquakin.identifiability(fim)
    w = np.asarray(r.eigenvalues)
    weak = r.directions[0]
    return {
        "ev": np.sort(np.clip(w, 1e-30, None))[::-1].tolist(),
        "kappa": float(w[-1] / max(w[0], 1e-300)),
        "weak_rel": float(weak.relative_eigenvalue),
        "weak_load": {n: float(weak.loadings[n]) for n in FREE},
        "crb_rel_pct": {n: float(100.0 * r.relative_std[n]) for n in FREE},
        "corr_me": float(r.correlation[FREE.index("muH"), FREE.index("etag")]),
        "D": float(aquakin.design_criterion(fim)),
    }


def _noise(specs, rng):
    return [b["clean"] + SIG * rng.standard_normal(b["clean"].shape) for b in specs]


def _mc(jnp, aquakin, rx, specs, truth, pi, seed0):
    """Perturbed-start recovery over N_MC noise realizations for one design."""
    C0s = [b["C0"] for b in specs]; tobs = [b["t"] for b in specs]
    out = []
    for s in range(N_MC):
        rng = np.random.default_rng(seed0 + s)
        obs = _noise(specs, rng)
        full = _full_init(pi, truth, rng)
        c = _fit(aquakin, rx, C0s, obs, tobs, initial=jnp.asarray(full))
        out.append([float(c.params_named[n]) for n in FREE])
    return np.array(out)


def _profile_muH(aquakin, jnp, rx, specs, obs, truth_all, pi, mu_grid):
    """Profiled objective: fix muH across the grid, re-optimise the other free
    parameters, and read the best-attainable data loss. A flat profile means muH is
    not identifiable (the other parameters absorb the change); a sharp minimum means
    it is. This does not rely on the local (Fisher) quadratic approximation."""
    C0s = [b["C0"] for b in specs]; tobs = [b["t"] for b in specs]
    free = [n for n in FREE if n != "muH"]
    losses = []
    for mu in mu_grid:
        init = np.asarray(truth_all["_p"]).copy(); init[pi["muH"]] = float(mu)
        c = aquakin.calibrate(
            rx, C0s, observations=obs, t_obs=tobs, free_params=free,
            observed_species=OBS, transforms=_transforms(free), loss="nll",
            sigma=[SIG] * len(C0s), optimizer=aquakin.OptimizerConfig(method="lbfgsb", n_starts=1),
            diff=aquakin.DifferentiationConfig(mode="reverse", method="stable"),
            laplace=False, use_priors=False,
            initial_params=jnp.asarray(init))
        losses.append(float(c.loss))
    return np.array(losses)


def _profile_interval(mu_grid, losses, delta=1.92):
    """The muH interval whose profiled objective is within the 95% likelihood-ratio
    threshold (delta=1.92 for one parameter) of the minimum. Each end is linearly
    interpolated to the threshold crossing, so the width is not quantised to the grid
    step; an end that does not cross inside the grid is returned at the grid edge
    and flagged open (the width is then a lower bound)."""
    d = np.asarray(losses) - float(np.min(losses))
    i0 = int(np.argmin(d))

    def _cross(idx):          # walk from the minimum until d exceeds delta
        for a, b in zip(idx[:-1], idx[1:]):
            if d[b] > delta:
                return float(np.interp(delta, [d[a], d[b]], [mu_grid[a], mu_grid[b]])), False
        return float(mu_grid[idx[-1]]), True

    lo, lo_open = _cross(list(range(i0, -1, -1)))
    hi, hi_open = _cross(list(range(i0, len(d))))
    return lo, hi, lo_open, hi_open


def _full_init(pi, truth_all, rng):
    """Full parameter vector with the free params perturbed off truth."""
    init = np.asarray(truth_all["_p"]).copy()
    for n in FREE:
        val = truth_all[n] * float(np.exp(0.4 * rng.standard_normal()))
        if n in ("etag", "Y_H"):
            val = min(max(val, 0.05), 0.95)
        init[pi[n]] = val
    return init


def _compute():
    jnp, aquakin, asm1, si, pi, rx, p_true, specs, oidx = _setup()
    truth = {n: float(p_true[pi[n]]) for n in FREE}
    truth_all = {**truth, "_p": np.asarray(p_true)}
    full_specs = specs                       # aerobic + two anoxic
    anox_specs = [b for b in specs if b["name"] != "aer"]

    rng = np.random.default_rng(0)
    obs_full = _noise(full_specs, rng)
    obs_anox = [o for b, o in zip(full_specs, obs_full) if b["name"] != "aer"]

    # --- (a,b) Fisher information of each design, at the true values ---------
    ex = _experiments(aquakin, jnp, specs)
    designs = {"full": [ex["aer"], ex["Nlim"], ex["Slim"]],
               "anox": [ex["Nlim"], ex["Slim"]], "aer": [ex["aer"]]}
    fkw = dict(params=jnp.asarray(p_true), transforms=_transforms(FREE))
    fims = {k: aquakin.fisher_information(rx, v, FREE, **fkw) for k, v in designs.items()}
    rep = {k: _report(aquakin, f) for k, f in fims.items()}
    # independent assays add: F_full = F_aerobic + F_anoxic (to round-off)
    Ff = fims["full"].matrix
    additivity = float(np.linalg.norm(Ff - fims["anox"].matrix - fims["aer"].matrix)
                       / np.linalg.norm(Ff))
    # What to add to the anoxic pair: the aerobic respirogram, or a repeat of
    # either anoxic assay (more of the same data).
    cand = ["aer", "Nlim", "Slim"]
    ranked = aquakin.rank_experiments(rx, [ex[c] for c in cand], FREE,
                                      base=designs["anox"], **fkw)
    rank = [{"candidate": cand[c.index], "D": float(c.value)} for c in ranked]

    # --- (c) recovery under each design (perturbed starts) ------------------
    mc_full = _mc(jnp, aquakin, rx, full_specs, truth_all, pi, 3000)
    mc_anox = _mc(jnp, aquakin, rx, anox_specs, truth_all, pi, 5000)

    def cv(mc, n):
        col = mc[:, FREE.index(n)]
        return float(col.std() / abs(col.mean()) * 100.0)

    prod = lambda mc: mc[:, FREE.index("muH")] * mc[:, FREE.index("etag")]
    prod_true = truth["muH"] * truth["etag"]
    prod_cv_anox = float(prod(mc_anox).std() / prod_true * 100.0)

    # --- profile likelihood of muH (beyond the local Fisher approximation) ---
    # 61 points (~0.13 1/d apart) so the 95% interval is resolved, not grid-limited.
    mu_grid = np.linspace(0.5 * truth["muH"], 1.8 * truth["muH"], 61)
    prof_full = _profile_muH(aquakin, jnp, rx, full_specs, obs_full, truth_all, pi, mu_grid)
    prof_anox = _profile_muH(aquakin, jnp, rx, anox_specs, obs_anox, truth_all, pi, mu_grid)
    ci_full = _profile_interval(mu_grid, prof_full)
    ci_anox = _profile_interval(mu_grid, prof_anox)
    w_full = ci_full[1] - ci_full[0]
    w_anox = ci_anox[1] - ci_anox[0]
    mu_span = float(mu_grid[-1] - mu_grid[0])

    return {
        "free": FREE, "names": list(fims["full"].names), "truth": truth,
        "fisher": rep, "additivity_relerr": additivity, "rank_vs_anox": rank,
        "ev_f": rep["full"]["ev"], "ev_a": rep["anox"]["ev"],
        "kappa_f": rep["full"]["kappa"], "kappa_a": rep["anox"]["kappa"],
        "corr_full_me": rep["full"]["corr_me"], "corr_anox_me": rep["anox"]["corr_me"],
        "mc_full": mc_full.tolist(), "mc_anox": mc_anox.tolist(),
        "muH_cv_full": cv(mc_full, "muH"), "muH_cv_anox": cv(mc_anox, "muH"),
        "etag_cv_full": cv(mc_full, "etag"), "etag_cv_anox": cv(mc_anox, "etag"),
        "prod_cv_anox": prod_cv_anox, "n_mc": N_MC,
        "mu_grid": mu_grid.tolist(), "prof_full": prof_full.tolist(),
        "prof_anox": prof_anox.tolist(), "prof_w_full": w_full,
        "prof_w_anox": w_anox, "mu_span": mu_span,
        "prof_ci_full": list(ci_full), "prof_ci_anox": list(ci_anox),
    }


def main():
    d = common.load_or_compute("identifiability", _compute)
    plt = common.setup_mpl()
    common.scale_fonts(plt, 11.0)
    fig, axs = plt.subplots(2, 2, figsize=(11.0, 8.6))
    ax = axs.ravel()

    def _pow10(x):
        """Condition number as a power-of-ten LaTeX label, e.g. 5\times10^{4}."""
        m, e = f"{x:.1e}".split("e")
        if m == "1.0":
            return rf"10^{{{int(e)}}}"
        return rf"{m}\times10^{{{int(e)}}}"

    # (a) Fisher eigenvalue spectrum
    ev_f = np.asarray(d["ev_f"]); ev_a = np.asarray(d["ev_a"])
    ax[0].semilogy(range(1, len(ev_f) + 1), ev_f, "o-", color="C0",
                   label=f"full ($\\kappa\\approx{_pow10(d['kappa_f'])}$)")
    ax[0].semilogy(range(1, len(ev_a) + 1), ev_a, "s--", color="C3",
                   label=(f"anoxic only ($\\kappa\\approx{_pow10(d['kappa_a'])}$)"
                          if d["kappa_a"] < 1.0 / np.finfo(float).eps
                          else "anoxic only (singular)"))
    ax[0].set(xlabel="eigenvalue index", ylabel="Fisher eigenvalue",
              title="(a) Fisher spectrum")
    ax[0].legend(frameon=False, loc="lower left")

    # (b) loadings of the weakest Fisher direction in each design: the same
    #     log muH / logit etag combination, weak in the full design, null without
    #     the aerobic assay.
    xs_ = np.arange(len(FREE)); wbar = 0.38
    for off, key, lab, col in ((-wbar / 2, "full", "full", "C0"),
                               (wbar / 2, "anox", "anoxic only", "C3")):
        f = d["fisher"][key]
        load = [f["weak_load"][n] for n in FREE]
        ax[1].bar(xs_ + off, load, width=wbar, color=col,
                  label=rf"{lab} ($\lambda/\lambda_{{max}}={_pow10(f['weak_rel'])}$)")
    ax[1].axhline(0.0, color="0.3", lw=0.8)
    ax[1].set_xticks(xs_); ax[1].set_xticklabels([PL[n] for n in FREE])
    ax[1].set(ylabel="loading (log / logit coordinates)", ylim=(-1.15, 0.85),
              title="(b) weakest Fisher direction")
    ax[1].legend(frameon=False, loc="upper right")

    # (c) recovered etag vs muH: full clusters at truth, anoxic slides on the ridge
    mf = np.asarray(d["mc_full"]); ma = np.asarray(d["mc_anox"])
    tm, te = d["truth"]["muH"], d["truth"]["etag"]
    xr = mf[:, FREE.index("muH")] / tm; yr = mf[:, FREE.index("etag")] / te
    xa = ma[:, FREE.index("muH")] / tm; ya = ma[:, FREE.index("etag")] / te
    xs = np.linspace(0.55, 1.8, 100)
    ax[2].plot(xs, 1.0 / xs, "-", color="0.6", lw=1.0,
               label=r"$\mu_H\eta_g=$ const")
    ax[2].plot(xa, ya, "s", color="C3", ms=5, alpha=0.75, label="anoxic only")
    ax[2].plot(xr, yr, "o", color="C0", ms=5, alpha=0.85, label="full")
    ax[2].plot(1.0, 1.0, "*", color="k", ms=13, zorder=5)
    ax[2].set(xlabel=r"recovered $\mu_H$ / true", ylabel=r"recovered $\eta_g$ / true",
              title="(c) the $\\mu_H$–$\\eta_g$ degeneracy")
    ax[2].legend(frameon=False, loc="upper right")

    # (d) profile likelihood of muH: the objective re-minimised over the other free
    # parameters at each fixed muH, relative to its minimum, with the 95% LR threshold.
    mg = np.asarray(d["mu_grid"])
    for key, lab, col, ls in (("full", "full", "C0", "-"), ("anox", "anoxic only", "C3", "--")):
        prof = np.asarray(d[f"prof_{key}"]); dp = prof - prof.min()
        lo, hi, _, hi_open = d[f"prof_ci_{key}"]
        ax[3].plot(mg, dp, ls, color=col, lw=1.6, label=lab)
        ax[3].axvspan(lo, hi, color=col, alpha=0.10, lw=0)
    ax[3].axhline(1.92, color="0.4", ls=":", lw=1.0)
    ax[3].text(mg[-1] - 0.1, 1.92 + 0.3, "95% threshold", color="0.35", va="bottom",
               ha="right")
    ax[3].axvline(tm, color="k", lw=0.8)
    ax[3].set(xlabel=r"fixed $\mu_H$ (d$^{-1}$)",
              ylabel=r"profiled $-\log L$ above minimum", ylim=(-0.5, 12.0),
              xlim=(mg[0], mg[-1]), title=r"(d) profile likelihood of $\mu_H$")
    # Right of the full-design arm and above the threshold label, the one clear region.
    ax[3].legend(frameon=False, loc="center right", bbox_to_anchor=(1.0, 0.37),
                 handlelength=1.2, labelspacing=0.3)

    common.save(fig, "fig_identifiability")
    print(f"  kappa full {d['kappa_f']:.2e}  anoxic {d['kappa_a']:.2e} "
          f"(ratio {d['kappa_a'] / d['kappa_f']:.1f})")
    print(f"  corr(muH,etag): full {d['corr_full_me']:+.3f}  anoxic {d['corr_anox_me']:+.4f}")
    print(f"  muH  CV%: full {d['muH_cv_full']:.0f}  anoxic {d['muH_cv_anox']:.0f}")
    print(f"  etag CV%: full {d['etag_cv_full']:.0f}  anoxic {d['etag_cv_anox']:.0f}")
    print(f"  product muH*etag CV% (anoxic) {d['prod_cv_anox']:.0f}")
    print(f"  profile 95% CI muH: full {d['prof_ci_full']}  anoxic {d['prof_ci_anox']}")


if __name__ == "__main__":
    main()
