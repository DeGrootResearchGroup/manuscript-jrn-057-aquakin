"""Figure: real-data model-structure comparison on aerobic OUR respirograms.

The real-measurement counterpart to the synthetic twin-experiment calibration
(fig_calibration). The same exact-gradient estimator calibrates TWO different
heterotroph model structures -- swapped by editing the YAML network, nothing else
-- to the same three digitized OUR respirograms (Gupta 2018 thesis, Fig. 4-1, Run #1; filtered
municipal wastewater, ATU-inhibited so heterotrophs only):

  * REDUCED (ASM1 heterotrophic aerobic core with death-regeneration): growth on
    readily biodegradable S_S, first-order hydrolysis of a soluble fraction S_H,
    death-regeneration decay. Growth-only structures are known to misfit the
    famine phase of real OUR (Guisasola et al. 2005); here they capture the peak
    but leave a slow-tail residual, and muH is recovered.
  * STORAGE (modified ASM3, simultaneous storage and growth; Karahan-Gul et al.
    2003): heterotrophs both grow directly on S_S and store it as X_STO, then grow
    on the stored polymer. Storage rounds the peak and its oxidation sustains the
    decaying tail, fitting the whole curve -- but the direct-growth muH is then
    non-identifiable from OUR alone (storage absorbs it), which the exact Fisher
    information exposes as a flat profile likelihood.

This is a real-data instance of the identifiability structure shown on twin data
in fig_identifiability: a richer structure that fits better can render a parameter
unidentifiable, and the exact gradient reads that off directly. Goodness of fit is
reported as RMSE on the OUR (R^2 alongside), following respirometric practice where
R^2 on a smooth OUR curve is uninformative.

Cache key: "respirometry".
"""
from __future__ import annotations
import os
import re
import numpy as np
import common

# name: (muH_thesis, X_H, S_S, S_H)  mg COD/L  (Gupta Table 4-2, 3-run average)
WW = {
    "RWW":  (2.30, 18.0, 239.0, 64.0),
    "PCE":  (3.46, 8.0, 235.0, 84.0),
    "RBFE": (2.48, 17.0, 222.0, 74.0),
}
PROFILE_WW = "RWW"                       # wastewater used for the muH profile panel
# muH values profiled: log-spaced (the fit works in log muH, and panel (d) is on a
# log axis), from the storage model's muH lower bound, with extra points just
# above it so the storage profile's 95% crossing is resolved, not interpolated.
MU_GRID = np.unique(np.round(np.concatenate([np.geomspace(0.2, 0.4, 9),
                                             np.geomspace(0.4, 6.0, 24)]), 6))
PROF_DELTA = 1.92                        # 95% profile threshold on the nll, chi2_{1,0.95}/2
PROF_YTOP_LOG = 8.0                      # y-limit of the profile panel (nll units)

# ---- reduced (ASM1 death-regeneration) fixed parameters --------------------
DR_YH, DR_BH, DR_FP, DR_KS = 0.66, 0.40, 0.20, 20.0
# ---- storage (modified ASM3) fixed parameters ------------------------------
ST_YSTO, ST_YH, ST_KS, ST_KSTO = 0.85, 0.66, 2.0, 1.0
ST_FXI, ST_BH, ST_BSTO = 0.20, 0.40, 0.20

# The two model structures, declared as aquakin YAML in data/.
DR_MODEL = os.path.join(common.DATA_DIR, "respirometry_reduced.yaml")
ST_MODEL = os.path.join(common.DATA_DIR, "respirometry_storage.yaml")


def load_curves():
    path = os.path.join(common.DATA_DIR, "gupta2018_fig4-1_OUR_digitized.csv")
    rows = [ln.split(",") for ln in open(path).read().splitlines()[2:]]
    out = {}
    for name, (i, j) in [("RWW", (0, 1)), ("PCE", (2, 3)), ("RBFE", (4, 5))]:
        xs, ys = [], []
        for r in rows:
            r = (r + [""] * 6)[:6]
            try:
                xs.append(float(r[i])); ys.append(float(r[j]))
            except ValueError:
                pass
        x, y = np.array(xs), np.array(ys)
        o = np.argsort(x)
        out[name] = (x[o] / 24.0, y[o] * 24.0)   # h->d, mg/L/h -> g/m3/d
    return out


def with_fit(mg, prof, mu_fit, d_fit):
    """The profile with the free fit (muH, nll rise) inserted at its place on the grid."""
    mg, prof = np.append(mg, mu_fit), np.append(prof, d_fit)
    o = np.argsort(mg)
    return mg[o], prof[o]


def _compute():
    import jax
    import jax.numpy as jnp
    import aquakin
    from aquakin import FreeICConfig, LaplaceConfig, OptimizerConfig

    # ---- reduced model -----------------------------------------------------
    net_dr = aquakin.load_model_from_file(DR_MODEL); si, pi = net_dr.species_index, net_dr.param_index
    SO = si["SO"]; cond = net_dr.default_conditions(1)
    rx_dr = aquakin.BatchReactor(net_dr, cond, integrator=aquakin.IntegratorConfig(max_steps=1_000_000))

    def dr_params(muH, khS):
        p = jnp.asarray(net_dr.default_parameters())
        return (p.at[pi["muH"]].set(muH).at[pi["khS"]].set(khS).at[pi["Y_H"]].set(DR_YH)
                .at[pi["bH"]].set(DR_BH).at[pi["f_P"]].set(DR_FP).at[pi["KS"]].set(DR_KS)
                .at[pi["KOH"]].set(0.2).at[pi["KX"]].set(0.03))

    def dr_C0(X_H, S_S, S_H):
        return (jnp.zeros(net_dr.n_species).at[si["XH"]].set(X_H).at[si["SS"]].set(S_S)
                .at[si["SH"]].set(S_H).at[si["SO"]].set(500.0))

    def our_dr(C, p):
        """Observation function: OUR (g O2/m3/d) = minus the dissolved-O2 rate at state C."""
        return -net_dr.dCdt(C, p, cond.fields, 0)[SO]

    def dr_curve(muH, khS, X_H, S_S, S_H, t):
        p = dr_params(muH, khS)
        sol = rx_dr.solve(dr_C0(X_H, S_S, S_H), (0.0, float(t[-1])), t_eval=t, params=p)
        return jax.vmap(lambda c: our_dr(c, p))(sol.C)

    # ---- storage model -----------------------------------------------------
    net_st = aquakin.load_model_from_file(ST_MODEL); sj, pj = net_st.species_index, net_st.param_index
    SOj = sj["SO"]; condj = net_st.default_conditions(1)
    rx_st = aquakin.BatchReactor(net_st, condj, integrator=aquakin.IntegratorConfig(max_steps=2_000_000))

    def st_params(muH, kSTO, muSTO):
        p = jnp.asarray(net_st.default_parameters())
        for nm, v in [("muH", muH), ("kSTO", kSTO), ("muSTO", muSTO), ("KS", ST_KS),
                      ("KSTO", ST_KSTO), ("YSTO", ST_YSTO), ("Y_H", ST_YH),
                      ("bH", ST_BH), ("bSTO", ST_BSTO), ("f_XI", ST_FXI), ("KOH", 0.2)]:
            p = p.at[pj[nm]].set(v)
        return p

    def st_C0(X_H, S_S):
        return jnp.zeros(net_st.n_species).at[sj["XH"]].set(X_H).at[sj["SS"]].set(S_S).at[sj["SO"]].set(500.0)

    def our_st(C, p):
        """Observation function: OUR (g O2/m3/d) = minus the dissolved-O2 rate at state C."""
        return -net_st.dCdt(C, p, condj.fields, 0)[SOj]

    def st_curve(muH, kSTO, muSTO, X_H, S_S, t):
        p = st_params(muH, kSTO, muSTO)
        sol = rx_st.solve(st_C0(X_H, S_S), (0.0, float(t[-1])), t_eval=t, params=p)
        return jax.vmap(lambda c: our_st(c, p))(sol.C)

    # ---- the library estimator ---------------------------------------------
    # Every fit and profile point is one aquakin.calibrate call over a fixed grid
    # of explicit starts (OptimizerConfig(starts=...), one compiled objective; the
    # lowest-loss optimum is kept). The loss is the Gaussian nll with the noise
    # level estimated (sigma="estimate", the concentrated likelihood), so one fit
    # gives the estimates, the noise std and the Laplace std together. The rate
    # parameters are held to the bounds declared in the model files. The initial
    # S_S (and, reduced model, S_H) are free_ic pools, fitted in log space and
    # boxed to the fSS range sigmoid(+-3) of the total COD; they start from the
    # same C0 for every start.
    F_LO, F_HI = float(jax.nn.sigmoid(-3.0)), float(jax.nn.sigmoid(3.0))

    def _calib(rx, obs, free, pools, COD, t_d, our_d, p0, C0, starts, laplace=False):
        """One multistart fit from the explicit `starts` ({name: value} over `free`)."""
        return aquakin.calibrate(
            rx, C0, our_d, t_d, free, initial_params=p0, observation_fn=obs,
            loss="nll", sigma="estimate", use_priors=False,
            free_ic=FreeICConfig(pools, bounds=(F_LO * COD, F_HI * COD)),
            optimizer=OptimizerConfig(max_iter=300, starts=starts),
            laplace=LaplaceConfig(method="gauss_newton", include_ic=True) if laplace else False)

    def metrics(model, our):
        resid = np.asarray(model) - our
        rmse = float(np.sqrt(np.mean(resid ** 2)))
        r2 = 1.0 - float(np.sum(resid ** 2)) / float(np.sum((our - our.mean()) ** 2))
        return rmse, r2

    # ---- fit reduced: free {muH, khS, S_S0, S_H0} ----------------------------
    def fit_dr(X_H, COD, t_d, our_d):
        starts = [{"muH": m, "khS": k} for m in (1.5, 2.5, 4.0) for k in (0.3, 1.0, 3.0)]
        r = _calib(rx_dr, our_dr, ["muH", "khS"], ["SS", "SH"], COD, t_d, our_d,
                   dr_params(2.5, 1.0), dr_C0(X_H, 0.25 * COD, 0.75 * COD), starts, laplace=True)
        sig = float(np.asarray(r.sigma_estimate)[0])
        ic, sd, icsd = r.ic_named[0], r.params_named_std, r.ic_named_std[0]
        return dict(muH=r.params_named["muH"], khS=r.params_named["khS"],
                    S_S0=ic["SS"], S_H0=ic["SH"], fSS=ic["SS"] / (ic["SS"] + ic["SH"]),
                    cod_ratio=(ic["SS"] + ic["SH"]) / COD, sigma=sig, nll=r.loss,
                    muH_std=sd["muH"], khS_std=sd["khS"], S_S0_std=icsd["SS"], S_H0_std=icsd["SH"])

    # ---- fit storage: free {muH, kSTO, muSTO, S_S0} --------------------------
    def fit_st(X_H, COD, t_d, our_d):
        starts = [{"muH": m, "kSTO": k, "muSTO": 2.0} for m in (1.0, 2.5, 4.0) for k in (3.0, 10.0)]
        r = _calib(rx_st, our_st, ["muH", "kSTO", "muSTO"], ["SS"], COD, t_d, our_d,
                   st_params(2.5, 3.0, 2.0), st_C0(X_H, 0.5 * COD), starts, laplace=True)
        sig = float(np.asarray(r.sigma_estimate)[0])
        ic, sd, icsd = r.ic_named[0], r.params_named_std, r.ic_named_std[0]
        return dict(muH=r.params_named["muH"], kSTO=r.params_named["kSTO"],
                    muSTO=r.params_named["muSTO"], S_S0=ic["SS"], fSS=ic["SS"] / COD,
                    sigma=sig, nll=r.loss, muH_std=sd["muH"], kSTO_std=sd["kSTO"],
                    muSTO_std=sd["muSTO"], S_S0_std=icsd["SS"])

    curves = load_curves()
    panels, per_dr, per_st = [], {}, {}
    for name, (muH_rep, X_H, S_S, S_H) in WW.items():
        t_np, our_np = curves[name]; COD = S_S + S_H
        t_d, our_d = jnp.asarray(t_np), jnp.asarray(our_np)
        fd = fit_dr(X_H, COD, t_d, our_d)
        fs = fit_st(X_H, COD, t_d, our_d)
        print(f"  {name}: reduced {fd}\n  {name}: storage {fs}", flush=True)
        tt = jnp.linspace(0.0, float(t_d[-1]), 240)
        dr_dense = np.asarray(dr_curve(fd["muH"], fd["khS"], X_H, fd["S_S0"], fd["S_H0"], tt))
        st_dense = np.asarray(st_curve(fs["muH"], fs["kSTO"], fs["muSTO"], X_H, fs["S_S0"], tt))
        dr_rmse, dr_r2 = metrics(dr_curve(fd["muH"], fd["khS"], X_H, fd["S_S0"], fd["S_H0"], t_d), our_np)
        st_rmse, st_r2 = metrics(st_curve(fs["muH"], fs["kSTO"], fs["muSTO"], X_H, fs["S_S0"], t_d), our_np)
        # forward-sim the reduced model at the source study's OWN reported parameters
        # (muH_rep, reported S_S/S_H); its graphical S_S overshoots the measured peak.
        g_dense = np.asarray(dr_curve(muH_rep, 1.0, X_H, S_S, S_H, tt))
        gupta_overshoot = float(g_dense.max() / our_np.max())
        per_dr[name] = dict(**fd, rmse=dr_rmse, r2=dr_r2, muH_rep=muH_rep,
                            err_pct=abs(fd["muH"]-muH_rep)/muH_rep*100.0,
                            gupta_overshoot=gupta_overshoot)
        per_st[name] = dict(**fs, rmse=st_rmse, r2=st_r2)
        panels.append(dict(name=name, muH_rep=muH_rep,
                           t=t_np.tolist(), our=our_np.tolist(), td=np.asarray(tt).tolist(),
                           dr=dr_dense.tolist(), st=st_dense.tolist(),
                           dr_muH=fd["muH"], dr_r2=dr_r2, st_r2=st_r2))

    # ---- muH profile likelihood on PROFILE_WW for both models --------------
    # muH is held at each grid value (left out of free_params) and the rest,
    # noise level included, is re-fitted by calibrate on the concentrated nll; the
    # profile is the nll rise above the fit, (N/2) log(SSE/SSE_min), read against
    # the 95% threshold PROF_DELTA = chi2_{1,0.95}/2.
    muH_rep, X_H, S_S, S_H = WW[PROFILE_WW]; COD = S_S + S_H
    t_np, our_np = curves[PROFILE_WW]; t_d, our_d = jnp.asarray(t_np), jnp.asarray(our_np)
    n_obs = int(t_np.size)

    def profile_dr(muH_fixed):
        return float(_calib(rx_dr, our_dr, ["khS"], ["SS", "SH"], COD, t_d, our_d,
                            dr_params(muH_fixed, 1.0), dr_C0(X_H, 0.5 * COD, 0.5 * COD),
                            [{"khS": k} for k in (0.3, 1.0, 3.0)]).loss)

    def profile_st(muH_fixed):
        return float(_calib(rx_st, our_st, ["kSTO", "muSTO"], ["SS"], COD, t_d, our_d,
                            st_params(muH_fixed, 3.0, 2.0), st_C0(X_H, 0.5 * COD),
                            [{"kSTO": k, "muSTO": 2.0} for k in (3.0, 10.0)]).loss)

    nll_dr = np.array([profile_dr(float(m)) for m in MU_GRID])
    nll_st = np.array([profile_st(float(m)) for m in MU_GRID])
    # rise above the free fit (or above the profile itself, if a grid point beat it)
    nll_dr_ref = min(nll_dr.min(), per_dr[PROFILE_WW]["nll"])
    nll_st_ref = min(nll_st.min(), per_st[PROFILE_WW]["nll"])
    prof_dr, prof_st = nll_dr - nll_dr_ref, nll_st - nll_st_ref
    # the same profiles as SSE, for the old "within 10% of the minimum SSE" reading
    # (inverting the concentrated nll = (N/2) log(SSE/N) + N/2)
    sse = lambda nll: n_obs * np.exp(2.0 * nll / n_obs - 1.0)
    sse_dr, sse_st = sse(nll_dr), sse(nll_st)

    # the free fit is the profile's own minimum (for the storage model it sits on
    # the muH bound, below the grid), so it is added to the grid for the intervals
    mg_dr, pf_dr = with_fit(MU_GRID, prof_dr, per_dr[PROFILE_WW]["muH"], per_dr[PROFILE_WW]["nll"] - nll_dr_ref)
    mg_st, pf_st = with_fit(MU_GRID, prof_st, per_st[PROFILE_WW]["muH"], per_st[PROFILE_WW]["nll"] - nll_st_ref)

    def ident_width(mg, prof, thresh=PROF_DELTA):
        """Width of the muH grid interval whose profile is within `thresh` of the minimum."""
        good = mg[prof <= thresh]
        return float(good.max() - good.min()) if good.size else float(mg[-1] - mg[0])

    def ident_ci(mg, prof, thresh=PROF_DELTA):
        """95% profile interval (linear interpolation of the crossings); None = open at the grid edge."""
        k = int(np.argmin(prof)); lo = hi = None
        for j in range(k, 0, -1):
            if prof[j - 1] > thresh:
                lo = float(np.interp(thresh, [prof[j], prof[j - 1]], [mg[j], mg[j - 1]])); break
        for j in range(k, len(prof) - 1):
            if prof[j + 1] > thresh:
                hi = float(np.interp(thresh, [prof[j], prof[j + 1]], [mg[j], mg[j + 1]])); break
        return [lo, hi]

    reps = np.array([WW[n][0] for n in WW]); fitv = np.array([per_dr[n]["muH"] for n in WW])
    return {
        "panels": panels, "per_dr": per_dr, "per_st": per_st,
        "profile_ww": PROFILE_WW, "mu_grid": MU_GRID.tolist(),
        "prof_dr": prof_dr.tolist(), "prof_st": prof_st.tolist(),
        "prof_delta": PROF_DELTA, "prof_n_obs": n_obs,
        "prof_sse_dr": sse_dr.tolist(), "prof_sse_st": sse_st.tolist(),
        "prof_fit_dr": per_dr[PROFILE_WW]["nll"] - nll_dr_ref,
        "prof_fit_st": per_st[PROFILE_WW]["nll"] - nll_st_ref,
        "ident_w_dr": ident_width(mg_dr, pf_dr), "ident_w_st": ident_width(mg_st, pf_st),
        "ident_ci_dr": ident_ci(mg_dr, pf_dr), "ident_ci_st": ident_ci(mg_st, pf_st),
        # the old reading, SSE within 10% of its minimum, on the grid only as before
        "ident_w_dr_sse110": ident_width(MU_GRID, sse_dr / sse_dr.min() - 1.0, 0.10),
        "ident_w_st_sse110": ident_width(MU_GRID, sse_st / sse_st.min() - 1.0, 0.10),
        "dr_err_best": float(min(per_dr[n]["err_pct"] for n in WW)),
        "dr_err_worst": float(max(per_dr[n]["err_pct"] for n in WW)),
        "dr_r2_lo": float(min(per_dr[n]["r2"] for n in WW)),
        "dr_r2_hi": float(max(per_dr[n]["r2"] for n in WW)),
        "st_r2_lo": float(min(per_st[n]["r2"] for n in WW)),
        "st_r2_hi": float(max(per_st[n]["r2"] for n in WW)),
        "dr_rmse_hi": float(max(per_dr[n]["rmse"] for n in WW)),
        "st_rmse_hi": float(max(per_st[n]["rmse"] for n in WW)),
        "mu_mean_fit": float(fitv.mean()), "mu_mean_rep": float(reps.mean()),
        "top_fit": list(WW)[int(np.argmax(fitv))], "top_rep": list(WW)[int(np.argmax(reps))],
        "gupta_overshoot_lo": float(min(per_dr[n]["gupta_overshoot"] for n in WW)),
        "gupta_overshoot_hi": float(max(per_dr[n]["gupta_overshoot"] for n in WW)),
        "n_ww": len(WW),
    }


def main():
    d = common.load_or_compute("respirometry", _compute)
    plt = common.setup_mpl()
    common.scale_fonts(plt, 11.0)
    fig, axs = plt.subplots(2, 2, figsize=(11.0, 7.6))
    axs = axs.ravel()
    colors = {"RWW": "C0", "PCE": "C3", "RBFE": "C2"}
    tags = {"RWW": "(a) RWW", "PCE": "(b) PCE", "RBFE": "(c) RBFE"}
    for ax, p in zip(axs[:3], d["panels"]):
        name = p["name"]; c = colors[name]
        t = np.asarray(p["t"]) * 24.0; our = np.asarray(p["our"]) / 24.0
        td = np.asarray(p["td"]) * 24.0
        ax.plot(t, our, "o", color="0.55", ms=2.6, alpha=0.7, label="data")
        ax.plot(td, np.asarray(p["st"]) / 24.0, "-", color=c, lw=1.7, label="storage")
        ax.plot(td, np.asarray(p["dr"]) / 24.0, "--", color=c, lw=1.4, label="reduced")
        ax.set_title(tags[name])
        ax.set_xlabel("time (h)")
        ax.text(0.96, 0.94, f"$R^2$ storage {p['st_r2']:.2f}\n$R^2$ reduced {p['dr_r2']:.2f}",
                transform=ax.transAxes, ha="right", va="top")
        ax.set_ylabel(r"OUR (mg O$_2$ L$^{-1}$ h$^{-1}$)")
        if name == "RWW":
            ax.legend(frameon=False, loc="center right")

    # ---- profile-likelihood panel -----------------------------------------
    # On a log muH axis (the fit's own coordinate) both 95% intervals are visible:
    # each is shaded in its model's colour, the best fits are dots, and the region
    # below the storage model's declared muH bound is greyed out.
    ax = axs[3]
    mg = np.asarray(d["mu_grid"])
    pw = d["profile_ww"]
    mdr, pdr = with_fit(mg, np.asarray(d["prof_dr"]), d["per_dr"][pw]["muH"], d["prof_fit_dr"])
    mst, pst = with_fit(mg, np.asarray(d["prof_st"]), d["per_st"][pw]["muH"], d["prof_fit_st"])
    st_lo = float(re.search(r"muH:\s*\{[^}]*bounds:\s*\[([0-9.]+)", open(ST_MODEL).read()).group(1))
    lo_dr, hi_dr = d["ident_ci_dr"]
    lo_st, hi_st = d["ident_ci_st"]
    lo_st = st_lo if lo_st is None else lo_st          # open at the grid edge = the bound
    ytop = PROF_YTOP_LOG
    ax.set_xscale("log"); ax.set_xlim(0.15, 7.0)
    ax.set_xticks([0.2, 0.5, 1, 2, 5]); ax.set_xticklabels(["0.2", "0.5", "1", "2", "5"])
    ax.minorticks_off()
    ax.set_ylim(-0.04 * ytop, ytop)
    ax.axvspan(0.01, st_lo, color="0.88", lw=0, zorder=0)
    ax.text(np.sqrt(0.15 * st_lo), 0.5 * ytop, r"storage $\mu_H$ bound", rotation=90,
            ha="center", va="center", color="0.4", fontsize="small")
    ax.axvspan(lo_dr, hi_dr, color="C0", alpha=0.15, lw=0)
    ax.axvspan(lo_st, hi_st, color="C1", alpha=0.25, lw=0)
    ax.plot(mdr, pdr, "--", color="C0", lw=1.7, label="reduced")
    ax.plot(mst, pst, "-", color="C1", lw=1.8, label="storage")
    ax.plot([d["per_dr"][pw]["muH"]], [d["prof_fit_dr"]], "o", color="C0", ms=5)
    ax.plot([d["per_st"][pw]["muH"]], [d["prof_fit_st"]], "o", color="C1", ms=5)
    ax.axhline(d["prof_delta"], color="0.45", lw=0.9, ls=":")
    ax.text(0.9, d["prof_delta"] + 0.02 * ytop, "95% threshold", color="0.35", ha="center",
            va="bottom", fontsize="small")
    rep = WW[pw][0]
    ax.axvline(rep, color="C3", lw=1.0, alpha=0.8)
    ax.text(rep * 0.95, 0.95 * ytop, "Gupta et al.", color="C3", va="top", ha="right",
            fontsize="small")
    ax.set_title(rf"(d) $\mu_H$ profile ({pw})")
    ax.set_xlabel(r"$\mu_H$ (d$^{-1}$)")
    ax.set_ylabel(r"$\Delta$ negative log-likelihood")
    ax.legend(frameon=False, loc="center", bbox_to_anchor=(0.5, 0.6), handlelength=1.6)

    common.save(fig, "fig_respirometry")
    for n in WW:
        a, b = d["per_dr"][n], d["per_st"][n]
        print(f"  {n:5} reduced muH={a['muH']:.2f}(Gupta {a['muH_rep']:.2f},{a['err_pct']:.0f}%) "
              f"R2={a['r2']:.2f} | storage muH={b['muH']:.2f}(rail) R2={b['r2']:.2f}")
    print(f"  muH 95% profile width: reduced {d['ident_w_dr']:.2f} vs storage {d['ident_w_st']:.2f} d^-1 "
          f"(intervals {d['ident_ci_dr']} vs {d['ident_ci_st']}; None = open at the grid edge)")
    print(f"  reduced err {d['dr_err_best']:.0f}-{d['dr_err_worst']:.0f}%, R2 {d['dr_r2_lo']:.2f}-{d['dr_r2_hi']:.2f}; "
          f"storage R2 {d['st_r2_lo']:.2f}-{d['st_r2_hi']:.2f}")


if __name__ == "__main__":
    main()
