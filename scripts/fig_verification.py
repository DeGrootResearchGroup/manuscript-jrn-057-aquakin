"""Figure: verification of the engine.

(a) First-order decay A->B integrated by the stiff solver vs the closed form.
(b) The reverse-mode derivative d[A](T)/dk vs the analytic derivative.
(c) Conservation and gradient residuals on a log axis: total-mass drift on the
    decay network, and the maximum reverse-mode vs central finite-difference
    gradient disagreement on the assembled BSM1 plant.

Cache key: "verification".
"""
from __future__ import annotations
import os
import numpy as np
import common


# Explicit tolerances of the closed-form decay and stiff-chain checks (a, b, stiff).
DECAY_RTOL, DECAY_ATOL = 1e-6, 1e-9


def _compute():
    import jax, jax.numpy as jnp
    import aquakin
    from aquakin.plant.bsm import build_bsm1, bsm1_warm_start

    # --- (a,b) closed-form decay A -> B, rate k[A], [A](t)=A0 exp(-k t) -------
    #     Solution and gradient checks share one explicit tolerance (DECAY_RTOL,
    #     DECAY_ATOL) rather than each falling back to a different implicit
    #     default (the reactor's auto-scaled atol vs. esdirk_adjoint_solve's own
    #     fixed atol), so the two reported residuals are directly comparable.
    net = aquakin.load_model_from_file(os.path.join(common.DATA_DIR, "decay.yaml"))
    cond = net.default_conditions(1)
    rx = aquakin.BatchReactor(net, cond, rtol=DECAY_RTOL, atol=DECAY_ATOL)
    C0 = net.default_concentrations()            # [A0, 0]
    p = net.default_parameters()                 # [k]
    k = float(p[0]); A0 = float(C0[0])
    iA, iB = net.species_index["A"], net.species_index["B"]

    tgrid = np.linspace(0.0, 40.0, 81)
    sol = rx.solve(C0, (0.0, 40.0), t_eval=jnp.asarray(tgrid), params=p)
    Ct = np.asarray(sol.C)
    A_model = Ct[:, iA]
    A_exact = A0 * np.exp(-k * tgrid)
    nz = A_exact > 1e-9
    decay_max_rel = float(np.max(np.abs(A_model[nz] - A_exact[nz]) / A_exact[nz]))

    # total mass A+B conserved (A0) over the trajectory
    total = Ct[:, iA] + Ct[:, iB]
    cons_total_max = float(np.max(np.abs(total - A0) / A0))

    # gradient d[A](T)/dk by the cap-free discrete adjoint (the reverse-mode
    # gradient the calibration and plant use) vs analytic -T A0 exp(-k T)
    T = 25.0
    dfields = cond.fields
    def _decay_rhs(t, y, pm):
        return net.dCdt(y, pm, dfields, 0)

    def A_at_T(params):
        ys = aquakin.esdirk_adjoint_solve(_decay_rhs, C0, params, (0.0, T),
                                          jnp.asarray([T]),
                                          rtol=DECAY_RTOL, atol=DECAY_ATOL)
        return ys[-1, iA]
    g = float(jax.grad(A_at_T)(p)[0])
    g_exact = -T * A0 * np.exp(-k * T)
    grad_rel = float(abs(g - g_exact) / abs(g_exact))

    # --- (b) gradient & solution error vs solver tolerance -------------------
    #     The reverse-mode gradient is the exact gradient of the discrete solve,
    #     so it introduces no error of its own: its disagreement with the analytic
    #     derivative is set by the solver tolerance and falls with it, plateauing
    #     at float64 machine precision. Tightening the tolerance shows this.
    A_exact_T = A0 * np.exp(-k * T)
    tols = [10.0 ** -e for e in range(3, 16)]        # down to the float64 roundoff floor
    tol_grad, tol_sol = [], []
    for tol in tols:
        def A_at_T_t(params, _tol=tol):
            ys = aquakin.esdirk_adjoint_solve(_decay_rhs, C0, params, (0.0, T),
                                              jnp.asarray([T]), rtol=_tol, atol=_tol)
            return ys[-1, iA]

        A_num = float(A_at_T_t(p))
        gt = float(jax.grad(A_at_T_t)(p)[0])
        tol_sol.append(abs(A_num - A_exact_T) / abs(A_exact_T))
        tol_grad.append(abs(gt - g_exact) / abs(g_exact))
    grad_tol_floor = float(min(tol_grad))

    # --- stiff analytic check: AD gradient vs closed-form sensitivity ---------
    #     A stiff linear chain A->B->C (k2 >> k1) has a closed-form solution and
    #     hence a closed-form parameter sensitivity, so the reverse-mode gradient
    #     of the stiff solve can be verified against an analytic reference -- on a
    #     stiff problem, unlike the non-stiff decay above.
    import tempfile
    stiff_yaml = (
        "model: {name: stiff_chain, version: \"1.0\", description: x, references: []}\n"
        "species:\n  - {name: A, default_concentration: 1.0}\n"
        "  - {name: B, default_concentration: 0.0}\n"
        "  - {name: C, default_concentration: 0.0}\n"
        "conditions:\n  - {name: T, default: 293.15}\n"
        "reactions:\n"
        "  - {name: r1, rate: \"k1 * [A]\", parameters: {k1: {value: 1.0}},"
        " stoichiometry: {A: -1, B: 1}}\n"
        "  - {name: r2, rate: \"k2 * [B]\", parameters: {k2: {value: 1000.0}},"
        " stoichiometry: {B: -1, C: 1}}\n"
    )
    with tempfile.TemporaryDirectory() as _d:
        _p = os.path.join(_d, "stiff_chain.yaml")
        open(_p, "w").write(stiff_yaml)
        snet = aquakin.load_model_from_file(_p)
    sC0 = snet.default_concentrations(); siB = snet.species_index["B"]
    sk1, sk2 = snet.param_index["r1.k1"], snet.param_index["r2.k2"]
    sp0 = snet.default_parameters(); sT = 3.0
    stiff_ratio = float(sp0[sk2] / sp0[sk1])
    sfields = snet.default_conditions(1).fields

    def _srhs(t, y, pm):
        return snet.dCdt(y, pm, sfields, 0)

    def _B_closed(pm):                      # B(T) = k1/(k2-k1) (e^{-k1 T} - e^{-k2 T})
        k1, k2 = pm[sk1], pm[sk2]
        return k1 / (k2 - k1) * (jnp.exp(-k1 * sT) - jnp.exp(-k2 * sT))

    #     Gradient by the cap-free hand-written discrete adjoint -- the object the
    #     calibration and plant gradients rely on (esdirk_adjoint_solve, the plant's
    #     "stable_adjoint") -- verified against the closed-form sensitivity on a
    #     stiff, large-implicit-step solve, the regime it exists for.
    def _B_discrete(pm):
        ys = aquakin.esdirk_adjoint_solve(_srhs, sC0, pm, (0.0, sT), jnp.array([sT]),
                                          rtol=DECAY_RTOL, atol=DECAY_ATOL)
        return ys[-1, siB]

    gsd = np.asarray(jax.grad(_B_discrete)(sp0)); gc = np.asarray(jax.grad(_B_closed)(sp0))
    stiff_grad_rel = float(np.linalg.norm(gsd - gc) / np.linalg.norm(gc))
    # median implicit step relative to the fast (1/k2) timescale, to show the
    # discrete adjoint is verified in the large-step (uncapped) regime.
    import diffrax as _dfx
    _sol = _dfx.diffeqsolve(
        _dfx.ODETerm(lambda t, y, a: _srhs(t, y, a)), _dfx.Kvaerno5(),
        0.0, sT, 0.01, sC0, args=sp0,
        stepsize_controller=_dfx.PIDController(rtol=1e-6, atol=1e-9),
        saveat=_dfx.SaveAt(steps=True), max_steps=100_000)
    _ts = np.asarray(_sol.ts); _dts = np.diff(_ts[np.isfinite(_ts)])
    stiff_step_ratio = float(np.median(_dts) * float(sp0[sk2]))

    # --- (c) AD vs central finite difference on the assembled BSM1 plant -----
    #     The official BSM1 open-loop steady-state constant influent and 15C
    #     ASM1 parameters -- the same operating point Section~\ref{sec:results-steady}
    #     reproduces the published benchmark at -- rather than a separate,
    #     inconsistent condition: one plant configuration serves every check in
    #     this section, including the mass-balance check below (a time-varying
    #     load would also make the loss depend on exactly when it is read off,
    #     which swamps the true parameter sensitivity and was earlier mistaken
    #     for finite-difference solver noise).
    from aquakin.plant.influent import InfluentSeries
    asm1 = aquakin.load_model("asm1")
    plant = build_bsm1(model=asm1, use_takacs=True, settler_soluble_holdup=True,
                       settler_composition_mode="lumped_tss")
    plant.add_influent("feed", InfluentSeries.constant(
        asm1, Q=18446.0, SI=30.0, SS=69.5, XI=51.2, XS=202.32, XB_H=28.17,
        XB_A=0.0, XP=0.0, SO=0.0, SNO=0.0, SNH=31.56, SND=6.95, XND=10.59, SALK=7.0))
    y0 = bsm1_warm_start(plant)
    p0 = common.bsm1_params(plant)
    # differentiate a scalar effluent loss wrt several ASM1 kinetic/yield params
    free = ["muH", "muA", "bH", "Y_H", "etag"]
    pidx = [asm1.param_index[n] for n in free]
    # Long enough for the plant to actually reach steady state under the
    # official influent above: the scaled RHS residual |dy/dt| there is
    # ~1.1e-6 at Th=120 (reported as such; looser than the 1e-8 PTC tolerance)
    # in the paper (checked separately against Plant.steady_state()).
    Th = 120.0

    # Tight solver tolerance so the loss is accurate enough for finite
    # differencing: at the default rtol the loss carries ~1e-6 relative noise,
    # which swamps a small-step central difference. atol is pinned explicitly
    # too, not left at the auto-scaled default: leaving it unset made atol the
    # binding constraint once rtol was tightened, capping how well two exact
    # gradients (forward-mode vs. the discrete adjoint, below) could agree --
    # confirmed by a direct sweep of both tolerances together. rtol=atol=1e-10
    # is the tightest pair that still solves without breaking down (1e-12
    # already returns NaN on this stiff, recycle-coupled plant).
    def loss(theta):
        params = p0.at[jnp.asarray(pidx)].set(theta)
        s = plant.solve((0.0, Th), t_eval=jnp.asarray([Th]), params=params,
                        y0=y0, rtol=1e-10, atol=1e-10, forward_fast=False)
        eff = plant.effluent_stream(s)
        return jnp.sum(eff.C[-1] ** 2)

    theta0 = jnp.asarray([float(p0[i]) for i in pidx])
    g_ad = np.asarray(jax.grad(loss)(theta0))

    # Scaled RHS residual at the horizon above (common.scaled_residual, the
    # quantity PTC tests for convergence), quantifying "long enough to be at
    # steady state" the same way Section~\ref{sec:methods-steady} does.
    y_at_Th = plant.solve((0.0, Th), t_eval=jnp.asarray([Th]), params=p0,
                          y0=y0, rtol=1e-10, atol=1e-10, forward_fast=False).state[-1]
    adfd_horizon_resid = common.scaled_residual(
        plant._rhs(jnp.asarray(Th), y_at_Th, p0), y_at_Th)

    # Plant-wide mass balance at this same steady operating point: a short
    # continuation from the already-converged baseline state (not a separate
    # long run under a different condition), matching the 2-day window
    # Section~\ref{sec:results-steady} uses for the same computation.
    sol_mb = plant.solve((Th, Th + 2.0), t_eval=jnp.linspace(Th, Th + 2.0, 11),
                         params=p0, y0=y_at_Th, rtol=1e-10, atol=1e-10, forward_fast=False)
    mb = plant.mass_balance(sol_mb, params=p0)
    mb_cod = abs(float(mb["COD"].relative_imbalance))
    mb_n = abs(float(mb["N"].relative_imbalance))

    # Step-size sweep, all 5 parameters at each step. The reverse-mode gradient
    # above is a single adjoint pass regardless of parameter count, but each
    # finite-difference point costs two solves per parameter (10 solves per
    # step here). The sweep is what actually shows the FD gradient going flat
    # at small step (limited by solver noise, not by truncation) -- the
    # evidence for the "residual is the finite-difference step's, not the
    # gradient's" claim in the text, on this check itself rather than by
    # analogy to the batch/plant sweep of Figure~\ref{fig:adfd}.
    ADFD_H_GRID = [3e-1, 1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6]
    # The pinned atol above pushed each curve's own floor to a much smaller
    # step than 1e-3 (previously the default at the looser, auto-scaled atol).
    # 1e-4 is the largest step in the grid where every parameter is still on
    # the descending (truncation-limited) side of its own curve or right at
    # its minimum, not yet past it into the noise-dominated rise.
    ADFD_H_DEFAULT = 1e-4
    sweep_gfd, sweep_rel = [], []
    for h_rel in ADFD_H_GRID:
        g_fd_h = np.zeros_like(g_ad)
        for j in range(len(theta0)):
            h = h_rel * max(abs(float(theta0[j])), 1.0)
            tp = theta0.at[j].add(h); tm = theta0.at[j].add(-h)
            g_fd_h[j] = float((loss(tp) - loss(tm)) / (2 * h))
        sweep_gfd.append(g_fd_h)
        sweep_rel.append(np.abs(g_fd_h - g_ad) / np.abs(g_ad))
    sweep_rel = np.asarray(sweep_rel)   # (n_h, n_free)

    h0 = ADFD_H_GRID.index(ADFD_H_DEFAULT)
    g_fd = sweep_gfd[h0]
    # standard gradient-check metric: ||g_ad - g_fd|| / ||g_fd||
    adfd_max_rel = float(np.linalg.norm(g_ad - g_fd) / max(np.linalg.norm(g_fd), 1e-30))
    # per-parameter relative difference at the reported step, for Figure
    # fig:plantfd -- the aggregate (2-norm) metric above is not their max: it
    # is dominated by the large-magnitude, small-relative-error parameters
    # (Y_H, bH), not the worst one.
    adfd_comp_rel = sweep_rel[h0]
    adfd_worst_i = int(np.argmax(adfd_comp_rel))
    adfd_worst_name = free[adfd_worst_i]
    adfd_worst_rel = float(adfd_comp_rel[adfd_worst_i])

    # --- plant gradient verified independently: forward vs reverse AD --------
    #     Forward-mode (variational) and reverse-mode (adjoint) AD are separate
    #     algorithms for the same exact derivative, so their agreement checks the
    #     plant adjoint without any finite-difference reference. Same loss, same
    #     operating point, same five parameters as the finite-difference check
    #     above (reusing g_ad for the reverse-mode side), so this check and that
    #     one are directly comparable rather than being different gradients that
    #     happen to both be small.
    def loss_fwd(theta):
        params = p0.at[jnp.asarray(pidx)].set(theta)
        s = plant.solve((0.0, Th), t_eval=jnp.asarray([Th]), params=params,
                        y0=y0, rtol=1e-10, atol=1e-10, forward_fast=False,
                        diff=aquakin.DifferentiationConfig(mode="forward",
                                                           method="through_solve"))
        eff = plant.effluent_stream(s)
        return jnp.sum(eff.C[-1] ** 2)

    g_fwd = np.asarray(jax.jacfwd(loss_fwd)(theta0))
    plant_fwdrev_rel = float(np.linalg.norm(g_fwd - g_ad) / np.linalg.norm(g_ad))
    # how much tighter the no-finite-difference check is than the FD comparison
    # above, both now the same gradient at the same operating point -- quoted
    # in the text instead of a hand-typed "orders of magnitude" claim.
    adfd_fwdrev_ratio = adfd_max_rel / plant_fwdrev_rel
    # per-parameter, for Figure fig:plantfd: how far below each FD step-size
    # curve's own floor the no-finite-difference (forward-vs-reverse) agreement
    # sits, shown as a reference line on each panel.
    adfd_fwdrev_comp = np.abs(g_fwd - g_ad) / np.abs(g_ad)

    return {
        "tgrid": tgrid, "A_model": A_model, "A_exact": A_exact,
        "decay_max_rel": decay_max_rel, "cons_total_max": cons_total_max,
        "grad_ad": g, "grad_exact": g_exact, "grad_rel": grad_rel,
        "tols": tols, "tol_grad": tol_grad, "tol_sol": tol_sol,
        "grad_tol_floor": grad_tol_floor,
        "stiff_grad_rel": stiff_grad_rel, "stiff_ratio": stiff_ratio,
        "stiff_step_ratio": stiff_step_ratio,
        "plant_fwdrev_rel": plant_fwdrev_rel, "adfd_fwdrev_ratio": adfd_fwdrev_ratio,
        "adfd_fwdrev_comp": adfd_fwdrev_comp,
        "adfd_free": free, "g_ad": g_ad, "g_fd": g_fd,
        "adfd_max_rel": adfd_max_rel, "adfd_n": len(free),
        "adfd_worst_name": adfd_worst_name, "adfd_worst_rel": adfd_worst_rel,
        "adfd_h_grid": ADFD_H_GRID, "adfd_sweep_rel": sweep_rel,
        "adfd_horizon": Th, "adfd_horizon_resid": adfd_horizon_resid,
        "adfd_h_default": ADFD_H_DEFAULT,
        "mb_cod": mb_cod, "mb_n": mb_n,
    }


# SI check: on single stiff units, where differentiating *through* the solve can stay
# finite, the discrete-adjoint gradient is compared with the through-solve gradient
# at a range of step-size caps, down to none. Kept in its own cache ("si_adjoint")
# so it recomputes without the rest of the verification.
SI_ADJ_RTOL, SI_ADJ_ATOL = 1e-10, 1e-12          # tight, so the two discretizations agree
SI_ADJ_BATCH_T = 0.1                              # d, an aerobic ASM1 batch assay
SI_ADJ_BATCH_CAPS = [1e-4, 1e-3, 1e-2, None]      # d; None = no step-size limitation
SI_ADJ_DIG_T = 5.0                                # d, the standalone ADM1 digester
SI_ADJ_DIG_CAPS = [1e-2, 1e-1, 1.0, None]


def _rel(g, ref):
    g, ref = np.asarray(g, float), np.asarray(ref, float)
    if not np.all(np.isfinite(g)):
        return None                               # non-finite: reported as such
    return float(np.linalg.norm(g - ref) / np.linalg.norm(ref))


def _compute_si_adjoint():
    import jax, jax.numpy as jnp
    import aquakin
    from aquakin.integrate.discrete_adjoint import esdirk_adjoint_solve
    from aquakin.integrate._common import DifferentiationConfig
    from aquakin.plant import Plant, InfluentSeries
    from aquakin.plant.digester import ADM1DigesterUnit
    import fig_steady_solver as F

    out = {"rtol": SI_ADJ_RTOL, "atol": SI_ADJ_ATOL}

    # --- ASM1 aerobic batch: esdirk discrete adjoint vs reactor.solve's own adjoint
    asm1 = aquakin.load_model("asm1"); si = asm1.species_index
    cond = asm1.default_conditions(1); fields = cond.fields
    C = np.asarray(asm1.default_concentrations()).copy()
    for k, v in {"SS": 60.0, "SO": 8.0, "XB_H": 40.0, "SNH": 20.0, "SNO": 0.1}.items():
        C[si[k]] = v
    C0 = jnp.asarray(C); p0 = asm1.default_parameters()
    tobs = jnp.linspace(0.1 * SI_ADJ_BATCH_T, SI_ADJ_BATCH_T, 10)
    obs = jnp.asarray([si["SS"], si["SO"], si["SNO"]])
    rhs = lambda t, y, p: asm1.dCdt(y, p, fields, 0)

    def batch_loss_da(p):
        Y = esdirk_adjoint_solve(rhs, C0, p, (0.0, SI_ADJ_BATCH_T), tobs, order=5,
                                 rtol=SI_ADJ_RTOL, atol=SI_ADJ_ATOL, max_steps=200_000)
        return jnp.sum(Y[:, obs] ** 2)

    g_ref = np.asarray(jax.grad(batch_loss_da)(p0))
    rows = []
    for cap in SI_ADJ_BATCH_CAPS:
        rx = aquakin.BatchReactor(asm1, cond, rtol=SI_ADJ_RTOL, atol=SI_ADJ_ATOL,
                                  integrator=aquakin.IntegratorConfig(order=5, dtmax=cap,
                                                                      max_steps=1_000_000))
        def loss_ts(p, _rx=rx):
            return jnp.sum(_rx.solve(C0, (0.0, SI_ADJ_BATCH_T), t_eval=tobs, params=p).C[:, obs] ** 2)
        g = np.asarray(jax.grad(loss_ts)(p0))
        rows.append({"dtmax": cap, "rel": _rel(g, g_ref), "finite": bool(np.all(np.isfinite(g)))})
        print(f"    batch dtmax={cap}: rel {rows[-1]['rel']}  finite {rows[-1]['finite']}")
    out["batch"] = rows

    # --- standalone ADM1 digester: plant stable adjoint vs through-solve
    adm1 = aquakin.load_model("adm1"); sa = adm1.species_index
    plant = Plant("digester")
    plant.add_unit(ADM1DigesterUnit(name="digester", model=adm1, volume=F.V_LIQ,
                                    conditions={"T": 308.15}, input_port_names=["feed"]))
    plant.add_influent("feed", InfluentSeries.constant(adm1, Q=F.Q_FEED, **F.INFLUENT),
                       to="digester.feed")
    Cd = np.asarray(adm1.default_concentrations()).copy()
    for k, v in F.REF_SS.items():
        Cd[sa[k]] = v
    y0 = jnp.asarray(plant.initial_state(overrides={"digester": jnp.asarray(Cd)}))
    pp = jnp.asarray(plant._coerce_params(adm1.default_parameters()))
    scale = jnp.maximum(jnp.abs(y0), 1e-3)

    def dig_loss(p, method, cap):
        s = plant.solve((0.0, SI_ADJ_DIG_T), t_eval=jnp.asarray([SI_ADJ_DIG_T]), params=p,
                        y0=y0, rtol=SI_ADJ_RTOL * 100, diff=DifferentiationConfig(method=method),
                        integrator=aquakin.IntegratorConfig(dtmax=cap, max_steps=1_000_000))
        return jnp.sum((s.state[-1] / scale) ** 2)

    gd_ref = np.asarray(jax.grad(lambda p: dig_loss(p, "stable", None))(pp))
    rows = []
    for cap in SI_ADJ_DIG_CAPS:
        g = np.asarray(jax.grad(lambda p, _c=cap: dig_loss(p, "through_solve", _c))(pp))
        rows.append({"dtmax": cap, "rel": _rel(g, gd_ref), "finite": bool(np.all(np.isfinite(g)))})
        print(f"    digester dtmax={cap}: rel {rows[-1]['rel']}  finite {rows[-1]['finite']}")
    out["digester"] = rows
    out["dig_rtol"] = SI_ADJ_RTOL * 100
    return out


def main():
    d = common.load_or_compute("verification", _compute)
    common.load_or_compute("si_adjoint", _compute_si_adjoint)
    plt = common.setup_mpl()
    common.scale_fonts(plt, 9.5)
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 4.6))
    fig.get_layout_engine().set(wspace=0.16)

    tg = np.asarray(d["tgrid"])
    ax[0].plot(tg, np.asarray(d["A_exact"]), color="0.4", lw=2.0, label="analytical")
    ax[0].plot(tg, np.asarray(d["A_model"]), "o", ms=3, color="C3",
               markevery=4, label="aquakin")
    ax[0].set(xlabel="time (s)", ylabel="[A] (mol/L)", title="(a) analytical vs. aquakin")
    ax[0].legend(frameon=False)

    tols = np.asarray(d["tols"])
    ax[1].loglog(tols, np.maximum(d["tol_grad"], 1e-18), "o-", color="C4",
                 label="gradient")
    ax[1].loglog(tols, np.maximum(d["tol_sol"], 1e-18), "s-", color="C0",
                 label="solution")
    ax[1].loglog(tols, tols, ":", color="0.35", lw=1.4, label="error = tolerance")
    ax[1].axhline(2.2e-16, color="0.45", ls="--", lw=1.2)
    ax[1].text(0.5, 3e-16, "float64 $\\epsilon$", transform=ax[1].get_yaxis_transform(),
               color="0.35", va="bottom", ha="center")
    ax[1].invert_xaxis()          # tighter tolerance to the right
    ax[1].set(xlabel="solver tolerance", ylabel="relative error",
              title="(b) errors vs. tolerance")
    ax[1].legend(frameon=False, loc="upper right")

    common.save(fig, "fig_verification")
    print(f"  decay max rel err   {d['decay_max_rel']:.2e}")
    print(f"  decay total-mass    {d['cons_total_max']:.2e}")
    print(f"  grad AD vs exact    {d['grad_rel']:.2e} (AD {d['grad_ad']:.6f} vs {d['grad_exact']:.6f})")
    print(f"  stiff AD vs exact   {d['stiff_grad_rel']:.2e}")
    print(f"  plant fwd vs rev    {d['plant_fwdrev_rel']:.2e}")
    print(f"  plant AD vs FD      {d['adfd_max_rel']:.2e} over {d['adfd_n']} params")


if __name__ == "__main__":
    main()
