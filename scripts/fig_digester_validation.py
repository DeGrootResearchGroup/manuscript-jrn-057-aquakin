"""Standalone ADM1 digester verification for JRN-057.

Runs the anaerobic digester (ADM1 in its BSM2 form) as a *single reacting unit*
fed the published BSM2 digester feed, without the coupled BSM2 plant, and checks
two things the paper needs on the same footing as the BSM1 reproduction:

  (a) numerical equivalence of the digester steady state (found as the algebraic
      root of dy/dt = 0 by pseudo-transient continuation, cross-checked against
      forward integration to the same attractor) against the published BSM2
      open-loop digester steady state; and

  (b) correctness of the state-derived charge-balance pH, by comparing the pH the
      engine solves at the converged composition against an *independent*
      electroneutrality solve coded from scratch here with the published ADM1
      acid-base constants (carbonate, ammonia, and the four VFA systems, plus the
      strong ions), so the check tests the speciation physics, not the engine's
      own solver.

The full coupled-plant BSM2 validation lives in the companion (JRN-056); this
file is deliberately the standalone-unit check only. Cached to
``results/digester_validation.json``; consumed by ``make_paper_values.py`` and
``make_validation_tables.py``.
"""

import numpy as np
import common
from fig_steady_solver import INFLUENT, REF_SS, Q_FEED, V_LIQ, FWD_HORIZON_D, PTC_TOL

T_DIG = 308.15  # digester temperature (K), 35 C

# Published ADM1 acid-base constants at 35 C (Batstone et al. 2002; the BSM2
# van't Hoff values). Used ONLY by the independent pH solve below, so its
# agreement with the engine's pH is a physics check, not a self-consistency one.
PKA = {"co2": 6.3065, "IN": 8.955, "ac": 4.76, "pro": 4.88, "bu": 4.82, "va": 4.86}
PKW_35 = 13.6805  # water at 35 C (van't Hoff, dH_w = 55900 J/mol)
# VFA molar masses (gCOD/mol) that convert the COD-based states to mol/L.
MW = {"ac": 64.0, "pro": 112.0, "bu": 160.0, "va": 208.0}


def _independent_pH(state, si):
    """Solve electroneutrality for [H+] from scratch, published ADM1 chemistry.

    Charge balance (eq/L): [H+] + [NH4+] + S_cat
        - [OH-] - [HCO3-] - [Ac-] - [Pro-] - [Bu-] - [Va-] - S_an = 0,
    with each acid/base fraction the standard monoprotic form at the given [H+].
    Bracketed bisection over pH in [1, 13]; no dependence on the engine's solver.
    """
    Ka = {k: 10.0 ** (-v) for k, v in PKA.items()}
    Kw = 10.0 ** (-PKW_35)
    S_IC = float(state[si["S_IC"]])          # kmolC/m3 = mol/L
    S_IN = float(state[si["S_IN"]])          # kmolN/m3 = mol/L
    S_cat = float(state[si["S_cat"]]) if "S_cat" in si else 0.0
    S_an = float(state[si["S_an"]]) if "S_an" in si else 0.0
    vfa = {k: float(state[si["S_" + k]]) / MW[k] for k in MW}  # gCOD/L -> mol/L

    def residual(pH):
        h = 10.0 ** (-pH)
        oh = Kw / h
        nh4 = S_IN * h / (h + Ka["IN"])                 # protonated (cation)
        hco3 = S_IC * Ka["co2"] / (h + Ka["co2"])       # deprotonated (anion)
        acid_anion = sum(vfa[k] * Ka[k] / (h + Ka[k]) for k in MW)
        cations = h + nh4 + S_cat
        anions = oh + hco3 + acid_anion + S_an
        return cations - anions

    lo, hi = 1.0, 13.0
    flo = residual(lo)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        fm = residual(mid)
        if flo * fm <= 0.0:
            hi = mid
        else:
            lo, flo = mid, fm
    return 0.5 * (lo + hi)


def _compute():
    import jax.numpy as jnp
    import aquakin
    from aquakin.plant import Plant, InfluentSeries
    from aquakin.plant.digester import ADM1DigesterUnit

    adm1 = aquakin.load_model("adm1")
    si = adm1.species_index

    plant = Plant("digester")
    plant.add_unit(ADM1DigesterUnit(name="digester", model=adm1, volume=V_LIQ,
                                    conditions={"T": T_DIG}, input_port_names=["feed"]))
    plant.add_influent("feed", InfluentSeries.constant(adm1, Q=Q_FEED, **INFLUENT),
                       to="digester.feed")
    plant._build_state_layout(); plant._build_parameter_layout()
    params = adm1.default_parameters()
    params_full = plant._coerce_params(params)

    C0 = np.asarray(adm1.default_concentrations()).copy()
    for k, v in REF_SS.items():
        C0[si[k]] = v
    y0 = plant.initial_state(overrides={"digester": jnp.asarray(C0)})

    # (a) algebraic steady state (PTC root of dy/dt = 0)
    # Converged on the paper's scaled residual (Eq. scaled-residual, floor eps).
    ss = plant.steady_state(params=params, y0=y0, max_iter=400, tol=PTC_TOL,
                            scale_floor=common.RESID_FLOOR)
    model = np.asarray(plant.states_by_unit(ss.state)["digester"])
    # cross-check: forward integration to the dynamic attractor, the same fixed-horizon
    # run as the Fig. 8 baseline, long enough that its scaled residual is below the
    # PTC tolerance (asserted), so both end states are converged on the same rho.
    fwd_state = plant.solve((0.0, FWD_HORIZON_D), t_eval=np.array([FWD_HORIZON_D]),
                            params=params, y0=y0, forward_fast=True, integrator=common.FAST_INTEGRATOR).final_state
    forward = np.asarray(plant.states_by_unit(fwd_state)["digester"])

    def _rho(y):
        y = jnp.asarray(y)
        return common.scaled_residual(plant.derivative(y, params=params), y)
    ptc_resid, fwd_resid = _rho(ss.state), _rho(fwd_state)
    assert fwd_resid <= PTC_TOL, f"forward run not converged: rho {fwd_resid:.1e}"
    # PTC-vs-forward agreement, defined as in fig_steady_solver: the largest relative
    # difference over the states above 1e-3 (near-zero states are excluded).
    y_p, y_f = np.asarray(ss.state), np.asarray(fwd_state)
    mask = np.abs(y_f) > 1e-3
    ptc_vs_fwd = float(np.max(np.abs(y_p[mask] - y_f[mask]) / np.abs(y_f[mask])))

    names, mvals, fvals, rvals = [], [], [], []
    for k, rv in REF_SS.items():
        j = si[k]
        names.append(k)
        mvals.append(float(model[j]))
        fvals.append(float(forward[j]))
        rvals.append(float(rv))
    model_a = np.asarray(mvals); ref_a = np.asarray(rvals); fwd_a = np.asarray(fvals)
    rel = np.abs(model_a - ref_a) / np.abs(ref_a)          # vs published reference
    w = int(np.argmax(rel))
    ch4_i = names.index("S_gas_ch4")

    # (b) state-derived pH vs the independent electroneutrality solve
    pH_engine = float(plant._unit_operating_pH("digester",
                                               plant.states_by_unit(ss.state), params_full))
    pH_indep = _independent_pH(model, si)

    return {
        "names": names, "model": mvals, "forward": fvals, "ref": rvals,
        "rel_pct": (rel * 100.0).tolist(),
        "worst_pct": float(rel[w] * 100.0), "worst_name": names[w],
        "methane_pct": float(rel[ch4_i] * 100.0),
        "n_resolvable": len(names),
        "ptc_vs_fwd": ptc_vs_fwd, "fwd_horizon_d": FWD_HORIZON_D,
        "ptc_resid": ptc_resid, "fwd_resid": fwd_resid,
        "pH_engine": pH_engine, "pH_independent": pH_indep,
        "pH_abs_err": float(abs(pH_engine - pH_indep)),
    }


def main():
    d = common.load_or_compute("digester_validation", _compute)
    print("Standalone ADM1 digester verification")
    print(f"  states compared:        {d['n_resolvable']}")
    print(f"  worst rel. diff vs ref: {d['worst_pct']:.3f}%  ({d['worst_name']})")
    print(f"  methane (gas) vs ref:   {d['methane_pct']:.3f}%")
    print(f"  PTC vs forward:         {d['ptc_vs_fwd']:.2e} (forward {d['fwd_horizon_d']:.0f} d, rho {d['fwd_resid']:.1e})")
    print(f"  pH (engine):            {d['pH_engine']:.4f}")
    print(f"  pH (independent):       {d['pH_independent']:.4f}")
    print(f"  |pH engine - indep|:    {d['pH_abs_err']:.4f}")


if __name__ == "__main__":
    main()
