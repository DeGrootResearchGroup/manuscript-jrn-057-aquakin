"""Figure: differentiability in action on a single activated-sludge reactor.

(a) A twin experiment: a true value of an ASM1 kinetic parameter is fixed, an
    observable species is recorded at a handful of time points, and the parameter
    is recovered by gradient-based optimization (L-BFGS-B with the exact
    reverse-mode gradient) from a perturbed start.
(b) Derivative-based local sensitivity (elasticities) of the observable to the
    kinetic parameters, from forward-mode sensitivities.

Cache key: "differentiability".
"""
from __future__ import annotations
import numpy as np
import common


def _compute():
    import jax, jax.numpy as jnp
    import scipy.optimize as so
    import aquakin

    asm1 = aquakin.load_model("asm1")
    si = asm1.species_index
    pi = asm1.param_index
    cond = asm1.default_conditions(1)
    rx = aquakin.BatchReactor(asm1, cond)

    # batch activated-sludge initial condition: substrate + heterotrophs (a closed
    # batch, so the initial oxygen is used within minutes)
    C0 = np.asarray(asm1.default_concentrations()).copy()
    overrides = {"SS": 60.0, "XB_H": 2000.0, "XB_A": 120.0, "XS": 80.0,
                 "SO": 8.0, "SNH": 25.0, "SND": 5.0, "XND": 10.0, "SALK": 7.0,
                 "SI": 30.0, "XI": 1000.0}
    for k, v in overrides.items():
        C0[si[k]] = v
    C0 = jnp.asarray(C0)

    p_true = asm1.default_parameters()
    obs_sp = "SS"
    obs_idx = si[obs_sp]
    par_name = "muH"
    par_idx = pi[par_name]
    mu_true = float(p_true[par_idx])

    # observations of SS at 10 time points across its drawdown, from the true
    # parameter. S_S falls from 60 to a quasi-steady ~8 mg/L within ~0.012 d (17 min)
    # and then stalls, so the samples span the transient, not the plateau after it.
    Tobs = 0.012
    t_obs = np.linspace(0.001, Tobs, 10)
    sol_true = rx.solve(C0, (0.0, Tobs), t_eval=jnp.asarray(t_obs), params=p_true)
    y_obs = np.asarray(sol_true.C)[:, obs_idx]

    # --- (a) gradient-based recovery of muH from a perturbed start ------------
    def predict(mu):
        params = p_true.at[par_idx].set(mu)
        s = rx.solve(C0, (0.0, Tobs), t_eval=jnp.asarray(t_obs), params=params)
        return s.C[:, obs_idx]

    def loss_log(u):                       # u = log(muH), keep positive
        mu = jnp.exp(u[0])
        r = predict(mu) - jnp.asarray(y_obs)
        return jnp.mean(r ** 2)

    loss_g = jax.jit(jax.value_and_grad(loss_log))
    path = []

    def fun(u):
        v, g = loss_g(jnp.asarray(u))
        path.append(float(np.exp(u[0])))
        return float(v), np.asarray(g, dtype=float)

    u0 = np.array([np.log(0.55 * mu_true)])    # perturbed start (-45%)
    res = so.minimize(fun, u0, jac=True, method="L-BFGS-B")
    mu_rec = float(np.exp(res.x[0]))
    rec_rel = abs(mu_rec - mu_true) / mu_true
    grad_finite = bool(np.all(np.isfinite(np.asarray(loss_g(res.x)[1]))))

    # --- (b) local derivative-based sensitivity (elasticities) ----------------
    sens = ["muH", "KS", "Y_H", "bH", "KOH", "etag"]
    fs = aquakin.forward_sensitivity(rx, C0, params=p_true, sens_params=sens,
                                     t_span=(0.0, Tobs), t_eval=jnp.asarray([Tobs]))
    S = np.asarray(fs.S)[-1, obs_idx, :]       # d(SS at T)/d(theta_j)
    y_end = float(np.asarray(sol_true.C)[-1, obs_idx])
    elasticity = {}
    for j, nm in enumerate(sens):
        th = float(p_true[pi[nm]])
        elasticity[nm] = float(th / y_end * S[j])
    top_name = max(elasticity, key=lambda k: abs(elasticity[k]))

    return {
        "obs_sp": obs_sp, "par_name": par_name, "mu_true": mu_true,
        "mu_rec": mu_rec, "rec_rel": rec_rel, "n_obs": len(t_obs),
        "n_steps": int(res.nit), "grad_finite": grad_finite,
        "path": path, "t_obs": t_obs, "y_obs": y_obs,
        "sens": sens, "elasticity": elasticity, "top_name": top_name,
        "top_val": elasticity[top_name],
    }


def main():
    d = common.load_or_compute("differentiability", _compute)
    plt = common.setup_mpl()
    common.scale_fonts(plt, 11.0, embed_frac=0.72)   # included at 0.72\linewidth
    fig, ax = plt.subplots(1, 2, figsize=(11.0, 4.3))

    path = np.asarray(d["path"])
    ax[0].plot(range(1, len(path) + 1), path, "o-", color="C0", ms=4)
    ax[0].axhline(d["mu_true"], color="0.4", ls="--", label="true value")
    ax[0].set(xlabel="optimizer evaluation", ylabel=fr"$\mu_H$ estimate",
              title=r"(a) recovery of $\mu_H$")
    ax[0].legend(frameon=False)

    el = d["elasticity"]
    names = sorted(el, key=lambda k: abs(el[k]))
    vals = [el[k] for k in names]
    ax[1].barh(range(len(names)), vals, color="C1")
    ax[1].set_yticks(range(len(names)))
    sym = {"muH": r"\mu_H", "etag": r"\eta_g", "bH": "b_H", "KS": "K_S", "KOH": "K_{OH}",
           "Y_H": "Y_H", "KNO": "K_{NO}", "kh": "k_h", "KX": "K_X"}
    ax[1].set_yticklabels([f"${sym.get(n, n)}$" for n in names])
    ax[1].axvline(0, color="0.5", lw=0.8)
    ax[1].set(xlabel=f"elasticity of ${d['obs_sp'][0]}_{{{d['obs_sp'][1:]}}}$ to parameter",
              title="(b) local sensitivity")

    common.save(fig, "fig_differentiability")
    print(f"  recover {d['par_name']}: true {d['mu_true']:.4f} rec {d['mu_rec']:.4f} "
          f"({d['rec_rel']*100:.3g}%) in {d['n_steps']} iters; grad finite={d['grad_finite']}")
    print(f"  top sensitivity: {d['top_name']} = {d['top_val']:.3f}")


if __name__ == "__main__":
    main()
