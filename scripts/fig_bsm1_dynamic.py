"""Figure: open-loop BSM1 dry-weather dynamic validation.

Protocol: bring the open-loop plant to steady state on the constant influent,
then drive it with the dry-weather influent for 14 days, and average the effluent
over days 7-14. Compares the averaged effluent (ammonia, total N, total COD, TSS)
and the effluent quality / operational cost indices to the published benchmark
reference (data/bsm1_openloop_dyn_reference.json).

Cache key: "bsm1_dynamic".
"""
from __future__ import annotations
import json
import os
import numpy as np
import common

_KMAP = {"XBH": "XB_H", "XBA": "XB_A"}
_COD = ("SI", "SS", "XI", "XS", "XB_H", "XB_A", "XP")
_PART = ("XI", "XS", "XB_H", "XB_A", "XP")


def _ref():
    with open(os.path.join(common.DATA_DIR, "bsm1_openloop_dyn_reference.json")) as f:
        return json.load(f)


def _compute():
    import jax.numpy as jnp
    import aquakin
    from aquakin.plant.bsm import build_bsm1, bsm1_warm_start, evaluate_bsm1
    from aquakin.plant import InfluentSeries

    ssref = json.load(open(os.path.join(common.DATA_DIR,
                                         "bsm1_openloop_ss_reference.json")))
    dref = _ref()
    asm1 = aquakin.load_model("asm1")
    si = asm1.species_index
    pi = asm1.param_index
    i_XB = float(asm1.default_parameters()[pi["i_XB"]])
    i_XP = float(asm1.default_parameters()[pi["i_XP"]])

    # 1) steady state under the constant influent
    ci = ssref["constant_influent"]
    species = {_KMAP.get(k, k): float(v) for k, v in ci.items() if k != "TSS"}
    Qc = float(ssref["flows_m3d"]["influent_wwtp"])
    plant_c = build_bsm1(model=asm1, use_takacs=True, settler_soluble_holdup=True, settler_composition_mode="lumped_tss")
    plant_c.add_influent("feed", InfluentSeries.constant(asm1, Q=Qc, **species))
    params_c = common.bsm1_params(plant_c)   # official BSM1 15 C parameters
    y0 = bsm1_warm_start(plant_c)
    yss = np.asarray(plant_c.solve((0.0, 150.0), t_eval=np.array([150.0]),
                                   params=params_c, y0=y0,
                                   forward_fast=True, integrator=common.FAST_INTEGRATOR).final_state)

    # 2) 14-day dry-weather dynamic from the steady state; save days 7-14
    plant = build_bsm1(model=asm1, use_takacs=True, settler_soluble_holdup=True, settler_composition_mode="lumped_tss")
    plant.add_influent("feed", common.official_dry_influent(asm1))
    params = common.bsm1_params(plant)
    # The benchmark warms the plant with 14 d of the dry-weather file before the
    # evaluated run, so the nitrifier population is dry-conditioned. Chain two
    # 14 d dry runs (warm-up, then the evaluated run from that state) and score
    # days 7-14 at the benchmark 15-min output resolution.
    y_warm = np.asarray(plant.solve((0.0, 14.0), t_eval=np.array([14.0]),
                                    params=params, y0=jnp.asarray(yss),
                                    forward_fast=True, integrator=common.FAST_INTEGRATOR).final_state)
    teval = np.linspace(7.0, 14.0, 7 * 96 + 1)
    sol = plant.solve((0.0, 14.0), t_eval=teval, params=params,
                      y0=jnp.asarray(y_warm), forward_fast=True, integrator=common.FAST_INTEGRATOR)

    eff = plant.stream(sol, "clarifier.overflow")
    Ce = np.asarray(eff.C)                    # (n_t, n_species)
    Qe = np.asarray(eff.Q).reshape(-1)
    Cbar = Ce.mean(axis=0)                    # time-averaged effluent composition
    Cbar_flow = (Ce * Qe[:, None]).sum(axis=0) / Qe.sum()   # flow-weighted (load)
    snh_time = float(Cbar[si["SNH"]]); snh_flow = float(Cbar_flow[si["SNH"]])

    def g(s):
        return float(Cbar_flow[si[s]])   # flow-weighted (load) avg, per perf_plant.m
    cod = sum(g(s) for s in _COD)
    tss = 0.75 * sum(g(s) for s in _PART)
    tkn = (g("SNH") + g("SND") + g("XND")
           + i_XB * (g("XB_H") + g("XB_A")) + i_XP * (g("XI") + g("XP")))
    tn = tkn + g("SNO")
    snh = g("SNH")

    # Settler soluble-holdup ablation: the same dynamic run with the pass-through
    # (no-holdup) settler, to quantify what the benchmark's soluble-carrying settler
    # buys dynamically (it damps the effluent-ammonia swing via the clarifier volume).
    # Same protocol as the benchmark run: steady state on the constant influent,
    # then the dry-weather warm-up and the evaluated run.
    plant_nh_c = build_bsm1(model=asm1, use_takacs=True, settler_soluble_holdup=False,
                            settler_composition_mode="lumped_tss")
    plant_nh_c.add_influent("feed", InfluentSeries.constant(asm1, Q=Qc, **species))
    yss_nh = np.asarray(plant_nh_c.solve((0.0, 150.0), t_eval=np.array([150.0]),
                                         params=common.bsm1_params(plant_nh_c),
                                         y0=bsm1_warm_start(plant_nh_c),
                                         forward_fast=True, integrator=common.FAST_INTEGRATOR).final_state)
    plant_nh = build_bsm1(model=asm1, use_takacs=True, settler_soluble_holdup=False,
                          settler_composition_mode="lumped_tss")
    plant_nh.add_influent("feed", common.official_dry_influent(asm1))
    params_nh = common.bsm1_params(plant_nh)
    yw_nh = np.asarray(plant_nh.solve((0.0, 14.0), t_eval=np.array([14.0]),
                                      params=params_nh, y0=jnp.asarray(yss_nh),
                                      forward_fast=True, integrator=common.FAST_INTEGRATOR).final_state)
    sol_nh = plant_nh.solve((0.0, 14.0), t_eval=teval, params=params_nh,
                            y0=jnp.asarray(yw_nh), forward_fast=True, integrator=common.FAST_INTEGRATOR)
    eff_nh = plant_nh.stream(sol_nh, "clarifier.overflow")
    Ce_nh = np.asarray(eff_nh.C); Qe_nh = np.asarray(eff_nh.Q).reshape(-1)
    snh_nh = float((Ce_nh[:, si["SNH"]] * Qe_nh).sum() / Qe_nh.sum())   # flow-weighted avg
    snh_series = Ce[:, si["SNH"]]; snh_series_nh = Ce_nh[:, si["SNH"]]
    # Settler feed (the last reactor's outflow to the clarifier) for both settlers, to
    # separate what the clarifier volume does to the signal from what the reactors do.
    feed = plant.stream(sol, "tank5_split.to_clarifier")
    feed_nh = plant_nh.stream(sol_nh, "tank5_split.to_clarifier")
    Cf = np.asarray(feed.C); Qf = np.asarray(feed.Q).reshape(-1)
    Cf_nh = np.asarray(feed_nh.C); Qf_nh = np.asarray(feed_nh.Q).reshape(-1)

    def _means(c, q):
        """(time average, flow-weighted average) of one series."""
        return float(c.mean()), float((c * q).sum() / q.sum())
    snh_amp = float(snh_series.max() - snh_series.min())            # with holdup
    snh_amp_nh = float(snh_series_nh.max() - snh_series_nh.min())   # without holdup
    holdup = {
        "snh_holdup": snh, "snh_noholdup": snh_nh,
        "snh_avg_reldiff_pct": abs(snh_nh - snh) / abs(snh) * 100,
        "snh_amp_holdup": snh_amp, "snh_amp_noholdup": snh_amp_nh,
        "amp_ratio": snh_amp_nh / snh_amp,
        "eff_SNH_noholdup": snh_series_nh.tolist(),
        "feed_SNH": Cf[:, si["SNH"]].tolist(), "feed_SNH_noholdup": Cf_nh[:, si["SNH"]].tolist(),
        "eff_Q": Qe.tolist(), "eff_Q_noholdup": Qe_nh.tolist(),
        "feed_Q": Qf.tolist(), "feed_Q_noholdup": Qf_nh.tolist(),
        # (time mean, flow-weighted mean) of each series, for the mechanism
        "means": {
            "feed_holdup": _means(Cf[:, si["SNH"]], Qf),
            "eff_holdup": _means(snh_series, Qe),
            "feed_noholdup": _means(Cf_nh[:, si["SNH"]], Qf_nh),
            "eff_noholdup": _means(snh_series_nh, Qe_nh),
        },
    }

    refc = dref["effluent_avg_conc"]
    comp = {"SNH": (snh, refc["SNH"]), "total N": (tn, refc["total_N"]),
            "total COD": (cod, refc["total_COD"]), "TSS": (tss, refc["TSS"])}
    rels = {k: abs(m - r) / abs(r) for k, (m, r) in comp.items()}
    worst_name = max(rels, key=rels.get)

    # 3) EQI / OCI via the benchmark evaluation
    ev = evaluate_bsm1(plant, sol, params)
    eqi_m = float(ev.eqi); oci_m = float(ev.oci)
    qi = dref["quality_indices"]; oc = dref["operational_cost_index"]
    # evaluate_bsm1 implements the updated benchmark indices (the OCI includes the
    # mixing energy and 5 x sludge production; see its docs), so compare against the
    # reference's updated values. The convention is fixed by the implementation, not
    # chosen by closeness to the result.
    eqi_conv = oci_conv = "updated"
    eqi_ref = qi["EQI_updated"]
    oci_ref = oc["OCI_updated"]

    return {
        "t": teval, "eff_SNH": Ce[:, si["SNH"]], "eff_SNO": Ce[:, si["SNO"]],
        "comp": {k: list(v) for k, v in comp.items()},
        "rels_pct": {k: v * 100 for k, v in rels.items()},
        "worst_name": worst_name, "worst_pct": rels[worst_name] * 100,
        "snh": snh, "snh_ref": refc["SNH"], "tn": tn, "tn_ref": refc["total_N"],
        "cod": cod, "cod_ref": refc["total_COD"], "tss": tss, "tss_ref": refc["TSS"],
        "eqi": eqi_m, "eqi_ref": eqi_ref, "eqi_conv": eqi_conv,
        "eqi_pct": abs(eqi_m - eqi_ref) / eqi_ref * 100,
        "oci": oci_m, "oci_ref": oci_ref, "oci_conv": oci_conv,
        "oci_pct": abs(oci_m - oci_ref) / oci_ref * 100,
        "influent_source": "official BSM1 dry-weather",
        "holdup": holdup,
    }


def main():
    d = common.load_or_compute("bsm1_dynamic", _compute)
    plt = common.setup_mpl()
    common.scale_fonts(plt, 9.0)
    fig, ax = plt.subplots(3, 1, figsize=(9.0, 8.6), sharex=True,
                           height_ratios=[0.75, 1.15, 1.0])
    t = np.asarray(d["t"]); hu = d["holdup"]; mn = hu["means"]

    # (a) context: the dry-weather influent flow and ammonia driving the diurnal cycle
    # (the evaluated run is driven by the same 14-day file from day 0).
    arr = np.loadtxt(os.path.join(common.DATA_DIR, "bsm1_dry_influent.csv"),
                     delimiter=",", skiprows=1)
    ti, Qi, Si = arr[:, 0], arr[:, -1], arr[:, 10]
    w = (ti >= t[0]) & (ti <= t[-1])
    ax[0].plot(ti[w], Qi[w] / 1e3, color="0.35", lw=1.3)
    ax[0].set_ylabel("influent flow\n(10$^3$ m$^3$/d)")
    axr = ax[0].twinx()
    axr.plot(ti[w], Si[w], color="C2", lw=1.3)
    axr.set_ylabel(r"influent $S_{NH}$" "\n(mg N/L)", color="C2")
    axr.tick_params(axis="y", colors="C2"); axr.grid(False)
    ax[0].set_title("(a) dry-weather influent", loc="left")

    # (b) effluent ammonia with and without the settler soluble holdup, against the
    # benchmark's flow-weighted average; the two dotted lines are the model averages.
    ax[1].plot(t, np.asarray(d["eff_SNH"]), color="C0", lw=1.4,
               label="with holdup (benchmark)")
    ax[1].plot(t, np.asarray(hu["eff_SNH_noholdup"]), color="C1", lw=1.4,
               label="pass-through")
    ax[1].axhline(d["snh_ref"], color="k", ls="--", lw=1.1, label="benchmark avg.")
    ax[1].axhline(hu["snh_holdup"], color="C0", ls=":", lw=1.3)
    ax[1].axhline(hu["snh_noholdup"], color="C1", ls=":", lw=1.3)
    ax[1].plot([], [], color="0.4", ls=":", lw=1.3, label="model avgs.")
    ax[1].set_ylabel(r"effluent $S_{NH}$ (mg N/L)")
    ax[1].legend(frameon=False, loc="upper left", ncol=4, bbox_to_anchor=(0.0, 1.16),
                 columnspacing=1.2, handlelength=1.6)
    ax[1].set_title(f"(b) effluent ammonia (pass-through average "
                    f"+{hu['snh_avg_reldiff_pct']:.0f}%)", loc="left", pad=28)

    # (c) mechanism: what the clarifier volume does to the signal. With the holdup the
    # effluent is a damped, delayed copy of the settler feed; the pass-through settler
    # passes its feed straight to the effluent.
    ax[2].plot(t, np.asarray(hu["feed_SNH"]), color="C0", lw=1.0, alpha=0.55,
               label="settler feed (holdup)")
    ax[2].plot(t, np.asarray(d["eff_SNH"]), color="C0", lw=1.6,
               label="effluent (holdup)")
    ax[2].plot(t, np.asarray(hu["feed_SNH_noholdup"]), color="C1", lw=1.0, ls="--",
               label="feed = effluent (pass-through)")
    ax[2].set(xlabel="time (d)", ylabel=r"$S_{NH}$ (mg N/L)")
    ax[2].legend(frameon=False, loc="upper left", ncol=3, bbox_to_anchor=(0.0, 1.16))
    ax[2].set_title("(c) clarifier damping of the settler feed", loc="left", pad=28)
    ax[2].set_xlim(t[0], t[-1])

    common.save(fig, "fig_bsm1_dynamic")
    print(f"  influent: {d['influent_source']}")
    for k, v in d["rels_pct"].items():
        print(f"   {k:10s} model {d['comp'][k][0]:.3f}  ref {d['comp'][k][1]:.3f}  ({v:.2f}%)")
    print(f"  worst effluent {d['worst_pct']:.2f}% ({d['worst_name']})")
    print(f"  EQI {d['eqi']:.1f} vs {d['eqi_ref']:.1f} [{d['eqi_conv']}] ({d['eqi_pct']:.2f}%)")
    print(f"  OCI {d['oci']:.1f} vs {d['oci_ref']:.1f} [{d['oci_conv']}] ({d['oci_pct']:.2f}%)")


if __name__ == "__main__":
    main()
