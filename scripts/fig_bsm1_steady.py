"""Figure: open-loop BSM1 steady-state validation.

Builds the open-loop BSM1 plant (five ASM1 reactors + 10-layer Takacs settler),
drives it to steady state by forward integration under the constant influent at
15 C, and compares the reactor composition, the settler effluent, the settler
underflow, and the 10-layer solids profile to the published benchmark reference
(data/bsm1_openloop_ss_reference.json). The plant COD/N mass-balance closure at
this same operating point is computed in fig_verification.py instead (reusing
its plant-AD-vs-FD baseline solve, which uses the identical official influent
and parameters), not duplicated here.

Cache key: "bsm1_steady".
"""
from __future__ import annotations
import json
import os
import numpy as np
import common

# ref keys (XBH/XBA) -> aquakin species names
_KMAP = {"XBH": "XB_H", "XBA": "XB_A"}
_PARTICULATE = ("XI", "XS", "XB_H", "XB_A", "XP")  # contribute to TSS at 0.75


def _ref():
    with open(os.path.join(common.DATA_DIR, "bsm1_openloop_ss_reference.json")) as f:
        return json.load(f)


def _tss(C, si):
    return 0.75 * float(sum(C[si[s]] for s in _PARTICULATE))


def _compute():
    import aquakin
    from aquakin.plant.bsm import build_bsm1, bsm1_warm_start
    from aquakin.plant import InfluentSeries

    ref = _ref()
    asm1 = aquakin.load_model("asm1")
    si = asm1.species_index

    # constant influent (the CONSTANTINPUT reference characteristics)
    ci = ref["constant_influent"]
    species = {_KMAP.get(k, k): float(v) for k, v in ci.items() if k != "TSS"}
    Q = float(ref["flows_m3d"]["influent_wwtp"])
    const = InfluentSeries.constant(asm1, Q=Q, **species)

    plant = build_bsm1(model=asm1, use_takacs=True, settler_soluble_holdup=True,
                       settler_composition_mode="lumped_tss")
    plant.add_influent("feed", const)
    # Official BSM1 15 C parameters applied at the engine reference temperature
    # (no Arrhenius correction): BSM1 uses a fixed 15 C parameter set.
    params = common.bsm1_params(plant)
    y0 = bsm1_warm_start(plant)

    # forward integration to steady state
    Tss = 300.0
    sol = plant.solve((0.0, Tss), t_eval=np.array([Tss]), params=params, y0=y0,
                      forward_fast=True)
    yss = np.asarray(sol.final_state)
    sb = plant.states_by_unit(yss)

    # --- reactor comparison ---------------------------------------------------
    rlabels, rmodel, rref = [], [], []
    for tk in ("tank1", "tank2", "tank3", "tank4", "tank5"):
        Cm = np.asarray(sb[tk])
        for k, rv in ref["reactors"][tk].items():
            if k == "TSS" or abs(float(rv)) < 1e-2:   # skip near-zero (e.g. anoxic SO)
                continue
            sp = _KMAP.get(k, k)
            rlabels.append(f"{tk}.{sp}")
            rmodel.append(float(Cm[si[sp]]))
            rref.append(float(rv))
    rmodel, rref = np.asarray(rmodel), np.asarray(rref)
    rrel = np.abs(rmodel - rref) / np.abs(rref)
    rw = int(np.argmax(rrel))

    # --- effluent (clarifier overflow) and underflow streams ------------------
    eff = plant.stream(sol, "clarifier.overflow")
    und = plant.stream(sol, "clarifier.underflow")
    Ce = np.asarray(eff.C)[-1]
    Cu = np.asarray(und.C)[-1]
    elabels, emodel, eref = [], [], []
    for k, rv in ref["settler_effluent"].items():
        if k == "TSS" or abs(float(rv)) < 1e-2:
            continue
        sp = _KMAP.get(k, k)
        elabels.append(sp); emodel.append(float(Ce[si[sp]])); eref.append(float(rv))
    emodel, eref = np.asarray(emodel), np.asarray(eref)
    erel = np.abs(emodel - eref) / np.abs(eref)
    ew = int(np.argmax(erel))

    eff_tss_m, eff_tss_r = _tss(Ce, si), float(ref["settler_effluent"]["TSS"])
    und_tss_m, und_tss_r = _tss(Cu, si), float(ref["settler_underflow"]["TSS"])
    tss_eff_rel = abs(eff_tss_m - eff_tss_r) / eff_tss_r
    tss_und_rel = abs(und_tss_m - und_tss_r) / und_tss_r

    # --- 10-layer settler TSS profile (best effort) ---------------------------
    prof_model, prof_ref = None, ref["settler_TSS_profile_top_to_bottom"]
    tss_prof_rel = None
    try:
        cl = plant.units["clarifier"]
        nL = int(getattr(cl, "n_layers", 10))
        raw = np.asarray(sb["clarifier"])
        if getattr(cl, "composition_mode", "per_species") == "lumped_tss":
            layer_tss = raw[:nL]               # one TSS value per layer (bottom..top)
        else:
            part = list(getattr(cl, "particulate_species"))
            block = raw[: nL * len(part)].reshape(nL, len(part))  # (layers, n_part)
            fac = np.array([0.75 if s != "XND" else 0.0 for s in part])
            layer_tss = (block * fac).sum(axis=1)  # index 0 = bottom
        prof_model = list(layer_tss[::-1])     # reverse to top..bottom
        pm, pr = np.asarray(prof_model), np.asarray(prof_ref)
        tss_prof_rel = float(np.max(np.abs(pm - pr) / np.abs(pr)))
    except Exception as e:  # pragma: no cover
        print("  [profile extraction skipped]", str(e)[:80])

    tss_worst = tss_prof_rel if tss_prof_rel is not None else max(tss_eff_rel, tss_und_rel)

    # Plant-wide mass balance at this steady operating point is reported in
    # Section~\ref{sec:results-verification-fd} (VerifMassBalCOD/N), computed
    # there from the same official steady-state condition rather than a
    # separate run here.

    # HRT check (total volume / influent flow), hours
    V = float(sum(plant.units[t].volume for t in
                  ("tank1", "tank2", "tank3", "tank4", "tank5")))
    V += float(ref["dimensions"]["settler_volume_m3"])
    hrt_h = V / Q * 24.0
    hrt_ref = float(ref["derived"]["total_HRT_h"])
    hrt_rel = abs(hrt_h - hrt_ref) / hrt_ref

    return {
        "rlabels": rlabels, "rmodel": rmodel, "rref": rref,
        "reactor_worst_pct": float(rrel[rw] * 100), "reactor_worst_name": rlabels[rw],
        "n_quant": len(rlabels),
        "elabels": elabels, "emodel": emodel, "eref": eref,
        "eff_worst_pct": float(erel[ew] * 100), "eff_worst_name": elabels[ew],
        "eff_tss_m": eff_tss_m, "eff_tss_r": eff_tss_r, "tss_eff_pct": tss_eff_rel * 100,
        "und_tss_m": und_tss_m, "und_tss_r": und_tss_r, "tss_und_pct": tss_und_rel * 100,
        "prof_model": prof_model, "prof_ref": prof_ref,
        "tss_worst_pct": float(tss_worst * 100),
        "tss_profile_available": tss_prof_rel is not None,
        "hrt_h": hrt_h, "hrt_ref": hrt_ref, "hrt_pct": hrt_rel * 100,
    }


# Panel (a) categories: ASM1 state order, and their typeset names.
_SPECIES_ORDER = ["SI", "SS", "XI", "XS", "XB_H", "XB_A", "XP", "SO", "SNO",
                  "SNH", "SND", "XND", "SALK"]
_SPECIES_TEX = {"SI": r"$S_I$", "SS": r"$S_S$", "XI": r"$X_I$", "XS": r"$X_S$",
                "XB_H": r"$X_{B,H}$", "XB_A": r"$X_{B,A}$", "XP": r"$X_P$",
                "SO": r"$S_O$", "SNO": r"$S_{NO}$", "SNH": r"$S_{NH}$",
                "SND": r"$S_{ND}$", "XND": r"$X_{ND}$", "SALK": r"$S_{ALK}$"}
_TANK_MARKERS = ["o", "s", "^", "D", "v"]


def _sci_pct(x):
    """A percentage in scientific notation for a title, e.g. 3.6x10^-3 %."""
    m, e = f"{x:.1e}".split("e")
    return rf"${m}\times10^{{{int(e)}}}$%"


def main():
    d = common.load_or_compute("bsm1_steady", _compute)
    plt = common.setup_mpl()
    common.scale_fonts(plt, 11.0)
    fig, ax = plt.subplots(1, 2, figsize=(11.0, 4.8), width_ratios=[1.7, 1.0])

    # Per-quantity relative error, which shows the round-off-level agreement that a
    # parity plot with a +/-5% band cannot resolve.
    # Categorical dot plot: species on x, one marker per reactor, dodged so the five
    # reactors of a species sit side by side. Errors below FLOOR (round-off level)
    # are drawn hollow on the floor line rather than at a misleading clipped value.
    # The floor is a display limit, well above float64 round-off.
    rref = np.asarray(d["rref"]); rmodel = np.asarray(d["rmodel"])
    rrel = np.abs(rmodel - rref) / np.maximum(np.abs(rref), 1e-30) * 100.0
    tanks = [lab.split(".")[0] for lab in d["rlabels"]]
    specs = [lab.split(".")[1] for lab in d["rlabels"]]
    sorder = [s for s in _SPECIES_ORDER if s in specs]
    torder = sorted(set(tanks))
    FLOOR = 1e-7
    dodge = np.linspace(-0.28, 0.28, len(torder))
    for j, tk in enumerate(torder):
        sel = [i for i, t in enumerate(tanks) if t == tk]
        x = np.array([sorder.index(specs[i]) for i in sel]) + dodge[j]
        y = rrel[sel]
        lo = y < FLOOR
        mk, col = _TANK_MARKERS[j], f"C{j}"
        ax[0].scatter(x[~lo], y[~lo], s=22, marker=mk, color=col,
                      edgecolor="white", linewidth=0.3, label=f"reactor {j + 1}")
        ax[0].scatter(x[lo], np.full(lo.sum(), FLOOR), s=22, marker=mk,
                      facecolor="none", edgecolor=col, linewidth=0.9)
    ax[0].set_yscale("log")
    ax[0].set_ylim(FLOOR / 3, 30.0)
    ax[0].axhline(5.0, color="0.6", ls="--", lw=0.9)
    ax[0].text(-0.4, 6.5, "5% band", color="0.45", va="bottom")
    ax[0].axhline(FLOOR, color="0.75", ls=":", lw=0.8)
    ax[0].scatter([], [], s=22, marker="o", facecolor="none", edgecolor="0.4",
                  linewidth=0.9, label=r"below $10^{-7}$%")
    ax[0].set_xticks(range(len(sorder)))
    ax[0].set_xticklabels([_SPECIES_TEX[s] for s in sorder])
    ax[0].set_xlim(-0.6, len(sorder) - 0.4)
    ax[0].grid(axis="x", visible=False)
    ax[0].legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 0.94),
                 handletextpad=0.2, columnspacing=1.0)
    ax[0].set(xlabel="ASM1 state", ylabel="relative error vs reference (%)",
              title=f"(a) ASM1 reactors (worst {_sci_pct(d['reactor_worst_pct'])})")

    if d["tss_profile_available"]:
        pm = np.asarray(d["prof_model"]); pr = np.asarray(d["prof_ref"])
        depth = np.arange(1, len(pr) + 1)
        ax[1].plot(pr, depth, "s-", color="0.4", label="reference", ms=5)
        ax[1].plot(pm, depth, "o--", color="C0", label="aquakin", ms=4)
        ax[1].set(xscale="log", xlabel="suspended solids (mg/L)",
                  ylabel="layer (1 = top)",
                  title=f"(b) settler profile (worst {_sci_pct(d['tss_worst_pct'])})")
        ax[1].invert_yaxis(); ax[1].legend(frameon=False)
    else:
        cats = ["effluent TSS", "underflow TSS"]
        ax[1].bar([0, 1], [d["eff_tss_m"], d["und_tss_m"]], width=0.35, label="aquakin")
        ax[1].bar([0.4, 1.4], [d["eff_tss_r"], d["und_tss_r"]], width=0.35, label="ref")
        ax[1].set_yscale("log"); ax[1].set_xticks([0.2, 1.2]); ax[1].set_xticklabels(cats)
        ax[1].set(title="(b) settler solids", ylabel="mg/L"); ax[1].legend(frameon=False)

    common.save(fig, "fig_bsm1_steady")
    print(f"  reactors worst {d['reactor_worst_pct']:.2e}% ({d['reactor_worst_name']}), n={d['n_quant']}")
    print(f"  effluent worst {d['eff_worst_pct']:.2f}% ({d['eff_worst_name']})")
    print(f"  eff TSS {d['tss_eff_pct']:.2f}%  underflow TSS {d['tss_und_pct']:.2f}%")
    print(f"  settler profile available={d['tss_profile_available']} worst {d['tss_worst_pct']:.2e}%")
    print(f"  HRT {d['hrt_h']:.2f} h vs {d['hrt_ref']:.2f} ({d['hrt_pct']:.2f}%)")


if __name__ == "__main__":
    main()
