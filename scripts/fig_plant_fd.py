"""Figure: plant gradient, discrete adjoint vs. central finite difference,
step-size sweep per calibration-style parameter.

Reads the "verification" cache written by fig_verification.py -- this script
only plots the plant-AD-vs-FD entries of that cache (adfd_h_grid,
adfd_sweep_rel), showing the full truncation-vs-noise trade-off curve behind
the single aggregate residual quoted in the text, for each of the five
parameters individually (small multiples: five very different curve shapes
crossing on one axis would be hard to read). Each panel also marks the
no-finite-difference reference (forward- vs. reverse-mode AD agreement,
adfd_fwdrev_comp): every curve's own floor sits above this line, showing that
finite differencing, not the gradient, sets the floor.

Cache key: "verification" (shared with fig_verification.py).
"""
from __future__ import annotations
import numpy as np
import common
import fig_verification

_SYM = {"muH": r"$\mu_H$", "muA": r"$\mu_A$", "bH": r"$b_H$",
        "Y_H": r"$Y_H$", "etag": r"$\eta_g$"}


def main():
    d = common.load_or_compute("verification", fig_verification._compute)
    plt = common.setup_mpl()
    common.scale_fonts(plt, 11.0)
    free = d["adfd_free"]
    h = np.asarray(d["adfd_h_grid"])
    rel = np.asarray(d["adfd_sweep_rel"])   # (n_h, n_free)
    h_default = d["adfd_h_default"]
    fwdrev = np.asarray(d["adfd_fwdrev_comp"])   # (n_free,) no-FD reference

    fig, axes = plt.subplots(1, len(free), figsize=(11.0, 2.6), sharey=True)
    fig.get_layout_engine().set(wspace=0.08)

    for j, ax in enumerate(axes):
        ax.loglog(h, np.maximum(rel[:, j], 1e-12), "o-", color="C4", ms=3)
        ax.axvline(h_default, color="0.5", ls=":", lw=1.0)
        ax.axhline(max(fwdrev[j], 1e-12), color="C0", ls="--", lw=1.0)
        ax.invert_xaxis()
        ax.set_title(_SYM.get(free[j], free[j]))
        ax.set_xlabel("step $h$")
        if j == 0:
            ax.set_ylabel("FD error\n(rel. to adjoint)")

    common.save(fig, "fig_plant_fd")
    worst_i = int(np.argmax(rel[list(h).index(h_default)]))
    print(f"  worst parameter at h={h_default:.0e}: {free[worst_i]} "
          f"rel diff {rel[list(h).index(h_default), worst_i]:.2e}")
    for j, n in enumerate(free):
        floor_i = int(np.argmin(rel[:, j]))
        print(f"    {n:6s} floor {rel[floor_i, j]:.2e} at h={h[floor_i]:.0e}")


if __name__ == "__main__":
    main()
