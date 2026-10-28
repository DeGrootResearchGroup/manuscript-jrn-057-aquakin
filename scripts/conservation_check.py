"""Stoichiometric conservation audit of the shipped ASM1 model (no figure).

Runs ``CompiledModel.check_conservation`` (the per-reaction dot product of the
stoichiometry with the per-species composition table, at the default parameter
values) and caches every reported residual. On ASM1 the only imbalance is anoxic
heterotroph growth, whose nitrogen and COD residuals are the dinitrogen gas the
reaction releases (N2 is not a model species); the cache also records that N2
amount, (1 - Y_H) / (2.86 Y_H) g N per unit biomass, and its COD content relative
to ammonia, so the paper can state the identification.

Cache key: "conservation".
"""
from __future__ import annotations
import common

TOL = 1e-2            # the check's default reporting tolerance


def _compute():
    import aquakin
    from aquakin.utils.balance import conservation_residuals
    asm1 = aquakin.load_model("asm1")
    p = asm1.default_parameters()
    comp = asm1.composition()
    violations = asm1.check_conservation(tol=TOL)
    # The full residual matrix, including the sub-tolerance entries the check does
    # not report (autotroph growth's small COD residual from the 4.57 rounding of
    # the nitrification oxygen demand in the ASM1 stoichiometry).
    rnames, qnames, R = conservation_residuals(asm1, comp)
    Y_H = float(p[asm1.param_index["Y_H"]])
    n2 = (1.0 - Y_H) / (2.86 * Y_H)              # g N2-N released per g biomass COD
    # COD carried by nitrogen leaving as N2, relative to the table's ammonia
    # reference: the nitrate COD content minus the N2 acceptor equivalent (2.86).
    n2_cod_per_n = -float(comp["SNO"]["COD"]) - 2.86
    return {
        "tol": TOL,
        "n_reactions": int(len(asm1.reaction_names)),
        "violations": [[r, q, float(v)] for r, q, v in violations],
        "reactions": list(rnames), "quantities": list(qnames),
        "residuals": [[float(x) for x in row] for row in R],
        "unbalanced_reactions": sorted({r for r, _, _ in violations}),
        "Y_H": Y_H, "n2_n_per_biomass": n2,
        "n2_cod_per_n": n2_cod_per_n, "n2_cod_per_biomass": n2 * n2_cod_per_n,
    }


def main():
    d = common.load_or_compute("conservation", _compute)
    for r, q, v in d["violations"]:
        print(f"  {r:32s} {q:4s} residual {v:+.4f}")
    print(f"  N2 released {d['n2_n_per_biomass']:.4f} g N, carrying "
          f"{d['n2_cod_per_biomass']:.4f} g COD (per unit biomass)")


if __name__ == "__main__":
    main()
