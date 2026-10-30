"""Regenerate every figure, value, and table for the paper.

    python3 scripts/make_all.py

Each figure caches its expensive computation under results/ and reads that cache
on later runs, so a cosmetic change needs no recompute. Force a recompute with
RECOMPUTE=all or RECOMPUTE=name1,name2 (the cache key, e.g. bsm1_steady).
"""
from __future__ import annotations
import importlib
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

FIGURES = [
    "fig_verification",
    "fig_plant_fd",
    "fig_bsm1_steady",
    "fig_bsm1_dynamic",
    "fig_steady_solver",
    "fig_digester_validation",
    "fig_differentiability",
    "fig_calibration",
    "fig_identifiability",
    "fig_design",
    "fig_respirometry",
    "fig_ad_vs_fd",
    "fig_performance",
]

TABLES = [
    "conservation_check",       # no figure: caches the ASM1 conservation audit
    "make_validation_tables",
    "make_paper_values",
]


def main():
    for name in FIGURES + TABLES:
        print(f"[{name}]")
        t0 = time.time()
        importlib.import_module(name).main()
        print(f"  done in {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
