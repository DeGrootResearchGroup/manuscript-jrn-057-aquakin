"""Extract the official BSM1 dry-weather influent into a shipped CSV.

Raw dryinfluent.ascii columns (per convert.m):
  a1=time(d) a2=SS a3=XBH a4=XS a5=XI a6=SNH a7=SI a8=SND a9=XND a10=Q
BSM1 influent state order: t SI SS XI XS XB_H XB_A XP SO SNO SNH SND XND SALK Q
"""
from __future__ import annotations
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ASCII = os.path.join(HERE, "..", "BSM1_R2019b", "influent_files", "dryinfluent.ascii")
OUT = os.path.join(HERE, "..", "data", "bsm1_dry_influent.csv")

a = np.loadtxt(ASCII)
t = a[:, 0]
z = np.zeros_like(t)
cols = {"SI": a[:, 6], "SS": a[:, 1], "XI": a[:, 4], "XS": a[:, 3],
        "XB_H": a[:, 2], "XB_A": z, "XP": z, "SO": z, "SNO": z,
        "SNH": a[:, 5], "SND": a[:, 7], "XND": a[:, 8],
        "SALK": np.full_like(t, 7.0), "Q": a[:, 9]}
order = ["SI", "SS", "XI", "XS", "XB_H", "XB_A", "XP", "SO", "SNO",
         "SNH", "SND", "XND", "SALK", "Q"]
M = np.column_stack([t] + [cols[k] for k in order])
np.savetxt(OUT, M, delimiter=",", header="t_d," + ",".join(order),
           comments="", fmt="%.6g")
print(f"wrote {os.path.relpath(OUT, os.path.join(HERE, '..'))}  ({M.shape[0]} rows, "
      f"SNH avg {cols['SNH'].mean():.2f}, Q avg {cols['Q'].mean():.0f})")
