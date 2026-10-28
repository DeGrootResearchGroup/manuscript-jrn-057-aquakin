"""Shared configuration, caching, and plotting helpers for the JRN-057 figures.

Everything paper-specific lives here so the individual ``fig_*.py`` scripts stay
short. Expensive computations (steady-state solves, model evaluations) are run
once and cached to ``results/<name>.json``; the plotting code reads the cache, so
a cosmetic plot change needs no recompute. Force a recompute with the env var
``RECOMPUTE=all`` or ``RECOMPUTE=name1,name2``.

Reproducibility: every analysis here is deterministic. The single module-level
``SEED`` is kept for any future sampling-based figure.
"""
from __future__ import annotations

import os

# A stiff solve can go non-finite in an extreme parameter region; ask
# equinox/diffrax to return NaN there rather than raise. Must be set before
# aquakin (hence jax/diffrax/equinox) is imported.
os.environ.setdefault("EQX_ON_ERROR", "nan")

import json
import sys

import numpy as np

# --- Make aquakin importable in the sibling-repo dev layout -----------------
# The pinned, reproducible path is ``pip install -e ../aquakin`` (see
# requirements.txt). If that has not been done, fall back to importing the
# sibling checkout directly so the figures still run from a fresh clone.
try:
    import aquakin  # noqa: F401
except ModuleNotFoundError:
    _SIBLING = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "aquakin")
    )
    if os.path.isdir(os.path.join(_SIBLING, "aquakin")):
        sys.path.insert(0, _SIBLING)
    import aquakin  # noqa: F401

import jax.numpy as jnp

# --- Reproducibility -------------------------------------------------------
SEED = 0

# --- Steady-state residual -------------------------------------------------
# Absolute floor eps in the paper's scaled residual max_i |F_i| / max(|y_i|, eps)
# (Eq. scaled-residual). Passed as ``scale_floor`` to every PTC / arclength solve
# and used by ``scaled_residual`` below, so the PTC convergence test and every
# residual the paper reports are the same quantity.
RESID_FLOOR = 1e-6


def scaled_residual(F, y):
    """The paper's scaled steady-state residual max_i |F_i| / max(|y_i|, eps)."""
    F, y = np.asarray(F), np.asarray(y)
    return float(np.max(np.abs(F) / np.maximum(np.abs(y), RESID_FLOOR)))

# --- Paths -----------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
FIG_DIR = os.path.join(ROOT, "figures")
RESULTS_DIR = os.path.join(ROOT, "results")   # cached computed results (JSON)
DATA_DIR = os.path.join(ROOT, "data")


# --- BSM1 official influent / parameters -----------------------------------
def official_dry_influent(network):
    """Build an InfluentSeries from the official BSM1 dry-weather influent
    (data/bsm1_dry_influent.csv, extracted from the benchmark distribution).

    The benchmark dynamic results are produced with the influent linearly
    interpolated between the 15-min samples (the reported effluent average flow
    equals the theoretical value, which the benchmark notes is reached only with
    interpolation), and ``InfluentSeries.at()`` linearly interpolates, so the
    samples are used directly."""
    import jax.numpy as jnp
    from aquakin.plant import InfluentSeries
    arr = np.loadtxt(os.path.join(DATA_DIR, "bsm1_dry_influent.csv"),
                     delimiter=",", skiprows=1)
    t, Q = arr[:, 0], arr[:, -1]
    order = ["SI", "SS", "XI", "XS", "XB_H", "XB_A", "XP", "SO", "SNO",
             "SNH", "SND", "XND", "SALK"]
    si = network.species_index
    C = np.zeros((arr.shape[0], network.n_species))
    for j, sp in enumerate(order):
        C[:, si[sp]] = arr[:, 1 + j]
    return InfluentSeries(jnp.asarray(t), jnp.asarray(Q), jnp.asarray(C), network)


def bsm1_params(plant):
    """Return the plant parameter vector with the ASM1 block set to the official
    BSM1 15 C parameter values (data/bsm1_asm1_parameters.json). BSM1 uses a
    fixed 15 C parameter set, so they are applied directly at the engine's
    reference temperature (where the Arrhenius factor is unity), reproducing the
    published benchmark. Keeps every non-ASM1 plant parameter at its default."""
    with open(os.path.join(DATA_DIR, "bsm1_asm1_parameters.json")) as f:
        d = json.load(f)
    overrides = {f"asm1.{k}": float(v) for k, v in d["asm1_parameters"].items()}
    return plant.parameter_values(overrides)


# --- Computed-result cache -------------------------------------------------
def _to_jsonable(x):
    """Recursively convert numpy/JAX scalars and arrays to JSON-native types."""
    if isinstance(x, dict):
        return {k: _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, jnp.floating)):
        return float(x)
    if isinstance(x, (np.integer, jnp.integer)):
        return int(x)
    return x


def _recompute_requested(name):
    env = os.environ.get("RECOMPUTE", "")
    return env == "all" or name in [s.strip() for s in env.split(",") if s.strip()]


# --- aquakin provenance ----------------------------------------------------
# The paper describes the aquakin 0.1.1 release, so every cached result must be
# computed against that commit. Point PYTHONPATH at a checkout of it (e.g.
# ``git worktree add --detach ../aquakin-v0.1.1 e198ec1``); a recompute on any other
# commit, or on a checkout with local edits, is refused unless
# AQUAKIN_ALLOW_UNPINNED=1 is set. 0.1.1 is not tagged yet: e198ec1 is the merged
# Laplace-covariance fix (#593); re-point the tag here once the release is cut.
AQUAKIN_PINNED_TAG = "v0.1.1"
AQUAKIN_PINNED_COMMIT = "e198ec1"


def aquakin_provenance():
    """The imported aquakin's version, git commit, dirty flag and location."""
    import subprocess
    src = os.path.dirname(os.path.dirname(os.path.abspath(aquakin.__file__)))

    def _git(*args):
        try:
            return subprocess.run(["git", "-C", src, *args], capture_output=True,
                                  text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    status = _git("status", "--porcelain", "--untracked-files=no")
    return {"version": getattr(aquakin, "__version__", None),
            "commit": _git("rev-parse", "HEAD"),
            "dirty": None if status is None else bool(status),
            "path": src}


def _check_pinned(prov):
    ok = (prov["commit"] or "").startswith(AQUAKIN_PINNED_COMMIT) and not prov["dirty"]
    if ok or os.environ.get("AQUAKIN_ALLOW_UNPINNED") == "1":
        return
    raise RuntimeError(
        f"aquakin at {prov['path']} is commit {prov['commit']} "
        f"(dirty={prov['dirty']}), not the {AQUAKIN_PINNED_TAG} release "
        f"{AQUAKIN_PINNED_COMMIT}. Set PYTHONPATH to a checkout of "
        f"{AQUAKIN_PINNED_TAG}, or AQUAKIN_ALLOW_UNPINNED=1 to override.")


def load_or_compute(name, compute_fn, recompute=None):
    """Return cached results for ``name``, computing and caching them if absent.

    Parameters
    ----------
    name : str
        Cache key; data is stored at ``results/<name>.json``.
    compute_fn : callable
        Zero-argument function returning a JSON-serializable result (numpy/JAX
        scalars and arrays are converted automatically). Run only on a cache miss
        or a forced recompute.
    recompute : bool, optional
        Force recomputation. Defaults to the ``RECOMPUTE`` env var
        (``all`` or a comma-separated list of names).

    Notes
    -----
    Arrays come back as nested lists; wrap with ``np.asarray`` at the call site.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, name + ".json")
    force = recompute if recompute is not None else _recompute_requested(name)
    if not force and os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    prov = aquakin_provenance()
    _check_pinned(prov)
    result = compute_fn()
    result = dict(result, _aquakin=prov)
    with open(path, "w") as f:
        json.dump(_to_jsonable(result), f, indent=1)
    print(f"  [computed + cached] {os.path.relpath(path, ROOT)}")
    return result


# --- Plotting --------------------------------------------------------------
def setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 130, "savefig.dpi": 200, "font.size": 10,
        "axes.grid": True, "grid.alpha": 0.3, "axes.titleweight": "bold",
        "legend.fontsize": 8, "figure.constrained_layout.use": True,
    })
    return plt


# Single-column text width of the document (inches). Figures are included at
# \linewidth, so a figure drawn wider than this is scaled down on the page.
TEXTWIDTH_IN = 6.5


def scale_fonts(plt, fig_width_in, target_pt=9.0, embed_frac=1.0):
    """Set font sizes so a figure of drawing width ``fig_width_in``, embedded at
    ``embed_frac`` of the text width (the ``\\includegraphics`` width as a fraction
    of ``\\linewidth``), renders its text near ``target_pt`` on the page."""
    s = target_pt * fig_width_in / (embed_frac * TEXTWIDTH_IN)
    plt.rcParams.update({
        "font.size": round(s, 1),
        "axes.titlesize": round(s * 1.12, 1),
        "axes.labelsize": round(s, 1),
        "xtick.labelsize": round(s * 0.92, 1),
        "ytick.labelsize": round(s * 0.92, 1),
        "legend.fontsize": round(s * 0.88, 1),
        "legend.title_fontsize": round(s * 0.88, 1),
    })
    return s


def save(fig, name: str):
    os.makedirs(FIG_DIR, exist_ok=True)
    for ext in ("png", "pdf"):
        # bbox=tight so edge panel titles/labels are not clipped by the canvas.
        fig.savefig(os.path.join(FIG_DIR, f"{name}.{ext}"), bbox_inches="tight")
    print(f"  wrote {os.path.join(FIG_DIR, name)}.png / .pdf")
