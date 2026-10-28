"""Figure: performance.

(a) Times the one-time compilation of the validated BSM1 plant solve, measured ahead
    of time (``jax.jit(...).lower().compile()``) so that no solve is included, the
    first execution, an amortized solve with changed parameters through the same
    executable, and the cost of an exact reverse-mode gradient relative to one solve.
(b) How the compile (again measured ahead of time) and the amortized solve scale
    with network size. A family of
    synthetic first-order chains of growing species count is compiled and solved,
    isolating the engine's cost from any one model's chemistry, and the measured
    wall times are fit to a power law in the state dimension.

These timings are machine-specific. Re-run with RECOMPUTE=performance on the
reference machine before submission.

Cache key: "performance".
"""
from __future__ import annotations
import os
import tempfile
import time
import numpy as np
import common

NRUNS = 10        # timed runs of the compiled plant solve/gradient; see _compute
SCALE_NRUNS = 10  # timed solves per network size in the (b) size-scaling sweep
SCALE_NCOMPILE = 5  # repeat compilations per size (from cleared caches) in (b)
# Sizes of the (b) sweep, powers of sqrt(2) from 4 to 256.
SCALE_SIZES = [4, 6, 8, 11, 16, 23, 32, 45, 64, 91, 128, 181, 256]


def _chain_yaml(n):
    """A first-order chain A0->A1->...->A_{n-1}, rate k[A_i], n states, n-1 reactions.

    A synthetic network whose only purpose is to grow the state dimension while
    keeping every rate law and the stiffness fixed, so the measured cost is the
    engine's and not a particular model's.
    """
    L = ["model:", f"  name: chain_{n}", '  version: "1.0"',
         '  description: "Synthetic first-order chain for size scaling."',
         "  references: []", "", "species:"]
    for i in range(n):
        L += [f"  - name: A{i}", "    units: mol/L",
              f"    default_concentration: {1.0 if i == 0 else 0.0}"]
    L += ["", "conditions:", "  - name: T", "    default: 293.15", "", "reactions:"]
    for i in range(n - 1):
        L += [f"  - name: r{i}", f'    rate: "k * [A{i}]"',
              "    parameters:", "      k:", "        value: 1.0",
              "    stoichiometry:", f"      A{i}: -1", f"      A{i + 1}: +1"]
    return "\n".join(L) + "\n"


def _aot_compile(fn, arg):
    """Compile ``fn`` ahead of time for ``arg`` without executing it.

    Returns (executable, trace+lower seconds, XLA compile seconds). The executable
    does not share jax.jit's cache, so every timed call must go through it."""
    import jax
    t0 = time.perf_counter()
    lowered = jax.jit(fn).lower(arg)
    t1 = time.perf_counter()
    compiled = lowered.compile()
    t2 = time.perf_counter()
    return compiled, t1 - t0, t2 - t1


def _time_network_size(n, tmpdir):
    """Compile times (SCALE_NCOMPILE repeats, each from cleared JAX caches and a fresh
    closure, so none reuses a compiled program) and amortized changed-parameter
    solve times (SCALE_NRUNS runs of the last compiled program) for a size-n chain."""
    import jax, jax.numpy as jnp
    import aquakin

    path = os.path.join(tmpdir, f"chain_{n}.yaml")
    with open(path, "w") as f:
        f.write(_chain_yaml(n))
    net = aquakin.load_model_from_file(path)
    rx = aquakin.BatchReactor(net, net.default_conditions(1))
    C0 = net.default_concentrations()
    p0 = net.default_parameters()
    Th = 8.0                       # a few chain timescales; stiffness is n-independent

    compile_s = []
    for _ in range(SCALE_NCOMPILE):
        jax.clear_caches()

        def solve_loss(params):
            s = rx.solve(C0, (0.0, Th), t_eval=jnp.asarray([Th]), params=params)
            return jnp.sum(s.C[-1] ** 2)

        solve_c, t_lower, t_xla = _aot_compile(solve_loss, p0)
        compile_s.append(t_lower + t_xla)

    times = []
    for i in range(SCALE_NRUNS):
        pp = p0 * (1.0 + 0.01 * (i + 1))
        t0 = time.perf_counter()
        out = float(solve_c(pp)); jax.block_until_ready(out)
        times.append(time.perf_counter() - t0)
    return compile_s, times


def _compute():
    import jax, jax.numpy as jnp
    import aquakin
    from aquakin.plant.bsm import build_bsm1, bsm1_warm_start, load_bsm1_influent

    # --- (a) BSM1 plant compile / solve / gradient costs --------------------
    asm1 = aquakin.load_model("asm1")
    # The validated plant of Figs. 6-7: five reactors and the ten-layer Takacs
    # settler with the benchmark's soluble holdup.
    plant = build_bsm1(model=asm1, use_takacs=True, settler_soluble_holdup=True,
                       settler_composition_mode="lumped_tss")
    plant.add_influent("feed", load_bsm1_influent("dry", asm1))
    y0 = bsm1_warm_start(plant)
    p0 = common.bsm1_params(plant)
    n_state = int(np.asarray(y0).shape[0])
    n_params = int(np.asarray(p0).shape[0])
    Th = 14.0

    def solve_loss(params):
        s = plant.solve((0.0, Th), t_eval=jnp.asarray([Th]), params=params, y0=y0)
        return jnp.sum(plant.effluent_stream(s).C[-1] ** 2)

    # Compilation alone (trace + lower, then XLA compile), measured ahead of time so
    # no solve is included; then the first execution of that same executable.
    solve_c, lower_s, xla_s = _aot_compile(solve_loss, p0)
    compile_s = lower_s + xla_s
    t0 = time.perf_counter()
    v = float(solve_c(p0)); jax.block_until_ready(v)
    first_solve_s = time.perf_counter() - t0

    # The plant solve and gradient are seconds-scale, so dispatch overhead is
    # negligible and a handful of warm runs already gives a stable median; we take
    # NRUNS runs and report the median with a 10th-90th-percentile band. (Contrast the
    # sub-millisecond digester root find, which needed far more runs to average out
    # dispatch noise.) The compile above is a one-time event and is not repeated.
    solve_times = []
    for i in range(NRUNS):
        pp = p0 * (1.0 + 0.001 * (i + 1))
        t0 = time.perf_counter()
        out = float(solve_c(pp)); jax.block_until_ready(out)
        solve_times.append(time.perf_counter() - t0)
    solve_times = np.asarray(solve_times)
    solve_s = float(np.median(solve_times))
    solve_p10 = float(np.percentile(solve_times, 10))
    solve_p90 = float(np.percentile(solve_times, 90))

    gradf, glower_s, gxla_s = _aot_compile(jax.grad(solve_loss), p0)
    grad_compile_s = glower_s + gxla_s
    g = gradf(p0); jax.block_until_ready(g)          # first execution (discarded)
    grad_times = []
    for i in range(NRUNS):
        pp = p0 * (1.0 + 0.001 * (i + 1))
        t0 = time.perf_counter()
        gg = gradf(pp); jax.block_until_ready(gg)
        grad_times.append(time.perf_counter() - t0)
    grad_times = np.asarray(grad_times)
    grad_s = float(np.median(grad_times))
    grad_p10 = float(np.percentile(grad_times, 10))
    grad_p90 = float(np.percentile(grad_times, 90))
    ad_factor = grad_s / solve_s
    fd_factor = n_params + 1

    # --- (b) compile / solve vs network size --------------------------------
    sizes = list(SCALE_SIZES)
    comp_runs, solve_runs = [], []
    with tempfile.TemporaryDirectory() as tmp:
        # warm-up (discarded): the first compile in a process carries one-time
        # tracing / backend start-up cost that would otherwise land on the first size
        _time_network_size(sizes[0], tmp)
        for n in sizes:
            c, s = _time_network_size(n, tmp)
            comp_runs.append(c); solve_runs.append(s)
            print(f"    n_state={n:4d}  compile {np.median(c):6.2f}s  solve {np.median(s) * 1e3:8.2f}ms")
    comp_sz = [float(np.median(c)) for c in comp_runs]
    solve_sz = [float(np.median(s)) for s in solve_runs]
    pct = lambda runs, q: [float(np.percentile(r, q)) for r in runs]
    ns = np.asarray(sizes, float)
    compile_exp = float(np.polyfit(np.log(ns), np.log(comp_sz), 1)[0])
    solve_exp = float(np.polyfit(np.log(ns), np.log(solve_sz), 1)[0])

    return {
        "n_state": n_state, "n_params": n_params,
        "compile_s": float(compile_s), "compile_lower_s": float(lower_s),
        "compile_xla_s": float(xla_s), "first_solve_s": float(first_solve_s),
        "first_call_s": float(compile_s + first_solve_s),
        "grad_compile_s": float(grad_compile_s),
        "solve_s": solve_s, "grad_s": grad_s,
        "solve_p10": solve_p10, "solve_p90": solve_p90,
        "grad_p10": grad_p10, "grad_p90": grad_p90, "n_runs": NRUNS,
        "scale_n_runs": SCALE_NRUNS, "scale_n_compile": SCALE_NCOMPILE,
        "scale_compile_runs": comp_runs, "scale_solve_runs": solve_runs,
        "scale_compile_p10": pct(comp_runs, 10), "scale_compile_p90": pct(comp_runs, 90),
        "scale_solve_p10": pct(solve_runs, 10), "scale_solve_p90": pct(solve_runs, 90),
        "ad_factor": float(ad_factor), "fd_factor": int(fd_factor),
        "solve_times": solve_times.tolist(), "grad_times": grad_times.tolist(),
        "scale_sizes": sizes, "scale_compile": comp_sz, "scale_solve": solve_sz,
        "scale_compile_exp": compile_exp, "scale_solve_exp": solve_exp,
    }


def main():
    d = common.load_or_compute("performance", _compute)
    plt = common.setup_mpl()
    common.scale_fonts(plt, 11.0)
    fig, ax = plt.subplots(1, 2, figsize=(10.4, 4.1))

    # (a) BSM1 compile / solve / gradient. Solve and gradient bars carry a
    #     10th-90th-percentile band over the timed runs; compile (measured ahead of
    #     time, no solve included) is a one-time event.
    heights = [d["compile_s"], d["solve_s"], d["grad_s"]]
    lo = [0.0, d["solve_s"] - d["solve_p10"], d["grad_s"] - d["grad_p10"]]
    hi = [0.0, d["solve_p90"] - d["solve_s"], d["grad_p90"] - d["grad_s"]]
    ax[0].bar([0, 1, 2], heights, yerr=[lo, hi], capsize=3,
              error_kw=dict(ecolor="0.3", lw=1.0), color=["C0", "C2", "C1"])
    ax[0].set_xticks([0, 1, 2])
    ax[0].set_xticklabels(["compile", "solve\n(amortized)", "gradient\n(reverse)"])
    ax[0].set(ylabel="wall time (s)", title="(a) BSM1 plant cost")

    # (b) scaling with network size
    ns = np.asarray(d["scale_sizes"], float)
    comp = np.asarray(d["scale_compile"]); solv = np.asarray(d["scale_solve"])
    # medians; bars span the 10th-90th percentile where the cache has repeats
    for y, key, mk, col, lab in (
            (comp, "compile", "o-", "C0", f"compile ($\\propto N^{{{d['scale_compile_exp']:.1f}}}$)"),
            (solv, "solve", "s-", "C2", f"solve ($\\propto N^{{{d['scale_solve_exp']:.1f}}}$)")):
        lo, hi = d.get(f"scale_{key}_p10"), d.get(f"scale_{key}_p90")
        yerr = None if lo is None else [y - np.asarray(lo), np.asarray(hi) - y]
        ax[1].errorbar(ns, y, yerr=yerr, fmt=mk, color=col, ms=5, capsize=2, lw=1.5,
                       elinewidth=0.9, label=lab)
    ax[1].set_xscale("log"); ax[1].set_yscale("log")
    ax[1].set(xlabel="model size $N$ (species)", ylabel="wall time (s)",
              title="(b) scaling with network size")
    ax[1].legend(frameon=False, loc="lower right")

    common.save(fig, "fig_performance")
    print(f"  n_state={d['n_state']} n_params={d['n_params']}")
    if "compile_lower_s" in d:   # caches from before the ahead-of-time compile split lack these
        print(f"  compile {d['compile_s']:.2f}s (trace+lower {d['compile_lower_s']:.2f}s, "
              f"XLA {d['compile_xla_s']:.2f}s)  first solve {d['first_solve_s']:.3f}s  "
              f"solve {d['solve_s']:.3f}s  grad {d['grad_s']:.3f}s (compile {d['grad_compile_s']:.2f}s)")
    else:
        print(f"  compile {d['compile_s']:.2f}s  solve {d['solve_s']:.3f}s  grad {d['grad_s']:.3f}s")
    print(f"  AD factor {d['ad_factor']:.2f}x  FD factor {d['fd_factor']}x")
    print(f"  size scaling: compile ~N^{d['scale_compile_exp']:.2f}, "
          f"solve ~N^{d['scale_solve_exp']:.2f} over N={d['scale_sizes'][0]}..{d['scale_sizes'][-1]}")


if __name__ == "__main__":
    main()
