#!/usr/bin/env python
"""
Waveform-Optimized FD Coefficients: Fixing the Objective
=========================================================
The objective Phi used by Wang (2014) and Wen & Wang (2025):
  * sums the squared dispersion relation over directions before taking the residual, so
    errors in different directions can cancel;
  * samples only 15 wavenumbers, and squaring the relation barely weights low wavenumbers.

What a simulation needs is for a band-limited pulse to arrive with the right shape. For a
plane-wave Ricker pulse with spectrum F(kappa) that travels D grid points in direction d,
the relative L2 waveform error is exactly

    e(D, d)^2 = sum |F|^2 * 4 sin^2(kappa * D * (sigma(kappa, d) - 1) / 2) / sum |F|^2,

because every Fourier component picks up the phase error kappa * D * (sigma - 1). This script
checks that formula against real leapfrog time stepping in 1D.

The waveform-optimized design ("WLS") minimises, over the source band kappa <= kappa_max and a
set of training directions,

    sum_d sum_kappa  w(kappa)^2 * kappa^2 * (sigma(kappa, d) - 1)^2      (w: Ricker spectrum)

subject to consistency (sigma(0) = 1) and exact von Neumann stability
(r * sqrt(dim) * max_x |S(x)| <= 0.95). It is still a nonlinear least-squares problem, so the
same Gauss-Newton + QUBO annealing pipeline as the original method can solve it ("QA-WLS").

Metric: accurate travel distance = the largest distance (grid points) a pulse can travel with
<= 10% waveform error, over held-out directions (worst and median), for the design source and
for sources the design did not see. All baselines get the same information (the source band).

Usage:
    python experiments/Waveform_Optimized_Design.py              # 1D, 2D, 3D
    python experiments/Waveform_Optimized_Design.py --dim 1      # one dimension
    python experiments/Waveform_Optimized_Design.py --qa-runs 0  # skip the annealing runs

Outputs:
    results/waveform_design_raw.csv
    results/waveform_design_summary.csv
    results/waveform_timestep_check.csv
    figures/waveform_design_distance.png
    figures/waveform_design_dispersion.png
    figures/waveform_design_trace_1d.png
"""

from __future__ import annotations

import argparse
import io
import time
from contextlib import redirect_stderr, redirect_stdout

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from neal import SimulatedAnnealingSampler
from scipy.optimize import linprog, minimize

from experiments_common import (
    FIGURES_DIR, PI, RESULTS_DIR, aggregate_mean_std, solve_taylor, solve_wang, write_csv,
)
from Dispersion_Optimized_Novelty import CONFIGS, run_dopp
from core.QUBO_solvers import IterSampleSolver

TOL = 0.10            # waveform error that defines "accurate"
STAB = 0.95           # stability margin: r * sqrt(dim) * max|S| <= STAB
D_MAX = 50000.0       # cap on the accurate-distance search (grid points)
X_STAB = np.linspace(0, PI, 400)
SOURCE_FACTORS = (0.7, 1.0, 1.3)   # Ricker peak relative to the design source


# --------------------------------------------------------------------------------------------
# Dispersion relation
# --------------------------------------------------------------------------------------------
def direction_sets(dim):
    """Training directions for the designs and a denser, disjoint held-out set for evaluation."""
    if dim == 1:
        return np.ones((1, 1)), np.ones((1, 1))
    if dim == 2:
        tr = np.linspace(0, PI / 4, 7)
        ho = np.array([t for t in np.linspace(0, PI / 2, 91)
                       if np.min(np.abs(tr - t)) > 1e-9 and np.min(np.abs((PI / 2 - tr) - t)) > 1e-9])
        comps = lambda th: np.stack([np.cos(th), np.sin(th)], axis=1)  # noqa: E731
        return comps(tr), comps(ho[:: max(1, len(ho) // 40)])
    g_tr = np.linspace(0, PI / 2, 5)
    g_ho = np.linspace(0, PI / 2, 17)
    trained = {(round(t, 9), round(p, 9)) for t in g_tr for p in g_tr}
    ho = [(t, p) for t in g_ho for p in g_ho if (round(t, 9), round(p, 9)) not in trained]
    comps = lambda tp: np.array([[np.cos(t) * np.cos(p), np.cos(t) * np.sin(p), np.sin(t)] for t, p in tp])  # noqa: E731
    return comps([(t, p) for t in g_tr for p in g_tr]), comps(ho[:: max(1, len(ho) // 40)])


def sigma_grad(a, kh, comps, r):
    """Phase-velocity ratio sigma (n_dirs, n_kh) and its gradient w.r.t. the coefficients."""
    a = np.asarray(a, float).reshape(-1)
    m = np.arange(1, a.size + 1) - 0.5
    Q, dQ = 0.0, 0.0
    for j in range(comps.shape[1]):
        sn = np.sin(kh[None, :, None] * comps[:, j][:, None, None] * m[None, None, :])
        S = sn @ a
        Q, dQ = Q + S ** 2, dQ + 2 * S[..., None] * sn
    x = r * np.sqrt(Q)
    with np.errstate(invalid="ignore", divide="ignore"):
        sig = 2.0 / (r * kh[None, :]) * np.arcsin(x)
        dsdQ = 1.0 / (kh[None, :] * np.sqrt(1 - x ** 2) * np.sqrt(Q))
    return sig, dsdQ[..., None] * dQ


def stability_index(a, r, dim):
    """Exact von Neumann index for this scheme; the scheme is stable when it is below 1."""
    S = np.sin(np.outer(X_STAB, np.arange(1, len(a) + 1) - 0.5)) @ np.asarray(a).reshape(-1)
    return float(r * np.sqrt(dim) * np.max(np.abs(S)))


def consistency_error(a):
    """|sigma(0) - 1| = |sum (2m - 1) a_m - 1|."""
    a = np.asarray(a).reshape(-1)
    return float(abs(np.sum((2 * np.arange(1, a.size + 1) - 1) * a) - 1))


def bandwidth(a, comps, r, tol=1e-3):
    kh = np.linspace(PI / 2000, PI, 2000)
    s, _ = sigma_grad(a, kh, comps, r)
    worst = np.max(np.where(np.isfinite(s), np.abs(s - 1), np.inf), axis=0)
    bad = np.nonzero(worst > tol)[0]
    return float(kh[-1] if bad.size == 0 else (kh[bad[0] - 1] if bad[0] > 0 else 0.0))


# --------------------------------------------------------------------------------------------
# Waveform error: closed form, accurate distance, and a real 1D simulation as a check
# --------------------------------------------------------------------------------------------
KAPPA_EVAL = np.linspace(PI / 4000, PI, 4000)


def ricker_spectrum(kappa, k0):
    return (kappa / k0) ** 2 * np.exp(-(kappa / k0) ** 2)


def waveform_error(sigma_row, k0, distance):
    F2 = ricker_spectrum(KAPPA_EVAL, k0) ** 2
    s = np.where(np.isfinite(sigma_row), sigma_row, 0.0)
    return float(np.sqrt(np.sum(F2 * 4 * np.sin(KAPPA_EVAL * distance * (s - 1) / 2) ** 2) / np.sum(F2)))


def accurate_distance(a, comps, r, k0):
    """Per direction: largest travel distance with waveform error <= TOL."""
    sig, _ = sigma_grad(a, KAPPA_EVAL, comps, r)
    out = []
    for row in sig:
        if waveform_error(row, k0, D_MAX) <= TOL:
            out.append(D_MAX)
            continue
        lo, hi = 0.0, D_MAX
        for _ in range(32):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if waveform_error(row, k0, mid) <= TOL else (lo, mid)
        out.append(lo)
    return np.array(out)


def ricker_space(x, k0):
    u = (k0 * x / 2.0) ** 2
    return (1 - 2 * u) * np.exp(-u)


def simulate_1d(a, r, k0, distance, n=8192):
    """Leapfrog time stepping p^{n+1} = 2p^n - p^{n-1} + r^2 D_- D_+ p^n on a periodic grid."""
    a = np.asarray(a, float).reshape(-1)
    x = np.arange(n, dtype=float)
    x0 = 0.15 * n
    f = lambda s: ricker_space(((s + n / 2) % n) - n / 2, k0)  # noqa: E731
    p_prev, p = f(x - x0 + r), f(x - x0)            # exact right-going pulse at t = -tau and 0
    steps = int(round(distance / r))
    for _ in range(steps):
        q = sum(am * (np.roll(p, -m) - np.roll(p, m - 1)) for m, am in enumerate(a, 1))
        lap = sum(am * (np.roll(q, -(m - 1)) - np.roll(q, m)) for m, am in enumerate(a, 1))
        p_prev, p = p, 2 * p - p_prev + r * r * lap
    exact = f(x - x0 - r * steps)
    return float(np.linalg.norm(p - exact) / np.linalg.norm(exact)), x, p, exact


# --------------------------------------------------------------------------------------------
# Designs
# --------------------------------------------------------------------------------------------
def _quiet(func, *args, **kwargs):
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        return func(*args, **kwargs)


def wang_for_band(dim, cfg, kh_band):
    taylor = solve_taylor(cfg["M"])["coeff"]
    w = _quiet(solve_wang, dim=dim, v=cfg["v"], h=cfg["h"], tao=cfg["tao"],
               freq_max=kh_band / PI * cfg["v"] / (2 * cfg["h"]), M=cfg["M"],
               max_iter_num=30 if dim == 3 else 20, alpha=1.0, alpha_decay=0.1, beta=1.0,
               epsilon=cfg["epsilon"], coeff_init=taylor)
    return np.asarray(w["coeff"]).reshape(-1), w["linear_sys"]


def wang_tuned(dim, cfg, k0, comps_tr, r, kh_band, kh_paper):
    """Strongest Wang baseline: the design band (narrower or wider than the source band) that
    maximises the worst-direction accurate distance for this source on the training directions."""
    top = min(1.8, 0.95 * PI / kh_band)
    fracs = sorted(set(np.round(np.r_[np.linspace(0.4, top, 9), 1.0, kh_paper / kh_band], 4)))
    best = None
    for frac in fracs:
        a, _ = wang_for_band(dim, cfg, frac * kh_band)
        score = accurate_distance(a, comps_tr, r, k0).min()
        if best is None or score > best[0]:
            best = (score, a, frac)
    return best[1], best[2]


def wls_problem(comps, r, k0, kh_band, floor=0.05):
    kap = np.linspace(kh_band / 300, kh_band, 300)
    w = ricker_spectrum(kap, k0)
    w = np.maximum(w, floor * w.max())

    def res(a):
        s, _ = sigma_grad(a, kap, comps, r)
        return (w[None, :] * kap[None, :] * (np.where(np.isfinite(s), s, 10.0) - 1)).ravel()

    def jac(a):
        _, G = sigma_grad(a, kap, comps, r)
        return (w[None, :, None] * kap[None, :, None] * np.where(np.isfinite(G), G, 0.0)).reshape(-1, len(a))
    return res, jac


def _constraints(M, r, dim):
    c = 2 * np.arange(1, M + 1) - 1.0
    P = np.sin(np.outer(X_STAB, np.arange(1, M + 1) - 0.5))
    cap = STAB / (r * np.sqrt(dim))
    return [{"type": "eq", "fun": lambda a: c @ a - 1.0, "jac": lambda a: c},
            {"type": "ineq", "fun": lambda a: cap - P @ a, "jac": lambda a: -P},
            {"type": "ineq", "fun": lambda a: cap + P @ a, "jac": lambda a: P}]


def wls_design(a0, comps_tr, r, dim, k0, kh_band):
    """Classical solution of the waveform-weighted objective (SLSQP with the two constraints)."""
    res, jac = wls_problem(comps_tr, r, k0, kh_band)
    scale = 1.0 / max(np.sum(res(a0) ** 2), 1e-300)
    sol = minimize(lambda a: scale * np.sum(res(a) ** 2), np.asarray(a0, float),
                   jac=lambda a: 2 * scale * jac(a).T @ res(a), constraints=_constraints(len(a0), r, dim),
                   method="SLSQP", options={"maxiter": 1000, "ftol": 1e-14})
    return sol.x


def qa_wls(a0, comps_tr, r, dim, k0, kh_band, iters=60, alpha0=1e-2, decay=0.1, patience=10):
    """The original annealing pipeline on the waveform-weighted objective.

    Consistency (sum (2m-1) a_m = 1) is built in by writing a = a_p + N z, where N spans the
    null space of the constraint, so every step is an ordinary unconstrained least-squares
    problem in z. Gauss-Newton outer loop; each linearised step goes to IterSampleSolver exactly
    as Wang's linear system does in the original method. A step is accepted only if it lowers
    the true objective and keeps the scheme stable; otherwise it is halved (backtracking).
    """
    res, jac = wls_problem(comps_tr, r, k0, kh_band)
    M = len(a0)
    c = 2 * np.arange(1, M + 1) - 1.0
    a_p = c / (c @ c)
    N = np.linalg.svd(c[None, :])[2][1:].T          # (M, M-1), orthonormal basis of c's null space
    to_a = lambda z: a_p + N @ z  # noqa: E731
    obj = lambda z: float(np.sum(res(to_a(z)) ** 2))  # noqa: E731
    ok = lambda z: stability_index(to_a(z), r, dim) <= STAB  # noqa: E731

    z = N.T @ (np.asarray(a0, float) - a_p)         # nearest consistent starting point
    scale = 1.0 / np.linalg.norm(res(to_a(z)))
    sis = IterSampleSolver(sampler=SimulatedAnnealingSampler(), x0=np.ones(M - 1).reshape(-1, 1) / 100,
                           num_sample=100, R=10, fixed_point=1, maxiter=3,
                           codingMod="pn", resultMod="lowest_energy")
    current, stalled = obj(z), 0
    for k in range(iters):
        A = scale * jac(to_a(z)) @ N
        b = -scale * res(to_a(z)).reshape(-1, 1)
        d = np.asarray(_quiet(sis.solve, A, b, alpha=alpha0 * np.linalg.norm(A, 2) ** 2 * decay ** k)).reshape(-1)
        t, accepted = 1.0, False
        while t >= 1 / 32:
            trial = z + t * d
            if ok(trial) and obj(trial) < current:
                z, current, accepted = trial, obj(trial), True
                break
            t *= 0.5
        stalled = 0 if accepted else stalled + 1
        if stalled >= patience:
            break
    return to_a(z)


def minimax_design(a0, comps_tr, r, dim, kh_band, iters=300):
    """Chebyshev (worst-case) design over the band, by sequential LP with the same constraints."""
    kh = np.linspace(kh_band / 150, kh_band, 150)
    a = np.asarray(a0, float).copy()
    M = a.size
    c = 2 * np.arange(1, M + 1) - 1.0
    a += c * (1.0 - c @ a) / (c @ c)            # start from the nearest consistent point
    P = np.sin(np.outer(X_STAB, np.arange(1, M + 1) - 0.5))
    cap = STAB / (r * np.sqrt(dim))
    err = lambda x: float(np.nanmax(np.abs(sigma_grad(x, kh, comps_tr, r)[0] - 1)))  # noqa: E731
    E, delta = err(a), 0.02
    for _ in range(iters):
        s, G = sigma_grad(a, kh, comps_tr, r)
        e, G = (s - 1).ravel(), G.reshape(-1, M)
        n = G.shape[0]
        A_ub = np.vstack([np.c_[G, -np.ones(n)], np.c_[-G, -np.ones(n)],
                          np.c_[P, np.zeros(len(P))], np.c_[-P, np.zeros(len(P))]])
        b_ub = np.r_[-e, e, cap - P @ a, cap + P @ a]
        sol = linprog(np.r_[np.zeros(M), 1.0], A_ub=A_ub, b_ub=b_ub,
                      A_eq=np.r_[c, 0.0][None, :], b_eq=[1.0 - c @ a],
                      bounds=[(-delta, delta)] * M + [(0, None)], method="highs")
        if sol.status != 0:
            delta *= 0.5
        else:
            d = sol.x[:M]
            E_new = err(a + d)
            if E_new < E:
                pred = E - sol.x[M]
                ratio = (E - E_new) / max(pred, 1e-300)
                a, E = a + d, E_new
                delta = min(2 * delta, 1.0) if ratio > 0.75 else (0.5 * delta if ratio < 0.25 else delta)
            else:
                delta *= 0.5
        if delta < 1e-13:
            break
    return a


# --------------------------------------------------------------------------------------------
# Experiment
# --------------------------------------------------------------------------------------------
def run(dims, qa_runs):
    rows, check_rows, curves = [], [], {}
    for dim in dims:
        cfg = CONFIGS[dim]
        r = cfg["v"] * cfg["tao"] / cfg["h"]
        kh_paper = cfg["freq_max"] / (cfg["v"] / (2 * cfg["h"])) * PI
        comps_tr, comps_ho = direction_sets(dim)
        wang_paper, ls_paper = wang_for_band(dim, cfg, kh_paper)
        dopp = np.asarray(_quiet(run_dopp, dim, cfg, wang_paper.reshape(-1, 1), ls_paper, "Wang")["coeff"]).reshape(-1)
        taylor = np.asarray(solve_taylor(cfg["M"])["coeff"]).reshape(-1)
        print(f"\n{'=' * 78}\n  {dim}D  r={r:.3f}  M={cfg['M']}  paper design band kh<={kh_paper:.2f}  "
              f"held-out directions: {len(comps_ho)}\n{'=' * 78}", flush=True)

        for band_frac in (0.6, 1.0):
            kh_band = band_frac * kh_paper
            k0 = kh_band / 2.5
            t0 = time.perf_counter()
            wang_m, _ = wang_for_band(dim, cfg, kh_band)
            wang_t, frac = wang_tuned(dim, cfg, k0, comps_tr, r, kh_band, kh_paper)
            designs = [("Taylor", taylor, 0), ("Wang (paper band)", wang_paper, 0),
                       ("Wang (matched band)", wang_m, 0), ("Wang (tuned band)", wang_t, 0),
                       ("DOPP(Wang) (paper band)", dopp, 0),
                       ("Minimax (matched)", minimax_design(wang_m, comps_tr, r, dim, kh_band), 0),
                       ("WLS (ours)", wls_design(wang_m, comps_tr, r, dim, k0, kh_band), 0)]
            for rep in range(qa_runs):
                designs.append(("QA-WLS (ours, annealing)", qa_wls(wang_m, comps_tr, r, dim, k0, kh_band), rep))
            print(f"\n  source band kh<={kh_band:.2f} (Ricker peak {k0:.2f}; tuned Wang uses {frac:.1f}x the band; "
                  f"designs took {time.perf_counter() - t0:.0f}s)", flush=True)
            print(f"    {'design':30s} " + "  ".join(f"src x{sf:.1f}: worst / median" for sf in SOURCE_FACTORS)
                  + "    stab   |s(0)-1|")
            for name, a, rep in designs:
                row = {"dim": dim, "band_frac": band_frac, "kh_band": kh_band, "method": name, "repeat": rep}
                cells = []
                for sf in SOURCE_FACTORS:
                    dist = accurate_distance(a, comps_ho, r, sf * k0)
                    row[f"worst_dist_src{sf}"] = float(dist.min())
                    row[f"median_dist_src{sf}"] = float(np.median(dist))
                    cells.append(f"{dist.min():9.0f} / {np.median(dist):9.0f}")
                row.update({"bandwidth_1e-3": bandwidth(a, comps_ho, r), "stability_index": stability_index(a, r, dim),
                            "consistency_error": consistency_error(a)})
                rows.append(row)
                print(f"    {name:30s} " + "    ".join(cells)
                      + f"    {row['stability_index']:.3f}  {row['consistency_error']:.1e}", flush=True)
                if band_frac == 1.0 and name in ("Wang (paper band)", "WLS (ours)", "DOPP(Wang) (paper band)"):
                    curves[(dim, name)] = a
            if dim == 1 and band_frac == 1.0:
                for name, a, _ in designs[:7]:
                    for dist in (300, 1000):
                        sim, *_ = simulate_1d(a, r, k0, dist)
                        closed = waveform_error(sigma_grad(a, KAPPA_EVAL, comps_ho, r)[0][0], k0, dist)
                        check_rows.append({"method": name, "distance": dist, "time_stepping_error": sim,
                                           "closed_form_error": closed})
        curves[(dim, "meta")] = (r, kh_paper, comps_ho, cfg)

    write_csv(RESULTS_DIR / "waveform_design_raw.csv", rows)
    value_keys = [k for k in rows[0] if k.startswith(("worst_", "median_"))] + ["bandwidth_1e-3", "stability_index"]
    write_csv(RESULTS_DIR / "waveform_design_summary.csv",
              aggregate_mean_std(rows, ["dim", "band_frac", "method"], value_keys))
    if check_rows:
        write_csv(RESULTS_DIR / "waveform_timestep_check.csv", check_rows)
        print("\n  1D check, time stepping vs closed form:")
        for c in check_rows:
            print(f"    {c['method']:30s} D={c['distance']:5d}  stepping={c['time_stepping_error']:.4f}  "
                  f"closed form={c['closed_form_error']:.4f}")
    plot_distance(rows, dims)
    plot_dispersion(curves, dims)
    if 1 in dims:
        plot_trace(curves)


# --------------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------------
COLORS = {"Taylor": "#9E9E9E", "Wang (paper band)": "#4CAF50", "Wang (matched band)": "#81C784",
          "Wang (tuned": "#2E7D32", "DOPP(Wang) (paper band)": "#E91E63", "Minimax (matched)": "#FF9800",
          "WLS (ours)": "#3F51B5", "QA-WLS (ours, annealing)": "#9C27B0"}


def _color(name):
    return next((c for k, c in COLORS.items() if name.startswith(k)), "#666666")


def plot_distance(rows, dims):
    fig, axes = plt.subplots(1, len(dims), figsize=(6.2 * len(dims), 4.8), dpi=200, squeeze=False)
    for ax, dim in zip(axes[0], dims):
        methods = list(dict.fromkeys(r["method"] for r in rows if r["dim"] == dim and r["method"] != "Taylor"))
        bands = sorted({r["band_frac"] for r in rows if r["dim"] == dim})
        width = 0.8 / len(methods)
        for i, m in enumerate(methods):
            vals = [[r["worst_dist_src1.0"] for r in rows if r["dim"] == dim and r["method"] == m
                     and r["band_frac"] == b] for b in bands]
            mean = [np.mean(v) for v in vals]
            err = [[np.mean(v) - np.min(v) for v in vals], [np.max(v) - np.mean(v) for v in vals]]
            ax.bar(np.arange(len(bands)) + (i - (len(methods) - 1) / 2) * width, mean, width, yerr=err,
                   capsize=2, color=_color(m), edgecolor="black", linewidth=0.4, label=m)
        ax.set_yscale("log")
        ax.set_xticks(np.arange(len(bands)))
        ax.set_xticklabels([f"source band\n{b:.0%} of paper's" for b in bands], fontsize=9)
        ax.set_ylabel("Accurate travel distance (grid points)\nworst held-out direction, <=10% error", fontsize=9)
        ax.set_title(f"{dim}D", fontsize=12, fontweight="bold")
        ax.grid(True, axis="y", which="both", linestyle="--", linewidth=0.4, alpha=0.4)
    axes[0][0].legend(fontsize=7, loc="upper left")
    fig.suptitle("How far a pulse travels before its waveform error reaches 10% (higher is better)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "waveform_design_distance.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_dispersion(curves, dims):
    fig, axes = plt.subplots(1, len(dims), figsize=(6.2 * len(dims), 4.4), dpi=200, squeeze=False)
    kh = np.linspace(PI / 600, PI, 600)
    for ax, dim in zip(axes[0], dims):
        r, kh_paper, comps_ho, _ = curves[(dim, "meta")]
        for name in ("Wang (paper band)", "DOPP(Wang) (paper band)", "WLS (ours)"):
            if (dim, name) not in curves:
                continue
            err = np.abs(sigma_grad(curves[(dim, name)], kh, comps_ho, r)[0] - 1)
            lo, hi = np.nanmin(err, axis=0), np.nanmax(err, axis=0)
            ax.fill_between(kh, np.maximum(lo, 1e-9), np.maximum(hi, 1e-9), color=_color(name), alpha=0.25)
            ax.plot(kh, np.maximum(hi, 1e-9), color=_color(name), linewidth=1.4, label=f"{name} (worst direction)")
        ax.axvspan(0, kh_paper, color="#FFF59D", alpha=0.35, label="design band")
        ax.set_yscale("log")
        ax.set_ylim(1e-7, 1e-1)
        ax.set_xlim(0, PI)
        ax.set_xlabel("kh")
        ax.set_ylabel("|sigma - 1|  (phase-velocity error)")
        ax.set_title(f"{dim}D: dispersion error over held-out directions", fontsize=11, fontweight="bold")
        ax.grid(True, which="both", linestyle="--", linewidth=0.4, alpha=0.4)
    axes[0][0].legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "waveform_design_dispersion.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_trace(curves):
    """Real 1D time stepping: (left) after 10 km, (right) where Wang's error reaches 10%."""
    r, kh_paper, comps, cfg = curves[(1, "meta")]
    k0 = kh_paper / 2.5
    n = 8192
    d_wang = int(accurate_distance(curves[(1, "Wang (paper band)")], comps, r, k0)[0])
    panels = [(1000, ("Wang (paper band)", "DOPP(Wang) (paper band)", "WLS (ours)")),
              (d_wang, ("Wang (paper band)", "WLS (ours)"))]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.2), dpi=200)
    for ax, (dist, names) in zip(axes, panels):
        for name in names:
            err, x, p, exact = simulate_1d(curves[(1, name)], r, k0, dist, n=n)
            ax.plot(x * cfg["h"] / 1000, p, color=_color(name), linewidth=1.2, label=f"{name}: error {err:.1%}")
        ax.plot(x * cfg["h"] / 1000, exact, "k--", linewidth=1.0, label="exact")
        centre = ((0.15 * n + dist) % n) * cfg["h"] / 1000
        ax.set_xlim(centre - 0.35, centre + 0.35)
        ax.set_xlabel("position on the periodic grid (km)")
        ax.set_ylabel("pressure")
        ax.set_title(f"after {dist * cfg['h'] / 1000:.0f} km of travel", fontsize=11, fontweight="bold")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.4)
    fig.suptitle("1D Ricker pulse, real leapfrog time stepping vs the exact solution", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "waveform_design_trace_1d.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, nargs="+", default=[1, 2, 3], choices=[1, 2, 3])
    parser.add_argument("--qa-runs", type=int, default=3, help="annealing runs per configuration")
    args = parser.parse_args()
    run(args.dim, args.qa_runs)
