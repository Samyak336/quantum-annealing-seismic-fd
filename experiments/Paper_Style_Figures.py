#!/usr/bin/env python
"""
Paper-Style Figures 9 & 10 — Salt Dome Wavefield & Seismic Records
===================================================================
Generates publication-quality 3x3 panel figures matching the paper's layout:
  Fig 9: Wavefield snapshots + horizontal slices at 2 depths
  Fig 10: Wiggle-trace seismic records + single-trace slices at 2 offsets

Usage:
    python experiments/Paper_Style_Figures.py
"""
from __future__ import annotations
import io, sys, time
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize as scipy_minimize

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments_common import (
    PI, SolverSettings, compute_metrics, make_linear_system,
    solve_quantum, solve_taylor, solve_wang, _direction_grid,
)

FIGURES_DIR = ROOT.parent / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

def _silent(func, **kw):
    sink = io.StringIO()
    with redirect_stdout(sink), redirect_stderr(sink):
        return func(**kw)

def ricker_wavelet(t, f0):
    u = (PI * f0 * t) ** 2
    return (1.0 - 2.0 * u) * np.exp(-u)

def second_deriv_stencil(coeff):
    M = len(coeff)
    s = np.zeros(2 * M + 1)
    c = M
    for m in range(M):
        am = float(coeff[m].flat[0]) if hasattr(coeff[m], 'flat') else float(coeff[m])
        s[c + m + 1] += am; s[c - m - 1] += am
        s[c + m] -= am; s[c - m] -= am
    return s

def _sigma_mae_obj(coeff_flat, linear_sys, dim, cfg):
    coeff = coeff_flat.reshape(-1, 1)
    kh = np.linspace(PI / 1000, PI, 500)
    r = cfg["v"] * cfg["tao"] / cfg["h"]
    errs = []
    for theta, phi in _direction_grid(dim):
        if dim == 1:
            sigma = np.asarray(linear_sys.cal_sigma(r, kh, coeff)).reshape(-1)
        elif dim == 2:
            sigma = np.asarray(linear_sys.cal_sigma(r, kh, coeff, theta=theta)).reshape(-1)
        else:
            sigma = np.asarray(linear_sys.cal_sigma(r, kh, coeff, theta=theta, phi=phi)).reshape(-1)
        errs.append(np.abs(sigma - 1.0))
    all_err = np.concatenate(errs)
    return 1e10 if np.any(np.isnan(all_err)) else float(np.mean(all_err))

# ---- Velocity model ----
def build_salt_model(nx=800, nz=450):
    """Synthetic salt dome model. Returns vel[nx, nz] in m/s."""
    vel = np.zeros((nx, nz))
    layer1 = nz // 4
    layer2 = 3 * nz // 4
    vel[:, :layer1] = 1500.0
    vel[:, layer1:layer2] = 3000.0
    vel[:, layer2:] = 4000.0
    cx, cz = nx // 2, nz // 2
    rx, rz = nx // 5, nz // 4
    Y, X = np.meshgrid(np.arange(nz), np.arange(nx))
    mask = ((X - cx) / rx) ** 2 + ((Y - cz) / rz) ** 2 <= 1.0
    vel[mask] = 4500.0
    return vel

# ---- 2D Simulation ----
def simulate_2d(coeff, vel, h, dt, nt, src_x, src_z, f0, rec_z=5):
    stencil = second_deriv_stencil(coeff)
    M = len(coeff)
    nx, nz = vel.shape
    v2dt2 = (vel * dt) ** 2
    inv_h2 = 1.0 / (h * h)
    p_prev = np.zeros((nx, nz))
    p_curr = np.zeros((nx, nz))
    p_next = np.zeros((nx, nz))
    t_delay = 1.5 / f0
    snap_time = int(0.7 * nt)
    snapshot = None
    n_rec = nx - 2 * M
    seismograms = np.zeros((n_rec, nt))
    for it in range(nt):
        t = it * dt
        lap = np.zeros_like(p_curr)
        for k in range(-M, M + 1):
            c = stencil[k + M] * inv_h2
            if c == 0.0: continue
            lap[M:nx-M, M:nz-M] += c * p_curr[M+k:nx-M+k, M:nz-M]
            lap[M:nx-M, M:nz-M] += c * p_curr[M:nx-M, M+k:nz-M+k]
        p_next[M:nx-M, M:nz-M] = (2.0 * p_curr[M:nx-M, M:nz-M]
                                   - p_prev[M:nx-M, M:nz-M]
                                   + v2dt2[M:nx-M, M:nz-M] * lap[M:nx-M, M:nz-M])
        p_next[src_x, src_z] += dt * dt * ricker_wavelet(t - t_delay, f0)
        seismograms[:, it] = p_curr[M:nx-M, rec_z]
        if it == snap_time:
            snapshot = p_curr.copy()
        p_prev, p_curr, p_next = p_curr, p_next, p_prev
    return {"snapshot": snapshot, "seismograms": seismograms,
            "snap_time": snap_time}

# ---- Get coefficients ----
def get_coefficients():
    M = 10; dim = 2
    cfg = {"v": 4500, "h": 20, "tao": 0.001, "freq_max": 90, "M": M, "epsilon": 1e-10}
    taylor = solve_taylor(M)["coeff"]
    common = dict(dim=dim, v=cfg["v"], h=cfg["h"], tao=cfg["tao"],
                  freq_max=cfg["freq_max"], M=M, max_iter_num=20,
                  alpha=1.0, alpha_decay=0.1, beta=1.0, epsilon=cfg["epsilon"],
                  coeff_init=taylor)
    settings = SolverSettings(num_sample=100, R=10, fixed_point=1, inner_maxiter=3)

    print("  Computing Wang..."); wang = _silent(solve_wang, **common)
    print("  Computing QA..."); qa = _silent(solve_quantum, **common, settings=settings)
    print("  Computing DOPP(Wang)...")
    ls = wang["linear_sys"]
    dopp = scipy_minimize(_sigma_mae_obj, x0=wang["coeff"].flatten(),
                          args=(ls, dim, cfg), method="Nelder-Mead",
                          options={"maxiter": 500, "adaptive": True})
    return {"Traditional": taylor, "QA": qa["coeff"],
            "Wang": wang["coeff"], "DOPP": dopp.x.reshape(-1, 1)}

# ---- Figure 9: Wavefield snapshots + slices ----
def plot_fig9(results, vel, h, methods, slice_depths_km):
    nm = len(methods)
    fig = plt.figure(figsize=(5 * nm, 4.5 * 3), dpi=200)
    gs = fig.add_gridspec(3, nm, hspace=0.35, wspace=0.25)
    nx, nz = vel.shape
    ext = [0, nx * h / 1000, nz * h / 1000, 0]  # km, depth increasing down

    # Row 1: Wavefield snapshots
    for col, name in enumerate(methods):
        ax = fig.add_subplot(gs[0, col])
        snap = results[name]["snapshot"]
        vmax = np.percentile(np.abs(snap), 99.5)
        vmax = vmax if vmax > 0 else 1.0
        ax.imshow(snap.T, cmap="gray", vmin=-vmax, vmax=vmax,
                  aspect="auto", extent=ext, interpolation="bilinear")
        for d in slice_depths_km:
            color = "blue" if d == slice_depths_km[0] else "red"
            ax.axhline(d, color=color, linestyle="--", linewidth=1.2)
        ax.set_title(f"({chr(97+col)}) {name}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Distance (km)", fontsize=9)
        if col == 0: ax.set_ylabel("Depth (km)", fontsize=9)
        ax.tick_params(labelsize=7)

    # Row 2 & 3: Horizontal slices
    for row_idx, depth_km in enumerate(slice_depths_km):
        depth_idx = int(depth_km * 1000 / h)
        depth_idx = min(depth_idx, nz - 1)
        color = "blue" if row_idx == 0 else "red"
        for col, name in enumerate(methods):
            ax = fig.add_subplot(gs[1 + row_idx, col])
            snap = results[name]["snapshot"]
            trace = snap[:, depth_idx]
            x_km = np.arange(nx) * h / 1000
            ax.plot(x_km, trace, color=color, linewidth=0.6)
            ax.axhline(0, color="black", linewidth=0.3)
            ax.set_title(f"({chr(97 + nm + row_idx * nm + col)}) "
                         f"{name} z={depth_km} km", fontsize=9)
            ax.set_xlabel("Distance (km)", fontsize=8)
            if col == 0: ax.set_ylabel("Amplitude", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.legend([f"{name} z={depth_km}km"], fontsize=6, loc="upper right",
                      framealpha=0.7)

    fig.suptitle("Wavefield Snapshots -- Salt Dome Model (Paper Fig 9 Style)",
                 fontsize=13, fontweight="bold", y=0.98)
    out = FIGURES_DIR / "paper_fig9_wavefield_snapshots.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out.name}")

# ---- Figure 10: Seismic records + single traces ----
def plot_fig10(results, h, dt, nt, methods, trace_offsets_km, nx, M):
    nm = len(methods)
    fig = plt.figure(figsize=(5 * nm, 4.5 * 3), dpi=200)
    gs = fig.add_gridspec(3, nm, hspace=0.35, wspace=0.25)
    n_rec = nx - 2 * M
    t_arr = np.arange(nt) * dt

    # Row 1: Wiggle-trace seismic records
    for col, name in enumerate(methods):
        ax = fig.add_subplot(gs[0, col])
        seis = results[name]["seismograms"]
        n_show = min(n_rec, 100)
        step = max(1, n_rec // n_show)
        for i in range(0, n_rec, step):
            trace = seis[i, :]
            norm = np.max(np.abs(trace))
            if norm > 0:
                trace_n = trace / norm * (step * h / 1000 * 0.4)
            else:
                trace_n = trace
            x_km = (i + M) * h / 1000
            ax.plot(trace_n + x_km, t_arr, color="black", linewidth=0.3)
            ax.fill_betweenx(t_arr, x_km, trace_n + x_km,
                             where=trace_n > 0, color="black", alpha=0.4,
                             linewidth=0)
        for off in trace_offsets_km:
            color = "blue" if off == trace_offsets_km[0] else "red"
            ax.axvline(off, color=color, linestyle="--", linewidth=1.0)
        ax.set_xlim([(M) * h / 1000, (nx - M) * h / 1000])
        ax.set_ylim([t_arr[-1], 0])
        ax.set_title(f"({chr(97+col)}) {name}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Distance (km)", fontsize=9)
        if col == 0: ax.set_ylabel("Time (s)", fontsize=9)
        ax.tick_params(labelsize=7)
        ax.set_facecolor("#d0d0d8")

    # Row 2 & 3: Single traces at specified offsets
    for row_idx, off_km in enumerate(trace_offsets_km):
        rec_idx = int(off_km * 1000 / h) - M
        rec_idx = max(0, min(rec_idx, n_rec - 1))
        color = "blue" if row_idx == 0 else "red"
        for col, name in enumerate(methods):
            ax = fig.add_subplot(gs[1 + row_idx, col])
            trace = results[name]["seismograms"][rec_idx, :]
            ax.plot(t_arr, trace, color=color, linewidth=0.6)
            ax.axhline(0, color="black", linewidth=0.3)
            ax.set_title(f"({chr(97 + nm + row_idx * nm + col)}) "
                         f"{name} x={off_km}km", fontsize=9)
            ax.set_xlabel("Time (s)", fontsize=8)
            if col == 0: ax.set_ylabel("Amplitude", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.legend([f"{name} x={off_km}km"], fontsize=6, loc="upper right",
                      framealpha=0.7)

    fig.suptitle("Seismic Records -- Salt Dome Model (Paper Fig 10 Style)",
                 fontsize=13, fontweight="bold", y=0.98)
    out = FIGURES_DIR / "paper_fig10_seismic_records.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out.name}")

# ---- Velocity model figure ----
def plot_velocity(vel, h):
    nx, nz = vel.shape
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=200)
    im = ax.imshow(vel.T, cmap="jet", aspect="auto",
                   extent=[0, nx*h/1000, nz*h/1000, 0])
    ax.set_xlabel("Distance (km)", fontsize=11)
    ax.set_ylabel("Depth (km)", fontsize=11)
    ax.set_title("Synthetic Salt Dome Velocity Model", fontsize=13, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("Velocity (m/s)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "paper_fig8_velocity_model.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: paper_fig8_velocity_model.png")

# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("  PAPER-STYLE FIGURES 8, 9, 10")
    print("=" * 60)

    h = 20.0; dt = 0.001; f0 = 15.0
    nx, nz = 800, 450
    nt = 2400
    src_x, src_z = nx // 2, 5
    rec_z = 5

    print("\n  Building salt dome model...")
    vel = build_salt_model(nx, nz)
    plot_velocity(vel, h)

    print("\n  Computing FD coefficients...")
    coeffs = get_coefficients()
    M = 10

    methods = ["Traditional", "QA", "Wang"]
    results = {}
    for name in methods:
        print(f"\n  Simulating {name} ({nx}x{nz} grid, {nt} steps)...")
        t0 = time.perf_counter()
        results[name] = simulate_2d(coeffs[name], vel, h, dt, nt,
                                    src_x, src_z, f0, rec_z=rec_z)
        print(f"    Done in {time.perf_counter()-t0:.1f}s")

    # Depths for Fig 9 horizontal slices (in km)
    slice_depths = [2.0, 7.0]
    # Offsets for Fig 10 single traces (in km)
    trace_offsets = [9.0, 6.4]

    print("\n  Generating Fig 9 (wavefield snapshots)...")
    plot_fig9(results, vel, h, methods, slice_depths)

    print("  Generating Fig 10 (seismic records)...")
    plot_fig10(results, h, dt, nt, methods, trace_offsets, nx, M)

    # Also do DOPP version
    print(f"\n  Simulating DOPP...")
    t0 = time.perf_counter()
    results["DOPP"] = simulate_2d(coeffs["DOPP"], vel, h, dt, nt,
                                  src_x, src_z, f0, rec_z=rec_z)
    print(f"    Done in {time.perf_counter()-t0:.1f}s")

    methods_dopp = ["Traditional", "Wang", "DOPP"]
    print("  Generating DOPP wavefield comparison...")
    plot_fig9(results, vel, h, methods_dopp, slice_depths)
    # Rename
    import shutil
    shutil.move(str(FIGURES_DIR / "paper_fig9_wavefield_snapshots.png"),
                str(FIGURES_DIR / "paper_fig9_dopp_wavefield.png"))
    print("  Renamed -> paper_fig9_dopp_wavefield.png")

    plot_fig10(results, h, dt, nt, methods_dopp, trace_offsets, nx, M)
    shutil.move(str(FIGURES_DIR / "paper_fig10_seismic_records.png"),
                str(FIGURES_DIR / "paper_fig10_dopp_records.png"))
    print("  Renamed -> paper_fig10_dopp_records.png")

    print(f"\n{'='*60}")
    print("  All paper-style figures generated!")
    print(f"{'='*60}")
