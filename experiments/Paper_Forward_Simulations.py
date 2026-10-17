#!/usr/bin/env python
"""
Paper-Style Forward Simulation Figures
=======================================

Reproduces the paper's forward simulation experiments (Figs 6-10 style):
  1. Homogeneous medium: wavefield snapshots + seismograms + pseudo-spectral reference
  2. Synthetic layered model: 3-layer velocity model with salt-like high-velocity body
  3. Multi-receiver seismic record sections (like paper Fig 7, 10)

Methods compared: Taylor, Wang, QA, Full Pipeline (DOPP)

Usage:
    python experiments/Paper_Forward_Simulations.py                  # full run
    python experiments/Paper_Forward_Simulations.py --homogeneous     # homogeneous only
    python experiments/Paper_Forward_Simulations.py --layered         # layered only
"""

from __future__ import annotations
import argparse, io, sys, time
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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

# ═══════════════════════════════════════════════════════════════════
# Simulation helpers
# ═══════════════════════════════════════════════════════════════════
def ricker_wavelet(t, f0):
    u = (PI * f0 * t) ** 2
    return (1.0 - 2.0 * u) * np.exp(-u)


def second_deriv_stencil(coeff):
    M = len(coeff)
    s = np.zeros(2 * M + 1)
    c = M
    for m in range(M):
        am = float(coeff[m].flat[0]) if hasattr(coeff[m], 'flat') else float(coeff[m])
        s[c + m + 1] += am
        s[c - m - 1] += am
        s[c + m]     -= am
        s[c - m]     -= am
    return s


def _silent(func, **kw):
    sink = io.StringIO()
    with redirect_stdout(sink), redirect_stderr(sink):
        return func(**kw)


def _sigma_mae_obj(coeff_flat, linear_sys, dim, cfg):
    coeff = coeff_flat.reshape(-1, 1)
    kh = np.linspace(PI / 1000, PI, 500)
    r = cfg["v"] * cfg["tao"] / cfg["h"]
    errs = []
    for theta, phi in _direction_grid(dim):
        if dim == 1:
            sigma = np.asarray(linear_sys.cal_sigma(r, kh, coeff)).reshape(-1)
        elif dim == 2:
            sigma = np.asarray(
                linear_sys.cal_sigma(r, kh, coeff, theta=theta)).reshape(-1)
        else:
            sigma = np.asarray(
                linear_sys.cal_sigma(r, kh, coeff, theta=theta, phi=phi)
            ).reshape(-1)
        errs.append(np.abs(sigma - 1.0))
    all_err = np.concatenate(errs)
    if np.any(np.isnan(all_err)) or np.any(np.isinf(all_err)):
        return 1e10
    return float(np.mean(all_err))


# ═══════════════════════════════════════════════════════════════════
# 2D Wave Simulation — Homogeneous OR Heterogeneous velocity
# ═══════════════════════════════════════════════════════════════════
def simulate_2d(coeff, vel_model, h, dt, nt, src_x, src_y, f0,
                receivers=None):
    """
    2D acoustic wave simulation with variable velocity model.
    
    vel_model: 2D array (nx, ny) of velocities in m/s
               OR scalar float for homogeneous
    """
    stencil = second_deriv_stencil(coeff)
    M = len(coeff)
    inv_h2 = 1.0 / (h * h)

    if np.isscalar(vel_model):
        nx, ny = 300, 300
        vel = np.full((nx, ny), vel_model)
    else:
        vel = vel_model
        nx, ny = vel.shape

    v2dt2 = (vel * dt) ** 2

    p_prev = np.zeros((nx, ny))
    p_curr = np.zeros((nx, ny))
    p_next = np.zeros((nx, ny))

    t_delay = 1.5 / f0
    snap_times = [nt // 5, 2 * nt // 5, 3 * nt // 5, 4 * nt // 5]
    snapshots = {}

    # Multi-receiver seismograms
    if receivers is None:
        receivers = []
    seismograms = {i: np.zeros(nt) for i in range(len(receivers))}

    for it in range(nt):
        t = it * dt
        lap = np.zeros_like(p_curr)
        for k in range(-M, M + 1):
            c = stencil[k + M] * inv_h2
            if c == 0.0:
                continue
            lap[M:nx-M, M:ny-M] += c * p_curr[M+k:nx-M+k, M:ny-M]
            lap[M:nx-M, M:ny-M] += c * p_curr[M:nx-M, M+k:ny-M+k]

        p_next[M:nx-M, M:ny-M] = (2.0 * p_curr[M:nx-M, M:ny-M]
                                   - p_prev[M:nx-M, M:ny-M]
                                   + v2dt2[M:nx-M, M:ny-M] * lap[M:nx-M, M:ny-M])
        p_next[src_x, src_y] += dt * dt * ricker_wavelet(t - t_delay, f0)

        for ri, (rx, ry) in enumerate(receivers):
            seismograms[ri][it] = p_curr[rx, ry]

        if it in snap_times:
            snapshots[it] = p_curr.copy()

        p_prev, p_curr, p_next = p_curr, p_next, p_prev

    return {
        "snapshots": snapshots, "snap_times": snap_times,
        "seismograms": seismograms, "final": p_curr.copy(),
    }


def pseudo_spectral_2d(vel_model, h, dt, nt, src_x, src_y, f0,
                       receivers=None):
    """
    Pseudo-spectral method — the gold standard reference.
    Uses FFT for spatial derivatives (no FD error at all).
    Includes a sponge absorbing layer to prevent periodic wrap-around.
    """
    if np.isscalar(vel_model):
        nx, ny = 300, 300
        vel = np.full((nx, ny), vel_model)
    else:
        vel = vel_model
        nx, ny = vel.shape

    v2dt2 = (vel * dt) ** 2
    t_delay = 1.5 / f0

    # Build sponge absorbing layer (cosine taper at boundaries)
    sponge_width = 30  # grid points
    damping = np.ones((nx, ny))
    for i in range(sponge_width):
        taper = 0.5 * (1.0 - np.cos(PI * i / sponge_width))
        damping[i, :] *= taper
        damping[nx - 1 - i, :] *= taper
        damping[:, i] *= taper
        damping[:, ny - 1 - i] *= taper

    p_prev = np.zeros((nx, ny))
    p_curr = np.zeros((nx, ny))
    p_next = np.zeros((nx, ny))

    # Wavenumber grids for pseudo-spectral
    kx = 2 * PI * np.fft.fftfreq(nx, d=h)
    ky = 2 * PI * np.fft.fftfreq(ny, d=h)
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    K2 = KX**2 + KY**2

    snap_times = [nt // 5, 2 * nt // 5, 3 * nt // 5, 4 * nt // 5]
    snapshots = {}
    if receivers is None:
        receivers = []
    seismograms = {i: np.zeros(nt) for i in range(len(receivers))}

    for it in range(nt):
        t = it * dt
        # Laplacian via FFT
        P_hat = np.fft.fft2(p_curr)
        lap = np.real(np.fft.ifft2(-K2 * P_hat))

        p_next = 2.0 * p_curr - p_prev + v2dt2 * lap
        p_next[src_x, src_y] += dt * dt * ricker_wavelet(t - t_delay, f0)

        # Apply sponge absorbing layer to prevent periodic wrap
        p_next *= damping

        for ri, (rx, ry) in enumerate(receivers):
            seismograms[ri][it] = p_curr[rx, ry]
        if it in snap_times:
            snapshots[it] = p_curr.copy()

        p_prev, p_curr, p_next = p_curr, p_next, p_prev

    return {
        "snapshots": snapshots, "snap_times": snap_times,
        "seismograms": seismograms, "final": p_curr.copy(),
    }


# ═══════════════════════════════════════════════════════════════════
# Build synthetic velocity models
# ═══════════════════════════════════════════════════════════════════
def build_layered_salt_model(nx=400, ny=400):
    """
    Synthetic 3-layer model with a salt-like high-velocity body.
    Mimics the BP2004 salt model structure used in the paper.
    
    Layer 1 (top):    v = 1500 m/s  (water/sediment)
    Layer 2 (middle): v = 3000 m/s  (consolidated sediment)
    Salt body:        v = 4500 m/s  (embedded in layer 2)
    Layer 3 (bottom): v = 4000 m/s  (basement)
    """
    vel = np.zeros((nx, ny))

    # Layer boundaries (in grid points)
    layer1_bottom = ny // 4       # top 25%
    layer2_bottom = 3 * ny // 4   # middle 50%

    # Fill layers
    vel[:, :layer1_bottom] = 1500.0
    vel[:, layer1_bottom:layer2_bottom] = 3000.0
    vel[:, layer2_bottom:] = 4000.0

    # Add salt body (elliptical high-velocity intrusion)
    cx, cy = nx // 2, ny // 2
    rx_salt, ry_salt = nx // 6, ny // 5
    for i in range(nx):
        for j in range(ny):
            if ((i - cx) / rx_salt) ** 2 + ((j - cy) / ry_salt) ** 2 <= 1.0:
                vel[i, j] = 4500.0

    return vel


# ═══════════════════════════════════════════════════════════════════
# Get coefficients for all methods
# ═══════════════════════════════════════════════════════════════════
def get_coefficients(M=10):
    """Get Taylor, Wang, QA, DOPP(QA) coefficients for 2D."""
    cfg = {"v": 4500, "h": 20, "tao": 0.001, "freq_max": 90,
           "M": M, "epsilon": 1e-10}
    dim = 2

    taylor_coeff = solve_taylor(M)["coeff"]

    print("  Computing Wang coefficients...")
    wang = _silent(solve_wang, dim=dim, v=cfg["v"], h=cfg["h"],
                   tao=cfg["tao"], freq_max=cfg["freq_max"], M=M,
                   max_iter_num=20, alpha=1.0, alpha_decay=0.1,
                   beta=1.0, epsilon=cfg["epsilon"], coeff_init=taylor_coeff)

    print("  Computing QA coefficients...")
    qa = _silent(solve_quantum, dim=dim, v=cfg["v"], h=cfg["h"],
                 tao=cfg["tao"], freq_max=cfg["freq_max"], M=M,
                 max_iter_num=20, alpha=1.0, alpha_decay=0.1,
                 beta=1.0, epsilon=cfg["epsilon"], coeff_init=taylor_coeff,
                 settings=SolverSettings(num_sample=100, R=10,
                                         fixed_point=1, inner_maxiter=3))

    # DOPP seeded from QA (not Wang) — this is the novel pipeline
    print("  Computing DOPP(QA) coefficients...")
    linear_sys = wang["linear_sys"]
    dopp_result = scipy_minimize(
        _sigma_mae_obj, x0=qa["coeff"].flatten(),
        args=(linear_sys, dim, cfg), method="Nelder-Mead",
        options={"maxiter": 500, "maxfev": 2000, "xatol": 1e-10,
                 "fatol": 1e-10, "adaptive": True})
    dopp_coeff = dopp_result.x.reshape(-1, 1)

    # Print metrics
    for name, coeff in [("Taylor", taylor_coeff), ("Wang", wang["coeff"]),
                        ("QA", qa["coeff"]), ("DOPP(QA)", dopp_coeff)]:
        m = compute_metrics(dim=dim, linear_sys=linear_sys, coeff=coeff,
                           v=cfg["v"], h=cfg["h"], tao=cfg["tao"])
        print(f"    {name:10s}: Phi={m['final_obj']:.4e}  sigma_MAE={m['sigma_mae']:.6f}")

    return {
        "Taylor": taylor_coeff,
        "Wang": wang["coeff"],
        "QA": qa["coeff"],
        "DOPP(QA)": dopp_coeff,
        "_linear_sys": linear_sys,
        "_cfg": cfg,
    }


# ═══════════════════════════════════════════════════════════════════
# Plotting functions
# ═══════════════════════════════════════════════════════════════════
METHOD_COLORS = {
    "Taylor": "#888888",
    "Wang": "forestgreen",
    "QA": "royalblue", "DOPP(QA)": "#9C27B0",
    "Fine-Grid Ref": "black",
}


def plot_wavefield_snapshots(results, methods, h, snap_idx, out_path,
                             title="", vel_model=None):
    """Paper Fig 6/8 style: wavefield snapshot panels."""
    n = len(methods)
    fig, axes = plt.subplots(1, n, figsize=(4.0 * n, 3.8), dpi=200)
    if n == 1:
        axes = [axes]

    vmax = max(np.abs(results[m]["snapshots"][snap_idx]).max()
               for m in methods if snap_idx in results[m]["snapshots"])
    vmax = vmax * 0.7 if vmax > 0 else 1.0

    for ax, name in zip(axes, methods):
        snap = results[name]["snapshots"].get(snap_idx)
        if snap is None:
            ax.text(0.5, 0.5, "N/A", transform=ax.transAxes, ha="center")
        else:
            nx, ny = snap.shape
            ax.imshow(snap.T, cmap="seismic", vmin=-vmax, vmax=vmax,
                      aspect="equal", origin="lower",
                      extent=[0, nx*h, 0, ny*h])
        ax.set_title(name, fontsize=11, fontweight="bold",
                     color=METHOD_COLORS.get(name, "black"))
        ax.set_xlabel("x (m)", fontsize=9)
        if ax is axes[0]:
            ax.set_ylabel("z (m)", fontsize=9)
        ax.tick_params(labelsize=7)

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_dispersion_curves(coeffs_dict, linear_sys, cfg, out_path, title=""):
    """Paper Figs 3-5 style: dispersion ratio sigma vs normalized wavenumber kh.
    This is the correct way to compare FD coefficient quality."""
    r = cfg["v"] * cfg["tao"] / cfg["h"]
    kh = np.linspace(0.01, np.pi, 500)
    methods = ["Taylor", "Wang", "QA", "DOPP(QA)"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=200)

    # Left: Full dispersion curves sigma(kh)
    ax1 = axes[0]
    for name in methods:
        coeff = coeffs_dict[name]
        sigmas = []
        for theta in np.linspace(0, np.pi/4, 5):
            sigma = np.asarray(
                linear_sys.cal_sigma(r, kh, coeff, theta=theta)).reshape(-1)
            sigmas.append(sigma)
        sigma_avg = np.mean(sigmas, axis=0)
        ax1.plot(kh, sigma_avg, label=name, linewidth=1.5,
                 color=METHOD_COLORS.get(name, "black"))

    ax1.axhline(y=1.0, color="red", linestyle="--", linewidth=0.8,
                label="Exact (\u03c3=1)")
    ax1.set_xlabel("Normalized wavenumber kh", fontsize=11)
    ax1.set_ylabel("Dispersion ratio \u03c3 = v_num/v", fontsize=11)
    ax1.set_title("Full Dispersion Curves", fontsize=11, fontweight="bold")
    ax1.legend(fontsize=9, loc="lower left")
    ax1.set_xlim(0, np.pi)
    ax1.grid(True, alpha=0.3)

    # Right: Zoomed dispersion ERROR |sigma - 1| (log scale)
    ax2 = axes[1]
    for name in methods:
        coeff = coeffs_dict[name]
        sigmas = []
        for theta in np.linspace(0, np.pi/4, 5):
            sigma = np.asarray(
                linear_sys.cal_sigma(r, kh, coeff, theta=theta)).reshape(-1)
            sigmas.append(sigma)
        sigma_avg = np.mean(sigmas, axis=0)
        err = np.abs(sigma_avg - 1.0)
        err = np.maximum(err, 1e-16)
        ax2.semilogy(kh, err, label=name, linewidth=1.5,
                     color=METHOD_COLORS.get(name, "black"))

    ax2.set_xlabel("Normalized wavenumber kh", fontsize=11)
    ax2.set_ylabel("|\u03c3 \u2212 1| (dispersion error)", fontsize=11)
    ax2.set_title("Dispersion Error (log scale)", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.set_xlim(0, np.pi)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_seismic_record(results, methods, dt, receivers, h, out_path,
                        title=""):
    """Paper Fig 7/10 style: multi-receiver seismic record section."""
    n_methods = len(methods)
    fig, axes = plt.subplots(1, n_methods, figsize=(5.0 * n_methods, 6),
                             dpi=200, sharey=True)
    if n_methods == 1:
        axes = [axes]

    for ax, name in zip(axes, methods):
        seis = results[name]["seismograms"]
        n_rec = len(seis)
        for ri in range(n_rec):
            trace = seis[ri]
            t = np.arange(len(trace)) * dt * 1000  # ms
            norm = np.max(np.abs(trace)) if np.max(np.abs(trace)) > 0 else 1.0
            offset = ri * h * (receivers[1][0] - receivers[0][0]) if n_rec > 1 else 0
            ax.plot(trace / norm * 50 + offset, t,
                    color=METHOD_COLORS.get(name, "black"),
                    linewidth=0.5)

        ax.set_title(name, fontsize=11, fontweight="bold",
                     color=METHOD_COLORS.get(name, "black"))
        ax.set_xlabel("Receiver offset (m)", fontsize=9)
        ax.invert_yaxis()
        ax.set_facecolor("#f8f8f8")
        ax.grid(True, axis="y", linestyle="--", linewidth=0.3, alpha=0.4)

    axes[0].set_ylabel("Time (ms)", fontsize=10)
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_velocity_model(vel, h, out_path):
    """Plot the velocity model."""
    fig, ax = plt.subplots(figsize=(6, 5), dpi=200)
    nx, ny = vel.shape
    im = ax.imshow(vel.T, cmap="jet", aspect="equal", origin="lower",
                   extent=[0, nx*h, 0, ny*h])
    ax.set_xlabel("x (m)", fontsize=11)
    ax.set_ylabel("z (m)", fontsize=11)
    ax.set_title("Synthetic Layered Salt Model\n(Velocity in m/s)",
                 fontsize=12, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label("Velocity (m/s)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ═══════════════════════════════════════════════════════════════════
# Main experiments
# ═══════════════════════════════════════════════════════════════════
def run_homogeneous(coeffs):
    """Paper Figs 6-7 style: homogeneous medium simulation."""
    print("\n" + "=" * 60)
    print("  HOMOGENEOUS MEDIUM SIMULATION (Paper Figs 6-7 style)")
    print("=" * 60)

    # Paper parameters: v=1500 m/s, h=20m, dt=1ms, f0=14Hz (max ~44Hz)
    v = 1500.0  # m/s — matches paper exactly
    h = 20.0    # grid spacing
    dt = 0.001  # time step
    nx, ny = 300, 300
    nt = 800    # enough propagation to see wavefront
    src_x, src_y = nx // 2, ny // 2
    f0 = 14.0   # Ricker dominant frequency — matches paper

    # Receivers: line of 40 receivers at z=3/4
    n_rec = 40
    receivers = [(nx // 4 + i * (nx // 2) // n_rec, 3 * ny // 4)
                 for i in range(n_rec)]

    methods = ["Taylor", "Wang", "QA", "DOPP(QA)"]
    results = {}

    for name in methods:
        print(f"  Simulating {name}...")
        results[name] = simulate_2d(
            coeffs[name], v, h, dt, nt, src_x, src_y, f0,
            receivers=receivers)

    snap_idx = results["Wang"]["snap_times"][2]  # t = 3/5

    # Wavefield snapshots (4 FD methods only)
    plot_wavefield_snapshots(
        results, methods, h, snap_idx,
        FIGURES_DIR / "paper_homogeneous_snapshots.png",
        "Homogeneous Medium: Wavefield Snapshots (v=1500 m/s, f\u2080=14 Hz)")

    # Dispersion curves (the correct way to compare FD coefficient quality)
    plot_dispersion_curves(
        coeffs, coeffs["_linear_sys"], coeffs["_cfg"],
        FIGURES_DIR / "paper_homogeneous_dispersion.png",
        "2D Dispersion Analysis: Coefficient Quality Comparison (M=10)")

    # Seismic record sections
    plot_seismic_record(
        results, methods, dt, receivers, h,
        FIGURES_DIR / "paper_homogeneous_seismic_record.png",
        "Homogeneous: Seismic Record Section")


def run_layered_salt(coeffs):
    """Paper Figs 8-10 style: layered model with salt body."""
    print("\n" + "=" * 60)
    print("  LAYERED SALT MODEL SIMULATION (Paper Figs 8-10 style)")
    print("=" * 60)

    # Paper uses BP2004 salt model, f0=20Hz, M=10
    h = 10.0
    dt = 0.0008  # smaller dt for stability with high velocities
    nx, ny = 400, 400
    nt = 800
    src_x, src_y = nx // 2, ny // 6  # source near top
    f0 = 20.0   # matches paper salt model

    vel = build_layered_salt_model(nx, ny)
    plot_velocity_model(vel, h, FIGURES_DIR / "paper_salt_velocity_model.png")

    n_rec = 50
    receivers = [(nx // 8 + i * (3 * nx // 4) // n_rec, ny // 6)
                 for i in range(n_rec)]

    methods = ["Taylor", "Wang", "QA", "DOPP(QA)"]
    results = {}

    for name in methods:
        print(f"  Simulating {name} on salt model...")
        results[name] = simulate_2d(
            coeffs[name], vel, h, dt, nt, src_x, src_y, f0,
            receivers=receivers)

    snap_idx = results["Wang"]["snap_times"][2]

    # Wavefield snapshots (4 FD methods only)
    plot_wavefield_snapshots(
        results, methods, h, snap_idx,
        FIGURES_DIR / "paper_salt_snapshots.png",
        "Layered Salt Model: Wavefield Snapshots (f\u2080=20 Hz)")

    # Dispersion curves (same coefficients, shown for salt context)
    plot_dispersion_curves(
        coeffs, coeffs["_linear_sys"], coeffs["_cfg"],
        FIGURES_DIR / "paper_salt_dispersion.png",
        "2D Dispersion Analysis: Coefficient Quality (M=10, v=4500, h=20)")

    # Seismic records
    plot_seismic_record(
        results, methods, dt, receivers, h,
        FIGURES_DIR / "paper_salt_seismic_record.png",
        "Layered Salt Model: Seismic Record Section")

    # Multiple time snapshots for Taylor vs Wang vs DOPP
    fig, axes = plt.subplots(3, 4, figsize=(16, 12), dpi=200)
    snap_list = results["Wang"]["snap_times"]
    for row, name in enumerate(["Taylor", "Wang", "DOPP(QA)"]):
        for col, si in enumerate(snap_list):
            snap = results[name]["snapshots"].get(si)
            if snap is None:
                continue
            vmax = np.abs(snap).max() * 0.7
            vmax = vmax if vmax > 0 else 1.0
            axes[row, col].imshow(snap.T, cmap="seismic", vmin=-vmax,
                                  vmax=vmax, aspect="equal", origin="lower",
                                  extent=[0, nx*h, 0, ny*h])
            axes[row, col].set_title(f"{name} (t={si*dt*1000:.0f}ms)",
                                     fontsize=8, fontweight="bold",
                                     color=METHOD_COLORS.get(name, "black"))
            if col == 0:
                axes[row, col].set_ylabel("z (m)", fontsize=8)
            if row == 2:
                axes[row, col].set_xlabel("x (m)", fontsize=8)
            axes[row, col].tick_params(labelsize=6)

    fig.suptitle("Salt Model: Time Evolution of Wavefield",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "paper_salt_time_evolution.png",
               dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: figures/paper_salt_time_evolution.png")


# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--homogeneous", action="store_true")
    parser.add_argument("--layered", action="store_true")
    args = parser.parse_args()

    do_both = not args.homogeneous and not args.layered

    print("Computing FD coefficients for all methods...")
    coeffs = get_coefficients(M=10)

    if args.homogeneous or do_both:
        run_homogeneous(coeffs)

    if args.layered or do_both:
        run_layered_salt(coeffs)

    print(f"\n{'='*60}")
    print("All paper-style forward simulation figures generated!")
    print(f"{'='*60}")
