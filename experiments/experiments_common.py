#!/usr/bin/env python
"""Shared utilities for baseline replication and ablation studies."""

from __future__ import annotations

import csv
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from neal import SimulatedAnnealingSampler

# Repository root: holds the `core` package and the results/ and figures/ folders.
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core import stagger_FD_coeff as sfci
from core.QUBO_solvers import IterSampleSolver
from core.utils import cholesky_solver

PI = np.pi
RESULTS_DIR = ROOT_DIR / "results"
FIGURES_DIR = ROOT_DIR / "figures"
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)


@dataclass
class SolverSettings:
    num_sample: int = 100
    R: int = 10
    fixed_point: int = 1
    inner_maxiter: int = 3
    coding_mod: str = "pn"
    result_mod: str = "lowest_energy"
    ensemble_k: int = 1  # ESE: number of top-K samples to average (1 = original)
    momentum_beta: float = 0.0  # Momentum L-update (0 = original steepest descent)
    greedy_revert: bool = False  # Revert to best-known on objective degradation


DimType = Literal[1, 2, 3]


def make_linear_system_1d(v: float, h: float, tao: float, freq_max: float, L: int = 20):
    return sfci.Stagger_FD_coeff_1D(v=v, h=h, tao=tao, freq_max=freq_max, L=L)


def make_linear_system(
    dim: DimType, v: float, h: float, tao: float, freq_max: float, L: int = 20
):
    if dim == 1:
        return sfci.Stagger_FD_coeff_1D(v=v, h=h, tao=tao, freq_max=freq_max, L=L)
    if dim == 2:
        return sfci.Stagger_FD_coeff_2D(v=v, h=h, tao=tao, freq_max=freq_max, L=L)
    if dim == 3:
        return sfci.Stagger_FD_coeff_3D(v=v, h=h, tao=tao, freq_max=freq_max, L=L)
    raise ValueError(f"Unsupported dimension: {dim}")


def solve_taylor(M: int) -> Dict[str, np.ndarray | float]:
    start = time.perf_counter()
    coeff = sfci.Taylor_Coef(M)
    return {"coeff": coeff, "runtime_sec": time.perf_counter() - start}


def solve_wang_1d(
    *,
    v: float,
    h: float,
    tao: float,
    freq_max: float,
    M: int,
    max_iter_num: int,
    alpha: float,
    alpha_decay: float,
    beta: float,
    epsilon: float,
    coeff_init: np.ndarray,
) -> Dict[str, np.ndarray | float | List[float]]:
    return solve_wang(
        dim=1,
        v=v,
        h=h,
        tao=tao,
        freq_max=freq_max,
        M=M,
        max_iter_num=max_iter_num,
        alpha=alpha,
        alpha_decay=alpha_decay,
        beta=beta,
        epsilon=epsilon,
        coeff_init=coeff_init,
    )


def solve_wang(
    *,
    dim: DimType,
    v: float,
    h: float,
    tao: float,
    freq_max: float,
    M: int,
    max_iter_num: int,
    alpha: float,
    alpha_decay: float,
    beta: float,
    epsilon: float,
    coeff_init: np.ndarray,
) -> Dict[str, np.ndarray | float | List[float]]:
    linear_sys = make_linear_system(dim=dim, v=v, h=h, tao=tao, freq_max=freq_max)
    history: List[float] = []

    def _cb(_coeff: np.ndarray, obj: float) -> None:
        history.append(float(obj))

    start = time.perf_counter()
    coeff = linear_sys.solve(
        M=M,
        coeff_vec=coeff_init,
        alpha=alpha,
        alpha_decay=alpha_decay,
        beta=beta,
        epsilon=epsilon,
        max_iter_num=max_iter_num,
        mode_of_construct="lstsq",
        solver=cholesky_solver,
        show_process=False,
        call_back=_cb,
    )
    runtime = time.perf_counter() - start
    return {
        "coeff": coeff,
        "runtime_sec": runtime,
        "history": history,
        "iterations": len(history),
        "linear_sys": linear_sys,
    }


def solve_quantum_1d(
    *,
    v: float,
    h: float,
    tao: float,
    freq_max: float,
    M: int,
    max_iter_num: int,
    alpha: float,
    alpha_decay: float,
    beta: float,
    epsilon: float,
    coeff_init: np.ndarray,
    settings: SolverSettings,
    x0: np.ndarray | None = None,
) -> Dict[str, np.ndarray | float | List[float]]:
    return solve_quantum(
        dim=1,
        v=v,
        h=h,
        tao=tao,
        freq_max=freq_max,
        M=M,
        max_iter_num=max_iter_num,
        alpha=alpha,
        alpha_decay=alpha_decay,
        beta=beta,
        epsilon=epsilon,
        coeff_init=coeff_init,
        settings=settings,
        x0=x0,
    )


def solve_quantum(
    *,
    dim: DimType,
    v: float,
    h: float,
    tao: float,
    freq_max: float,
    M: int,
    max_iter_num: int,
    alpha: float,
    alpha_decay: float,
    beta: float,
    epsilon: float,
    coeff_init: np.ndarray,
    settings: SolverSettings,
    x0: np.ndarray | None = None,
) -> Dict[str, np.ndarray | float | List[float]]:
    linear_sys = make_linear_system(dim=dim, v=v, h=h, tao=tao, freq_max=freq_max)
    history: List[float] = []

    def _cb(_coeff: np.ndarray, obj: float) -> None:
        history.append(float(obj))

    sampler = SimulatedAnnealingSampler()
    x0_use = np.ones(M).reshape(-1, 1) / 100 if x0 is None else x0.reshape(-1, 1)

    sis = IterSampleSolver(
        sampler=sampler,
        x0=x0_use,
        num_sample=settings.num_sample,
        R=settings.R,
        fixed_point=settings.fixed_point,
        maxiter=settings.inner_maxiter,
        codingMod=settings.coding_mod,
        resultMod=settings.result_mod,
        ensemble_k=settings.ensemble_k,
        momentum_beta=settings.momentum_beta,
        greedy_revert=settings.greedy_revert,
    )

    start = time.perf_counter()
    coeff = linear_sys.solve(
        M=M,
        coeff_vec=coeff_init,
        alpha=alpha,
        alpha_decay=alpha_decay,
        beta=beta,
        epsilon=epsilon,
        max_iter_num=max_iter_num,
        mode_of_construct="direct",
        solver=sis.solve,
        show_process=False,
        call_back=_cb,
    )
    runtime = time.perf_counter() - start
    return {
        "coeff": coeff,
        "runtime_sec": runtime,
        "history": history,
        "iterations": len(history),
        "linear_sys": linear_sys,
    }


def compute_metrics_1d(
    *,
    linear_sys: sfci.Stagger_FD_coeff_1D,
    coeff: np.ndarray,
    v: float,
    h: float,
    tao: float,
    n_points: int = 1000,
) -> Dict[str, np.ndarray | float]:
    return compute_metrics(
        dim=1,
        linear_sys=linear_sys,
        coeff=coeff,
        v=v,
        h=h,
        tao=tao,
        n_points=n_points,
    )


def _direction_grid(dim: DimType) -> List[Tuple[float | None, float | None]]:
    if dim == 1:
        return [(None, None)]
    if dim == 2:
        return [(0.0, None), (PI / 8, None), (PI / 4, None), (3 * PI / 8, None)]
    if dim == 3:
        return [
            (0.0, 0.0),
            (PI / 8, 0.0),
            (PI / 8, PI / 8),
            (PI / 4, PI / 8),
            (PI / 4, PI / 4),
        ]
    raise ValueError(f"Unsupported dimension: {dim}")


def compute_metrics(
    *,
    dim: DimType,
    linear_sys,
    coeff: np.ndarray,
    v: float,
    h: float,
    tao: float,
    n_points: int = 1000,
) -> Dict[str, np.ndarray | float]:
    kh = np.linspace(PI / 1000, PI, n_points)
    r = v * tao / h

    sigma_curves: List[np.ndarray] = []
    for theta, phi in _direction_grid(dim):
        if dim == 1:
            sigma = np.asarray(linear_sys.cal_sigma(r, kh, coeff)).reshape(-1)
        elif dim == 2:
            sigma = np.asarray(linear_sys.cal_sigma(r, kh, coeff, theta=theta)).reshape(-1)
        else:
            sigma = np.asarray(
                linear_sys.cal_sigma(r, kh, coeff, theta=theta, phi=phi)
            ).reshape(-1)
        sigma_curves.append(sigma)

    sigma_stack = np.vstack(sigma_curves)
    sigma_abs_err = np.abs(sigma_stack - 1.0)

    return {
        "final_obj": float(linear_sys.cal_Phi(coeff)),
        "sigma_mae": float(np.mean(sigma_abs_err)),
        "sigma_max_abs": float(np.max(sigma_abs_err)),
        "kh": kh,
        "sigma": sigma_stack[0],
        "sigma_mean_curve": np.mean(sigma_stack, axis=0),
        "num_directions": int(sigma_stack.shape[0]),
    }


def write_csv(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_mean_std(
    rows: List[Dict[str, object]], group_keys: List[str], value_keys: List[str]
) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[object, ...], Dict[str, List[float]]] = {}
    for row in rows:
        gkey = tuple(row[k] for k in group_keys)
        if gkey not in grouped:
            grouped[gkey] = {k: [] for k in value_keys}
        for vk in value_keys:
            grouped[gkey][vk].append(float(row[vk]))

    out: List[Dict[str, object]] = []
    for gkey, values in grouped.items():
        rec: Dict[str, object] = {k: v for k, v in zip(group_keys, gkey)}
        for vk in value_keys:
            arr = np.asarray(values[vk], dtype=float)
            rec[f"{vk}_mean"] = float(arr.mean())
            rec[f"{vk}_std"] = float(arr.std(ddof=0))
        out.append(rec)
    return out


def plot_dispersion(
    *,
    kh: np.ndarray,
    curves: List[Tuple[str, np.ndarray]],
    title: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=140)
    for label, sigma in curves:
        ax.plot(kh, sigma, label=label, linewidth=1.4)
    ax.axhline(1.0, color="black", linewidth=1.0, linestyle="--", alpha=0.8)
    ax.set_xlim(0.0, PI)
    ax.set_ylim(0.9, 1.05)
    ax.set_xlabel("kh")
    ax.set_ylabel("sigma")
    ax.set_title(title)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def plot_line(
    *,
    x: List[float],
    ys: List[Tuple[str, List[float]]],
    title: str,
    x_label: str,
    y_label: str,
    out_path: Path,
    log_y: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.0), dpi=140)
    for label, y in ys:
        ax.plot(x, y, marker="o", linewidth=1.4, label=label)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if log_y:
        ax.set_yscale("log")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def plot_bar(
    *,
    categories: List[str],
    values: List[float],
    title: str,
    y_label: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.0), dpi=140)
    ax.bar(categories, values, color=["#39568C", "#55C667", "#FDE725", "#8F2D56", "#2A9D8F", "#E76F51"][: len(categories)])
    ax.set_title(title)
    ax.set_ylabel(y_label)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.4)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
