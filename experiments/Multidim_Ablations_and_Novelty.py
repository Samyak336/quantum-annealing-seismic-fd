#!/usr/bin/env python
"""Run compact 2D/3D ablations plus an additional novelty study."""

from __future__ import annotations

import argparse
from collections import defaultdict

from experiments_common import (
    FIGURES_DIR,
    RESULTS_DIR,
    SolverSettings,
    aggregate_mean_std,
    compute_metrics,
    plot_bar,
    plot_line,
    solve_quantum,
    solve_taylor,
    solve_wang,
    write_csv,
)


CONFIGS = {
    2: {
        "tag": "2d_ref",
        "v": 4500,
        "h": 20,
        "tao": 0.001,
        "freq_max": 90,
        "M": 10,
        "max_iter_num": 20,
        "epsilon": 1e-10,
        "repeats": 2,
    },
    3: {
        "tag": "3d_ref",
        "v": 2500,
        "h": 10,
        "tao": 0.001,
        "freq_max": 100,
        "M": 10,
        "max_iter_num": 20,
        "epsilon": 1e-10,
        "repeats": 1,
    },
}


def _baseline_for_dim(dim: int, cfg: dict) -> list[dict[str, object]]:
    v, h, tao = cfg["v"], cfg["h"], cfg["tao"]
    freq_max, M = cfg["freq_max"], cfg["M"]

    print(f"\n[dim={dim}] baseline run")

    taylor = solve_taylor(M)
    wang = solve_wang(
        dim=dim,
        v=v,
        h=h,
        tao=tao,
        freq_max=freq_max,
        M=M,
        max_iter_num=cfg["max_iter_num"],
        alpha=1.0,
        alpha_decay=0.1,
        beta=1.0,
        epsilon=cfg["epsilon"],
        coeff_init=taylor["coeff"],
    )

    qa_settings = SolverSettings(
        num_sample=100,
        R=10,
        fixed_point=1,
        inner_maxiter=3,
        coding_mod="pn",
        result_mod="lowest_energy",
    )
    qa = solve_quantum(
        dim=dim,
        v=v,
        h=h,
        tao=tao,
        freq_max=freq_max,
        M=M,
        max_iter_num=cfg["max_iter_num"],
        alpha=1.0,
        alpha_decay=0.1,
        beta=1.0,
        epsilon=cfg["epsilon"],
        coeff_init=taylor["coeff"],
        settings=qa_settings,
    )

    taylor_metrics = compute_metrics(
        dim=dim, linear_sys=wang["linear_sys"], coeff=taylor["coeff"], v=v, h=h, tao=tao
    )
    wang_metrics = compute_metrics(
        dim=dim, linear_sys=wang["linear_sys"], coeff=wang["coeff"], v=v, h=h, tao=tao
    )
    qa_metrics = compute_metrics(
        dim=dim, linear_sys=qa["linear_sys"], coeff=qa["coeff"], v=v, h=h, tao=tao
    )

    rows = [
        {
            "dim": dim,
            "tag": cfg["tag"],
            "method": "Taylor",
            "runtime_sec": float(taylor["runtime_sec"]),
            "iterations": 1,
            "final_obj": taylor_metrics["final_obj"],
            "sigma_mae": taylor_metrics["sigma_mae"],
            "sigma_max_abs": taylor_metrics["sigma_max_abs"],
            "num_directions": int(taylor_metrics["num_directions"]),
        },
        {
            "dim": dim,
            "tag": cfg["tag"],
            "method": "Wang",
            "runtime_sec": float(wang["runtime_sec"]),
            "iterations": int(wang["iterations"]),
            "final_obj": wang_metrics["final_obj"],
            "sigma_mae": wang_metrics["sigma_mae"],
            "sigma_max_abs": wang_metrics["sigma_max_abs"],
            "num_directions": int(wang_metrics["num_directions"]),
        },
        {
            "dim": dim,
            "tag": cfg["tag"],
            "method": "QA_baseline",
            "runtime_sec": float(qa["runtime_sec"]),
            "iterations": int(qa["iterations"]),
            "final_obj": qa_metrics["final_obj"],
            "sigma_mae": qa_metrics["sigma_mae"],
            "sigma_max_abs": qa_metrics["sigma_max_abs"],
            "num_directions": int(qa_metrics["num_directions"]),
        },
    ]
    return rows


def _encoding_ablation_for_dim(dim: int, cfg: dict) -> list[dict[str, object]]:
    v, h, tao = cfg["v"], cfg["h"], cfg["tao"]
    freq_max, M = cfg["freq_max"], cfg["M"]

    R_values = [6, 10, 12]
    fixed_points = [0, 1, 2]
    repeats = cfg["repeats"]

    rows: list[dict[str, object]] = []
    taylor = solve_taylor(M)

    for fp in fixed_points:
        for R in R_values:
            for rep in range(repeats):
                print(
                    f"[dim={dim}] encoding fp={fp}, R={R}, rep={rep + 1}/{repeats}"
                )
                settings = SolverSettings(
                    num_sample=100,
                    R=R,
                    fixed_point=fp,
                    inner_maxiter=3,
                    coding_mod="pn",
                    result_mod="lowest_energy",
                )
                qa = solve_quantum(
                    dim=dim,
                    v=v,
                    h=h,
                    tao=tao,
                    freq_max=freq_max,
                    M=M,
                    max_iter_num=cfg["max_iter_num"],
                    alpha=1.0,
                    alpha_decay=0.1,
                    beta=1.0,
                    epsilon=cfg["epsilon"],
                    coeff_init=taylor["coeff"],
                    settings=settings,
                )
                metrics = compute_metrics(
                    dim=dim,
                    linear_sys=qa["linear_sys"],
                    coeff=qa["coeff"],
                    v=v,
                    h=h,
                    tao=tao,
                )
                rows.append(
                    {
                        "dim": dim,
                        "tag": cfg["tag"],
                        "fixed_point": fp,
                        "R": R,
                        "qubit_count": M * R,
                        "repeat": rep,
                        "runtime_sec": float(qa["runtime_sec"]),
                        "iterations": int(qa["iterations"]),
                        "final_obj": metrics["final_obj"],
                        "sigma_mae": metrics["sigma_mae"],
                        "sigma_max_abs": metrics["sigma_max_abs"],
                    }
                )

    summary = aggregate_mean_std(
        rows,
        group_keys=["dim", "tag", "fixed_point", "R", "qubit_count"],
        value_keys=["runtime_sec", "iterations", "final_obj", "sigma_mae", "sigma_max_abs"],
    )
    write_csv(RESULTS_DIR / f"encoding_sensitivity_dim{dim}_raw.csv", rows)
    write_csv(RESULTS_DIR / f"encoding_sensitivity_dim{dim}_summary.csv", summary)

    by_fp_obj = defaultdict(list)
    for fp in fixed_points:
        fp_rows = [r for r in summary if int(r["fixed_point"]) == fp]
        fp_rows = sorted(fp_rows, key=lambda x: int(x["R"]))
        by_fp_obj[fp] = [float(r["final_obj_mean"]) for r in fp_rows]

    plot_line(
        x=R_values,
        ys=[(f"fixed_point={fp}", by_fp_obj[fp]) for fp in fixed_points],
        title=f"Dim {dim}: Encoding sensitivity (objective vs R)",
        x_label="Encoding bits R",
        y_label="Final objective (mean)",
        out_path=FIGURES_DIR / f"encoding_obj_vs_R_dim{dim}.png",
        log_y=True,
    )

    return rows


def _regularization_ablation_for_dim(dim: int, cfg: dict) -> list[dict[str, object]]:
    v, h, tao = cfg["v"], cfg["h"], cfg["tao"]
    freq_max, M = cfg["freq_max"], cfg["M"]

    strategies = {
        "no_reg": {"alpha": 0.0, "alpha_decay": 1.0},
        "standard": {"alpha": 1.0, "alpha_decay": 0.1},
        "adaptive": {"alpha": 1.0, "alpha_decay": 0.5},
    }

    rows: list[dict[str, object]] = []
    taylor = solve_taylor(M)
    repeats = cfg["repeats"]

    qa_settings = SolverSettings(
        num_sample=100,
        R=10,
        fixed_point=1,
        inner_maxiter=3,
        coding_mod="pn",
        result_mod="lowest_energy",
    )

    for sname, s in strategies.items():
        for rep in range(repeats):
            print(f"[dim={dim}] regularization={sname}, rep={rep + 1}/{repeats}")
            qa = solve_quantum(
                dim=dim,
                v=v,
                h=h,
                tao=tao,
                freq_max=freq_max,
                M=M,
                max_iter_num=cfg["max_iter_num"],
                alpha=s["alpha"],
                alpha_decay=s["alpha_decay"],
                beta=1.0,
                epsilon=cfg["epsilon"],
                coeff_init=taylor["coeff"],
                settings=qa_settings,
            )
            metrics = compute_metrics(
                dim=dim,
                linear_sys=qa["linear_sys"],
                coeff=qa["coeff"],
                v=v,
                h=h,
                tao=tao,
            )
            rows.append(
                {
                    "dim": dim,
                    "tag": cfg["tag"],
                    "strategy": sname,
                    "repeat": rep,
                    "runtime_sec": float(qa["runtime_sec"]),
                    "iterations": int(qa["iterations"]),
                    "final_obj": metrics["final_obj"],
                    "sigma_mae": metrics["sigma_mae"],
                    "sigma_max_abs": metrics["sigma_max_abs"],
                }
            )

    summary = aggregate_mean_std(
        rows,
        group_keys=["dim", "tag", "strategy"],
        value_keys=["runtime_sec", "iterations", "final_obj", "sigma_mae", "sigma_max_abs"],
    )
    write_csv(RESULTS_DIR / f"regularization_dim{dim}_raw.csv", rows)
    write_csv(RESULTS_DIR / f"regularization_dim{dim}_summary.csv", summary)

    return rows


def _hybrid_ablation_for_dim(dim: int, cfg: dict) -> list[dict[str, object]]:
    v, h, tao = cfg["v"], cfg["h"], cfg["tao"]
    freq_max, M = cfg["freq_max"], cfg["M"]

    total_iter_budget = cfg["max_iter_num"]
    warmstart_steps = [2, 5]

    rows: list[dict[str, object]] = []
    taylor = solve_taylor(M)

    qa_settings = SolverSettings(
        num_sample=100,
        R=10,
        fixed_point=1,
        inner_maxiter=3,
        coding_mod="pn",
        result_mod="lowest_energy",
    )

    pure_wang = solve_wang(
        dim=dim,
        v=v,
        h=h,
        tao=tao,
        freq_max=freq_max,
        M=M,
        max_iter_num=total_iter_budget,
        alpha=1.0,
        alpha_decay=0.1,
        beta=1.0,
        epsilon=cfg["epsilon"],
        coeff_init=taylor["coeff"],
    )
    pure_wang_metrics = compute_metrics(
        dim=dim,
        linear_sys=pure_wang["linear_sys"],
        coeff=pure_wang["coeff"],
        v=v,
        h=h,
        tao=tao,
    )
    rows.append(
        {
            "dim": dim,
            "tag": cfg["tag"],
            "method": "pure_wang",
            "warmstart_k": 0,
            "runtime_sec": float(pure_wang["runtime_sec"]),
            "iterations": int(pure_wang["iterations"]),
            "final_obj": pure_wang_metrics["final_obj"],
            "sigma_mae": pure_wang_metrics["sigma_mae"],
            "sigma_max_abs": pure_wang_metrics["sigma_max_abs"],
        }
    )

    pure_qa = solve_quantum(
        dim=dim,
        v=v,
        h=h,
        tao=tao,
        freq_max=freq_max,
        M=M,
        max_iter_num=total_iter_budget,
        alpha=1.0,
        alpha_decay=0.1,
        beta=1.0,
        epsilon=cfg["epsilon"],
        coeff_init=taylor["coeff"],
        settings=qa_settings,
    )
    pure_qa_metrics = compute_metrics(
        dim=dim,
        linear_sys=pure_qa["linear_sys"],
        coeff=pure_qa["coeff"],
        v=v,
        h=h,
        tao=tao,
    )
    rows.append(
        {
            "dim": dim,
            "tag": cfg["tag"],
            "method": "pure_qa",
            "warmstart_k": 0,
            "runtime_sec": float(pure_qa["runtime_sec"]),
            "iterations": int(pure_qa["iterations"]),
            "final_obj": pure_qa_metrics["final_obj"],
            "sigma_mae": pure_qa_metrics["sigma_mae"],
            "sigma_max_abs": pure_qa_metrics["sigma_max_abs"],
        }
    )

    for k in warmstart_steps:
        print(f"[dim={dim}] hybrid warmstart k={k}")
        warm = solve_wang(
            dim=dim,
            v=v,
            h=h,
            tao=tao,
            freq_max=freq_max,
            M=M,
            max_iter_num=k,
            alpha=1.0,
            alpha_decay=0.1,
            beta=1.0,
            epsilon=cfg["epsilon"],
            coeff_init=taylor["coeff"],
        )

        qa_refine = solve_quantum(
            dim=dim,
            v=v,
            h=h,
            tao=tao,
            freq_max=freq_max,
            M=M,
            max_iter_num=max(total_iter_budget - k, 1),
            alpha=1.0,
            alpha_decay=0.1,
            beta=1.0,
            epsilon=cfg["epsilon"],
            coeff_init=warm["coeff"],
            settings=qa_settings,
        )
        metrics = compute_metrics(
            dim=dim,
            linear_sys=qa_refine["linear_sys"],
            coeff=qa_refine["coeff"],
            v=v,
            h=h,
            tao=tao,
        )
        rows.append(
            {
                "dim": dim,
                "tag": cfg["tag"],
                "method": f"hybrid_k{k}",
                "warmstart_k": k,
                "runtime_sec": float(warm["runtime_sec"] + qa_refine["runtime_sec"]),
                "iterations": int(warm["iterations"] + qa_refine["iterations"]),
                "final_obj": metrics["final_obj"],
                "sigma_mae": metrics["sigma_mae"],
                "sigma_max_abs": metrics["sigma_max_abs"],
            }
        )

    write_csv(RESULTS_DIR / f"hybrid_dim{dim}.csv", rows)
    return rows


def _progressive_encoding_novelty_for_dim(dim: int, cfg: dict) -> list[dict[str, object]]:
    """Novelty: progressive encoding continuation (R: 6 -> 8 -> 12)."""

    v, h, tao = cfg["v"], cfg["h"], cfg["tao"]
    freq_max, M = cfg["freq_max"], cfg["M"]

    rows: list[dict[str, object]] = []
    taylor = solve_taylor(M)

    # Baseline high-resolution single-stage QA.
    fixed_settings = SolverSettings(
        num_sample=100,
        R=12,
        fixed_point=2,
        inner_maxiter=3,
        coding_mod="pn",
        result_mod="lowest_energy",
    )
    fixed = solve_quantum(
        dim=dim,
        v=v,
        h=h,
        tao=tao,
        freq_max=freq_max,
        M=M,
        max_iter_num=cfg["max_iter_num"],
        alpha=1.0,
        alpha_decay=0.1,
        beta=1.0,
        epsilon=cfg["epsilon"],
        coeff_init=taylor["coeff"],
        settings=fixed_settings,
    )
    fixed_metrics = compute_metrics(
        dim=dim,
        linear_sys=fixed["linear_sys"],
        coeff=fixed["coeff"],
        v=v,
        h=h,
        tao=tao,
    )

    rows.append(
        {
            "dim": dim,
            "tag": cfg["tag"],
            "method": "fixed_R12",
            "runtime_sec": float(fixed["runtime_sec"]),
            "iterations": int(fixed["iterations"]),
            "final_obj": fixed_metrics["final_obj"],
            "sigma_mae": fixed_metrics["sigma_mae"],
            "sigma_max_abs": fixed_metrics["sigma_max_abs"],
        }
    )

    # Progressive schedule with the same total big-loop iterations.
    schedule = [(6, 7), (8, 7), (12, 6)]
    coeff_cur = taylor["coeff"]
    total_runtime = 0.0
    total_iter = 0
    qa_last = None

    for r_val, iters in schedule:
        print(f"[dim={dim}] progressive stage R={r_val}, iters={iters}")
        stage_settings = SolverSettings(
            num_sample=100,
            R=r_val,
            fixed_point=2,
            inner_maxiter=3,
            coding_mod="pn",
            result_mod="lowest_energy",
        )
        qa_stage = solve_quantum(
            dim=dim,
            v=v,
            h=h,
            tao=tao,
            freq_max=freq_max,
            M=M,
            max_iter_num=iters,
            alpha=1.0,
            alpha_decay=0.1,
            beta=1.0,
            epsilon=cfg["epsilon"],
            coeff_init=coeff_cur,
            settings=stage_settings,
        )
        coeff_cur = qa_stage["coeff"]
        total_runtime += float(qa_stage["runtime_sec"])
        total_iter += int(qa_stage["iterations"])
        qa_last = qa_stage

    prog_metrics = compute_metrics(
        dim=dim,
        linear_sys=qa_last["linear_sys"],
        coeff=coeff_cur,
        v=v,
        h=h,
        tao=tao,
    )

    rows.append(
        {
            "dim": dim,
            "tag": cfg["tag"],
            "method": "progressive_R6_8_12",
            "runtime_sec": total_runtime,
            "iterations": total_iter,
            "final_obj": prog_metrics["final_obj"],
            "sigma_mae": prog_metrics["sigma_mae"],
            "sigma_max_abs": prog_metrics["sigma_max_abs"],
        }
    )

    write_csv(RESULTS_DIR / f"novelty_progressive_dim{dim}.csv", rows)

    plot_bar(
        categories=[r["method"] for r in rows],
        values=[float(r["final_obj"]) for r in rows],
        title=f"Dim {dim}: Novelty objective comparison",
        y_label="Final objective",
        out_path=FIGURES_DIR / f"novelty_progressive_obj_dim{dim}.png",
    )
    plot_bar(
        categories=[r["method"] for r in rows],
        values=[float(r["runtime_sec"]) for r in rows],
        title=f"Dim {dim}: Novelty runtime comparison",
        y_label="Runtime (s)",
        out_path=FIGURES_DIR / f"novelty_progressive_runtime_dim{dim}.png",
    )

    return rows


def run_for_dimension(dim: int) -> None:
    cfg = CONFIGS[dim]
    baseline_rows = _baseline_for_dim(dim, cfg)
    encoding_rows = _encoding_ablation_for_dim(dim, cfg)
    reg_rows = _regularization_ablation_for_dim(dim, cfg)
    hybrid_rows = _hybrid_ablation_for_dim(dim, cfg)
    novelty_rows = _progressive_encoding_novelty_for_dim(dim, cfg)

    write_csv(RESULTS_DIR / f"multidim_baseline_dim{dim}.csv", baseline_rows)

    # Combined compact table for quick discussion/presentation.
    combined = []
    combined.extend([{"experiment": "baseline", **r} for r in baseline_rows])
    combined.extend([{"experiment": "encoding", **r} for r in encoding_rows])
    combined.extend([{"experiment": "regularization", **r} for r in reg_rows])
    combined.extend([{"experiment": "hybrid", **r} for r in hybrid_rows])
    combined.extend([{"experiment": "novelty", **r} for r in novelty_rows])

    unified_keys = sorted({k for row in combined for k in row.keys()})
    normalized = [{k: row.get(k, "") for k in unified_keys} for row in combined]
    write_csv(RESULTS_DIR / f"multidim_all_dim{dim}.csv", normalized)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dims",
        nargs="+",
        type=int,
        default=[2, 3],
        help="Dimensions to run (2 and/or 3)",
    )
    args = parser.parse_args()

    for d in args.dims:
        if d not in CONFIGS:
            raise ValueError(f"Unsupported dim: {d}. Valid: 2, 3")
        run_for_dimension(d)
