#!/usr/bin/env python
"""Ablation study: sensitivity to encoding bits (R) and fixed-point location."""

from __future__ import annotations

from collections import defaultdict

from experiments_common import (
    FIGURES_DIR,
    RESULTS_DIR,
    SolverSettings,
    aggregate_mean_std,
    compute_metrics_1d,
    plot_line,
    solve_quantum_1d,
    solve_taylor,
    write_csv,
)


def run_encoding_sensitivity() -> None:
    v = 4500
    h = 10
    tao = 0.001
    freq_max = 174
    M = 8
    max_iter_num = 20
    epsilon = 1e-8

    R_values = [4, 6, 8, 10, 12, 14]
    fixed_points = [0, 1, 2]
    repeats = 2

    taylor = solve_taylor(M)

    rows = []
    for fixed_point in fixed_points:
        for R in R_values:
            for rep in range(repeats):
                print(
                    f"Encoding study: fixed_point={fixed_point}, R={R}, repeat={rep + 1}/{repeats}"
                )
                settings = SolverSettings(
                    num_sample=100,
                    R=R,
                    fixed_point=fixed_point,
                    inner_maxiter=3,
                    coding_mod="pn",
                    result_mod="lowest_energy",
                )
                qa = solve_quantum_1d(
                    v=v,
                    h=h,
                    tao=tao,
                    freq_max=freq_max,
                    M=M,
                    max_iter_num=max_iter_num,
                    alpha=1.0,
                    alpha_decay=0.1,
                    beta=1.0,
                    epsilon=epsilon,
                    coeff_init=taylor["coeff"],
                    settings=settings,
                )
                metrics = compute_metrics_1d(
                    linear_sys=qa["linear_sys"], coeff=qa["coeff"], v=v, h=h, tao=tao
                )
                rows.append(
                    {
                        "fixed_point": fixed_point,
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

    raw_csv = RESULTS_DIR / "encoding_sensitivity_raw.csv"
    write_csv(raw_csv, rows)

    summary = aggregate_mean_std(
        rows,
        group_keys=["fixed_point", "R", "qubit_count"],
        value_keys=["runtime_sec", "iterations", "final_obj", "sigma_mae", "sigma_max_abs"],
    )
    summary_csv = RESULTS_DIR / "encoding_sensitivity_summary.csv"
    write_csv(summary_csv, summary)

    by_fp_obj = defaultdict(list)
    by_fp_iter = defaultdict(list)
    for fp in fixed_points:
        fp_rows = [r for r in summary if int(r["fixed_point"]) == fp]
        fp_rows = sorted(fp_rows, key=lambda x: int(x["R"]))
        by_fp_obj[fp] = [float(r["final_obj_mean"]) for r in fp_rows]
        by_fp_iter[fp] = [float(r["iterations_mean"]) for r in fp_rows]

    plot_line(
        x=R_values,
        ys=[(f"fixed_point={fp}", by_fp_obj[fp]) for fp in fixed_points],
        title="Encoding Sensitivity: final objective vs R",
        x_label="Encoding bits R",
        y_label="Final objective (mean)",
        out_path=FIGURES_DIR / "encoding_obj_vs_R.png",
    )

    plot_line(
        x=R_values,
        ys=[(f"fixed_point={fp}", by_fp_iter[fp]) for fp in fixed_points],
        title="Encoding Sensitivity: iterations vs R",
        x_label="Encoding bits R",
        y_label="Outer iterations (mean)",
        out_path=FIGURES_DIR / "encoding_iters_vs_R.png",
    )

    print(f"Saved encoding raw results: {raw_csv}")
    print(f"Saved encoding summary: {summary_csv}")


if __name__ == "__main__":
    run_encoding_sensitivity()
