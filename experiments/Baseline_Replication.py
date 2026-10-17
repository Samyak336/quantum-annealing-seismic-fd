#!/usr/bin/env python
"""Replicate core 1D results from the authors' setup (Taylor vs Wang vs QA-SA)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from experiments_common import (
    FIGURES_DIR,
    RESULTS_DIR,
    SolverSettings,
    compute_metrics_1d,
    plot_dispersion,
    solve_quantum_1d,
    solve_taylor,
    solve_wang_1d,
    write_csv,
)


def run_baseline() -> None:
    baseline_cases = [
        {
            "case_id": "v1500_lowfreq",
            "v": 1500,
            "h": 10,
            "tao": 0.001,
            "freq_max_set": [39, 51, 57],
            "M_set": [4, 6, 8],
            "max_iter_num": 20,
            "epsilon": 1e-8,
        },
        {
            "case_id": "v4500_midfreq",
            "v": 4500,
            "h": 10,
            "tao": 0.001,
            "freq_max_set": [120, 156, 174],
            "M_set": [4, 6, 8],
            "max_iter_num": 20,
            "epsilon": 1e-8,
        },
    ]

    qa_settings = SolverSettings(
        num_sample=100,
        R=10,
        fixed_point=1,
        inner_maxiter=3,
        coding_mod="pn",
        result_mod="lowest_energy",
    )

    rows = []

    for case in baseline_cases:
        v = case["v"]
        h = case["h"]
        tao = case["tao"]

        for M, freq_max in zip(case["M_set"], case["freq_max_set"]):
            print(f"\n=== Baseline case={case['case_id']} M={M} freq_max={freq_max} ===")

            taylor = solve_taylor(M)

            wang = solve_wang_1d(
                v=v,
                h=h,
                tao=tao,
                freq_max=freq_max,
                M=M,
                max_iter_num=case["max_iter_num"],
                alpha=1.0,
                alpha_decay=0.1,
                beta=1.0,
                epsilon=case["epsilon"],
                coeff_init=taylor["coeff"],
            )

            qa = solve_quantum_1d(
                v=v,
                h=h,
                tao=tao,
                freq_max=freq_max,
                M=M,
                max_iter_num=case["max_iter_num"],
                alpha=1.0,
                alpha_decay=0.1,
                beta=1.0,
                epsilon=case["epsilon"],
                coeff_init=taylor["coeff"],
                settings=qa_settings,
            )

            taylor_metrics = compute_metrics_1d(
                linear_sys=wang["linear_sys"],
                coeff=taylor["coeff"],
                v=v,
                h=h,
                tao=tao,
            )
            wang_metrics = compute_metrics_1d(
                linear_sys=wang["linear_sys"],
                coeff=wang["coeff"],
                v=v,
                h=h,
                tao=tao,
            )
            qa_metrics = compute_metrics_1d(
                linear_sys=qa["linear_sys"],
                coeff=qa["coeff"],
                v=v,
                h=h,
                tao=tao,
            )

            rows.extend(
                [
                    {
                        "case_id": case["case_id"],
                        "method": "Taylor",
                        "v": v,
                        "h": h,
                        "tao": tao,
                        "freq_max": freq_max,
                        "M": M,
                        "runtime_sec": float(taylor["runtime_sec"]),
                        "final_obj": taylor_metrics["final_obj"],
                        "sigma_mae": taylor_metrics["sigma_mae"],
                        "sigma_max_abs": taylor_metrics["sigma_max_abs"],
                        "iterations": 1,
                        "num_sample": 0,
                        "R": 0,
                        "fixed_point": 0,
                    },
                    {
                        "case_id": case["case_id"],
                        "method": "Wang",
                        "v": v,
                        "h": h,
                        "tao": tao,
                        "freq_max": freq_max,
                        "M": M,
                        "runtime_sec": float(wang["runtime_sec"]),
                        "final_obj": wang_metrics["final_obj"],
                        "sigma_mae": wang_metrics["sigma_mae"],
                        "sigma_max_abs": wang_metrics["sigma_max_abs"],
                        "iterations": int(wang["iterations"]),
                        "num_sample": 0,
                        "R": 0,
                        "fixed_point": 0,
                    },
                    {
                        "case_id": case["case_id"],
                        "method": "QA-SA",
                        "v": v,
                        "h": h,
                        "tao": tao,
                        "freq_max": freq_max,
                        "M": M,
                        "runtime_sec": float(qa["runtime_sec"]),
                        "final_obj": qa_metrics["final_obj"],
                        "sigma_mae": qa_metrics["sigma_mae"],
                        "sigma_max_abs": qa_metrics["sigma_max_abs"],
                        "iterations": int(qa["iterations"]),
                        "num_sample": qa_settings.num_sample,
                        "R": qa_settings.R,
                        "fixed_point": qa_settings.fixed_point,
                    },
                ]
            )

            plot_dispersion(
                kh=taylor_metrics["kh"],
                curves=[
                    ("Taylor", np.asarray(taylor_metrics["sigma"])),
                    ("Wang", np.asarray(wang_metrics["sigma"])),
                    ("QA-SA", np.asarray(qa_metrics["sigma"])),
                ],
                title=f"{case['case_id']} M={M} freq={freq_max}",
                out_path=FIGURES_DIR / f"baseline_disp_{case['case_id']}_M{M}.png",
            )

            print(
                f"Taylor obj={taylor_metrics['final_obj']:.3e}, "
                f"Wang obj={wang_metrics['final_obj']:.3e}, "
                f"QA obj={qa_metrics['final_obj']:.3e}"
            )

    out_csv = RESULTS_DIR / "baseline_metrics.csv"
    write_csv(out_csv, rows)
    print(f"\nSaved baseline metrics: {out_csv}")


if __name__ == "__main__":
    run_baseline()
