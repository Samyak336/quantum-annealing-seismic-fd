#!/usr/bin/env python
"""Novelty study: QA exploration followed by Wang polishing for stability."""

from __future__ import annotations

from experiments_common import (
    FIGURES_DIR,
    RESULTS_DIR,
    SolverSettings,
    aggregate_mean_std,
    compute_metrics,
    plot_bar,
    solve_quantum,
    solve_taylor,
    solve_wang,
    write_csv,
)


CONFIGS = {
    1: {
        "tag": "1d_ref",
        "v": 4500,
        "h": 10,
        "tao": 0.001,
        "freq_max": 174,
        "M": 8,
        "epsilon": 1e-8,
        "repeat": 3,
    },
    2: {
        "tag": "2d_ref",
        "v": 4500,
        "h": 20,
        "tao": 0.001,
        "freq_max": 90,
        "M": 10,
        "epsilon": 1e-10,
        "repeat": 3,
    },
    3: {
        "tag": "3d_ref",
        "v": 2500,
        "h": 10,
        "tao": 0.001,
        "freq_max": 100,
        "M": 10,
        "epsilon": 1e-10,
        "repeat": 3,
    },
}


def _eval(dim: int, cfg: dict, run_result):
    return compute_metrics(
        dim=dim,
        linear_sys=run_result["linear_sys"],
        coeff=run_result["coeff"],
        v=cfg["v"],
        h=cfg["h"],
        tao=cfg["tao"],
    )


def run_stabilization_novelty() -> None:
    qa_settings = SolverSettings(
        num_sample=100,
        R=10,
        fixed_point=1,
        inner_maxiter=3,
        coding_mod="pn",
        result_mod="lowest_energy",
    )

    rows = []

    for dim, cfg in CONFIGS.items():
        qa_alpha = 1.0  # Paper uses alpha=1.0 for ALL dims, alpha_decay=0.1
        max_iter = 30 if dim == 3 else 20
        qa_coarse_iter = 12 if dim == 3 else 8
        wang_polish_iter = max_iter - qa_coarse_iter
        for rep in range(cfg["repeat"]):
            print(f"[dim={dim}] repetition {rep + 1}/{cfg['repeat']}")
            taylor = solve_taylor(cfg["M"])

            # Method A: pure QA for full budget.
            qa_only = solve_quantum(
                dim=dim,
                v=cfg["v"],
                h=cfg["h"],
                tao=cfg["tao"],
                freq_max=cfg["freq_max"],
                M=cfg["M"],
                max_iter_num=max_iter,
                alpha=qa_alpha,
                alpha_decay=0.1,
                beta=1.0,
                epsilon=cfg["epsilon"],
                coeff_init=taylor["coeff"],
                settings=qa_settings,
            )
            qa_only_m = _eval(dim, cfg, qa_only)
            rows.append(
                {
                    "dim": dim,
                    "tag": cfg["tag"],
                    "repeat": rep,
                    "method": "qa_only",
                    "runtime_sec": float(qa_only["runtime_sec"]),
                    "iterations": int(qa_only["iterations"]),
                    "final_obj": qa_only_m["final_obj"],
                    "sigma_mae": qa_only_m["sigma_mae"],
                    "sigma_max_abs": qa_only_m["sigma_max_abs"],
                }
            )

            # Method B (novel): QA coarse exploration then deterministic Wang polish.
            qa_coarse = solve_quantum(
                dim=dim,
                v=cfg["v"],
                h=cfg["h"],
                tao=cfg["tao"],
                freq_max=cfg["freq_max"],
                M=cfg["M"],
                max_iter_num=qa_coarse_iter,
                alpha=qa_alpha,
                alpha_decay=0.1,
                beta=1.0,
                epsilon=cfg["epsilon"],
                coeff_init=taylor["coeff"],
                settings=qa_settings,
            )
            wang_polish = solve_wang(
                dim=dim,
                v=cfg["v"],
                h=cfg["h"],
                tao=cfg["tao"],
                freq_max=cfg["freq_max"],
                M=cfg["M"],
                max_iter_num=wang_polish_iter,
                alpha=1.0,
                alpha_decay=0.1,
                beta=1.0,
                epsilon=cfg["epsilon"],
                coeff_init=qa_coarse["coeff"],
            )
            qaw_m = _eval(dim, cfg, wang_polish)
            rows.append(
                {
                    "dim": dim,
                    "tag": cfg["tag"],
                    "repeat": rep,
                    "method": "qa_to_wang",
                    "runtime_sec": float(qa_coarse["runtime_sec"] + wang_polish["runtime_sec"]),
                    "iterations": int(qa_coarse["iterations"] + wang_polish["iterations"]),
                    "final_obj": qaw_m["final_obj"],
                    "sigma_mae": qaw_m["sigma_mae"],
                    "sigma_max_abs": qaw_m["sigma_max_abs"],
                }
            )

            # Method C: deterministic Wang only.
            wang_only = solve_wang(
                dim=dim,
                v=cfg["v"],
                h=cfg["h"],
                tao=cfg["tao"],
                freq_max=cfg["freq_max"],
                M=cfg["M"],
                max_iter_num=max_iter,
                alpha=1.0,
                alpha_decay=0.1,
                beta=1.0,
                epsilon=cfg["epsilon"],
                coeff_init=taylor["coeff"],
            )
            wang_only_m = _eval(dim, cfg, wang_only)
            rows.append(
                {
                    "dim": dim,
                    "tag": cfg["tag"],
                    "repeat": rep,
                    "method": "wang_only",
                    "runtime_sec": float(wang_only["runtime_sec"]),
                    "iterations": int(wang_only["iterations"]),
                    "final_obj": wang_only_m["final_obj"],
                    "sigma_mae": wang_only_m["sigma_mae"],
                    "sigma_max_abs": wang_only_m["sigma_max_abs"],
                }
            )

    raw_path = RESULTS_DIR / "novelty_qa_to_wang_raw.csv"
    write_csv(raw_path, rows)

    summary = aggregate_mean_std(
        rows,
        group_keys=["dim", "tag", "method"],
        value_keys=["runtime_sec", "iterations", "final_obj", "sigma_mae", "sigma_max_abs"],
    )
    summary_path = RESULTS_DIR / "novelty_qa_to_wang_summary.csv"
    write_csv(summary_path, summary)

    # Quick plots by dimension.
    for dim in CONFIGS:
        dim_rows = [r for r in summary if int(r["dim"]) == dim]
        order = ["qa_only", "qa_to_wang", "wang_only"]
        dim_rows = sorted(dim_rows, key=lambda r: order.index(r["method"]))

        plot_bar(
            categories=[r["method"] for r in dim_rows],
            values=[float(r["final_obj_mean"]) for r in dim_rows],
            title=f"Dim {dim}: QA->Wang stability novelty (objective)",
            y_label="Final objective mean",
            out_path=FIGURES_DIR / f"novelty_qaw_obj_dim{dim}.png",
        )
        plot_bar(
            categories=[r["method"] for r in dim_rows],
            values=[float(r["runtime_sec_mean"]) for r in dim_rows],
            title=f"Dim {dim}: QA->Wang stability novelty (runtime)",
            y_label="Runtime mean (s)",
            out_path=FIGURES_DIR / f"novelty_qaw_runtime_dim{dim}.png",
        )

    print(f"Saved raw novelty results: {raw_path}")
    print(f"Saved novelty summary: {summary_path}")


if __name__ == "__main__":
    run_stabilization_novelty()
