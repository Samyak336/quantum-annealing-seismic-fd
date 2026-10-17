#!/usr/bin/env python
"""Novel study: classical Wang warm-start before QA refinement."""

from __future__ import annotations

from experiments_common import (
    FIGURES_DIR,
    RESULTS_DIR,
    SolverSettings,
    compute_metrics_1d,
    plot_bar,
    solve_quantum_1d,
    solve_taylor,
    solve_wang_1d,
    write_csv,
)


def run_hybrid_study() -> None:
    v = 4500
    h = 10
    tao = 0.001
    freq_max = 174
    M = 8
    epsilon = 1e-8

    total_iter_budget = 20
    warmstart_steps = [1, 2, 3, 5]

    qa_settings = SolverSettings(
        num_sample=100,
        R=10,
        fixed_point=1,
        inner_maxiter=3,
        coding_mod="pn",
        result_mod="lowest_energy",
    )

    rows = []
    taylor = solve_taylor(M)

    pure_wang = solve_wang_1d(
        v=v,
        h=h,
        tao=tao,
        freq_max=freq_max,
        M=M,
        max_iter_num=total_iter_budget,
        alpha=1.0,
        alpha_decay=0.1,
        beta=1.0,
        epsilon=epsilon,
        coeff_init=taylor["coeff"],
    )
    pure_wang_metrics = compute_metrics_1d(
        linear_sys=pure_wang["linear_sys"],
        coeff=pure_wang["coeff"],
        v=v,
        h=h,
        tao=tao,
    )
    rows.append(
        {
            "method": "pure_wang",
            "warmstart_k": 0,
            "runtime_sec": float(pure_wang["runtime_sec"]),
            "iterations": int(pure_wang["iterations"]),
            "final_obj": pure_wang_metrics["final_obj"],
            "sigma_mae": pure_wang_metrics["sigma_mae"],
            "sigma_max_abs": pure_wang_metrics["sigma_max_abs"],
        }
    )

    pure_qa = solve_quantum_1d(
        v=v,
        h=h,
        tao=tao,
        freq_max=freq_max,
        M=M,
        max_iter_num=total_iter_budget,
        alpha=1.0,
        alpha_decay=0.1,
        beta=1.0,
        epsilon=epsilon,
        coeff_init=taylor["coeff"],
        settings=qa_settings,
    )
    pure_qa_metrics = compute_metrics_1d(
        linear_sys=pure_qa["linear_sys"],
        coeff=pure_qa["coeff"],
        v=v,
        h=h,
        tao=tao,
    )
    rows.append(
        {
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
        print(f"Hybrid study: warmstart k={k}")
        warm = solve_wang_1d(
            v=v,
            h=h,
            tao=tao,
            freq_max=freq_max,
            M=M,
            max_iter_num=k,
            alpha=1.0,
            alpha_decay=0.1,
            beta=1.0,
            epsilon=epsilon,
            coeff_init=taylor["coeff"],
        )

        qa_refine = solve_quantum_1d(
            v=v,
            h=h,
            tao=tao,
            freq_max=freq_max,
            M=M,
            max_iter_num=max(total_iter_budget - k, 1),
            alpha=1.0,
            alpha_decay=0.1,
            beta=1.0,
            epsilon=epsilon,
            coeff_init=warm["coeff"],
            settings=qa_settings,
        )

        metrics = compute_metrics_1d(
            linear_sys=qa_refine["linear_sys"],
            coeff=qa_refine["coeff"],
            v=v,
            h=h,
            tao=tao,
        )

        rows.append(
            {
                "method": f"hybrid_k{k}",
                "warmstart_k": k,
                "runtime_sec": float(warm["runtime_sec"] + qa_refine["runtime_sec"]),
                "iterations": int(warm["iterations"] + qa_refine["iterations"]),
                "final_obj": metrics["final_obj"],
                "sigma_mae": metrics["sigma_mae"],
                "sigma_max_abs": metrics["sigma_max_abs"],
            }
        )

    out_csv = RESULTS_DIR / "hybrid_classical_quantum.csv"
    write_csv(out_csv, rows)

    categories = [r["method"] for r in rows]
    plot_bar(
        categories=categories,
        values=[float(r["final_obj"]) for r in rows],
        title="Hybrid study: final objective",
        y_label="Final objective",
        out_path=FIGURES_DIR / "hybrid_final_obj_bar.png",
    )
    plot_bar(
        categories=categories,
        values=[float(r["runtime_sec"]) for r in rows],
        title="Hybrid study: total runtime",
        y_label="Runtime (s)",
        out_path=FIGURES_DIR / "hybrid_runtime_bar.png",
    )

    print(f"Saved hybrid results: {out_csv}")


if __name__ == "__main__":
    run_hybrid_study()
