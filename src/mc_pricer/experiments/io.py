"""
I/O utilities for saving and loading experiment results.
"""

import json
from pathlib import Path

from mc_pricer.experiments.types import ExperimentResult


def save_results(
    results: list[ExperimentResult],
    out_dir: Path,
    experiment_name: str
) -> None:
    """
    Save experiment results to JSON and summary text files.

    Creates:
    - results.json: Full machine-readable results
    - summary.txt: Human-readable table summary

    Parameters
    ----------
    results : list[ExperimentResult]
        Experiment results to save
    out_dir : Path
        Output directory
    experiment_name : str
        Name of experiment for headers
    """
    # Create output directory
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save JSON
    json_path = out_dir / "results.json"
    json_data = {
        "experiment_name": experiment_name,
        "n_results": len(results),
        "results": [r.to_dict() for r in results]
    }

    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)

    # Create summary table
    summary_path = out_dir / "summary.txt"
    with open(summary_path, "w") as f:
        f.write("=" * 120 + "\n")
        f.write(f"Experiment: {experiment_name}\n")
        f.write("=" * 120 + "\n")
        f.write(f"\nTotal runs: {len(results)}\n")

        if results:
            # Show metadata from first result
            meta = results[0].metadata
            f.write("\nMetadata:\n")
            f.write(f"  Timestamp:      {meta.timestamp}\n")
            f.write(f"  Python:         {meta.python_version}\n")
            f.write(f"  NumPy:          {meta.numpy_version}\n")
            f.write(f"  Platform:       {meta.os_platform}\n")
            f.write(f"  Git commit:     {meta.git_commit or 'N/A'}\n")
            f.write(f"  Model:          {meta.model}\n")
            f.write(f"  Option:         {meta.option_type} {meta.style}\n")

        f.write("\n" + "-" * 120 + "\n")
        f.write(f"{'Method':<25} {'n_paths':>10} {'n_steps':>10} {'Price':>12} "
                f"{'Stderr':>12} {'CI Width':>12} {'Rel Err %':>10} {'Runtime (s)':>12}\n")
        f.write("-" * 120 + "\n")

        for r in results:
            n_steps_str = str(r.n_steps) if r.n_steps else "N/A"
            f.write(f"{r.notes:<25} {r.n_paths:>10} {n_steps_str:>10} {r.price:>12.6f} "
                    f"{r.stderr:>12.6f} {r.ci_width:>12.6f} {r.relative_error * 100:>10.4f} "
                    f"{r.runtime_seconds:>12.3f}\n")

        f.write("-" * 120 + "\n")

        # Aggregate statistics by method
        f.write("\nAggregate Statistics by Method:\n")
        f.write("-" * 120 + "\n")
        f.write(f"{'Method':<25} {'Count':>10} {'Mean Price':>12} {'Mean Stderr':>12} "
                f"{'Mean CI Width':>14} {'Mean Runtime':>14}\n")
        f.write("-" * 120 + "\n")

        # Group by notes (method)
        method_groups: dict[str, list[ExperimentResult]] = {}
        for r in results:
            if r.notes not in method_groups:
                method_groups[r.notes] = []
            method_groups[r.notes].append(r)

        for method, group in sorted(method_groups.items()):
            count = len(group)
            mean_price = sum(r.price for r in group) / count
            mean_stderr = sum(r.stderr for r in group) / count
            mean_ci_width = sum(r.ci_width for r in group) / count
            mean_runtime = sum(r.runtime_seconds for r in group) / count

            f.write(f"{method:<25} {count:>10} {mean_price:>12.6f} {mean_stderr:>12.6f} "
                    f"{mean_ci_width:>14.6f} {mean_runtime:>14.3f}\n")

        f.write("-" * 120 + "\n")

        # Add paper-ready table if multiple methods
        if len(method_groups) > 1:
            f.write("\n" + "=" * 80 + "\n")
            f.write("PAPER TABLE (Mean ± Stderr [95% CI])\n")
            f.write("=" * 80 + "\n")
            f.write(f"{'Method':<30} {'Price (Mean ± SE)':<35} {'95% CI':<20}\n")
            f.write("-" * 80 + "\n")

            for method, group in sorted(method_groups.items()):
                mean_price = sum(r.price for r in group) / len(group)
                mean_stderr = sum(r.stderr for r in group) / len(group)
                mean_ci_lower = sum(r.ci_lower for r in group) / len(group)
                mean_ci_upper = sum(r.ci_upper for r in group) / len(group)

                f.write(f"{method:<30} {mean_price:.6f} ± {mean_stderr:.6f}    "
                        f"[{mean_ci_lower:.6f}, {mean_ci_upper:.6f}]\n")

            f.write("=" * 80 + "\n")

    print(f"\n✓ Results saved to {out_dir}")
    print(f"  - {json_path.name}")
    print(f"  - {summary_path.name}")


def load_results(results_dir: Path) -> dict:
    """
    Load experiment results from JSON file.

    Parameters
    ----------
    results_dir : Path
        Directory containing results.json

    Returns
    -------
    dict
        Loaded experiment data
    """
    json_path = results_dir / "results.json"
    with open(json_path) as f:
        return json.load(f)
