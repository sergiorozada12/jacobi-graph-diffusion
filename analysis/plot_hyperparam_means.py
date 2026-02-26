#!/usr/bin/env python3
import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median, pstdev

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot mean ± std objective for each hyperparameter value in a tuning JSON file."
    )
    parser.add_argument("json_path", type=Path, help="Path to tuning results JSON.")
    parser.add_argument(
        "--objective-key",
        type=str,
        default="tree_acc",
        help="Objective key under top-level 'objectives' (default: tree_acc).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where plots are saved (default: analysis/<json_stem>_mean_std_plots).",
    )
    return parser.parse_args()


def value_sort_key(value):
    if isinstance(value, bool):
        return (0, int(value))
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return (1, str(value))
        return (1, float(value))
    return (2, str(value))


def value_label(value) -> str:
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def load_trials(json_path: Path, objective_key: str):
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    trials = data["objectives"][objective_key]["trials"]
    rows = []
    for trial in trials.values():
        score = trial.get("objective")
        try:
            score = float(score) if score is not None else None
        except (TypeError, ValueError):
            score = None
        rows.append(
            {
                "status": trial.get("status", "unknown"),
                "params": trial.get("params", {}),
                "objective": score,
                "metrics": trial.get("metrics", {}) or {},
                "run_id": trial.get("run_id", ""),
                "seed": trial.get("seed", ""),
                "trial_index": trial.get("trial_index", ""),
                "timestamp": trial.get("timestamp", ""),
            }
        )
    return rows


def build_stats(rows):
    param_names = sorted(rows[0]["params"].keys())
    stats = {}

    for param in param_names:
        grouped = defaultdict(list)
        for row in rows:
            grouped[row["params"][param]].append(row["objective"])

        values = sorted(grouped.keys(), key=value_sort_key)
        stats[param] = {
            "values": values,
            "means": [mean(grouped[v]) for v in values],
            "stds": [pstdev(grouped[v]) if len(grouped[v]) > 1 else 0.0 for v in values],
            "counts": [len(grouped[v]) for v in values],
            "mins": [min(grouped[v]) for v in values],
            "maxs": [max(grouped[v]) for v in values],
        }

    return stats


def write_summary_json(all_trials: list, ok_rows: list, stats: dict, objective_key: str, output_path: Path):
    status_counts = Counter(row["status"] for row in all_trials)
    total_trials = len(all_trials)
    ok_trials = status_counts.get("ok", 0)
    unsuccessful_trials = total_trials - ok_trials
    ok_missing_objective = sum(1 for row in all_trials if row["status"] == "ok" and row["objective"] is None)

    objectives = [row["objective"] for row in ok_rows]
    best_value = max(row["objective"] for row in ok_rows)
    worst_value = min(row["objective"] for row in ok_rows)

    best_rows = [row for row in ok_rows if math.isclose(row["objective"], best_value, rel_tol=0.0, abs_tol=1e-12)]
    worst_rows = [row for row in ok_rows if math.isclose(row["objective"], worst_value, rel_tol=0.0, abs_tol=1e-12)]

    def _as_float(value, default):
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def best_tie_key(row: dict):
        metrics = row.get("metrics", {})
        frac_valid = _as_float(metrics.get("sampling/frac_unic_non_iso_valid"), float("-inf"))
        average_ratio = _as_float(metrics.get("average_ratio"), float("inf"))
        # Maximize frac_valid, then minimize average_ratio.
        return (frac_valid, -average_ratio)

    def worst_tie_key(row: dict):
        metrics = row.get("metrics", {})
        frac_valid = _as_float(metrics.get("sampling/frac_unic_non_iso_valid"), float("inf"))
        average_ratio = _as_float(metrics.get("average_ratio"), float("-inf"))
        # Minimize frac_valid, then maximize average_ratio.
        return (-frac_valid, average_ratio)

    best_representative = max(best_rows, key=best_tie_key)
    worst_representative = max(worst_rows, key=worst_tie_key)

    def serialize_trial(row: dict) -> dict:
        return {
            "objective": row["objective"],
            "run_id": row["run_id"],
            "seed": row["seed"],
            "trial_index": row["trial_index"],
            "timestamp": row["timestamp"],
            "params": row["params"],
            "metrics": row.get("metrics", {}),
        }

    param_value_stats = []
    for param_name, param_stats in stats.items():
        for value, count, mu, sigma, vmin, vmax in zip(
            param_stats["values"],
            param_stats["counts"],
            param_stats["means"],
            param_stats["stds"],
            param_stats["mins"],
            param_stats["maxs"],
        ):
            param_value_stats.append(
                {
                    "parameter": param_name,
                    "value": value_label(value),
                    "count": count,
                    "mean": mu,
                    "std": sigma,
                    "min": vmin,
                    "max": vmax,
                }
            )

    summary = {
        "objective_key": objective_key,
        "trial_counts": {
            "total_trials": total_trials,
            "ok_trials": ok_trials,
            "unsuccessful_trials": unsuccessful_trials,
            "ok_missing_objective": ok_missing_objective,
        },
        "status_counts": dict(sorted(status_counts.items())),
        "objective_distribution": {
            "mean": mean(objectives),
            "std": pstdev(objectives) if len(objectives) > 1 else 0.0,
            "median": median(objectives),
        },
        # Keep singular keys for compatibility with previous consumers.
        "best_trial": serialize_trial(best_representative),
        "worst_trial": serialize_trial(worst_representative),
        # New keys: include every tied combination at the global best/worst objective.
        "best_trials_count": len(best_rows),
        "worst_trials_count": len(worst_rows),
        "best_trials": [serialize_trial(row) for row in best_rows],
        "worst_trials": [serialize_trial(row) for row in worst_rows],
        "tie_break_policy": {
            "best_trial": [
                "objective (higher is better)",
                "sampling/frac_unic_non_iso_valid (higher is better)",
                "average_ratio (lower is better)",
            ],
            "worst_trial": [
                "objective (lower is worse)",
                "sampling/frac_unic_non_iso_valid (lower is worse)",
                "average_ratio (higher is worse)",
            ],
        },
        "param_value_stats": param_value_stats,
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=False)


def plot_combined(stats: dict, ok_rows: list, output_path: Path):
    param_names = list(stats.keys())
    n_params = len(param_names)
    n_cols = 3
    n_rows = math.ceil(n_params / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.3 * n_cols, 3.8 * n_rows))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for i, param_name in enumerate(param_names):
        ax = axes[i]
        param_stats = stats[param_name]
        values = param_stats["values"]
        counts = param_stats["counts"]
        grouped = defaultdict(list)
        for row in ok_rows:
            grouped[row["params"][param_name]].append(row["objective"])

        x = list(range(len(values)))
        x_labels = [value_label(v) for v in values]
        data = [grouped[v] for v in values]

        bp = ax.boxplot(
            data,
            positions=x,
            widths=0.6,
            patch_artist=True,
            showfliers=True,
        )
        for box in bp["boxes"]:
            box.set(facecolor="#8fbce6", edgecolor="#1f4e79", linewidth=1.0)
        for median_line in bp["medians"]:
            median_line.set(color="#0d2d4d", linewidth=1.2)
        for whisker in bp["whiskers"]:
            whisker.set(color="#1f4e79", linewidth=0.9)
        for cap in bp["caps"]:
            cap.set(color="#1f4e79", linewidth=0.9)
        for flier in bp["fliers"]:
            flier.set(marker="o", markersize=3, markerfacecolor="#1f4e79", alpha=0.5, markeredgewidth=0)

        ax.set_title(param_name)
        ax.set_xlabel(param_name)
        ax.set_ylabel("objective")
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=35, ha="right")
        ax.grid(axis="y", linestyle="--", alpha=0.35)

        for xi, c in zip(x, counts):
            ax.annotate(
                f"n={c}",
                (xi, 1.01),
                xycoords=("data", "axes fraction"),
                textcoords="offset points",
                xytext=(0, -2),
                ha="center",
                fontsize=7,
            )

    for j in range(n_params, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle("Hyperparameter sweep: objective distribution by value (boxplots)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    json_path = args.json_path.resolve()
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    output_dir = args.output_dir
    if output_dir is None:
        analysis_dir = Path(__file__).resolve().parent
        output_dir = analysis_dir / f"{json_path.stem}_mean_std_plots"
    else:
        output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    all_trials = load_trials(json_path, args.objective_key)
    ok_rows = [row for row in all_trials if row["status"] == "ok" and row["objective"] is not None]
    if not ok_rows:
        raise ValueError("No successful trials with objective found.")

    stats = build_stats(ok_rows)
    stats = {k: v for k, v in stats.items() if len(v["values"]) > 1}
    if not stats:
        raise ValueError("No parameters with more than one unique value were found.")

    summary_json_path = output_dir / "analysis_summary.json"
    write_summary_json(all_trials, ok_rows, stats, args.objective_key, summary_json_path)

    combined_path = output_dir / "all_params_boxplots.png"
    plot_combined(stats, ok_rows, combined_path)
    print(f"Saved combined plot to: {combined_path}")
    print(f"Saved summary JSON to: {summary_json_path}")


if __name__ == "__main__":
    main()
