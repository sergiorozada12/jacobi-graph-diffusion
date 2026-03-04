import argparse
import ast
import re
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors


STATS_PATTERN = re.compile(r"Sampling statistics (\{.*\})")

DATASET_CONFIG = {
    "tree": {
        "log_pattern": re.compile(r"tree-graphon-gen-epoch-(\d+)(?:-([a-z0-9-]+))?-n(\d+)\.log$"),
        "glob": "tree-graphon-gen-epoch-*-n*.log",
        "required_metrics": ("tree_acc", "average_ratio", "forest_acc"),
        "plot_metrics": ("tree_acc", "average_ratio", "forest_acc"),
        "bounded_metrics": ("tree_acc", "forest_acc"),
        "in_dist_range": (20, 80),
        "default_logs_dir": Path("samples/tree_graphon_eval"),
        "default_output": Path("analysis/results/tree_num_nodes_sweep/plot_tree_metrics_sweep.png"),
        "title": "tree",
    },
    "sbm": {
        "log_pattern": re.compile(r"sbm-2comms-gen-epoch-(\d+)(?:-([a-z0-9-]+))?-n(\d+)\.log$"),
        "glob": "sbm-2comms-gen-epoch-*-n*.log",
        "required_metrics": ("sbm_acc", "average_ratio", "sampling/frac_unic_non_iso_valid"),
        "plot_metrics": ("sbm_acc", "average_ratio", "sampling/frac_unic_non_iso_valid"),
        "bounded_metrics": ("sbm_acc", "sampling/frac_unic_non_iso_valid"),
        "in_dist_range": (40, 80),
        "default_logs_dir": Path("samples/sbm_2comms_eval"),
        "default_output": Path("analysis/results/sbm_num_nodes_sweep/plot_sbm_metrics_sweep.png"),
        "title": "sbm",
    },
}


def epoch_label(epoch: int) -> str:
    return f"{epoch // 1000}k" if epoch % 1000 == 0 else str(epoch)


def add_background(ax, x_min, x_max, in_dist_min, in_dist_max):
    left = max(x_min, in_dist_min)
    right = min(x_max, in_dist_max)
    ax.axvspan(x_min, left, color="#d99a9a", alpha=0.5, lw=0)
    ax.axvspan(left, right, color="#97d39b", alpha=0.6, lw=0)
    ax.axvspan(right, x_max, color="#d99a9a", alpha=0.5, lw=0)
    ax.set_xlim(x_min, x_max)


def parse_log(log_path: Path, cfg: dict):
    match = cfg["log_pattern"].match(log_path.name)
    if not match:
        return None

    epoch = int(match.group(1))
    run_tag = match.group(2) or "base"
    n_nodes = int(match.group(3))
    text = log_path.read_text(encoding="utf-8", errors="ignore")

    stats_matches = STATS_PATTERN.findall(text)
    if not stats_matches:
        return None

    try:
        stats = ast.literal_eval(stats_matches[-1])
    except (SyntaxError, ValueError):
        return None

    if any(k not in stats for k in cfg["required_metrics"]):
        return None

    row = {"epoch": epoch, "run_tag": run_tag, "n": n_nodes}
    for key in cfg["plot_metrics"]:
        row[key] = float(stats[key])
    return row


def collect_results(logs_dir: Path, cfg: dict):
    by_variant = {}
    for log_path in sorted(logs_dir.glob(cfg["glob"])):
        row = parse_log(log_path, cfg)
        if row is None:
            continue
        key = (row["epoch"], row["run_tag"])
        by_variant.setdefault(key, []).append(row)

    for key in by_variant:
        by_variant[key].sort(key=lambda r: r["n"])
    return by_variant


def _mix_with_white(color, alpha: float):
    r, g, b = mcolors.to_rgb(color)
    return (r + (1.0 - r) * alpha, g + (1.0 - g) * alpha, b + (1.0 - b) * alpha)


def build_epoch_color_map(by_variant: dict):
    epochs = sorted({epoch for epoch, _ in by_variant.keys()})
    if not epochs:
        return {}
    if len(epochs) == 1:
        return {epochs[0]: plt.cm.viridis(0.75)}

    # Earlier checkpoints are lighter; later checkpoints are darker.
    color_map = {}
    for i, epoch in enumerate(epochs):
        t = i / (len(epochs) - 1)
        color_map[epoch] = plt.cm.viridis(0.85 - 0.55 * t)
    return color_map


def plot_all_metrics(by_variant: dict, out_path: Path, cfg: dict):
    metric_keys = cfg["plot_metrics"]
    bounded = set(cfg["bounded_metrics"])
    all_n = sorted({row["n"] for rows in by_variant.values() for row in rows})
    if not all_n:
        raise ValueError("No valid sweep logs found.")

    x_min, x_max = min(all_n), max(all_n)
    x_ticks = list(range(x_min, x_max + 1, 10))
    in_dist_min, in_dist_max = cfg["in_dist_range"]
    epoch_colors = build_epoch_color_map(by_variant)
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    for ax, metric_key in zip(axes, metric_keys):
        add_background(ax, x_min, x_max, in_dist_min, in_dist_max)
        for epoch, run_tag in sorted(by_variant.keys()):
            rows = by_variant[(epoch, run_tag)]
            x = [r["n"] for r in rows]
            y = [r[metric_key] for r in rows]
            label = f"{epoch_label(epoch)} ({run_tag})" if run_tag != "base" else epoch_label(epoch)
            linestyle = "--" if run_tag != "base" else "-"
            marker = "*" if run_tag != "base" else "o"
            markersize = 8.0 if run_tag != "base" else 4.5
            base_color = epoch_colors[epoch]
            color = _mix_with_white(base_color, 0.30) if run_tag != "base" else base_color
            ax.plot(
                x,
                y,
                color=color,
                linestyle=linestyle,
                marker=marker,
                lw=1.8,
                ms=markersize,
                label=label,
            )

        ax.set_title(metric_key)
        ax.set_ylabel(metric_key)
        ax.grid(alpha=0.2)
        if metric_key in bounded:
            ax.set_ylim(0.0, 1.05)

    axes[-1].set_xlabel("n")
    axes[-1].set_xticks(x_ticks)
    axes[0].legend(title="checkpoint", loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def run(dataset: str, logs_dir: Optional[Path] = None, output: Optional[Path] = None):
    if dataset not in DATASET_CONFIG:
        raise ValueError(f"Unsupported dataset '{dataset}'. Choose from {list(DATASET_CONFIG)}")

    cfg = DATASET_CONFIG[dataset]
    repo_root = Path(__file__).resolve().parents[1]
    resolved_logs = logs_dir.resolve() if logs_dir is not None else (repo_root / cfg["default_logs_dir"])
    resolved_output = output.resolve() if output is not None else (repo_root / cfg["default_output"])
    resolved_output.parent.mkdir(parents=True, exist_ok=True)

    by_variant = collect_results(resolved_logs, cfg)
    if not by_variant:
        raise RuntimeError(f"No valid {cfg['title']} logs were parsed in {resolved_logs}.")

    plot_all_metrics(by_variant, resolved_output, cfg)
    parsed = sum(len(rows) for rows in by_variant.values())
    print(f"Parsed {parsed} logs across variants: {sorted(by_variant.keys())}")
    print(f"Saved: {resolved_output}")


def parse_args():
    parser = argparse.ArgumentParser(description="Plot sweep metrics for tree or sbm generation logs.")
    parser.add_argument("--dataset", choices=("tree", "sbm"), required=True, help="Dataset family to plot.")
    parser.add_argument("--logs-dir", type=Path, default=None, help="Directory containing generation logs.")
    parser.add_argument("--output", type=Path, default=None, help="Output PNG path.")
    return parser.parse_args()


def main():
    args = parse_args()
    run(dataset=args.dataset, logs_dir=args.logs_dir, output=args.output)


if __name__ == "__main__":
    main()
