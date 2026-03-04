import argparse
import ast
import re
from pathlib import Path

import matplotlib.pyplot as plt


LOG_PATTERN = re.compile(r"tree-graphon-gen-epoch-(\d+)(?:-([a-z0-9-]+))?-n(\d+)\.log$")
STATS_PATTERN = re.compile(r"Sampling statistics (\{.*\})")


def parse_log(log_path: Path):
    match = LOG_PATTERN.match(log_path.name)
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

    required = ("tree_acc", "average_ratio", "forest_acc")
    if any(k not in stats for k in required):
        return None

    return {
        "epoch": epoch,
        "run_tag": run_tag,
        "n": n_nodes,
        "tree_acc": float(stats["tree_acc"]),
        "average_ratio": float(stats["average_ratio"]),
        "forest_acc": float(stats["forest_acc"]),
    }


def collect_results(base_dir: Path):
    by_variant = {}
    for log_path in sorted(base_dir.glob("tree-graphon-gen-epoch-*-n*.log")):
        row = parse_log(log_path)
        if row is None:
            continue
        key = (row["epoch"], row["run_tag"])
        by_variant.setdefault(key, []).append(row)

    for key in by_variant:
        by_variant[key].sort(key=lambda r: r["n"])
    return by_variant


def add_background(ax, x_min, x_max):
    ax.axvspan(x_min, 20, color="#d99a9a", alpha=0.5, lw=0)
    ax.axvspan(20, 80, color="#97d39b", alpha=0.6, lw=0)
    ax.axvspan(80, x_max, color="#d99a9a", alpha=0.5, lw=0)
    ax.set_xlim(x_min, x_max)


def epoch_label(epoch: int) -> str:
    if epoch % 1000 == 0:
        return f"{epoch // 1000}k"
    return str(epoch)


def plot_all_metrics(by_variant, out_path: Path):
    metric_keys = ("tree_acc", "average_ratio", "forest_acc")
    all_n = sorted({row["n"] for rows in by_variant.values() for row in rows})
    if not all_n:
        raise ValueError("No valid sweep logs found.")

    x_min, x_max = min(all_n), max(all_n)
    x_ticks = list(range(x_min, x_max + 1, 10))

    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    for ax, metric_key in zip(axes, metric_keys):
        add_background(ax, x_min, x_max)
        for epoch, run_tag in sorted(by_variant.keys()):
            rows = by_variant[(epoch, run_tag)]
            x = [r["n"] for r in rows]
            y = [r[metric_key] for r in rows]
            label = f"{epoch_label(epoch)} ({run_tag})" if run_tag != "base" else epoch_label(epoch)
            linestyle = "--" if run_tag != "base" else "-"
            marker = "*" if run_tag != "base" else "o"
            markersize = 8.0 if run_tag != "base" else 4.5
            ax.plot(x, y, linestyle=linestyle, marker=marker, lw=1.8, ms=markersize, label=label)

        ax.set_title(metric_key)
        ax.set_ylabel(metric_key)
        ax.grid(alpha=0.2)
        if metric_key in ("tree_acc", "forest_acc"):
            ax.set_ylim(0.0, 1.05)

    axes[-1].set_xlabel("n")
    axes[-1].set_xticks(x_ticks)
    axes[0].legend(title="checkpoint", loc="best")

    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Plot tree_graphon sweep metrics from generation logs.")
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=None,
        help="Directory containing tree-graphon-gen-epoch-*-n*.log files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG path.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    logs_dir = (args.logs_dir.resolve() if args.logs_dir is not None else (repo_root / "samples" / "tree_graphon_eval"))
    by_variant = collect_results(logs_dir)

    if not by_variant:
        raise RuntimeError(f"No valid logs were parsed in {logs_dir}.")

    out_plot = (
        args.output.resolve()
        if args.output is not None
        else (
            Path(__file__).resolve().parent
            / "results"
            / "tree_num_nodes_sweep"
            / "plot_tree_metrics_sweep.png"
        )
    )
    out_plot.parent.mkdir(parents=True, exist_ok=True)
    plot_all_metrics(by_variant, out_plot)

    parsed = sum(len(rows) for rows in by_variant.values())
    print(f"Parsed {parsed} logs across variants: {sorted(by_variant.keys())}")
    print(f"Saved: {out_plot}")


if __name__ == "__main__":
    main()
