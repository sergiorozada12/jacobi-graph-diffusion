import ast
import re
from pathlib import Path

import matplotlib.pyplot as plt


LOG_PATTERN = re.compile(r"tree-graphon-gen-epoch-(\d+)-n(\d+)\.log$")
STATS_PATTERN = re.compile(r"Sampling statistics (\{.*\})")


def parse_log(log_path: Path):
    match = LOG_PATTERN.match(log_path.name)
    if not match:
        return None

    epoch = int(match.group(1))
    n_nodes = int(match.group(2))
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
        "n": n_nodes,
        "tree_acc": float(stats["tree_acc"]),
        "average_ratio": float(stats["average_ratio"]),
        "forest_acc": float(stats["forest_acc"]),
    }


def collect_results(base_dir: Path):
    by_epoch = {}
    for log_path in sorted(base_dir.glob("tree-graphon-gen-epoch-*-n*.log")):
        row = parse_log(log_path)
        if row is None:
            continue
        by_epoch.setdefault(row["epoch"], []).append(row)

    for epoch in by_epoch:
        by_epoch[epoch].sort(key=lambda r: r["n"])
    return by_epoch


def add_background(ax, x_min, x_max):
    ax.axvspan(x_min, 20, color="#d99a9a", alpha=0.5, lw=0)
    ax.axvspan(20, 80, color="#97d39b", alpha=0.6, lw=0)
    ax.axvspan(80, x_max, color="#d99a9a", alpha=0.5, lw=0)
    ax.set_xlim(x_min, x_max)


def epoch_label(epoch: int) -> str:
    if epoch % 1000 == 0:
        return f"{epoch // 1000}k"
    return str(epoch)


def plot_all_metrics(by_epoch, out_path: Path):
    metric_keys = ("tree_acc", "average_ratio", "forest_acc")
    all_n = sorted({row["n"] for rows in by_epoch.values() for row in rows})
    if not all_n:
        raise ValueError("No valid sweep logs found.")

    x_min, x_max = min(all_n), max(all_n)
    x_ticks = list(range(x_min, x_max + 1, 10))

    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    for ax, metric_key in zip(axes, metric_keys):
        add_background(ax, x_min, x_max)
        for epoch in sorted(by_epoch.keys()):
            rows = by_epoch[epoch]
            x = [r["n"] for r in rows]
            y = [r[metric_key] for r in rows]
            ax.plot(x, y, marker="o", lw=1.8, ms=4.5, label=f"{epoch_label(epoch)}")

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


def main():
    repo_root = Path(__file__).resolve().parents[1]
    logs_dir = repo_root / "samples" / "tree_graphon_eval"
    by_epoch = collect_results(logs_dir)

    if not by_epoch:
        raise RuntimeError(f"No valid logs were parsed in {logs_dir}.")

    out_plot = Path(__file__).resolve().parent / "plot_tree_metrics_sweep.png"
    plot_all_metrics(by_epoch, out_plot)

    parsed = sum(len(rows) for rows in by_epoch.values())
    print(f"Parsed {parsed} logs across epochs: {sorted(by_epoch.keys())}")
    print(f"Saved: {out_plot}")


if __name__ == "__main__":
    main()
