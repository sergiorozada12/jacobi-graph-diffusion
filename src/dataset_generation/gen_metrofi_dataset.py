#!/usr/bin/env python
"""
Convert the MetroFi wireless measurements into a train/val/test graph dataset.

Each location is represented as a graph whose nodes correspond to the
globally-observed MAC addresses and whose edge weights capture the inferred
interference between two access points at that location.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import torch
from src.visualization.plots import (
    plot_edge_weight_histograms,
    plot_weighted_adj_and_graph,
    save_figure,
)


DEFAULT_INPUT_PATH = Path("data/metrofi/stumble_filtered.txt")
DEFAULT_OUTPUT_PATH = Path("data/metrofi/metrofi.pkl")


@dataclass(frozen=True)
class MetroFiDataset:
    graphs: Mapping[int, nx.Graph]
    mac_id_to_address: Sequence[str]
    location_id_to_coords: Mapping[int, Tuple[float, float]]
    max_interference: float
    max_interference_watts: float
    interference_units: str


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Path to the MetroFi stumble log (default: %(default)s).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Destination pickle file (default: %(default)s).",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Fraction of locations reserved for validation (default: %(default)s).",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Fraction of locations reserved for testing (default: %(default)s).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=17,
        help="Random seed for the train/val/test split (default: %(default)s).",
    )
    parser.add_argument(
        "--max-locations",
        type=int,
        default=None,
        help="Optional cap on the number of locations to process (useful for smoke tests).",
    )
    parser.add_argument(
        "--model",
        choices=["sum", "product", "min"],
        default="min",
        help="Interference aggregation model (default: %(default)s).",
    )
    parser.add_argument(
        "--max-interference",
        type=float,
        default=None,
        help="Cap the interference (in Watts) when computing edges (default: no cap).",
    )
    parser.add_argument(
        "--interference-output",
        choices=["w", "dbm"],
        default="dbm",
        help="Unit for stored interference values (default: %(default)s).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output file if present.",
    )
    parser.add_argument(
        "--metadata-json",
        type=Path,
        default=None,
        help="Optional path to dump metadata (mac addresses, coordinates) as JSON.",
    )
    return parser.parse_args(argv)


def load_metrofi_dataset(path: Path) -> Tuple[pd.DataFrame, Dict[int, str], Dict[int, Tuple[float, float]]]:
    df_raw = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=["mac", "lat", "lon", "rssi"],
    )

    mac_codes, mac_values = pd.factorize(df_raw["mac"])
    df_raw["mac_id"] = mac_codes

    locations = list(zip(df_raw["lat"], df_raw["lon"]))
    location_codes, location_values = pd.factorize(locations)
    df_raw["location_id"] = location_codes

    mac_id_to_mac: Dict[int, str] = {idx: mac for idx, mac in enumerate(mac_values)}
    location_id_to_coords: Dict[int, Tuple[float, float]] = {
        idx: loc for idx, loc in enumerate(location_values)
    }

    dataset = df_raw[["mac_id", "location_id", "rssi"]].copy()
    return dataset, mac_id_to_mac, location_id_to_coords


def rssi_raw_to_dbm(rssi_raw: float) -> float:
    # Convert chipset RSSI to dBm (calibration offset derived from exploratory analysis).
    return rssi_raw - 95.0


def dbm_to_w(dbm: float) -> float:
    # Convert dBm to Watts.
    return 0.001 * (10.0 ** (dbm / 10.0))


def w_to_dbm(p_w: float) -> float:
    # Convert Watts to dBm.
    return 10 * np.log10(p_w / 0.001)


def build_location_graphs(
    dataset: pd.DataFrame,
    mac_id_to_mac: Mapping[int, str],
    location_id_to_coords: Mapping[int, Tuple[float, float]],
    max_locations: Optional[int] = None,
    model: str = "min",
    max_interference: Optional[float] = None,
    output: str = "dbm",
) -> MetroFiDataset:
    def interference_value(p_a: float, p_b: float) -> float:
        if model == "sum":
            val = p_a + p_b
        elif model == "product":
            val = p_a * p_b
        elif model == "min":
            val = min(p_a, p_b)
        else:
            raise ValueError(f"Unknown model '{model}'")

        if max_interference is not None:
            val = min(val, max_interference)

        return val

    graphs: Dict[int, nx.Graph] = {}
    all_mac_ids = sorted(mac_id_to_mac.keys())
    max_interference_out: Optional[float] = None
    max_interference_watts = 0.0

    grouped = dataset.groupby("location_id")
    all_location_ids = list(grouped.groups.keys())
    if max_locations is not None:
        iter_location_ids = sorted(all_location_ids)[:max_locations]
    else:
        iter_location_ids = all_location_ids

    for location_id in iter_location_ids:
        if location_id not in location_id_to_coords:
            continue
        try:
            group = grouped.get_group(location_id)
        except KeyError:
            continue
        if group.empty:
            continue

        graph = nx.Graph(location_id=int(location_id))
        coordinates = location_id_to_coords[location_id]
        graph.graph["coordinates"] = coordinates

        # Add all MAC nodes so that every graph lives in the same 70-node universe.
        for mac_id in all_mac_ids:
            graph.add_node(
                int(mac_id),
                mac_address=mac_id_to_mac[mac_id],
                observed=False,
                mean_rssi=np.nan,
                mean_rssi_raw=np.nan,
                mean_rssi_dbm=np.nan,
                sample_count=0,
            )

        mac_stats = (
            group.groupby("mac_id")["rssi"]
            .agg(["mean", "count"])
            .rename(columns={"mean": "mean_rssi_raw", "count": "sample_count"})
        )

        observed_ids = mac_stats.index.tolist()
        for mac_id in observed_ids:
            stats = mac_stats.loc[mac_id]
            mean_rssi_raw = float(stats["mean_rssi_raw"])
            mean_rssi_dbm = rssi_raw_to_dbm(mean_rssi_raw)
            node_data = graph.nodes[int(mac_id)]
            node_data["observed"] = True
            node_data["mean_rssi_raw"] = mean_rssi_raw
            node_data["mean_rssi_dbm"] = mean_rssi_dbm
            node_data["mean_rssi"] = mean_rssi_dbm  # retain legacy key
            node_data["sample_count"] = int(stats["sample_count"])

        for mac_a, mac_b in combinations(observed_ids, 2):
            mean_rssi_a_raw = float(mac_stats.loc[mac_a, "mean_rssi_raw"])
            mean_rssi_b_raw = float(mac_stats.loc[mac_b, "mean_rssi_raw"])

            mean_rssi_a_dbm = rssi_raw_to_dbm(mean_rssi_a_raw)
            mean_rssi_b_dbm = rssi_raw_to_dbm(mean_rssi_b_raw)

            power_a = dbm_to_w(mean_rssi_a_dbm)
            power_b = dbm_to_w(mean_rssi_b_dbm)
            interference_w = float(interference_value(power_a, power_b))

            if output == "dbm":
                interference_out = w_to_dbm(interference_w)
            elif output == "w":
                interference_out = interference_w
            else:
                raise ValueError("output must be 'w' or 'dbm'")

            graph.add_edge(
                int(mac_a),
                int(mac_b),
                weight=float(interference_out),
                interference=float(interference_out),
                interference_watts=interference_w,
                interference_dbm=w_to_dbm(interference_w),
                mean_rssi_a_dbm=mean_rssi_a_dbm,
                mean_rssi_b_dbm=mean_rssi_b_dbm,
            )

            if max_interference_out is None or interference_out > max_interference_out:
                max_interference_out = float(interference_out)
            if interference_w > max_interference_watts:
                max_interference_watts = float(interference_w)

        graphs[int(location_id)] = graph

        if max_locations is not None and len(graphs) >= max_locations:
            break

    return MetroFiDataset(
        graphs=graphs,
        mac_id_to_address=tuple(mac_id_to_mac[i] for i in all_mac_ids),
        location_id_to_coords=location_id_to_coords,
        max_interference=float(0.0 if max_interference_out is None else max_interference_out),
        max_interference_watts=float(max_interference_watts),
        interference_units=output,
    )


def train_val_test_split(
    keys: Iterable[int],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[List[int], List[int], List[int]]:
    keys = list(keys)
    if not keys:
        return [], [], []

    if val_ratio < 0 or test_ratio < 0 or (val_ratio + test_ratio) >= 1.0:
        raise ValueError("Require val_ratio >= 0, test_ratio >= 0, and val_ratio + test_ratio < 1.")

    rng = np.random.default_rng(seed)
    rng.shuffle(keys)

    n_total = len(keys)
    n_test = int(round(test_ratio * n_total))
    n_val = int(round(val_ratio * n_total))
    n_test = min(n_test, n_total)
    n_val = min(n_val, max(0, n_total - n_test))
    n_train = n_total - n_val - n_test
    if n_train <= 0:
        raise ValueError("Train split ended up empty; adjust val/test ratios.")

    test_keys = keys[:n_test]
    val_keys = keys[n_test : n_test + n_val]
    train_keys = keys[n_test + n_val :]
    return train_keys, val_keys, test_keys


def serialize_dataset(
    dataset: MetroFiDataset,
    output_path: Path,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Dict[str, object]:
    train_ids, val_ids, test_ids = train_val_test_split(
        dataset.graphs.keys(),
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    payload = {
        "train": [dataset.graphs[i] for i in train_ids],
        "val": [dataset.graphs[i] for i in val_ids],
        "test": [dataset.graphs[i] for i in test_ids],
        "metadata": {
            "mac_id_to_address": list(dataset.mac_id_to_address),
            "location_id_to_coords": {
                int(idx): (float(lat), float(lon))
                for idx, (lat, lon) in dataset.location_id_to_coords.items()
            },
            "max_interference": float(dataset.max_interference),
            "max_interference_watts": float(dataset.max_interference_watts),
            "interference_units": dataset.interference_units,
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(payload, output_path)
    return payload


def maybe_dump_metadata(metadata: Mapping[str, object], metadata_path: Optional[Path]) -> None:
    if metadata_path is None:
        return
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)

    if args.output.exists() and not args.force:
        raise FileExistsError(f"{args.output} already exists. Use --force to overwrite.")

    dataset_df, mac_id_to_mac, location_id_to_coords = load_metrofi_dataset(args.input)
    metrofi_dataset = build_location_graphs(
        dataset_df,
        mac_id_to_mac=mac_id_to_mac,
        location_id_to_coords=location_id_to_coords,
        max_locations=args.max_locations,
        model=args.model,
        max_interference=args.max_interference,
        output=args.interference_output,
    )

    payload = serialize_dataset(
        metrofi_dataset,
        output_path=args.output,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    maybe_dump_metadata(payload["metadata"], args.metadata_json)

    # Save example plots for quick inspection.
    fig_dir = args.output.parent / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    def _graph_to_adj_and_flags(graph: nx.Graph) -> Tuple[torch.Tensor, torch.Tensor]:
        nodes_sorted = sorted(graph.nodes())
        adj = nx.to_numpy_array(graph, nodelist=nodes_sorted, weight="weight")
        flags = np.array([bool(graph.nodes[n].get("observed", False)) for n in nodes_sorted], dtype=bool)
        return torch.tensor(adj, dtype=torch.float32), torch.tensor(flags, dtype=torch.bool)

    # Weighted adjacency + graph for the first non-empty test graph.
    for split_name in ("test", "val", "train"):
        graphs = payload.get(split_name, [])
        graph = next((g for g in graphs if g.number_of_edges() > 0), None)
        if graph is None:
            continue
        adj_t, flags_t = _graph_to_adj_and_flags(graph)
        fig = plot_weighted_adj_and_graph(
            adj_t,
            flags=flags_t,
            dataset_name=f"metrofi-{split_name}",
        )
        save_figure(fig, fig_dir / f"metrofi_{split_name}_example.png", dpi=200)
        break

    # Edge weight histograms for the test split.
    if payload.get("test"):
        hist_fig = plot_edge_weight_histograms(
            payload["test"],
            generated_graphs=None,
            dataset_name="metrofi test weights",
            num_edges=9,
        )
        save_figure(hist_fig, fig_dir / "metrofi_edge_weight_hist_test.png", dpi=200)

    n_train = len(payload["train"])
    n_val = len(payload["val"])
    n_test = len(payload["test"])
    print(
        f"Saved MetroFi dataset to {args.output} "
        f"(train={n_train}, val={n_val}, test={n_test}, total={n_train + n_val + n_test})."
    )

# python src/dataset_generation/gen_metrofi_dataset.py --force

if __name__ == "__main__":
    main()
