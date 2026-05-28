#!/usr/bin/env python3
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> int:
    if len(sys.argv) != 3:
        print("Usage: python3 scripts/plot_benchmarks.py benchmark-results/results.csv plots/")
        return 1

    input_csv = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    rows = load_rows(input_csv)
    if not rows:
        raise SystemExit("CSV file is empty.")

    output_dir.mkdir(parents=True, exist_ok=True)
    grouped_rows: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped_rows[str(row["kernel_set"])].append(row)

    for kernel_set, kernel_rows in grouped_rows.items():
        slug = slugify(kernel_set)
        plot_metric(
            rows=kernel_rows,
            metric_key="avg_ms",
            y_label="Average time, ms",
            title=f"Convolution performance ({kernel_set})",
            output_path=output_dir / f"{slug}-avg-ms.svg",
        )
        plot_metric(
            rows=kernel_rows,
            metric_key="throughput_mp_s",
            y_label="Throughput, MP/s",
            title=f"Convolution throughput ({kernel_set})",
            output_path=output_dir / f"{slug}-throughput.svg",
        )

    print(f"Saved plots to {output_dir}")
    return 0


def load_rows(input_csv: Path) -> list[dict[str, object]]:
    with input_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows: list[dict[str, object]] = []
        for row in reader:
            rows.append(
                {
                    "kernel_set": row["kernel_set"],
                    "mode": row["mode"],
                    "strategy": row["strategy"],
                    "partition_kind": row["partition_kind"],
                    "partition_value": row["partition_value"],
                    "threads": int(row["threads"]),
                    "avg_ms": float(row["avg_ms"]),
                    "min_ms": float(row["min_ms"]),
                    "max_ms": float(row["max_ms"]),
                    "stddev_ms": float(row["stddev_ms"]),
                    "ci95_low_ms": float(row["ci95_low_ms"]),
                    "ci95_high_ms": float(row["ci95_high_ms"]),
                    "throughput_mp_s": float(row["throughput_mp_s"]),
                    "throughput_ci95_low_mp_s": float(row["throughput_ci95_low_mp_s"]),
                    "throughput_ci95_high_mp_s": float(row["throughput_ci95_high_mp_s"]),
                }
            )
    return rows


def plot_metric(
    rows: list[dict[str, object]],
    metric_key: str,
    y_label: str,
    title: str,
    output_path: Path,
) -> None:
    sequential_rows = [row for row in rows if row["mode"] == "sequential"]
    parallel_rows = [row for row in rows if row["mode"] == "parallel"]

    all_threads = sorted({int(row["threads"]) for row in parallel_rows})
    if not all_threads:
        all_threads = [1]

    fig, ax = plt.subplots(figsize=(12, 7), constrained_layout=True)
    ax.set_title(title, fontsize=18, fontweight="semibold")
    ax.set_xlabel("Threads", fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.grid(True, which="major", axis="both", linestyle="--", linewidth=0.7, alpha=0.35)
    ax.set_xticks(all_threads)

    if sequential_rows:
        baseline_row = sequential_rows[0]
        baseline = float(baseline_row[metric_key])
        baseline_low, baseline_high = ci_bounds(baseline_row, metric_key)
        ax.axhline(
            baseline,
            color="black",
            linestyle=":",
            linewidth=2.0,
            label="sequential",
        )
        ax.axhspan(
            baseline_low,
            baseline_high,
            color="black",
            alpha=0.08,
        )

    series_map: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in parallel_rows:
        series_map[series_name(row)].append(row)

    colors = plt.get_cmap("tab10")
    for index, name in enumerate(sorted(series_map)):
        values = series_map[name]
        thread_values = np.array([int(row["threads"]) for row in values], dtype=np.int32)
        metric_values = np.array([float(row[metric_key]) for row in values], dtype=np.float64)
        error_values = np.array([ci_half_width(row, metric_key) for row in values], dtype=np.float64)
        order = np.argsort(thread_values)
        thread_values = thread_values[order]
        metric_values = metric_values[order]
        error_values = error_values[order]

        ax.errorbar(
            thread_values,
            metric_values,
            yerr=error_values,
            marker="o",
            linewidth=2.2,
            markersize=6.0,
            label=name,
            color=colors(index % 10),
            capsize=4.0,
        )

    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    fig.savefig(output_path, format="svg")
    plt.close(fig)


def series_name(row: dict[str, object]) -> str:
    partition_kind = str(row["partition_kind"])
    partition_value = str(row["partition_value"])
    if partition_kind == "tile":
        return f"tile {partition_value}"
    if partition_kind == "grid":
        return f"grid {partition_value}"
    return str(row["strategy"])


def ci_half_width(row: dict[str, object], metric_key: str) -> float:
    low, high = ci_bounds(row, metric_key)
    return max(0.0, (high - low) / 2.0)


def ci_bounds(row: dict[str, object], metric_key: str) -> tuple[float, float]:
    if metric_key == "avg_ms":
        return float(row["ci95_low_ms"]), float(row["ci95_high_ms"])
    if metric_key == "throughput_mp_s":
        return (
            float(row["throughput_ci95_low_mp_s"]),
            float(row["throughput_ci95_high_mp_s"]),
        )
    raise ValueError(f"Unsupported metric key: {metric_key}")


def slugify(value: str) -> str:
    allowed: list[str] = []
    for char in value.lower():
        if char.isalnum():
            allowed.append(char)
        elif char in {",", ";", "+", " "}:
            allowed.append("-")
    slug = "".join(allowed).strip("-")
    return slug or "benchmark"


if __name__ == "__main__":
    raise SystemExit(main())
