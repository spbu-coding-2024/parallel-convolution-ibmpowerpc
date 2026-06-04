#!/usr/bin/env python3
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SINGLE_IMAGE_WORKLOAD = "single-image"
IMAGE_BATCH_WORKLOAD = "image-batch"


def main() -> int:
    if len(sys.argv) != 3:
        print("Usage: plot_benchmarks.py <benchmark.csv> <output-dir>", file=sys.stderr)
        return 1

    csv_path = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(csv_path)
    if not rows:
        print(f"No rows found in {csv_path}", file=sys.stderr)
        return 1

    grouped_rows = defaultdict(list)
    for row in rows:
        grouped_rows[(row["kernel_set"], row["workload"])].append(row)

    for (kernel_set, workload), kernel_rows in grouped_rows.items():
        slug = slugify(kernel_set)
        workload_suffix = "" if workload == SINGLE_IMAGE_WORKLOAD else f"-{slugify(workload)}"

        plot_metric(
            kernel_rows,
            metric="avg_ms",
            title=plot_title(workload, "performance", kernel_set),
            ylabel="Average time, ms",
            output_path=output_dir / f"{slug}{workload_suffix}-avg-ms.svg",
            workload=workload,
        )
        plot_metric(
            kernel_rows,
            metric="throughput_mp_s",
            title=plot_title(workload, "throughput", kernel_set),
            ylabel="Throughput, MP/s",
            output_path=output_dir / f"{slug}{workload_suffix}-throughput.svg",
            workload=workload,
        )

    return 0


def load_rows(csv_path: Path) -> list[dict]:
    rows = []
    with csv_path.open(newline="") as file:
        for row in csv.DictReader(file):
            rows.append(
                {
                    "workload": row.get("workload") or SINGLE_IMAGE_WORKLOAD,
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
                    "image_width": int(row.get("image_width") or 0),
                    "image_height": int(row.get("image_height") or 0),
                    "image_count": int(row.get("image_count") or 1),
                    "total_pixels": int(row.get("total_pixels") or 0),
                    "kernel_count": int(row.get("kernel_count") or 0),
                }
            )
    return rows


def plot_metric(rows: list[dict], metric: str, title: str, ylabel: str, output_path: Path, workload: str) -> None:
    sequential_rows = [row for row in rows if row["mode"] == "sequential"]
    parallel_rows = [row for row in rows if row["mode"] == "parallel"]

    threads = sorted({row["threads"] for row in parallel_rows})
    if not threads:
        threads = [1]

    figure, axes = plt.subplots(figsize=(10, 6))

    if sequential_rows:
        baseline = sequential_rows[0]
        baseline_value = baseline[metric]
        ci_low, ci_high = ci_bounds(baseline, metric)
        axes.axhline(
            baseline_value,
            color="black",
            linestyle="--",
            linewidth=1.5,
            label="sequential baseline",
        )
        axes.fill_between(
            threads,
            ci_low,
            ci_high,
            color="black",
            alpha=0.08,
        )

    series = defaultdict(list)
    for row in parallel_rows:
        series[series_name(row, workload)].append(row)

    for name, series_rows in sorted(series.items()):
        series_rows = sorted(series_rows, key=lambda row: row["threads"])
        x_values = np.array([row["threads"] for row in series_rows], dtype=float)
        y_values = np.array([row[metric] for row in series_rows], dtype=float)
        y_errors = np.array([ci_half_width(row, metric) for row in series_rows], dtype=float)
        axes.errorbar(
            x_values,
            y_values,
            yerr=y_errors,
            marker="o",
            capsize=4,
            linewidth=1.6,
            label=name,
        )

    axes.set_title(title)
    axes.set_xlabel("Image workers" if workload == IMAGE_BATCH_WORKLOAD else "Threads")
    axes.set_ylabel(ylabel)
    axes.set_xticks(threads)
    axes.grid(True, linestyle=":", alpha=0.55)
    axes.legend(fontsize="small", ncol=2)
    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)


def plot_title(workload: str, metric_name: str, kernel_set: str) -> str:
    if workload == IMAGE_BATCH_WORKLOAD:
        return f"Image batch convolution {metric_name} ({kernel_set})"
    return f"Convolution {metric_name} ({kernel_set})"


def series_name(row: dict, workload: str) -> str:
    if workload == IMAGE_BATCH_WORKLOAD and row["strategy"] == "images":
        return "parallel images"
    if row["partition_kind"] == "tile":
        return f"{row['strategy']} tile {row['partition_value']}"
    if row["partition_kind"] == "grid":
        return f"{row['strategy']} grid {row['partition_value']}"
    return row["strategy"]


def ci_half_width(row: dict, metric: str) -> float:
    low, high = ci_bounds(row, metric)
    return max((high - low) / 2.0, 0.0)


def ci_bounds(row: dict, metric: str) -> tuple[float, float]:
    if metric == "avg_ms":
        return row["ci95_low_ms"], row["ci95_high_ms"]
    if metric == "throughput_mp_s":
        return row["throughput_ci95_low_mp_s"], row["throughput_ci95_high_mp_s"]
    raise ValueError(f"Unsupported metric: {metric}")


def slugify(value: str) -> str:
    return "".join(character if character.isalnum() else "-" for character in value.lower()).strip("-")


if __name__ == "__main__":
    raise SystemExit(main())
