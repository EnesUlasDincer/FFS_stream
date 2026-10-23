#!/usr/bin/env python3
"""Render GitHub-visible PNG previews for the spatial-filter benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


RESOLUTION_ORDER = ("VGA", "HD720", "HD1080", "HD2K")
FILTER_ORDER = ("none", "bbox", "table", "both")
FILTER_LABELS = {
    "none": "Filters off",
    "bbox": "Bounding box",
    "table": "Table removed",
    "both": "Box + table",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "gallery_dir",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().parent / "filter_benchmark_gallery",
    )
    parser.add_argument("--preview-points", type=int, default=12_000)
    return parser.parse_args(argv)


def load_binary_ply(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("rb") as stream:
        vertex_count: int | None = None
        while True:
            line = stream.readline()
            if not line:
                raise ValueError(f"PLY header is incomplete: {path}")
            text = line.decode("ascii").strip()
            if text.startswith("element vertex "):
                vertex_count = int(text.rsplit(" ", 1)[1])
            if text == "end_header":
                break
        if vertex_count is None:
            raise ValueError(f"PLY has no vertex count: {path}")
        dtype = np.dtype(
            [
                ("x", "<f4"),
                ("y", "<f4"),
                ("z", "<f4"),
                ("red", "u1"),
                ("green", "u1"),
                ("blue", "u1"),
            ]
        )
        vertices = np.fromfile(stream, dtype=dtype, count=vertex_count)
    points = np.column_stack((vertices["x"], vertices["y"], vertices["z"]))
    colors = np.column_stack((vertices["red"], vertices["green"], vertices["blue"]))
    return points, colors.astype(np.float32) / 255.0


def render_resolution(
    gallery_dir: Path,
    results: list[dict[str, object]],
    resolution: str,
    point_limits: list[int],
    preview_points: int,
) -> None:
    lookup = {
        (entry["resolution"], entry["max_points"], entry["filter_mode"]): entry
        for entry in results
    }
    fig = plt.figure(figsize=(16, 10), facecolor="#07101b")
    fig.suptitle(
        f"Two-ZED calibrated point clouds · {resolution}",
        color="white",
        fontsize=18,
        y=0.995,
    )
    rng = np.random.default_rng(0)
    for row, point_limit in enumerate(point_limits):
        for column, filter_mode in enumerate(FILTER_ORDER):
            ax = fig.add_subplot(
                len(point_limits), len(FILTER_ORDER), row * len(FILTER_ORDER) + column + 1,
                projection="3d",
            )
            ax.set_facecolor("#07101b")
            entry = lookup.get((resolution, point_limit, filter_mode))
            if entry is None:
                ax.text2D(0.5, 0.5, "not captured", transform=ax.transAxes, ha="center")
                ax.set_axis_off()
                continue
            points, colors = load_binary_ply(gallery_dir / str(entry["ply"]))
            if len(points) > preview_points:
                selected = rng.choice(len(points), preview_points, replace=False)
                points, colors = points[selected], colors[selected]
            ax.scatter(
                points[:, 0], points[:, 1], points[:, 2],
                c=colors, s=0.18, marker=".", linewidths=0, depthshade=False,
            )
            ax.set_xlim(-1.45, -0.10)
            ax.set_ylim(-0.80, 0.70)
            ax.set_zlim(-0.20, 0.60)
            ax.set_box_aspect((1.35, 1.5, 0.8))
            ax.view_init(elev=24, azim=-58)
            ax.grid(False)
            ax.tick_params(colors="#90a4bd", labelsize=6, pad=-2)
            ax.xaxis.pane.set_alpha(0.0)
            ax.yaxis.pane.set_alpha(0.0)
            ax.zaxis.pane.set_alpha(0.0)
            if row == 0:
                ax.set_title(FILTER_LABELS[filter_mode], color="white", fontsize=11)
            if column == 0:
                ax.text2D(
                    -0.17, 0.5, f"{point_limit // 1000}K cap",
                    transform=ax.transAxes, rotation=90, va="center",
                    color="#7fc4ff", fontsize=11,
                )
            ax.text2D(
                0.03, 0.03,
                f"{int(float(entry['points_mean'])):,} pts · {float(entry['stream_fps']):.2f} FPS",
                transform=ax.transAxes, color="#dcecff", fontsize=7,
            )
    fig.subplots_adjust(
        left=0.04, right=0.99, bottom=0.03, top=0.90, wspace=0.02, hspace=0.08
    )
    fig.savefig(gallery_dir / f"preview_{resolution.lower()}.png", dpi=170, facecolor=fig.get_facecolor())
    plt.close(fig)


def render_fps_chart(gallery_dir: Path, results: list[dict[str, object]]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), facecolor="#07101b")
    colors = {"none": "#d9e5f2", "bbox": "#ffb454", "table": "#75d88a", "both": "#68a7ff"}
    for ax, resolution in zip(axes.flat, RESOLUTION_ORDER, strict=True):
        ax.set_facecolor("#0d1928")
        subset = [entry for entry in results if entry["resolution"] == resolution]
        point_limits = sorted({int(entry["max_points"]) for entry in subset})
        for filter_mode in FILTER_ORDER:
            values = []
            for point_limit in point_limits:
                matches = [
                    entry for entry in subset
                    if entry["filter_mode"] == filter_mode and entry["max_points"] == point_limit
                ]
                values.append(float(matches[0]["stream_fps"]) if matches else np.nan)
            ax.plot(
                [value / 1000 for value in point_limits], values,
                marker="o", linewidth=2, color=colors[filter_mode],
                label=FILTER_LABELS[filter_mode],
            )
        ax.set_title(resolution, color="white", fontsize=13)
        ax.set_xlabel("Point cap (thousands)", color="#a9bdd5")
        ax.set_ylabel("Stream FPS", color="#a9bdd5")
        ax.tick_params(colors="#a9bdd5")
        ax.grid(color="#29405d", alpha=0.5)
        for spine in ax.spines.values():
            spine.set_color("#29405d")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.95),
        ncol=4,
        frameon=False,
        labelcolor="white",
    )
    fig.suptitle(
        "Two-ZED TensorRT streaming benchmark", color="white", fontsize=18, y=0.995
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(gallery_dir / "fps_matrix.png", dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)


def render_point_reduction_chart(
    gallery_dir: Path, results: list[dict[str, object]]
) -> None:
    measured = [entry for entry in results if "points_generated_mean" in entry]
    if not measured:
        return
    point_limit = min(int(entry["max_points"]) for entry in measured)
    stages = (
        ("points_generated_mean", "Generated", "#d9e5f2"),
        ("points_after_bbox_mean", "After box", "#ffb454"),
        ("points_after_table_mean", "After table", "#75d88a"),
        ("points_mean", "Output", "#68a7ff"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), facecolor="#07101b")
    x = np.arange(len(FILTER_ORDER), dtype=np.float32)
    width = 0.19
    for ax, resolution in zip(axes.flat, RESOLUTION_ORDER, strict=True):
        ax.set_facecolor("#0d1928")
        for stage_index, (key, label, color) in enumerate(stages):
            values = []
            for filter_mode in FILTER_ORDER:
                matches = [
                    entry
                    for entry in measured
                    if entry["resolution"] == resolution
                    and int(entry["max_points"]) == point_limit
                    and entry["filter_mode"] == filter_mode
                ]
                values.append(float(matches[0][key]) if matches else np.nan)
            ax.bar(
                x + (stage_index - 1.5) * width,
                values,
                width=width,
                label=label,
                color=color,
            )
        ax.set_title(resolution, color="white", fontsize=13)
        ax.set_xticks(x, [FILTER_LABELS[value] for value in FILTER_ORDER])
        ax.set_ylabel("Mean points per frame", color="#a9bdd5")
        ax.tick_params(colors="#a9bdd5", labelsize=8)
        ax.grid(axis="y", color="#29405d", alpha=0.5)
        for spine in ax.spines.values():
            spine.set_color("#29405d")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.95),
        ncol=4,
        frameon=False,
        labelcolor="white",
    )
    fig.suptitle(
        f"Point reductions at the {point_limit // 1000}K output cap",
        color="white",
        fontsize=18,
        y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(
        gallery_dir / "point_reduction_matrix.png",
        dpi=180,
        facecolor=fig.get_facecolor(),
    )
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    gallery_dir = args.gallery_dir.expanduser().resolve()
    document = json.loads((gallery_dir / "benchmark_results.json").read_text(encoding="utf-8"))
    results = document["results"]
    point_limits = sorted({int(entry["max_points"]) for entry in results})
    for resolution in RESOLUTION_ORDER:
        if any(entry["resolution"] == resolution for entry in results):
            render_resolution(
                gallery_dir, results, resolution, point_limits, args.preview_points
            )
    render_fps_chart(gallery_dir, results)
    render_point_reduction_chart(gallery_dir, results)
    print(f"Rendered previews in {gallery_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
