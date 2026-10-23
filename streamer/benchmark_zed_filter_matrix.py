#!/usr/bin/env python3
"""Benchmark and capture the calibrated two-ZED spatial-filter matrix.

The default matrix covers every combination of four ZED resolutions, three
point caps (30K/50K/80K), and four spatial-filter modes (off, box, table,
box+table). Each case records a benchmark result and saves its first fused PLY.
One matching left-camera RGB reference is saved per resolution.
"""

from __future__ import annotations

import argparse
import itertools
import json
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
STREAMER = REPO_ROOT / "streamer" / "Streamer_real_two_cam.py"
RESULT_PREFIX = "BENCHMARK_RESULT "
CAMERA_FPS = {"VGA": 100, "HD720": 60, "HD1080": 30, "HD2K": 15}
DEFAULT_RESOLUTIONS = tuple(CAMERA_FPS)
DEFAULT_POINT_LIMITS = (30_000, 50_000, 80_000)
DEFAULT_FILTER_MODES = ("none", "bbox", "table", "both")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial", action="append", type=int, required=True)
    parser.add_argument("--transform", action="append", required=True)
    parser.add_argument("--engine-dir", type=Path, default=REPO_ROOT / "output")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "streamer" / "filter_benchmark_gallery",
    )
    parser.add_argument(
        "--resolution", action="append", choices=DEFAULT_RESOLUTIONS
    )
    parser.add_argument("--max-points", action="append", type=int)
    parser.add_argument(
        "--filter-mode", action="append", choices=DEFAULT_FILTER_MODES
    )
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--warmup-runs", type=int, default=2)
    parser.add_argument("--xyz-scale", type=float, default=1.5)
    parser.add_argument(
        "--flip-x", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--bbox-min", nargs=3, type=float, default=(-1.35, -0.70, -0.10)
    )
    parser.add_argument(
        "--bbox-max", nargs=3, type=float, default=(-0.20, 0.60, 0.50)
    )
    parser.add_argument("--table-clearance", type=float, default=0.01)
    parser.add_argument("--base-port", type=int, default=18000)
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument("--case-retries", type=int, default=3)
    parser.add_argument(
        "--case-delay",
        type=float,
        default=1.0,
        help="Camera cooldown between completed cases in seconds (default: 1)",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue an interrupted matrix from benchmark_results.json",
    )
    parser.add_argument(
        "--refresh-stage-counts",
        action="store_true",
        help=(
            "On resume, rerun the lowest point cap for each resolution/filter "
            "if its generated/box/table counts are missing"
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def unique(values: Sequence[Any]) -> list[Any]:
    return list(dict.fromkeys(values))


def parse_result(output: str) -> dict[str, Any]:
    for line in reversed(output.splitlines()):
        marker = line.find(RESULT_PREFIX)
        if marker >= 0:
            return json.loads(line[marker + len(RESULT_PREFIX) :])
    raise RuntimeError("Streamer did not emit BENCHMARK_RESULT")


def case_stem(resolution: str, points: int, filter_mode: str) -> str:
    return f"two_zed_{resolution.lower()}_{points // 1000}k_{filter_mode}"


def build_command(
    args: argparse.Namespace,
    resolution: str,
    points: int,
    filter_mode: str,
    ply_path: Path,
    rgb_path: Path | None,
    index: int,
) -> list[str]:
    command = [
        sys.executable,
        str(STREAMER),
        "--engine-dir",
        str(args.engine_dir.expanduser().resolve()),
        "--resolution",
        resolution,
        "--camera-fps",
        str(CAMERA_FPS[resolution]),
        "--max-points",
        str(points),
        "--voxel-size",
        "0",
        "--xyz-scale",
        str(args.xyz_scale),
        "--flip-x" if args.flip_x else "--no-flip-x",
        "--warmup-runs",
        str(args.warmup_runs),
        "--benchmark-frames",
        str(args.frames),
        "--port",
        str(args.base_port + index * 2),
        "--discovery-port",
        str(args.base_port + index * 2 + 1),
        "--save-first-cloud",
        str(ply_path),
    ]
    for serial in args.serial:
        command.extend(("--serial", str(serial)))
    for transform in args.transform:
        command.extend(("--transform", transform))
    if rgb_path is not None:
        command.extend(("--save-first-rgb", str(rgb_path)))
    if filter_mode in {"bbox", "both"}:
        command.extend(("--bbox-filter", "--bbox-min"))
        command.extend(str(value) for value in args.bbox_min)
        command.append("--bbox-max")
        command.extend(str(value) for value in args.bbox_max)
    else:
        command.append("--no-bbox-filter")
    if filter_mode in {"table", "both"}:
        command.extend(
            ("--table-filter", "--table-clearance", str(args.table_clearance))
        )
    else:
        command.append("--no-table-filter")
    return command


def write_document(path: Path, metadata: dict[str, Any], results: list[dict[str, Any]]) -> None:
    document = {**metadata, "completed_cases": len(results), "results": results}
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def validate_args(args: argparse.Namespace) -> None:
    if len(args.serial) != 2 or len(args.transform) != 2:
        raise ValueError("Provide exactly two --serial and two --transform arguments")
    if args.frames <= 0 or args.warmup_runs < 0 or args.case_retries <= 0:
        raise ValueError("Frames/retries must be positive and warmup cannot be negative")
    if args.case_delay < 0.0:
        raise ValueError("--case-delay cannot be negative")
    if args.xyz_scale <= 0.0 or args.table_clearance < 0.0:
        raise ValueError("Scale must be positive and table clearance cannot be negative")
    if any(a >= b for a, b in zip(args.bbox_min, args.bbox_max, strict=True)):
        raise ValueError("Every bbox-min coordinate must be below bbox-max")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    validate_args(args)
    resolutions = unique(args.resolution or DEFAULT_RESOLUTIONS)
    point_limits = unique(args.max_points or DEFAULT_POINT_LIMITS)
    filter_modes = unique(args.filter_mode or DEFAULT_FILTER_MODES)
    cases = list(itertools.product(resolutions, point_limits, filter_modes))
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "benchmark_results.json"
    metadata = {
        "bbox_max": args.bbox_max,
        "bbox_min": args.bbox_min,
        "camera_count": 2,
        "coordinate_units": "scaled metres",
        "filter_modes": filter_modes,
        "frames_per_case": args.frames,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "point_limits": point_limits,
        "resolutions": resolutions,
        "segmentation": False,
        "table_clearance": args.table_clearance,
        "xyz_scale": args.xyz_scale,
    }
    results: list[dict[str, Any]] = []
    if args.resume and result_path.exists() and not args.dry_run:
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        results = existing.get("results", [])
        if not isinstance(results, list):
            raise ValueError(f"Invalid results list in {result_path}")
    completed_names = {result["name"] for result in results}
    result_indices = {result["name"]: index for index, result in enumerate(results)}
    saved_rgb: set[str] = set()
    count_reference_limit = min(point_limits)

    for index, (resolution, points, filter_mode) in enumerate(cases):
        stem = case_stem(resolution, points, filter_mode)
        ply_path = output_dir / f"{stem}.ply"
        rgb_path = output_dir / f"two_zed_{resolution.lower()}_rgb.png"
        refresh_counts = (
            args.refresh_stage_counts
            and points == count_reference_limit
            and stem in completed_names
            and "points_generated_mean" not in results[result_indices[stem]]
        )
        if stem in completed_names and not refresh_counts:
            if not ply_path.exists():
                raise FileNotFoundError(f"Completed case is missing {ply_path}")
            saved_rgb.add(resolution)
            print(f"[{index + 1}/{len(cases)}] {stem}: already complete", flush=True)
            continue
        capture_rgb = None if resolution in saved_rgb or rgb_path.exists() else rgb_path
        if (
            ply_path.exists()
            and not args.overwrite
            and not refresh_counts
            and not args.dry_run
        ):
            raise FileExistsError(f"{ply_path} exists; use --overwrite")
        command = build_command(
            args, resolution, points, filter_mode, ply_path, capture_rgb, index
        )
        printable = [
            "<SERIAL>" if token in {str(value) for value in args.serial} else token
            for token in command
        ]
        print(
            f"[{index + 1}/{len(cases)}] {resolution} {points // 1000}K {filter_mode}",
            flush=True,
        )
        print("  " + shlex.join(printable), flush=True)
        if args.dry_run:
            continue

        completed: subprocess.CompletedProcess[str] | None = None
        for attempt in range(1, args.case_retries + 1):
            completed = subprocess.run(
                command,
                cwd=REPO_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=args.timeout,
                check=False,
            )
            if completed.returncode == 0:
                break
            if attempt < args.case_retries:
                print(f"  attempt {attempt} failed; retrying in 3 s", flush=True)
                time.sleep(3.0)
        assert completed is not None
        if completed.returncode != 0:
            tail = "\n".join(completed.stdout.splitlines()[-40:])
            raise RuntimeError(f"{stem} failed:\n{tail}")

        result = parse_result(completed.stdout)
        result.update(
            {
                "filter_mode": filter_mode,
                "name": stem,
                "ply": ply_path.name,
                "rgb_reference": rgb_path.name,
            }
        )
        if stem in result_indices:
            results[result_indices[stem]] = result
        else:
            result_indices[stem] = len(results)
            results.append(result)
            completed_names.add(stem)
        saved_rgb.add(resolution)
        write_document(result_path, metadata, results)
        print(
            f"  {result['stream_fps']:.2f} FPS | "
            f"inference {result['inference_mean_ms']:.2f} ms | "
            f"cloud {result['pointcloud_mean_ms']:.2f} ms | "
            f"points {int(result['points_generated_mean'])} -> "
            f"bbox {int(result['points_after_bbox_mean'])} -> "
            f"table {int(result['points_after_table_mean'])} -> "
            f"output {int(result['points_mean'])}",
            flush=True,
        )
        if args.case_delay > 0.0:
            time.sleep(args.case_delay)

    if args.dry_run:
        return 0
    stage_keys = (
        "points_generated_mean",
        "points_after_bbox_mean",
        "points_after_table_mean",
        "after_bbox_retained_percent",
        "after_table_retained_percent",
    )
    count_sources = {
        (result["resolution"], result["filter_mode"]): result
        for result in results
        if result["max_points"] == count_reference_limit
        and "points_generated_mean" in result
    }
    for result in results:
        source = count_sources.get((result["resolution"], result["filter_mode"]))
        if source is None:
            continue
        for key in stage_keys:
            result[key] = source[key]
        result["stage_counts_source_max_points"] = count_reference_limit
        after_table = float(result["points_after_table_mean"])
        result["points_output_retained_percent"] = round(
            0.0 if after_table == 0.0 else float(result["points_mean"]) / after_table * 100.0,
            2,
        )
    write_document(result_path, metadata, results)
    print(f"Saved {len(results)} cases to {result_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
