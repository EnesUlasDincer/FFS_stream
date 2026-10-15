#!/usr/bin/env python3
"""Run a reproducible Fast-FoundationStereo ZED streaming benchmark matrix."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULT_PREFIX = "BENCHMARK_RESULT "


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    camera_count: int
    resolution: str
    camera_fps: int
    max_points: int
    voxel_size: float = 0.0
    bbox_min: tuple[float, float, float] | None = None
    bbox_max: tuple[float, float, float] | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--serial",
        action="append",
        type=int,
        required=True,
        help="ZED serial number; provide one or two",
    )
    parser.add_argument("--engine-dir", type=Path, default=REPO_ROOT / "output")
    parser.add_argument("--frames", type=int, default=180)
    parser.add_argument("--warmup-runs", type=int, default=3)
    parser.add_argument("--base-port", type=int, default=16600)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument(
        "--suite",
        choices=("all", "resolution"),
        default="all",
        help="Benchmark matrix to run (default: all)",
    )
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def build_resolution_cases(include_two_camera: bool) -> list[BenchmarkCase]:
    cases = [
        BenchmarkCase("one_vga_100fps_15k", 1, "VGA", 100, 15_000),
        BenchmarkCase("one_hd720_60fps_15k", 1, "HD720", 60, 15_000),
        BenchmarkCase("one_hd1080_30fps_15k", 1, "HD1080", 30, 15_000),
        BenchmarkCase("one_hd2k_15fps_15k", 1, "HD2K", 15, 15_000),
    ]
    if include_two_camera:
        cases.extend(
            [
                BenchmarkCase("two_vga_100fps_15k", 2, "VGA", 100, 15_000),
                BenchmarkCase("two_hd720_60fps_15k", 2, "HD720", 60, 15_000),
                BenchmarkCase("two_hd1080_30fps_15k", 2, "HD1080", 30, 15_000),
                BenchmarkCase("two_hd2k_15fps_15k", 2, "HD2K", 15, 15_000),
            ]
        )
    return cases


def build_cases(
    include_two_camera: bool,
    suite: str = "all",
) -> list[BenchmarkCase]:
    if suite == "resolution":
        return build_resolution_cases(include_two_camera)

    cases = [
        BenchmarkCase("one_vga_30fps_15k", 1, "VGA", 30, 15_000),
        BenchmarkCase("one_vga_60fps_15k", 1, "VGA", 60, 15_000),
        BenchmarkCase("one_vga_100fps_5k", 1, "VGA", 100, 5_000),
        BenchmarkCase("one_vga_100fps_15k", 1, "VGA", 100, 15_000),
        BenchmarkCase(
            "one_vga_100fps_bbox",
            1,
            "VGA",
            100,
            15_000,
            bbox_min=(-0.5, -0.5, 0.2),
            bbox_max=(0.5, 0.5, 2.0),
        ),
        BenchmarkCase("one_vga_100fps_30k", 1, "VGA", 100, 30_000),
        BenchmarkCase("one_vga_100fps_unlimited", 1, "VGA", 100, 0),
        BenchmarkCase("one_vga_100fps_voxel5mm", 1, "VGA", 100, 15_000, 0.005),
        BenchmarkCase("one_hd720_60fps_15k", 1, "HD720", 60, 15_000),
    ]
    if include_two_camera:
        cases.extend(
            [
                BenchmarkCase("two_vga_100fps_5k", 2, "VGA", 100, 5_000),
                BenchmarkCase("two_vga_100fps_15k", 2, "VGA", 100, 15_000),
                BenchmarkCase("two_vga_100fps_30k", 2, "VGA", 100, 30_000),
                BenchmarkCase(
                    "two_vga_100fps_voxel5mm",
                    2,
                    "VGA",
                    100,
                    15_000,
                    0.005,
                ),
            ]
        )
    return cases


def build_command(
    case: BenchmarkCase,
    args: argparse.Namespace,
    index: int,
) -> list[str]:
    entry_point = (
        "streamer/Streamer_real_one_cam.py"
        if case.camera_count == 1
        else "streamer/Streamer_real_two_cam.py"
    )
    command = [
        sys.executable,
        entry_point,
        "--engine-dir",
        str(args.engine_dir),
        "--resolution",
        case.resolution,
        "--camera-fps",
        str(case.camera_fps),
        "--max-points",
        str(case.max_points),
        "--voxel-size",
        str(case.voxel_size),
        "--benchmark-frames",
        str(args.frames),
        "--warmup-runs",
        str(args.warmup_runs),
        "--port",
        str(args.base_port + index * 2),
        "--discovery-port",
        str(args.base_port + index * 2 + 1),
    ]
    for serial in args.serial[: case.camera_count]:
        command.extend(("--serial", str(serial)))
    if case.bbox_min is not None and case.bbox_max is not None:
        command.append("--bbox-filter")
        command.append("--bbox-min")
        command.extend(str(value) for value in case.bbox_min)
        command.append("--bbox-max")
        command.extend(str(value) for value in case.bbox_max)
    return command


def parse_result(output: str) -> dict[str, Any]:
    for line in reversed(output.splitlines()):
        marker = line.find(RESULT_PREFIX)
        if marker >= 0:
            return json.loads(line[marker + len(RESULT_PREFIX) :])
    raise RuntimeError("Streamer did not emit a BENCHMARK_RESULT line")


def run_case(
    case: BenchmarkCase,
    args: argparse.Namespace,
    index: int,
) -> dict[str, Any]:
    command = build_command(case, args, index)
    serial_tokens = {str(serial) for serial in args.serial}
    printable = ["<SERIAL>" if token in serial_tokens else token for token in command]
    print(f"[{index + 1}] {case.name}", flush=True)
    print("  " + shlex.join(printable), flush=True)
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=args.timeout,
        check=False,
    )
    if completed.returncode != 0:
        tail = "\n".join(completed.stdout.splitlines()[-30:])
        raise RuntimeError(f"{case.name} failed with code {completed.returncode}:\n{tail}")

    result = parse_result(completed.stdout)
    result["name"] = case.name
    print(
        f"  {result['stream_fps']:.2f} FPS, "
        f"inference {result['inference_mean_ms']:.2f} ms, "
        f"cloud {result['pointcloud_mean_ms']:.2f} ms",
        flush=True,
    )
    return result


def main() -> int:
    args = parse_args()
    if not 1 <= len(args.serial) <= 2:
        raise ValueError("Provide exactly one or two --serial arguments")
    if args.frames <= 0:
        raise ValueError("--frames must be positive")
    if args.warmup_runs < 0:
        raise ValueError("--warmup-runs cannot be negative")

    cases = build_cases(
        include_two_camera=len(args.serial) == 2,
        suite=args.suite,
    )
    results = [run_case(case, args, index) for index, case in enumerate(cases)]
    document = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "frames_per_case": args.frames,
        "suite": args.suite,
        "results": results,
    }
    encoded = json.dumps(document, indent=2, sort_keys=True) + "\n"
    if args.output_json is not None:
        output_path = args.output_json.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(encoded, encoding="utf-8")
        print(f"Saved {output_path}")
    else:
        print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
