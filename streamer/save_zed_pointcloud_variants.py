#!/usr/bin/env python3
"""Capture one-camera ZED point-cloud/RGB variants through TensorRT FFS.

Each selected resolution, point-count limit, and XYZ scale is captured as an
exact-frame PLY/PNG pair. The output directory also receives ``manifest.json``
so the generated variants remain easy to identify and reproduce.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
STREAMER = REPO_ROOT / "streamer" / "Streamer_real_one_cam.py"
RESULT_PREFIX = "BENCHMARK_RESULT "
DEFAULT_CAMERA_FPS = {
    "VGA": 100,
    "HD720": 60,
    "HD1080": 30,
    "HD2K": 15,
}
DEFAULT_POINT_LIMITS = (5_000, 10_000, 20_000, 30_000, 40_000, 50_000)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--serial",
        type=int,
        help="ZED serial number; omit to use the first available camera",
    )
    parser.add_argument("--engine-dir", type=Path, default=REPO_ROOT / "output")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "streamer" / "pointcloud_variants",
    )
    parser.add_argument(
        "--resolution",
        action="append",
        choices=tuple(DEFAULT_CAMERA_FPS),
        help="Resolution to capture; repeat as needed (default: all four)",
    )
    parser.add_argument(
        "--max-points",
        action="append",
        type=int,
        help=(
            "Point limit to capture; repeat as needed "
            "(default: 5000, 10000, 20000, 30000, 40000, and 50000)"
        ),
    )
    parser.add_argument(
        "--xyz-scale",
        action="append",
        type=float,
        help="Uniform XYZ scale; repeat as needed (default: 1.5)",
    )
    parser.add_argument(
        "--flip-x",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the real-world/legacy VR X correction (default: on)",
    )
    parser.add_argument("--z-min", type=float, default=0.10)
    parser.add_argument("--z-max", type=float, default=5.0)
    parser.add_argument("--warmup-runs", type=int, default=2)
    parser.add_argument("--base-port", type=int, default=17900)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument(
        "--capture-retries",
        type=int,
        default=3,
        help="Attempts per variant for transient ZED open failures (default: 3)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing PLY/RGB pair with the same variant name",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without opening the camera or writing files",
    )
    return parser.parse_args(argv)


def scale_token(value: float) -> str:
    """Return a filename-safe, human-readable floating-point token."""

    return format(value, ".8g").replace("-", "neg").replace(".", "p")


def variant_stem(
    resolution: str,
    max_points: int,
    xyz_scale: float,
    flip_x: bool,
) -> str:
    x_label = "flipx" if flip_x else "rawx"
    return (
        f"one_zed_{resolution.lower()}_{max_points}pts_"
        f"scale{scale_token(xyz_scale)}_{x_label}"
    )


def parse_result(output: str) -> dict[str, Any]:
    for line in reversed(output.splitlines()):
        marker = line.find(RESULT_PREFIX)
        if marker >= 0:
            return json.loads(line[marker + len(RESULT_PREFIX) :])
    raise RuntimeError("Streamer did not emit a BENCHMARK_RESULT line")


def build_command(
    args: argparse.Namespace,
    resolution: str,
    max_points: int,
    xyz_scale: float,
    ply_path: Path,
    rgb_path: Path,
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
        str(DEFAULT_CAMERA_FPS[resolution]),
        "--max-points",
        str(max_points),
        "--xyz-scale",
        str(xyz_scale),
        "--z-min",
        str(args.z_min),
        "--z-max",
        str(args.z_max),
        "--warmup-runs",
        str(args.warmup_runs),
        "--benchmark-frames",
        "1",
        "--port",
        str(args.base_port + index * 2),
        "--discovery-port",
        str(args.base_port + index * 2 + 1),
        "--save-first-cloud",
        str(ply_path),
        "--save-first-rgb",
        str(rgb_path),
        "--flip-x" if args.flip_x else "--no-flip-x",
    ]
    if args.serial is not None:
        command.extend(("--serial", str(args.serial)))
    return command


def validate_args(args: argparse.Namespace) -> None:
    point_limits = args.max_points or DEFAULT_POINT_LIMITS
    scales = args.xyz_scale or [1.5]
    if any(limit <= 0 for limit in point_limits):
        raise ValueError("Every --max-points value must be positive")
    if any(not math.isfinite(scale) or scale <= 0.0 for scale in scales):
        raise ValueError("Every --xyz-scale value must be a positive finite number")
    if not (0.0 < args.z_min < args.z_max):
        raise ValueError("Depth range must satisfy 0 < --z-min < --z-max")
    if args.warmup_runs < 0:
        raise ValueError("--warmup-runs cannot be negative")
    if args.base_port <= 0 or args.base_port > 65534:
        raise ValueError("--base-port must be between 1 and 65534")
    if args.timeout <= 0.0:
        raise ValueError("--timeout must be positive")
    if args.capture_retries <= 0:
        raise ValueError("--capture-retries must be positive")


def read_existing_manifest(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as stream:
        document = json.load(stream)
    variants = document.get("variants", [])
    if not isinstance(variants, list):
        raise ValueError(f"Invalid variants list in {path}")
    return variants


def unique(values: Sequence[Any]) -> list[Any]:
    """Deduplicate repeatable CLI values while retaining their input order."""

    return list(dict.fromkeys(values))


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    validate_args(args)
    resolutions = unique(args.resolution or list(DEFAULT_CAMERA_FPS))
    point_limits = unique(args.max_points or DEFAULT_POINT_LIMITS)
    scales = unique(args.xyz_scale or [1.5])
    combinations = list(itertools.product(resolutions, point_limits, scales))
    if args.base_port + len(combinations) * 2 - 1 > 65535:
        raise ValueError("The selected variants exceed the available port range")

    output_dir = args.output_dir.expanduser().resolve()
    manifest_path = output_dir / "manifest.json"
    existing_variants = (
        [] if args.dry_run else read_existing_manifest(manifest_path)
    )
    captured: list[dict[str, Any]] = []

    for index, (resolution, max_points, xyz_scale) in enumerate(combinations):
        stem = variant_stem(resolution, max_points, xyz_scale, args.flip_x)
        ply_path = output_dir / f"{stem}.ply"
        rgb_path = output_dir / f"{stem}.png"
        if not args.dry_run and not args.overwrite:
            existing = [path for path in (ply_path, rgb_path) if path.exists()]
            if existing:
                names = ", ".join(path.name for path in existing)
                raise FileExistsError(
                    f"Variant output already exists ({names}); use --overwrite to replace it"
                )

        command = build_command(
            args,
            resolution,
            max_points,
            xyz_scale,
            ply_path,
            rgb_path,
            index,
        )
        printable = [
            "<SERIAL>"
            if args.serial is not None and token == str(args.serial)
            else token
            for token in command
        ]
        print(
            f"[{index + 1}/{len(combinations)}] {resolution}, "
            f"{max_points} points, scale {xyz_scale:g}"
        )
        print("  " + shlex.join(printable), flush=True)
        if args.dry_run:
            continue

        output_dir.mkdir(parents=True, exist_ok=True)
        completed: subprocess.CompletedProcess[str] | None = None
        for attempt in range(1, args.capture_retries + 1):
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
            if attempt < args.capture_retries:
                print(
                    f"  Capture attempt {attempt} failed; retrying in 3 seconds",
                    flush=True,
                )
                time.sleep(3.0)
        assert completed is not None
        if completed.returncode != 0:
            tail = "\n".join(completed.stdout.splitlines()[-30:])
            raise RuntimeError(f"{stem} failed with code {completed.returncode}:\n{tail}")
        result = parse_result(completed.stdout)
        captured.append(
            {
                "camera_fps": DEFAULT_CAMERA_FPS[resolution],
                "flip_x": args.flip_x,
                "max_points": max_points,
                "ply": ply_path.name,
                "resolution": resolution,
                "rgb": rgb_path.name,
                "stream_result": result,
                "xyz_scale": xyz_scale,
                "z_max_m_before_scale": args.z_max,
                "z_min_m_before_scale": args.z_min,
            }
        )

    if args.dry_run:
        return 0

    captured_names = {variant["ply"] for variant in captured}
    retained = [
        variant
        for variant in existing_variants
        if variant.get("ply") not in captured_names
    ]
    document = {
        "coordinate_units": "scaled metres",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "schema_version": 1,
        "segmentation": False,
        "variants": retained + captured,
    }
    with manifest_path.open("w", encoding="utf-8") as stream:
        json.dump(document, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(f"Saved {len(captured)} PLY/RGB pair(s) and {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
