#!/usr/bin/env python3
"""View a saved PLY or the live ZMQ RGB point-cloud stream with Open3D."""

from __future__ import annotations

import argparse
import struct
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import zmq


def add_view_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--point-size", type=float, default=2.0)
    parser.add_argument(
        "--background",
        nargs=3,
        type=float,
        default=(0.05, 0.05, 0.05),
        metavar=("R", "G", "B"),
    )
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--no-axis", action="store_true")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)

    file_parser = subparsers.add_parser("file", help="Open a saved PLY file")
    file_parser.add_argument("path", type=Path)
    file_parser.add_argument(
        "--headless",
        action="store_true",
        help="Validate the cloud and print its bounds without opening a window",
    )
    add_view_arguments(file_parser)

    live_parser = subparsers.add_parser("live", help="View the live ZMQ stream")
    live_parser.add_argument(
        "--endpoint",
        default="tcp://127.0.0.1:5555",
        help="ZMQ publisher endpoint",
    )
    live_parser.add_argument("--topic", default="PointCloud")
    add_view_arguments(live_parser)
    return parser.parse_args()


def load_open3d() -> Any:
    try:
        import open3d as o3d
    except ImportError as exc:
        raise RuntimeError("Install Open3D with: pip install open3d") from exc
    return o3d


def decode_zmq_message(
    parts: Sequence[bytes],
    expected_topic: bytes = b"PointCloud",
) -> tuple[np.ndarray, np.ndarray]:
    """Decode the streamer's four-part point-cloud message."""

    if len(parts) != 4:
        raise ValueError(f"Expected 4 message parts, received {len(parts)}")
    topic, count_bytes, xyz_bytes, rgb_bytes = parts
    if topic != expected_topic:
        raise ValueError(f"Unexpected topic {topic!r}")
    if len(count_bytes) != 4:
        raise ValueError("Point-count frame must contain one little-endian uint32")

    count = struct.unpack("<I", count_bytes)[0]
    expected_xyz_bytes = count * 3 * np.dtype("<i2").itemsize
    expected_rgb_bytes = count * 3
    if len(xyz_bytes) != expected_xyz_bytes:
        raise ValueError(
            f"XYZ payload has {len(xyz_bytes)} bytes, expected {expected_xyz_bytes}"
        )
    if len(rgb_bytes) != expected_rgb_bytes:
        raise ValueError(
            f"RGB payload has {len(rgb_bytes)} bytes, expected {expected_rgb_bytes}"
        )

    points = np.frombuffer(xyz_bytes, dtype="<i2").reshape(count, 3)
    colors = np.frombuffer(rgb_bytes, dtype=np.uint8).reshape(count, 3)
    return points.astype(np.float64) / 1000.0, colors.astype(np.float64) / 255.0


def print_cloud_stats(points: np.ndarray) -> None:
    if len(points) == 0:
        print("Point cloud is empty")
        return
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    print(f"Points: {len(points)}")
    print(f"Minimum XYZ [m]: {minimum}")
    print(f"Maximum XYZ [m]: {maximum}")


def configure_window(visualizer: Any, args: argparse.Namespace) -> None:
    render = visualizer.get_render_option()
    render.point_size = args.point_size
    render.background_color = np.asarray(args.background, dtype=np.float64)


def add_axis(visualizer: Any, o3d: Any, hidden: bool) -> None:
    if not hidden:
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.25)
        visualizer.add_geometry(axis, reset_bounding_box=False)


def open_file(args: argparse.Namespace) -> int:
    o3d = load_open3d()
    path = args.path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Point-cloud file not found: {path}")

    cloud = o3d.io.read_point_cloud(str(path))
    points = np.asarray(cloud.points)
    if len(points) == 0:
        raise RuntimeError(f"No points could be read from {path}")
    print_cloud_stats(points)
    if args.headless:
        return 0

    visualizer = o3d.visualization.Visualizer()
    if not visualizer.create_window(
        window_name=f"Fast-FoundationStereo - {path.name}",
        width=args.width,
        height=args.height,
    ):
        raise RuntimeError("Open3D could not create a visualization window")
    try:
        configure_window(visualizer, args)
        visualizer.add_geometry(cloud)
        add_axis(visualizer, o3d, args.no_axis)
        visualizer.run()
    finally:
        visualizer.destroy_window()
    return 0


def open_live(args: argparse.Namespace) -> int:
    o3d = load_open3d()
    context = zmq.Context()
    subscriber = context.socket(zmq.SUB)
    subscriber.setsockopt(zmq.RCVHWM, 1)
    subscriber.setsockopt(zmq.LINGER, 0)
    topic = args.topic.encode("utf-8")
    subscriber.setsockopt(zmq.SUBSCRIBE, topic)
    subscriber.connect(args.endpoint)

    visualizer = o3d.visualization.Visualizer()
    if not visualizer.create_window(
        window_name="Fast-FoundationStereo live point cloud",
        width=args.width,
        height=args.height,
    ):
        subscriber.close()
        context.term()
        raise RuntimeError("Open3D could not create a visualization window")

    cloud = o3d.geometry.PointCloud()
    cloud_added = False
    received = 0
    report_started = time.perf_counter()
    configure_window(visualizer, args)
    add_axis(visualizer, o3d, args.no_axis)
    print(f"Waiting for {args.topic!r} at {args.endpoint}; press Q or Esc to close")
    try:
        while visualizer.poll_events():
            latest: Sequence[bytes] | None = None
            while True:
                try:
                    latest = subscriber.recv_multipart(flags=zmq.NOBLOCK)
                except zmq.Again:
                    break

            if latest is None:
                visualizer.update_renderer()
                time.sleep(0.005)
                continue

            points, colors = decode_zmq_message(latest, topic)
            if len(points) == 0:
                continue
            cloud.points = o3d.utility.Vector3dVector(points)
            cloud.colors = o3d.utility.Vector3dVector(colors)
            if cloud_added:
                visualizer.update_geometry(cloud)
            else:
                visualizer.add_geometry(cloud)
                cloud_added = True
            visualizer.update_renderer()

            received += 1
            elapsed = time.perf_counter() - report_started
            if elapsed >= 1.0:
                print(f"Viewer {received / elapsed:.1f} FPS | {len(points)} points")
                received = 0
                report_started = time.perf_counter()
    finally:
        visualizer.destroy_window()
        subscriber.close()
        context.term()
    return 0


def main() -> int:
    args = parse_args()
    if args.point_size <= 0.0:
        raise ValueError("--point-size must be positive")
    if args.width <= 0 or args.height <= 0:
        raise ValueError("Window dimensions must be positive")
    background = np.asarray(args.background)
    if not np.logical_and(0.0 <= background, background <= 1.0).all():
        raise ValueError("--background values must be between 0 and 1")
    return open_file(args) if args.mode == "file" else open_live(args)


if __name__ == "__main__":
    raise SystemExit(main())
