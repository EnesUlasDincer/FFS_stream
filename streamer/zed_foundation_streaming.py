"""Live ZED -> Fast-FoundationStereo TensorRT -> ZMQ point-cloud streaming.

The ZED SDK is used only to capture rectified left/right RGB images and read
the factory calibration. Depth is produced by the two-stage TensorRT version
of Fast-FoundationStereo. No segmentation model or segmentation server is
used in this pipeline.

The publisher keeps the wire format used by ``ZMQPublishPointCloudAction``:

    [b"PointCloud", uint32 point_count, int16 XYZ millimetres, uint8 RGB]

This module contains the shared implementation. Use
``Streamer_real_one_cam.py`` or ``Streamer_real_two_cam.py`` as entry points.
"""

from __future__ import annotations

import argparse
import json
import logging
import socket
import struct
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Sequence

import cv2
import numpy as np
import torch
import yaml
import zmq
from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.foundation_stereo import TrtRunner  # noqa: E402


# Intentionally not configurable: this streamer never starts segmentation.
SEGMENTATION_ENABLED: Final[bool] = False


@dataclass(frozen=True)
class CameraCalibration:
    """Rectified left-camera calibration at the captured resolution."""

    matrix: np.ndarray
    baseline_m: float

    def scaled(self, scale_x: float, scale_y: float) -> "CameraCalibration":
        matrix = self.matrix.copy()
        matrix[0, :] *= scale_x
        matrix[1, :] *= scale_y
        return CameraCalibration(matrix=matrix, baseline_m=self.baseline_m)


@dataclass(frozen=True)
class StereoFrame:
    left_rgb: np.ndarray
    right_rgb: np.ndarray
    calibration: CameraCalibration


class ZedStereoCamera:
    """Small owner wrapper around one ZED SDK camera."""

    def __init__(
        self,
        serial: int | None,
        resolution: str,
        fps: int,
    ) -> None:
        self.requested_serial = serial
        self.resolution = resolution
        self.fps = fps
        self.serial: int | None = None
        self.calibration: CameraCalibration | None = None
        self._sl: Any = None
        self._camera: Any = None
        self._runtime: Any = None
        self._left: Any = None
        self._right: Any = None

    def open(self) -> None:
        try:
            import pyzed.sl as sl
        except ImportError as exc:
            raise RuntimeError(
                "The ZED Python API is not installed. Install it with "
                "/usr/local/zed/get_python_api.py in the active environment."
            ) from exc

        camera = sl.Camera()
        init = sl.InitParameters()
        init.camera_resolution = getattr(sl.RESOLUTION, self.resolution)
        init.camera_fps = self.fps
        init.depth_mode = sl.DEPTH_MODE.NONE
        init.coordinate_units = sl.UNIT.METER
        if self.requested_serial is not None:
            init.set_from_serial_number(self.requested_serial)

        status = camera.open(init)
        if status != sl.ERROR_CODE.SUCCESS:
            camera.close()
            target = self.requested_serial if self.requested_serial is not None else "default"
            raise RuntimeError(f"Could not open ZED camera {target}: {status}")

        info = camera.get_camera_information()
        calibration = info.camera_configuration.calibration_parameters
        left = calibration.left_cam
        matrix = np.array(
            [[left.fx, 0.0, left.cx], [0.0, left.fy, left.cy], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )

        # InitParameters.coordinate_units is METER, so both APIs return metres.
        if hasattr(calibration, "get_camera_baseline"):
            baseline_m = abs(float(calibration.get_camera_baseline()))
        else:
            translation = calibration.stereo_transform.get_translation().get()
            baseline_m = abs(float(translation[0]))
        if baseline_m <= 0.0:
            camera.close()
            raise RuntimeError("ZED returned a non-positive stereo baseline")

        self._sl = sl
        self._camera = camera
        self._runtime = sl.RuntimeParameters()
        self._left = sl.Mat()
        self._right = sl.Mat()
        self.serial = int(info.serial_number)
        self.calibration = CameraCalibration(matrix=matrix, baseline_m=baseline_m)

        size = info.camera_configuration.resolution
        logging.info(
            "Opened ZED %s at %dx%d @ %d FPS (baseline %.3f m)",
            self.serial,
            size.width,
            size.height,
            self.fps,
            baseline_m,
        )

    def grab(self) -> StereoFrame | None:
        if self._camera is None or self.calibration is None:
            raise RuntimeError("Camera must be opened before grabbing frames")

        status = self._camera.grab(self._runtime)
        if status != self._sl.ERROR_CODE.SUCCESS:
            return None

        self._camera.retrieve_image(self._left, self._sl.VIEW.LEFT)
        self._camera.retrieve_image(self._right, self._sl.VIEW.RIGHT)

        # VIEW.LEFT/RIGHT are rectified. PyZED exposes their pixels as BGRA.
        left_rgb = cv2.cvtColor(self._left.get_data(), cv2.COLOR_BGRA2RGB)
        right_rgb = cv2.cvtColor(self._right.get_data(), cv2.COLOR_BGRA2RGB)
        return StereoFrame(
            left_rgb=np.ascontiguousarray(left_rgb),
            right_rgb=np.ascontiguousarray(right_rgb),
            calibration=self.calibration,
        )

    def close(self) -> None:
        if self._camera is not None:
            self._camera.close()
            self._camera = None


class FastFoundationStereoTensorRTRunner:
    """Runs the repository's fixed-shape, two-stage TensorRT pipeline."""

    def __init__(self, engine_dir: Path) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("Fast-FoundationStereo TensorRT inference requires a CUDA GPU")

        engine_dir = engine_dir.expanduser().resolve()
        config_path = engine_dir / "onnx.yaml"
        feature_path = engine_dir / "feature_runner.engine"
        post_path = engine_dir / "post_runner.engine"
        for path in (config_path, feature_path, post_path):
            if not path.is_file():
                raise FileNotFoundError(f"TensorRT pipeline file not found: {path}")

        with config_path.open("r", encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
        if "image_size" not in config:
            raise KeyError(f"image_size is missing from {config_path}")

        self.height, self.width = map(int, config["image_size"])
        self.args = OmegaConf.create(config)

        torch.set_grad_enabled(False)
        torch.backends.cudnn.benchmark = True

        logging.info(
            "Loading Fast-FoundationStereo TensorRT engines from %s (%dx%d)",
            engine_dir,
            self.width,
            self.height,
        )
        self.model = TrtRunner(self.args, str(feature_path), str(post_path))
        self._stream = torch.cuda.Stream()
        self._validate_engine_input_shape()

    def _validate_engine_input_shape(self) -> None:
        expected = (1, 3, self.height, self.width)
        for name in ("left", "right"):
            actual = tuple(self.model.feature_engine.get_tensor_shape(name))
            if actual != expected:
                raise ValueError(
                    f"TensorRT input '{name}' has shape {actual}, expected {expected} "
                    f"from onnx.yaml"
                )

    def _forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        # This is the actual TensorRT inference call. TrtRunner executes the
        # feature engine, builds GWC with Triton, then executes the post engine.
        caller_stream = torch.cuda.current_stream()
        self._stream.wait_stream(caller_stream)
        with torch.inference_mode():
            with torch.cuda.stream(self._stream):
                disparity = self.model.forward(left, right)
        caller_stream.wait_stream(self._stream)
        return disparity

    def warmup(self, runs: int) -> None:
        if runs <= 0:
            return
        logging.info("Warming up TensorRT pipeline for %d run(s)", runs)
        left = torch.zeros((1, 3, self.height, self.width), device="cuda")
        right = torch.zeros_like(left)
        for _ in range(runs):
            self._forward(left, right)
        torch.cuda.synchronize()

    def infer(self, frames: Sequence[StereoFrame]) -> tuple[list[np.ndarray], list[StereoFrame]]:
        disparities: list[np.ndarray] = []
        prepared_frames: list[StereoFrame] = []

        # The current engines have static batch size 1. Reuse their execution
        # contexts for each camera instead of pretending that batch 2 is valid.
        for frame in frames:
            prepared = resize_stereo_frame(frame, self.height, self.width)
            left = (
                torch.from_numpy(prepared.left_rgb)
                .cuda(non_blocking=True)
                .float()
                .permute(2, 0, 1)
                .unsqueeze(0)
                .contiguous()
            )
            right = (
                torch.from_numpy(prepared.right_rgb)
                .cuda(non_blocking=True)
                .float()
                .permute(2, 0, 1)
                .unsqueeze(0)
                .contiguous()
            )
            disparity = self._forward(left, right)
            if tuple(disparity.shape) != (1, 1, self.height, self.width):
                raise RuntimeError(f"Unexpected TensorRT disparity shape: {disparity.shape}")
            disparities.append(disparity[0, 0].float().cpu().numpy().clip(0.0, None))
            prepared_frames.append(prepared)

        return disparities, prepared_frames


def resize_stereo_frame(
    frame: StereoFrame,
    target_height: int,
    target_width: int,
) -> StereoFrame:
    height, width = frame.left_rgb.shape[:2]
    if (height, width) == (target_height, target_width):
        return frame

    size = (target_width, target_height)
    interpolation = (
        cv2.INTER_AREA
        if target_width < width and target_height < height
        else cv2.INTER_LINEAR
    )
    left = cv2.resize(frame.left_rgb, size, interpolation=interpolation)
    right = cv2.resize(frame.right_rgb, size, interpolation=interpolation)
    calibration = frame.calibration.scaled(target_width / width, target_height / height)
    return StereoFrame(left_rgb=left, right_rgb=right, calibration=calibration)


def disparity_to_point_cloud(
    disparity: np.ndarray,
    left_rgb: np.ndarray,
    calibration: CameraCalibration,
    z_min: float,
    z_max: float,
    remove_invisible: bool,
    camera_to_output: np.ndarray,
    sample_limit: int = 0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Back-project one disparity map and transform it to the output frame."""

    fx = float(calibration.matrix[0, 0])
    fy = float(calibration.matrix[1, 1])
    cx = float(calibration.matrix[0, 2])
    cy = float(calibration.matrix[1, 2])

    valid = np.isfinite(disparity) & (disparity > 1e-6)
    if remove_invisible:
        u_coordinates = np.arange(disparity.shape[1], dtype=np.float32)[None, :]
        valid &= (u_coordinates - disparity) >= 0.0

    depth = np.zeros_like(disparity, dtype=np.float32)
    depth[valid] = fx * calibration.baseline_m / disparity[valid]
    valid &= np.isfinite(depth) & (depth >= z_min) & (depth <= z_max)

    rows, columns = np.nonzero(valid)
    if rows.size == 0:
        empty = np.empty((0, 3), dtype=np.float32)
        return empty, empty.copy()
    if sample_limit > 0 and rows.size > sample_limit:
        generator = rng if rng is not None else np.random.default_rng()
        keep = generator.choice(rows.size, sample_limit, replace=False)
        rows = rows[keep]
        columns = columns[keep]

    z = depth[rows, columns]
    x = (columns.astype(np.float32) - cx) * z / fx
    y = (rows.astype(np.float32) - cy) * z / fy
    points = np.column_stack((x, y, z)).astype(np.float32, copy=False)
    colors = left_rgb[rows, columns].astype(np.float32) / 255.0

    # Identity is the benchmark/default path. Avoid an unnecessary matrix
    # multiplication when the cloud already uses the requested output frame.
    if not np.array_equal(camera_to_output, np.eye(4, dtype=camera_to_output.dtype)):
        rotation = camera_to_output[:3, :3].astype(np.float32)
        translation = camera_to_output[:3, 3].astype(np.float32)
        points = points @ rotation.T + translation
    return np.ascontiguousarray(points), np.ascontiguousarray(colors)


class PointCloudProcessor:
    """Merges, bounds, downsamples, and caps the streamed cloud."""

    def __init__(
        self,
        voxel_size: float,
        max_points: int,
        bbox_min: np.ndarray | None,
        bbox_max: np.ndarray | None,
        random_seed: int,
    ) -> None:
        self.voxel_size = voxel_size
        self.max_points = max_points
        self.bbox_min = bbox_min
        self.bbox_max = bbox_max
        self.rng = np.random.default_rng(random_seed)

    def process(
        self, clouds: Sequence[tuple[np.ndarray, np.ndarray]]
    ) -> tuple[np.ndarray, np.ndarray]:
        points = np.concatenate([cloud[0] for cloud in clouds], axis=0)
        colors = np.concatenate([cloud[1] for cloud in clouds], axis=0)

        valid = np.isfinite(points).all(axis=1) & np.isfinite(colors).all(axis=1)
        if self.bbox_min is not None and self.bbox_max is not None:
            valid &= (points >= self.bbox_min).all(axis=1)
            valid &= (points <= self.bbox_max).all(axis=1)
        points = points[valid]
        colors = colors[valid]

        if self.voxel_size > 0.0 and len(points) > 0:
            voxel_ids = np.floor(points / self.voxel_size).astype(np.int64)
            _, keep = np.unique(voxel_ids, axis=0, return_index=True)
            points = points[keep]
            colors = colors[keep]

        if self.max_points > 0 and len(points) > self.max_points:
            keep = self.rng.choice(len(points), self.max_points, replace=False)
            points = points[keep]
            colors = colors[keep]

        return np.ascontiguousarray(points), np.ascontiguousarray(colors)


def save_point_cloud_ply(
    path: Path,
    points_m: np.ndarray,
    colors_rgb: np.ndarray,
) -> None:
    """Write an RGB point cloud as a compact binary little-endian PLY file."""

    if points_m.shape != colors_rgb.shape or points_m.ndim != 2 or points_m.shape[1] != 3:
        raise ValueError("Points and colors must both have shape (N, 3)")

    output_path = path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    vertices = np.empty(
        len(points_m),
        dtype=np.dtype(
            [
                ("x", "<f4"),
                ("y", "<f4"),
                ("z", "<f4"),
                ("red", "u1"),
                ("green", "u1"),
                ("blue", "u1"),
            ]
        ),
    )
    vertices["x"] = points_m[:, 0]
    vertices["y"] = points_m[:, 1]
    vertices["z"] = points_m[:, 2]
    colors_u8 = np.rint(np.clip(colors_rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
    vertices["red"] = colors_u8[:, 0]
    vertices["green"] = colors_u8[:, 1]
    vertices["blue"] = colors_u8[:, 2]

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(vertices)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    with output_path.open("wb") as stream:
        stream.write(header.encode("ascii"))
        vertices.tofile(stream)


class PointCloudPublisher:
    """ZMQ publisher compatible with the repository's existing receiver."""

    def __init__(
        self,
        port: int,
        discovery_port: int,
        topic: str = "PointCloud",
    ) -> None:
        self.topic = topic.encode("utf-8")
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.PUB)
        self._socket.setsockopt(zmq.SNDHWM, 1)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.bind(f"tcp://*:{port}")

        self._discovery_stop = threading.Event()
        self._discovery_thread = threading.Thread(
            target=self._discovery_loop,
            args=(discovery_port,),
            name="point-cloud-discovery",
            daemon=True,
        )
        self._discovery_thread.start()
        logging.info("Point cloud PUB: tcp://*:%d", port)

    def _discovery_loop(self, port: int) -> None:
        udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        udp.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        udp.settimeout(0.25)
        try:
            udp.bind(("", port))
            while not self._discovery_stop.is_set():
                try:
                    data, address = udp.recvfrom(1024)
                except socket.timeout:
                    continue
                if data == b"DISCOVER_ZMQ_SERVER":
                    udp.sendto(b"ZMQ_SERVER_HERE", address)
        except OSError as exc:
            logging.warning("Discovery server disabled on UDP %d: %s", port, exc)
        finally:
            udp.close()

    def publish(self, points_m: np.ndarray, colors_rgb: np.ndarray) -> bool:
        if points_m.shape != colors_rgb.shape or points_m.ndim != 2 or points_m.shape[1] != 3:
            raise ValueError("Points and colors must both have shape (N, 3)")

        # The protocol stores coordinates as signed millimetres.
        within_int16 = (np.abs(points_m) <= 32.767).all(axis=1)
        if not within_int16.all():
            points_m = points_m[within_int16]
            colors_rgb = colors_rgb[within_int16]

        xyz_mm = np.ascontiguousarray(np.rint(points_m * 1000.0).astype("<i2"))
        rgb_u8 = np.ascontiguousarray(
            np.rint(np.clip(colors_rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
        )
        count = len(xyz_mm)
        try:
            self._socket.send_multipart(
                [
                    self.topic,
                    struct.pack("<I", count),
                    memoryview(xyz_mm),
                    memoryview(rgb_u8),
                ],
                flags=zmq.NOBLOCK,
                copy=False,
            )
            return True
        except zmq.Again:
            return False

    def close(self) -> None:
        self._discovery_stop.set()
        self._discovery_thread.join(timeout=1.0)
        self._socket.close(linger=0)
        self._context.term()


class StreamRateLogger:
    def __init__(self) -> None:
        now = time.perf_counter()
        self.total_started = now
        self.interval_started = now
        self.interval_frames = 0
        self.total_frames = 0
        self.inference_times_ms: list[float] = []
        self.pointcloud_times_ms: list[float] = []
        self.point_counts: list[int] = []

    def update(self, points: int, inference_ms: float, pointcloud_ms: float) -> None:
        self.interval_frames += 1
        self.total_frames += 1
        self.inference_times_ms.append(inference_ms)
        self.pointcloud_times_ms.append(pointcloud_ms)
        self.point_counts.append(points)
        now = time.perf_counter()
        elapsed = now - self.interval_started
        if elapsed >= 1.0:
            logging.info(
                "Stream %.1f FPS | inference %.1f ms | cloud %.1f ms | "
                "%d points | segmentation=%s",
                self.interval_frames / elapsed,
                inference_ms,
                pointcloud_ms,
                points,
                SEGMENTATION_ENABLED,
            )
            self.interval_frames = 0
            self.interval_started = now

    def benchmark_result(
        self,
        camera_count: int,
        resolution: str,
        requested_camera_fps: int,
        max_points: int,
        voxel_size: float,
        bbox_filter: bool,
    ) -> dict[str, int | float | str | bool]:
        if self.total_frames == 0:
            raise RuntimeError("Cannot summarize an empty stream")

        elapsed = time.perf_counter() - self.total_started
        inference = np.asarray(self.inference_times_ms, dtype=np.float64)
        pointcloud = np.asarray(self.pointcloud_times_ms, dtype=np.float64)
        points = np.asarray(self.point_counts, dtype=np.float64)
        return {
            "bbox_filter": bbox_filter,
            "camera_count": camera_count,
            "elapsed_s": round(elapsed, 4),
            "frames": self.total_frames,
            "inference_mean_ms": round(float(inference.mean()), 4),
            "inference_p95_ms": round(float(np.percentile(inference, 95)), 4),
            "max_points": max_points,
            "pointcloud_mean_ms": round(float(pointcloud.mean()), 4),
            "pointcloud_p95_ms": round(float(np.percentile(pointcloud, 95)), 4),
            "points_mean": round(float(points.mean()), 1),
            "requested_camera_fps": requested_camera_fps,
            "resolution": resolution,
            "segmentation": SEGMENTATION_ENABLED,
            "stream_fps": round(self.total_frames / elapsed, 4),
            "voxel_size_m": voxel_size,
        }


def _matrix_from_yaml(data: Any, key: str | None, source: Path) -> np.ndarray:
    if key:
        value = data
        for part in key.split("."):
            if not isinstance(value, dict) or part not in value:
                raise KeyError(f"Transform key '{key}' not found in {source}")
            value = value[part]
        return np.asarray(value, dtype=np.float64)

    direct = np.asarray(data, dtype=np.float64) if isinstance(data, list) else None
    if direct is not None and direct.shape == (4, 4):
        return direct

    if isinstance(data, dict):
        candidates: list[np.ndarray] = []
        for value in data.values():
            try:
                candidate = np.asarray(value, dtype=np.float64)
            except (TypeError, ValueError):
                continue
            if candidate.shape == (4, 4):
                candidates.append(candidate)
        if len(candidates) == 1:
            return candidates[0]

    raise ValueError(
        f"Could not choose one 4x4 matrix from {source}; append ':yaml_key' to the path"
    )


def load_transform(specification: str) -> np.ndarray:
    """Load camera-to-output 4x4 transform from identity, NPY, text, or YAML."""

    if specification.lower() == "identity":
        return np.eye(4, dtype=np.float64)

    path_text = specification
    yaml_key: str | None = None
    prefix, separator, suffix = specification.rpartition(":")
    if separator and Path(prefix).is_file():
        path_text = prefix
        yaml_key = suffix

    path = Path(path_text).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Transform file not found: {path}")

    if path.suffix.lower() == ".npy":
        matrix = np.load(path)
    elif path.suffix.lower() in {".yaml", ".yml"}:
        with path.open("r", encoding="utf-8") as stream:
            matrix = _matrix_from_yaml(yaml.safe_load(stream), yaml_key, path)
    else:
        matrix = np.loadtxt(path)

    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"Transform in {path} has shape {matrix.shape}, expected (4, 4)")
    if not np.isfinite(matrix).all():
        raise ValueError(f"Transform in {path} contains non-finite values")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
        raise ValueError(f"Transform in {path} has an invalid homogeneous last row")
    return matrix


def build_parser(camera_count: int) -> argparse.ArgumentParser:
    label = "one" if camera_count == 1 else "two"
    noun = "camera" if camera_count == 1 else "cameras"
    parser = argparse.ArgumentParser(
        description=(
            f"Stream a merged point cloud from {label} ZED {noun} using "
            "Fast-FoundationStereo depth. Segmentation is always disabled."
        )
    )
    parser.add_argument(
        "--engine-dir",
        type=Path,
        default=REPO_ROOT / "output",
        help="Directory containing feature_runner.engine, post_runner.engine, and onnx.yaml",
    )
    parser.add_argument(
        "--serial",
        action="append",
        type=int,
        help=f"ZED serial number; repeat exactly {camera_count} time(s)",
    )
    parser.add_argument(
        "--transform",
        action="append",
        default=[],
        metavar="PATH[:KEY]",
        help=(
            "Camera-to-output 4x4 transform."
            if camera_count == 1
            else (
                "Camera-to-output 4x4 transform. If omitted, both cameras use "
                "identity for an unregistered FPS test."
            )
        ),
    )
    if camera_count == 2:
        parser.add_argument(
            "--allow-unregistered",
            action="store_true",
            help="Deprecated compatibility flag; identity transforms are now the default",
        )
    parser.add_argument(
        "--resolution",
        choices=("VGA", "HD720", "HD1080", "HD2K"),
        default="VGA",
        help="ZED capture resolution; TensorRT resolution comes from --engine-dir",
    )
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--warmup-runs", type=int, default=2)
    parser.add_argument("--z-min", type=float, default=0.10, help="Minimum depth in metres")
    parser.add_argument("--z-max", type=float, default=5.0, help="Maximum depth in metres")
    parser.add_argument(
        "--remove-invisible",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reject pixels whose correspondence falls left of the right image",
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=0.0,
        help="Voxel size in metres; 0 disables",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=15_000,
        help="Maximum points per message; 0 disables",
    )
    parser.add_argument(
        "--bbox-filter",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable or disable the 3D axis-aligned bounding-box filter",
    )
    parser.add_argument(
        "--bbox-min",
        "--crop-min",
        dest="bbox_min",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        help="Bounding-box minimum XYZ in the output frame, in metres",
    )
    parser.add_argument(
        "--bbox-max",
        "--crop-max",
        dest="bbox_max",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        help="Bounding-box maximum XYZ in the output frame, in metres",
    )
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--discovery-port", type=int, default=5556)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument(
        "--save-first-cloud",
        type=Path,
        help="Save the first processed RGB cloud as a binary PLY file",
    )
    parser.add_argument(
        "--benchmark-frames",
        type=int,
        default=0,
        help="Stop after this many streamed frames and print BENCHMARK_RESULT JSON",
    )
    return parser


def _resolve_transforms(
    parser: argparse.ArgumentParser,
    specifications: Sequence[str],
    camera_count: int,
) -> list[np.ndarray]:
    specs = list(specifications)
    if not specs:
        specs = ["identity"] * camera_count
    elif camera_count == 2 and len(specs) == 1:
        specs.insert(0, "identity")

    if len(specs) != camera_count:
        parser.error(f"expected {camera_count} transform(s), got {len(specs)}")
    return [load_transform(spec) for spec in specs]


def _validate_args(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    camera_count: int,
) -> None:
    if camera_count == 1:
        if args.serial is not None and len(args.serial) != 1:
            parser.error("one-camera mode accepts zero or one --serial")
    elif args.serial is None or len(args.serial) != camera_count:
        parser.error(f"two-camera mode requires exactly {camera_count} --serial arguments")
    if args.warmup_runs < 0:
        parser.error("--warmup-runs cannot be negative")
    if args.camera_fps <= 0:
        parser.error("--camera-fps must be positive")
    if args.benchmark_frames < 0:
        parser.error("--benchmark-frames cannot be negative")
    if not (0.0 < args.z_min < args.z_max):
        parser.error("depth range must satisfy 0 < --z-min < --z-max")
    if args.voxel_size < 0.0 or args.max_points < 0:
        parser.error("--voxel-size and --max-points cannot be negative")
    bbox_values_supplied = args.bbox_min is not None or args.bbox_max is not None
    if args.bbox_filter and (args.bbox_min is None or args.bbox_max is None):
        parser.error("--bbox-filter requires both --bbox-min and --bbox-max")
    if not args.bbox_filter and bbox_values_supplied:
        parser.error("--bbox-min/--bbox-max require --bbox-filter")
    if args.bbox_filter and np.any(
        np.asarray(args.bbox_min) >= np.asarray(args.bbox_max)
    ):
        parser.error("every --bbox-min value must be smaller than --bbox-max")


def run_stream(camera_count: int, argv: Sequence[str] | None = None) -> int:
    parser = build_parser(camera_count)
    args = parser.parse_args(argv)
    _validate_args(parser, args, camera_count)
    transforms = _resolve_transforms(parser, args.transform, camera_count)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logging.info("Segmentation enabled: %s", SEGMENTATION_ENABLED)
    if args.bbox_filter:
        logging.info("Bounding-box filter: min=%s max=%s", args.bbox_min, args.bbox_max)
    else:
        logging.info("Bounding-box filter: disabled")
    if camera_count == 2 and not args.transform:
        logging.warning(
            "Using identity transforms for both cameras; clouds are unregistered (FPS test)"
        )

    serials: list[int | None] = args.serial or [None]
    cameras = [
        ZedStereoCamera(serial=serial, resolution=args.resolution, fps=args.camera_fps)
        for serial in serials
    ]
    publisher: PointCloudPublisher | None = None
    try:
        for camera in cameras:
            camera.open()

        runner = FastFoundationStereoTensorRTRunner(args.engine_dir)
        runner.warmup(args.warmup_runs)

        publisher = PointCloudPublisher(args.port, args.discovery_port)
        processor = PointCloudProcessor(
            voxel_size=args.voxel_size,
            max_points=args.max_points,
            bbox_min=(
                None
                if not args.bbox_filter
                else np.asarray(args.bbox_min, dtype=np.float32)
            ),
            bbox_max=(
                None
                if not args.bbox_filter
                else np.asarray(args.bbox_max, dtype=np.float32)
            ),
            random_seed=args.random_seed,
        )
        rate = StreamRateLogger()
        projection_rng = np.random.default_rng(args.random_seed)
        pending_cloud_path = args.save_first_cloud
        early_sample_limit = (
            args.max_points
            if args.voxel_size == 0.0 and not args.bbox_filter
            else 0
        )

        while True:
            frames = [camera.grab() for camera in cameras]
            if any(frame is None for frame in frames):
                time.sleep(0.001)
                continue
            stereo_frames = [frame for frame in frames if frame is not None]

            inference_start = time.perf_counter()
            disparities, prepared_frames = runner.infer(stereo_frames)
            inference_ms = (time.perf_counter() - inference_start) * 1000.0

            pointcloud_start = time.perf_counter()
            clouds = [
                disparity_to_point_cloud(
                    disparity=disparity,
                    left_rgb=frame.left_rgb,
                    calibration=frame.calibration,
                    z_min=args.z_min,
                    z_max=args.z_max,
                    remove_invisible=args.remove_invisible,
                    camera_to_output=transform,
                    sample_limit=early_sample_limit,
                    rng=projection_rng,
                )
                for disparity, frame, transform in zip(
                    disparities, prepared_frames, transforms, strict=True
                )
            ]
            points, colors = processor.process(clouds)
            pointcloud_ms = (time.perf_counter() - pointcloud_start) * 1000.0
            if pending_cloud_path is not None:
                save_point_cloud_ply(pending_cloud_path, points, colors)
                logging.info(
                    "Saved %d-point sample cloud to %s",
                    len(points),
                    pending_cloud_path.expanduser().resolve(),
                )
                pending_cloud_path = None
            publisher.publish(points, colors)
            rate.update(len(points), inference_ms, pointcloud_ms)
            if args.benchmark_frames and rate.total_frames >= args.benchmark_frames:
                result = rate.benchmark_result(
                    camera_count=camera_count,
                    resolution=args.resolution,
                    requested_camera_fps=args.camera_fps,
                    max_points=args.max_points,
                    voxel_size=args.voxel_size,
                    bbox_filter=args.bbox_filter,
                )
                logging.info("BENCHMARK_RESULT %s", json.dumps(result, sort_keys=True))
                return 0

    except KeyboardInterrupt:
        logging.info("Stopping point-cloud stream")
        return 0
    finally:
        if publisher is not None:
            publisher.close()
        for camera in reversed(cameras):
            camera.close()


__all__ = [
    "SEGMENTATION_ENABLED",
    "CameraCalibration",
    "FastFoundationStereoTensorRTRunner",
    "PointCloudProcessor",
    "PointCloudPublisher",
    "StereoFrame",
    "ZedStereoCamera",
    "disparity_to_point_cloud",
    "load_transform",
    "run_stream",
    "save_point_cloud_ply",
]
