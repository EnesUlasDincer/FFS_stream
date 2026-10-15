#!/usr/bin/env python3
"""Run the working two-stage Fast-FoundationStereo TensorRT engines.

This is the small, non-streaming reference for TensorRT inference. The live
ZED streamer uses the same ``TrtRunner.forward(left, right)`` call.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
import yaml
from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.foundation_stereo import TrtRunner  # noqa: E402
from Utils import depth2xyzmap, o3d, toOpen3dCloud, vis_disparity  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Fast-FoundationStereo with the two TensorRT engines"
    )
    parser.add_argument("--engine-dir", type=Path, default=REPO_ROOT / "output")
    parser.add_argument("--left-file", type=Path, default=REPO_ROOT / "demo_data/left.png")
    parser.add_argument("--right-file", type=Path, default=REPO_ROOT / "demo_data/right.png")
    parser.add_argument(
        "--intrinsic-file",
        type=Path,
        default=REPO_ROOT / "demo_data/K.txt",
    )
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "output_trt_demo")
    parser.add_argument("--warmup-runs", type=int, default=2)
    parser.add_argument("--z-max", type=float, default=100.0)
    parser.add_argument(
        "--xyz-scale",
        type=float,
        default=2.0,
        help="Uniform scale applied to saved/visualized point-cloud XYZ (default: 2.0)",
    )
    parser.add_argument(
        "--remove-invisible",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--get-point-cloud",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--visualize", action="store_true")
    return parser.parse_args()


def load_engine_config(engine_dir: Path) -> tuple[OmegaConf, int, int]:
    config_path = engine_dir / "onnx.yaml"
    feature_path = engine_dir / "feature_runner.engine"
    post_path = engine_dir / "post_runner.engine"
    for path in (config_path, feature_path, post_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required TensorRT file not found: {path}")

    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    height, width = map(int, config["image_size"])
    return OmegaConf.create(config), height, width


def load_rgb(path: Path, width: int, height: int) -> tuple[np.ndarray, float, float]:
    image = imageio.imread(path)
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    image = image[..., :3]
    original_height, original_width = image.shape[:2]
    scale_x = width / original_width
    scale_y = height / original_height
    if (original_height, original_width) != (height, width):
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray(image), scale_x, scale_y


def to_cuda_nchw(image: np.ndarray) -> torch.Tensor:
    # TrtFeatureRunner contains ImageNet normalization, so input stays 0..255.
    return (
        torch.from_numpy(image)
        .cuda(non_blocking=True)
        .float()
        .permute(2, 0, 1)
        .unsqueeze(0)
        .contiguous()
    )


def load_calibration(path: Path, scale_x: float, scale_y: float) -> tuple[np.ndarray, float]:
    lines = path.read_text(encoding="utf-8").splitlines()
    matrix = np.asarray([float(value) for value in lines[0].split()], dtype=np.float32)
    matrix = matrix.reshape(3, 3)
    matrix[0, :] *= scale_x
    matrix[1, :] *= scale_y
    return matrix, float(lines[1])


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.warmup_runs < 0:
        raise ValueError("--warmup-runs cannot be negative")
    if not np.isfinite(args.xyz_scale) or args.xyz_scale <= 0.0:
        raise ValueError("--xyz-scale must be a positive finite number")

    engine_dir = args.engine_dir.expanduser().resolve()
    config, height, width = load_engine_config(engine_dir)
    logging.info("Loading TensorRT engines from %s", engine_dir)
    logging.info("Static engine input: 1x3x%dx%d", height, width)
    runner = TrtRunner(
        config,
        str(engine_dir / "feature_runner.engine"),
        str(engine_dir / "post_runner.engine"),
    )

    left_rgb, scale_x, scale_y = load_rgb(args.left_file, width, height)
    right_rgb, right_scale_x, right_scale_y = load_rgb(args.right_file, width, height)
    if not np.allclose([scale_x, scale_y], [right_scale_x, right_scale_y]):
        raise ValueError("Left and right source images must have the same resolution")

    left = to_cuda_nchw(left_rgb)
    right = to_cuda_nchw(right_rgb)

    trt_stream = torch.cuda.Stream()

    def infer() -> torch.Tensor:
        caller_stream = torch.cuda.current_stream()
        trt_stream.wait_stream(caller_stream)
        with torch.cuda.stream(trt_stream):
            output = runner.forward(left, right)
        caller_stream.wait_stream(trt_stream)
        return output

    for _ in range(args.warmup_runs):
        infer()
    torch.cuda.synchronize()

    start = time.perf_counter()
    # The actual Fast-FoundationStereo TensorRT inference call:
    disparity_tensor = infer()
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    logging.info("TensorRT inference: %.2f ms (%.1f FPS)", elapsed_ms, 1000.0 / elapsed_ms)

    disparity = disparity_tensor[0, 0].float().cpu().numpy().clip(0.0, None)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    disparity_vis = vis_disparity(disparity, color_map=cv2.COLORMAP_TURBO)
    imageio.imwrite(args.out_dir / "left.png", left_rgb)
    imageio.imwrite(args.out_dir / "right.png", right_rgb)
    imageio.imwrite(args.out_dir / "disp_vis.png", disparity_vis)

    if args.get_point_cloud:
        matrix, baseline_m = load_calibration(args.intrinsic_file, scale_x, scale_y)
        valid = disparity > 1e-6
        if args.remove_invisible:
            columns = np.arange(width, dtype=np.float32)[None, :]
            valid &= (columns - disparity) >= 0.0

        depth = np.full_like(disparity, np.inf, dtype=np.float32)
        depth[valid] = matrix[0, 0] * baseline_m / disparity[valid]
        np.save(args.out_dir / "depth_meter.npy", depth)

        xyz = depth2xyzmap(depth, matrix)
        cloud = toOpen3dCloud(xyz.reshape(-1, 3), left_rgb.reshape(-1, 3))
        points = np.asarray(cloud.points)
        keep = np.isfinite(points).all(axis=1)
        keep &= (points[:, 2] > 0.0) & (points[:, 2] <= args.z_max)
        cloud = cloud.select_by_index(np.flatnonzero(keep))
        cloud.scale(args.xyz_scale, center=np.zeros(3))
        o3d.io.write_point_cloud(str(args.out_dir / "cloud.ply"), cloud)
        logging.info(
            "Saved point cloud to %s with XYZ scale %.6g",
            args.out_dir / "cloud.ply",
            args.xyz_scale,
        )

        if args.visualize:
            o3d.visualization.draw_geometries([cloud])

    if args.visualize:
        cv2.imshow("Fast-FoundationStereo TensorRT disparity", disparity_vis[..., ::-1])
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
