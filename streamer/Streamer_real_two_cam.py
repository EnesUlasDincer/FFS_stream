#!/usr/bin/env python3
"""Merge two ZED clouds using Fast-FoundationStereo TensorRT depth."""

from zed_foundation_streaming import run_stream


if __name__ == "__main__":
    raise SystemExit(run_stream(camera_count=2))
