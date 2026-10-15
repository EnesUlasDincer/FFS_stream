# ZED TensorRT RGB point-cloud streaming

This extension connects one or two ZED stereo cameras to the repository's
two-stage Fast-FoundationStereo TensorRT pipeline. It converts disparity to a
metric RGB point cloud, publishes the cloud over ZMQ, optionally saves a PLY,
and includes an Open3D viewer for saved and live clouds.

Segmentation is permanently disabled in this pipeline. The ZED SDK supplies
only rectified left/right images and calibration; ZED native depth is disabled.

## Pipeline

```text
ZED rectified left/right RGB
        |
        v
resize to the static TensorRT input (448x640) and scale intrinsics
        |
        v
feature_runner.engine -> Triton GWC volume -> post_runner.engine
        |
        v
disparity -> metric XYZ + left RGB -> crop/sample/merge
        |
        +--------------------+
        |                    |
        v                    v
ZMQ RGB cloud stream     optional binary PLY
        |
        v
Open3D live viewer
```

The current engines have static batch size 1. One camera requires one
TensorRT call per output frame. Two cameras reuse the same runner and execute
two calls sequentially before merging the clouds.

## Included files

| File | Purpose |
|---|---|
| `Streamer_real_one_cam.py` | One-ZED entry point |
| `Streamer_real_two_cam.py` | Two-ZED entry point and cloud merge |
| `zed_foundation_streaming.py` | Capture, TensorRT, projection, ZMQ, PLY, and metrics |
| `view_point_cloud.py` | Saved-PLY and live-ZMQ Open3D viewer |
| `benchmark_zed_trt.py` | Reproducible parameter-matrix runner |
| `TRT_BENCHMARK.md` | Hardware, methodology, results, and conclusions |
| `TRT_BENCHMARK_RESULTS.json` | Machine-readable results for every case |
| `sample_one_zed_15000.ply` | Example 15,000-point RGB cloud |

## Requirements

- NVIDIA GPU with CUDA and TensorRT
- Python environment containing `requirements.txt`
- ZED SDK and its matching `pyzed` Python package
- Two-stage TensorRT files in `output/`:
  - `feature_runner.engine`
  - `post_runner.engine`
  - `onnx.yaml`

If `pyzed` is missing, install it into the active environment using the helper
provided by the local ZED SDK installation:

```bash
/usr/local/zed/get_python_api.py
```

The tested environment used Python 3.12, TensorRT 10.16.1.11, PyTorch
2.11.0+cu130, and CUDA 13.0.

## Quick start: one camera

Use the first available camera at the default VGA/30 capture mode:

```bash
python streamer/Streamer_real_one_cam.py --engine-dir output
```

Select a camera and request VGA/100 for a throughput run:

```bash
python streamer/Streamer_real_one_cam.py \
  --serial <SERIAL> \
  --engine-dir output \
  --resolution VGA \
  --camera-fps 100 \
  --max-points 15000
```

`--camera-fps` is the requested ZED capture rate, not the TensorRT rate. A
request of 30 caps the complete loop near 30 FPS even if inference is faster.
The achieved rate can still be lower because capture, inference, projection,
sampling, and publishing all take time.

The one-camera transform defaults to the 4x4 identity matrix, so points remain
in the rectified left optical frame.

## Quick start: two cameras

For an FPS test, omit `--transform`. Both cameras use identity transforms:

```bash
python streamer/Streamer_real_two_cam.py \
  --serial <SERIAL_1> \
  --serial <SERIAL_2> \
  --engine-dir output \
  --resolution VGA \
  --camera-fps 100
```

Identity clouds from different physical cameras are not spatially registered.
For a correct merged cloud, camera 1 is the reference and the supplied matrix
must transform camera 2 into camera 1's frame:

```bash
python streamer/Streamer_real_two_cam.py \
  --serial <SERIAL_1> \
  --serial <SERIAL_2> \
  --engine-dir output \
  --transform /path/to/camera2_to_camera1.txt
```

Transforms may be whitespace-delimited text, `.npy`, or YAML. For YAML with
multiple matrices, append a dotted key to the filename:

```bash
--transform /path/to/extrinsics.yml:T_follower_to_main
```

## 3D point-cloud viewer

Open the sample included in the repository:

```bash
python streamer/view_point_cloud.py \
  file streamer/sample_one_zed_15000.ply
```

Validate a PLY and print its bounds without opening a window:

```bash
python streamer/view_point_cloud.py \
  file streamer/sample_one_zed_15000.ply \
  --headless
```

For live viewing, start a streamer in one terminal and the subscriber in a
second terminal:

```bash
# Terminal 1
python streamer/Streamer_real_one_cam.py \
  --serial <SERIAL> \
  --camera-fps 100

# Terminal 2
python streamer/view_point_cloud.py live \
  --endpoint tcp://127.0.0.1:5555 \
  --point-size 2.0
```

Use the mouse to rotate, pan, and zoom. Press `Q` or `Esc` to close the viewer.
For a remote publisher, replace `127.0.0.1` with the publisher's IP address and
allow the selected TCP port through the firewall.

## Save a point cloud

`--save-first-cloud` writes the first processed cloud as a compact binary
little-endian PLY and then continues streaming:

```bash
python streamer/Streamer_real_one_cam.py \
  --serial <SERIAL> \
  --save-first-cloud output_clouds/first_cloud.ply
```

Coordinates are stored in metres and colors as 8-bit RGB.

## Benchmark

Run the complete matrix with one or two camera serial numbers:

```bash
python streamer/benchmark_zed_trt.py \
  --serial <SERIAL_1> \
  --serial <SERIAL_2> \
  --frames 180 \
  --output-json streamer/TRT_BENCHMARK_RESULTS.json
```

Each subprocess stops deterministically after `--benchmark-frames`, prints a
`BENCHMARK_RESULT` JSON record, releases its camera, and starts the next case.

The measured results on an RTX 5080 were:

| Cameras | Capture | Points | Voxel | Stream FPS | TRT mean | Cloud mean |
|---:|---|---:|---:|---:|---:|---:|
| 1 | VGA/30 | 15k | off | 30.20 | 8.93 ms | 1.97 ms |
| 1 | VGA/60 | 15k | off | 60.13 | 9.04 ms | 1.92 ms |
| 1 | VGA/100 | 5k | off | 95.47 | 8.69 ms | 1.33 ms |
| 1 | VGA/100 | 15k | off | 88.38 | 8.74 ms | 1.91 ms |
| 1 | VGA/100 | 30k | off | 80.90 | 8.85 ms | 2.72 ms |
| 1 | VGA/100 | unlimited | off | 39.95 | 9.01 ms | 12.76 ms |
| 1 | VGA/100 | 15k | 5 mm | 10.95 | 8.92 ms | 81.49 ms |
| 1 | HD720/60 | 15k | off | 59.90 | 9.51 ms | 2.05 ms |
| 2 | VGA/100 | 5k | off | 46.52 | 17.55 ms | 2.94 ms |
| 2 | VGA/100 | 15k | off | 43.04 | 17.67 ms | 4.42 ms |
| 2 | VGA/100 | 30k | off | 39.52 | 17.59 ms | 6.38 ms |
| 2 | VGA/100 | 15k | 5 mm | 4.62 | 17.78 ms | 197.14 ms |

See [`TRT_BENCHMARK.md`](TRT_BENCHMARK.md) for the full methodology, p95
latencies, and conclusions. Exact values are in
[`TRT_BENCHMARK_RESULTS.json`](TRT_BENCHMARK_RESULTS.json).

### Recommended settings

- One-camera maximum throughput: VGA/100, 5k points, voxel filtering off.
- One-camera balanced output: VGA/100, 15k points, voxel filtering off.
- Two-camera maximum throughput: VGA/100, 5k points, voxel filtering off.
- Avoid the current NumPy voxel filter in the real-time loop. It processes the
  full dense cloud before limiting points and is the dominant bottleneck.
- HD720 does not increase model detail with the current engines because every
  frame is resized to the fixed `448x640` engine input.

Run an individual deterministic case with:

```bash
python streamer/Streamer_real_one_cam.py \
  --serial <SERIAL> \
  --camera-fps 100 \
  --max-points 15000 \
  --benchmark-frames 180
```

## Point-cloud controls

| Option | Default | Meaning |
|---|---:|---|
| `--z-min` | 0.10 m | Minimum retained depth |
| `--z-max` | 5.0 m | Maximum retained depth |
| `--max-points` | 15000 | Random output cap; `0` keeps all points |
| `--voxel-size` | 0 | Voxel size in metres; `0` disables filtering |
| `--crop-min X Y Z` | unset | Minimum output-frame crop corner |
| `--crop-max X Y Z` | unset | Maximum output-frame crop corner |
| `--remove-invisible` | enabled | Reject correspondences outside the right image |
| `--random-seed` | 0 | Reproducible point sampling |

When crop and voxel filtering are disabled, the implementation samples valid
pixels before constructing the complete dense cloud. This early sampling is
why `--max-points 5000` and `15000` are much faster than unlimited output.

## ZMQ wire format

The publisher binds to `tcp://*:5555` by default and sends four multipart
frames:

1. UTF-8 topic `PointCloud`
2. little-endian `uint32` point count
3. packed `(N, 3)` little-endian `int16` XYZ in millimetres
4. packed `(N, 3)` `uint8` RGB

UDP discovery remains compatible on port `5556`. Change both ports with
`--port` and `--discovery-port`.

## TensorRT inference path

The live runner executes:

```python
disparity = self.model.forward(left, right)
```

`core.foundation_stereo.TrtRunner.forward` executes `feature_runner.engine`,
builds the GWC volume with the Triton CUDA kernel, and then executes
`post_runner.engine`.

Run the standalone TensorRT example with:

```bash
python scripts_v2/run_demo_tensorrt.py --engine-dir output
```

## Notes and troubleshooting

- `Could not open ZED`: check the serial, USB connection, and whether another
  process owns the camera.
- TensorRT engine portability warning: rebuild the plans on the target GPU for
  final performance and deployment reliability.
- Empty or sparse cloud: inspect `--z-min`, `--z-max`, baseline calibration,
  and left/right image order.
- Incorrect two-camera geometry: identity transforms are only for FPS tests;
  provide measured camera extrinsics for real merging.
- No live visualization: confirm publisher/subscriber host and TCP port, then
  check firewall rules.
