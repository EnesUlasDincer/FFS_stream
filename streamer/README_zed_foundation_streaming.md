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
resize to the configured static TensorRT input and scale intrinsics
        |
        v
feature_runner.engine -> Triton GWC volume -> post_runner.engine
        |
        v
disparity -> metric XYZ + left RGB -> bounding box/sample/merge
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
| `TRT_RESOLUTION_BENCHMARK_RESULTS.json` | One/two-camera capture-resolution sweep |
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

## Change these values first

Run commands from the repository root after activating the environment:

```bash
cd /path/to/Fast-FoundationStereo
conda activate ffs
```

| Value | What to change | Recommended starting point |
|---|---|---|
| `<SERIAL>` | Your ZED serial from ZED Explorer | One physical camera |
| `--engine-dir` | Directory containing both engines and `onnx.yaml` | `output` |
| `--resolution` | ZED capture resolution | `VGA` for speed |
| `--camera-fps` | Requested capture-rate ceiling | `100` at VGA |
| `--max-points` | Output density versus speed | `15000` balanced, `5000` fastest |
| `--voxel-size` | CPU voxel filter | `0` for real time |
| `--transform` | Camera 2 to camera 1 extrinsic matrix | Required for real merging |
| `--port` | ZMQ publisher port | `5555` |

Segmentation is hard-disabled. You do not need to pass a segmentation option.

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

## Bounding-box filter

The box is disabled by default. Enable it and provide minimum and maximum XYZ
corners in metres:

```bash
python streamer/Streamer_real_one_cam.py \
  --serial <SERIAL> \
  --camera-fps 100 \
  --max-points 15000 \
  --bbox-filter \
  --bbox-min -0.5 -0.5 0.2 \
  --bbox-max  0.5  0.5 2.0
```

With an identity transform, X points right, Y points down, and Z points forward
from the rectified left camera. With camera transforms, the box is evaluated in
the merged output frame.

Disable it by omitting the option or explicitly using:

```bash
--no-bbox-filter
```

`--bbox-min` and `--bbox-max` are required when the filter is enabled. The
filter must construct and inspect the dense cloud before the final point cap;
the measured VGA/100, 15k case ran at 37.76 FPS versus 92.79 FPS without it.

## Increasing resolution

There are two independent resolutions:

1. `--resolution` controls ZED image capture.
2. `onnx.yaml` inside `--engine-dir` controls the TensorRT model resolution.

The camera modes now include `VGA`, `HD720`, `HD1080`, and `HD2K`. This tested
command opens HD2K capture at 2208x1242/15 FPS:

| `--resolution` | Camera pixels | Maximum supported `--camera-fps` |
|---|---:|---:|
| `VGA` | 672x376 | 100 |
| `HD720` | 1280x720 | 60 |
| `HD1080` | 1920x1080 | 30 |
| `HD2K` | 2208x1242 | 15 |

Use one of the listed rates for that resolution. These are capture dimensions;
the current `output` TensorRT engines still infer at their static 448x640 input.

```bash
python streamer/Streamer_real_one_cam.py \
  --serial <SERIAL> \
  --resolution HD2K \
  --camera-fps 15 \
  --engine-dir output
```

With the existing `output` engines, that HD2K image is still resized to
448x640. To increase the actual disparity resolution, build a new static engine
set. A practical HD720-aligned TensorRT size is 704x1280 because both dimensions
must be divisible by 32:

```bash
python scripts/make_onnx.py \
  --model_dir weights/20-30-48/model_best_bp2_serialize.pth \
  --save_path "$PWD/output_hd720/" \
  --height 704 \
  --width 1280 \
  --valid_iters 8 \
  --max_disp 192

/usr/src/tensorrt/bin/trtexec \
  --onnx=output_hd720/feature_runner.onnx \
  --saveEngine=output_hd720/feature_runner.engine \
  --fp16

/usr/src/tensorrt/bin/trtexec \
  --onnx=output_hd720/post_runner.onnx \
  --saveEngine=output_hd720/post_runner.engine \
  --fp16
```

Then run the higher-resolution engine:

```bash
python streamer/Streamer_real_one_cam.py \
  --serial <SERIAL> \
  --resolution HD720 \
  --camera-fps 60 \
  --engine-dir output_hd720 \
  --max-points 15000
```

Higher model resolution uses substantially more GPU memory and is slower. The
engine plans are GPU-specific and should be rebuilt on the deployment GPU.
Increasing horizontal resolution can also require a larger `--max_disp` to
preserve the same near-depth range; that further increases memory and latency.

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

Run only the capture-resolution sweep for both cameras:

```bash
python streamer/benchmark_zed_trt.py \
  --serial <SERIAL_1> \
  --serial <SERIAL_2> \
  --engine-dir output \
  --suite resolution \
  --frames 180 \
  --warmup-runs 3 \
  --output-json streamer/TRT_RESOLUTION_BENCHMARK_RESULTS.json
```

The measured RTX 5080 results use 15,000 points, identity transforms, and no
bounding-box or voxel filter:

| Capture | One camera | Two cameras |
|---|---:|---:|
| VGA/100 | 88.62 FPS | 42.74 FPS |
| HD720/60 | 59.84 FPS | 37.15 FPS |
| HD1080/30 | 29.66 FPS | 28.80 FPS |
| HD2K/15 | 15.11 FPS | 15.06 FPS |

The one-camera VGA result can vary between runs because this is a complete
live-camera pipeline, not a TensorRT-only microbenchmark. See
[`TRT_BENCHMARK.md`](TRT_BENCHMARK.md) for latency columns and interpretation.

Each subprocess stops deterministically after `--benchmark-frames`, prints a
`BENCHMARK_RESULT` JSON record, releases its camera, and starts the next case.

The measured results on an RTX 5080 were:

| Cameras | Capture | Points | Box | Voxel | Stream FPS | TRT mean | Cloud mean |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | VGA/30 | 15k | off | off | 30.12 | 8.45 ms | 1.96 ms |
| 1 | VGA/60 | 15k | off | off | 59.94 | 8.35 ms | 1.87 ms |
| 1 | VGA/100 | 5k | off | off | 99.96 | 8.32 ms | 1.29 ms |
| 1 | VGA/100 | 15k | off | off | 92.79 | 8.32 ms | 1.91 ms |
| 1 | VGA/100 | 15k | on | off | 37.76 | 8.58 ms | 17.15 ms |
| 1 | VGA/100 | 30k | off | off | 84.80 | 8.37 ms | 2.73 ms |
| 1 | VGA/100 | unlimited | off | off | 42.44 | 8.61 ms | 11.94 ms |
| 1 | VGA/100 | 15k | off | 5 mm | 10.88 | 8.60 ms | 82.50 ms |
| 1 | HD720/60 | 15k | off | off | 59.90 | 9.53 ms | 2.04 ms |
| 2 | VGA/100 | 5k | off | off | 48.75 | 16.70 ms | 2.87 ms |
| 2 | VGA/100 | 15k | off | off | 45.31 | 16.80 ms | 4.21 ms |
| 2 | VGA/100 | 30k | off | off | 41.32 | 16.86 ms | 6.13 ms |
| 2 | VGA/100 | 15k | off | 5 mm | 5.05 | 16.98 ms | 179.36 ms |

See [`TRT_BENCHMARK.md`](TRT_BENCHMARK.md) for the full methodology, p95
latencies, and conclusions. Exact values are in
[`TRT_BENCHMARK_RESULTS.json`](TRT_BENCHMARK_RESULTS.json).

### Recommended settings

- One-camera maximum throughput: VGA/100, 5k points, voxel filtering off.
- One-camera balanced output: VGA/100, 15k points, voxel filtering off.
- Two-camera maximum throughput: VGA/100, 5k points, voxel filtering off.
- Leave the bounding box disabled unless spatial rejection is required.
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
| `--bbox-filter` | disabled | Enable the axis-aligned 3D box |
| `--bbox-min X Y Z` | required when enabled | Minimum output-frame corner |
| `--bbox-max X Y Z` | required when enabled | Maximum output-frame corner |
| `--remove-invisible` | enabled | Reject correspondences outside the right image |
| `--random-seed` | 0 | Reproducible point sampling |

When box and voxel filtering are disabled, the implementation samples valid
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
