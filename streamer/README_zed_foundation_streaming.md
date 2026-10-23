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
disparity -> metric XYZ + left RGB -> output XYZ scale -> bounding box/sample/merge
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
| `save_zed_pointcloud_variants.py` | One-camera resolution/point-count/scale capture sweep |
| `TRT_BENCHMARK.md` | Hardware, methodology, results, and conclusions |
| `TRT_BENCHMARK_RESULTS.json` | Machine-readable results for every case |
| `TRT_RESOLUTION_BENCHMARK_RESULTS.json` | One/two-camera capture-resolution sweep |
| `TRT_30K_RESOLUTION_BENCHMARK_RESULTS.json` | One/two-camera 30k resolution sweep |
| `TRT_RESOLUTION_POINT_MATRIX_RESULTS.json` | Measured 48-case, 5k-50k resolution/camera-count sweep |
| `sample_one_zed_15000.ply` | Example 15,000-point RGB cloud at 1.5x XYZ scale |
| `pointcloud_samples/` | Interactive website, RGB references, and four-resolution 5k-50k PLY matrix |

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
| `--xyz-scale` | Uniform output geometry scale | `1.5` (current calibrated VR default) |
| `--voxel-size` | CPU voxel filter | `0` for real time |
| `--radius-outlier-filter` | CUDA 3D isolated-point rejection | Enable for a cleaner VR cloud |
| `--transform` | Camera 2 to camera 1 extrinsic matrix | Required for real merging |
| `--port` | ZMQ publisher port | `5555` |

Segmentation is hard-disabled. You do not need to pass a segmentation option.

## Quick command map

Stream one ZED at HD1080 using the calibrated 1.5x scale:

```bash
python streamer/Streamer_real_one_cam.py \
  --serial <SERIAL> \
  --engine-dir output \
  --resolution HD1080 \
  --camera-fps 30 \
  --max-points 30000 \
  --xyz-scale 1.5
```

View that live stream in a second terminal:

```bash
python streamer/view_point_cloud.py live \
  --endpoint tcp://127.0.0.1:5555 \
  --axis-size 1.0 \
  --point-size 2.0
```

Save the complete 24-variant matrix—four resolutions at 5k, 10k, 20k, 30k,
40k, and 50k points—using scale 1.5:

```bash
python streamer/save_zed_pointcloud_variants.py \
  --serial <SERIAL> \
  --engine-dir output \
  --output-dir output_clouds/scale_1p5
```

The variant command selects valid maximum camera rates automatically. It saves
an exact-frame RGB image beside each PLY and updates a manifest for later
comparison. Add `--overwrite` only when replacing files already in that output
directory; use `--dry-run` first to preview all 24 commands.

Stream two cameras with the same scale for an unregistered FPS test:

```bash
python streamer/Streamer_real_two_cam.py \
  --serial <SERIAL_1> \
  --serial <SERIAL_2> \
  --engine-dir output \
  --resolution VGA \
  --camera-fps 100 \
  --max-points 30000 \
  --xyz-scale 1.5
```

For a real two-camera merge, add
`--transform /path/to/camera2_to_camera1.txt`. Use `--xyz-scale 1` anywhere to
restore the original unscaled coordinates.

## One-camera command guide

Run every command from the repository root:

```bash
cd /path/to/Fast-FoundationStereo
conda activate ffs
```

### Stream one camera now

This is the recommended balanced command. It uses the first available ZED,
runs Fast-FoundationStereo with TensorRT, and publishes 15,000-point RGB clouds
on `tcp://*:5555`:

```bash
python streamer/Streamer_real_one_cam.py \
  --engine-dir output \
  --resolution VGA \
  --camera-fps 100 \
  --max-points 15000 \
  --xyz-scale 1.5
```

Press `Ctrl+C` to stop it. Segmentation, voxel filtering, and bounding-box
filtering are disabled. Output X is reflected by default to match the verified
real-world/legacy VR convention; use `--no-flip-x` only for raw ZED optical
coordinates.

### Control the XYZ scale

The streamer now uniformly multiplies X, Y, and Z by 1.5 by default because
that matches the current real-world/VR calibration. The scale is applied after
each camera-to-output transform, so transform translations are scaled too:

```bash
# Current default: 1.5 times the original point-cloud size
python streamer/Streamer_real_one_cam.py --xyz-scale 1.5

# Original unscaled geometry
python streamer/Streamer_real_one_cam.py --xyz-scale 1
```

Scaling happens before bounding-box filtering, voxel filtering, PLY saving, and
ZMQ publishing. Therefore `--bbox-min`, `--bbox-max`, and `--voxel-size` are
interpreted in the final scaled output coordinate system. Raw depth rejection
with `--z-min`/`--z-max` still happens before scaling.

### Select a specific camera

Add its ZED serial number when more than one camera is connected:

```bash
python streamer/Streamer_real_one_cam.py \
  --serial <SERIAL> \
  --engine-dir output \
  --resolution VGA \
  --camera-fps 100 \
  --max-points 15000
```

### Stream and view the live cloud

Use two terminals. The viewer subscribes to the stream created by terminal 1:

```bash
# Terminal 1: publish the point cloud
conda activate ffs
cd /path/to/Fast-FoundationStereo
python streamer/Streamer_real_one_cam.py \
  --engine-dir output \
  --resolution VGA \
  --camera-fps 100 \
  --max-points 15000
```

```bash
# Terminal 2: display the point cloud
conda activate ffs
cd /path/to/Fast-FoundationStereo
python streamer/view_point_cloud.py live \
  --endpoint tcp://127.0.0.1:5555 \
  --point-size 2.0
```

For a viewer on another computer, replace `127.0.0.1` with the streamer's IP
address and allow TCP port 5555 through the firewall.

### Save one cloud and its matching RGB image

This saves the processed cloud and camera 1's native-resolution rectified left
image from the same frame, then continues streaming:

```bash
python streamer/Streamer_real_one_cam.py \
  --engine-dir output \
  --resolution VGA \
  --camera-fps 100 \
  --max-points 15000 \
  --save-first-cloud output_clouds/one_camera.ply \
  --save-first-rgb output_clouds/one_camera_rgb.png
```

Open the saved cloud with:

```bash
python streamer/view_point_cloud.py file output_clouds/one_camera.ply
```

### Run a finite one-camera speed test

This streams 180 frames, prints a final `BENCHMARK_RESULT` JSON line, releases
the camera, and exits automatically:

```bash
python streamer/Streamer_real_one_cam.py \
  --engine-dir output \
  --resolution VGA \
  --camera-fps 100 \
  --max-points 15000 \
  --benchmark-frames 180
```

### Choose speed or density

| Goal | Change | Previously measured complete-loop rate |
|---|---|---:|
| Maximum speed | `--max-points 5000` | 99.96 FPS |
| Balanced | `--max-points 15000` | 88-93 FPS |
| Denser cloud | `--max-points 30000` | 84.80 FPS |

These rates were measured on the documented RTX 5080 system and will vary.
Keep `--voxel-size 0` and omit `--bbox-filter` for maximum throughput.

`--camera-fps` is the requested ZED capture-rate ceiling, not the TensorRT
rate. A request of 30 caps the complete loop near 30 FPS even if inference is
faster. The achieved rate can still be lower because capture, inference,
projection, sampling, and publishing all take time.

Show every available option with:

```bash
python streamer/Streamer_real_one_cam.py --help
```

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

## Fast Radius Outlier Removal

Add `--radius-outlier-filter` to run exact 3D Radius Outlier Removal on the
CUDA GPU. After the box, table, voxel, and point-cap stages, each point must
have the requested number of neighboring points within the specified radius.
Isolated points are removed from the published cloud and saved PLY.

Recommended real-time settings:

```bash
--radius-outlier-filter \
--radius-outlier-radius 0.01 \
--radius-outlier-min-neighbors 5
```

### Run one camera with noise reduction

```bash
CALIB=/home/enes/Documents/fast_mail_fork/configs/cameras/calibration/2026-10-24.yaml

python streamer/Streamer_real_one_cam.py \
  --serial 17008863 \
  --engine-dir output \
  --resolution HD1080 \
  --camera-fps 30 \
  --transform "${CALIB}:left_cam.extrinsics" \
  --xyz-scale 1.5 \
  --flip-x \
  --radius-outlier-filter \
  --radius-outlier-radius 0.01 \
  --radius-outlier-min-neighbors 5 \
  --bbox-filter \
  --bbox-min -1.35 -0.70 -0.10 \
  --bbox-max -0.20 0.60 0.50 \
  --table-filter \
  --max-points 30000
```

### Run two cameras with noise reduction

```bash
CALIB=/home/enes/Documents/fast_mail_fork/configs/cameras/calibration/2026-10-24.yaml

python streamer/Streamer_real_two_cam.py \
  --serial 17008863 --serial 16055244 \
  --engine-dir output \
  --resolution HD1080 \
  --camera-fps 30 \
  --transform "${CALIB}:left_cam.extrinsics" \
  --transform "${CALIB}:right_cam.extrinsics" \
  --xyz-scale 1.5 \
  --flip-x \
  --radius-outlier-filter \
  --radius-outlier-radius 0.01 \
  --radius-outlier-min-neighbors 5 \
  --bbox-filter \
  --bbox-min -1.35 -0.70 -0.10 \
  --bbox-max -0.20 0.60 0.50 \
  --table-filter \
  --max-points 30000
```

### View the live cloud locally

Start this in a second terminal after starting either streamer:

```bash
conda activate ffs
cd /home/enes/Documents/Fast-FoundationStereo

python streamer/view_point_cloud.py live \
  --endpoint tcp://127.0.0.1:5555 \
  --axis-size 0.5 \
  --point-size 2.0
```

### Save one filtered cloud and matching RGB image

Add these arguments to either streaming command:

```bash
--save-first-cloud output_clouds/radius_hd1080_30k.ply \
--save-first-rgb output_clouds/radius_hd1080_rgb.png
```

Add `--benchmark-frames 120` to stop automatically after 120 frames and print
the measured FPS plus every filter-stage point count.

The recommended radius is `0.01` in the final scaled coordinate system: with
`--xyz-scale 1.5`, that means 1 cm in the streamed cloud. A point must have five
other points inside that radius. Increase the radius or decrease the neighbor
count to retain more points. Decrease the radius or increase the neighbor count
for stronger removal. Disable it by omitting `--radius-outlier-filter`.

The periodic log and benchmark JSON report the complete order and reduction:
`generated -> median -> bbox -> table -> sample -> radius`.

Direct CUDA measurements on the saved HD1080 clouds with a 1 cm radius and five
neighbors were:

| Input points | Radius-filter time | Retained points |
|---:|---:|---:|
| 30K | 0.75 ms | 29,315 (97.72%) |
| 50K | 1.48 ms | 49,405 (98.81%) |
| 80K | 2.95 ms | 79,325 (99.16%) |

Live HD1080 measurements with calibrated box/table filters and a 30K cap:

| Cameras | Radius filter | Stream FPS | Cloud time | Output points |
|---:|---|---:|---:|---:|
| 1 | Off | 29.83 | 3.02 ms | 30,000 |
| 1 | On | 29.89 | 4.58 ms | 29,710 |
| 2 | Off | 28.93 | 6.77 ms | 30,000 |
| 2 | On | 28.32 | 7.93 ms | 29,399 |

Open3D CUDA is warmed up once before FPS measurement and publishing begins.

### Optional disparity median filter

The earlier four arguments control a different, image-space filter:

- `--noise-filter` enables disparity median smoothing.
- `--noise-kernel-size 3` uses a 3x3 pixel neighborhood (`5` is stronger).
- `--noise-relative-threshold 0.03` permits 3% local disparity deviation.
- `--noise-absolute-threshold 1.0` always permits at least one disparity pixel.

They are not required for Radius Outlier Removal. For ROR only, use the three
`--radius-outlier-*` arguments above and omit all four `--noise-*` arguments.

## Bounding-box filter

First preview the box in the viewer while the streamer publishes an unfiltered
cloud. The viewer always shows the coordinate frame (X red, Y green, Z blue)
and draws the requested box in orange:

```bash
python streamer/view_point_cloud.py live \
  --endpoint tcp://127.0.0.1:5555 \
  --axis-size 0.5 \
  --point-size 2.0 \
  --bbox-min -0.5 -0.5 0.2 \
  --bbox-max  0.5  0.5 2.0
```

Add `--apply-bbox` to the viewer command to preview only the points inside the
box. This is display-side filtering and does not slow or change the publisher.
The terminal reports `inside/total` points once per second. Adjust the six box
coordinates and restart the viewer until the orange box contains the desired
workspace.

Do not use viewer-only `--display-scale` or `--flip-axis` while choosing values
that will be copied to the streamer; those options change only the displayed
coordinates. The streamer's own `--xyz-scale`, `--flip-x`, and camera transforms
are already present in the received cloud and therefore remain consistent.

Then copy exactly the same coordinates to the streamer to apply the real
publisher-side filter:

```bash
python streamer/Streamer_real_one_cam.py \
  --serial <SERIAL> \
  --camera-fps 100 \
  --max-points 15000 \
  --bbox-filter \
  --bbox-min -0.5 -0.5 0.2 \
  --bbox-max  0.5  0.5 2.0
```

In raw ZED optical coordinates, X points right, Y points down, and Z points
forward. Because the default output reflects X, default bounding-box X is the
verified real-world/VR X direction. With camera transforms, the box is evaluated
in the final merged, reflected, and scaled output frame. With the default
`--xyz-scale 1.5`, bounding-box coordinates therefore use the scaled output.

Disable it by omitting the option or explicitly using:

```bash
--no-bbox-filter
```

`--bbox-min` and `--bbox-max` are required when the filter is enabled. The
optimized path evaluates each XYZ bound directly during per-camera projection.
Rejected points never enter the merged point array and never receive an RGB
float allocation.

The same viewer flags work for saved files:

```bash
python streamer/view_point_cloud.py \
  file output_clouds/calibrated_cloud.ply \
  --axis-size 0.5 \
  --bbox-min -0.5 -0.5 0.2 \
  --bbox-max  0.5  0.5 2.0 \
  --apply-bbox
```

## Remove the tabletop

Add `--table-filter` to remove the calibrated tabletop and every point below
it. The filter uses the final streamed coordinate frame, so it runs after the
camera transform, `--xyz-scale`, `--flip-x`, and bounding-box filter, but before
voxel filtering and `--max-points`. It therefore works for VR streaming and for
clouds written with `--save-first-cloud`.

For the current two-ZED calibration, default `--xyz-scale 1.5`, and default
`--flip-x`, the measured table plane is already the default. This is the
one-camera command:

```bash
CALIB=/home/enes/Documents/fast_mail_fork/configs/cameras/calibration/2026-10-24.yaml

python streamer/Streamer_real_one_cam.py \
  --serial 17008863 \
  --engine-dir output \
  --resolution HD1080 \
  --camera-fps 30 \
  --transform "${CALIB}:left_cam.extrinsics" \
  --xyz-scale 1.5 \
  --flip-x \
  --bbox-filter \
  --bbox-min -1.35 -0.70 -0.10 \
  --bbox-max -0.20 0.60 0.50 \
  --table-filter \
  --table-clearance 0.01 \
  --max-points 30000
```

For two cameras, use the same filter after both calibrated clouds are merged:

```bash
CALIB=/home/enes/Documents/fast_mail_fork/configs/cameras/calibration/2026-10-24.yaml

python streamer/Streamer_real_two_cam.py \
  --serial 17008863 --serial 16055244 \
  --engine-dir output \
  --resolution HD1080 \
  --camera-fps 30 \
  --transform "${CALIB}:left_cam.extrinsics" \
  --transform "${CALIB}:right_cam.extrinsics" \
  --xyz-scale 1.5 \
  --flip-x \
  --bbox-filter \
  --bbox-min -1.35 -0.70 -0.10 \
  --bbox-max -0.20 0.60 0.50 \
  --table-filter \
  --table-clearance 0.01 \
  --max-points 30000
```

The calibrated default plane is:

```text
0.056345*x + 0.010991*y + 0.998351*z - 0.01902 = 0
```

Only the plane's positive side is retained. Increase `--table-clearance` if a
thin layer of table points remains; decrease it if the filter trims the bottoms
of objects. Override the measured plane with `--table-plane A B C D` if the
camera transforms, XYZ scale, X reflection, or table position changes. Disable
the filter with `--no-table-filter` or by omitting `--table-filter`.

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

The viewer shows a coordinate frame by default: **X is red, Y is green, and Z
is blue**. Use `--axis-size` to change its size in metres.

[Open the interactive browser explorer](https://raw.githack.com/EnesUlasDincer/FFS_stream/zed-trt-pointcloud-streaming/streamer/pointcloud_samples/index.html).
It reads `manifest.json` and switches between VGA, HD720, HD1080, and HD2K at
5k, 10k, 15k, 20k, 30k, 40k, and 50k points. It shows the native RGB reference,
RGB axes, capture metadata, point-size and display-scale controls, auto-rotation,
fullscreen, a download link for the selected PLY, and the measured one-/two-
camera FPS for every requested density. The 30k PLY and RGB are an exact-frame
pair; other densities are separate captures of the same scene.
Repository PLY samples use `--xyz-scale 1.5`.
The [sample README](pointcloud_samples/README.md) also renders the four RGB
references directly on GitHub.

Corrected samples are included for the six requested point limits, plus the
existing 15k comparison:

| ZED capture mode | RGB | Point clouds |
|---|---|---|
| VGA, 672x376 | `one_zed_vga_rgb.png` | `one_zed_vga_{5000,10000,15000,20000,30000,40000,50000}.ply` |
| HD720, 1280x720 | `one_zed_hd720_rgb.png` | `one_zed_hd720_{5000,10000,15000,20000,30000,40000,50000}.ply` |
| HD1080, 1920x1080 | `one_zed_hd1080_rgb.png` | `one_zed_hd1080_{5000,10000,15000,20000,30000,40000,50000}.ply` |
| HD2K, 2208x1242 | `one_zed_hd2k_rgb.png` | `one_zed_hd2k_{5000,10000,15000,20000,30000,40000,50000}.ply` |

Open the HD1080 sample with a one-metre coordinate frame:

```bash
python streamer/view_point_cloud.py \
  file streamer/pointcloud_samples/one_zed_hd1080_15000.ply \
  --axis-size 1.0
```

The viewer can apply a temporary scale without changing a PLY or live stream.
This is useful for comparison; do not add `--display-scale 1.5` when the
streamer already used `--xyz-scale 1.5` unless a total 2.25x display scale is
wanted:

```bash
python streamer/view_point_cloud.py \
  file output_clouds/unscaled.ply \
  --display-scale 1.5
```

The raw ZED optical frame is `+X` right, `+Y` down, and `+Z` forward. Testing
confirmed that the real-world/legacy VR convention requires an X reflection,
so new streams and samples apply it by default. To display raw ZED X without
changing the PLY, reflect the already-corrected sample once in the viewer:

```bash
python streamer/view_point_cloud.py \
  file streamer/pointcloud_samples/one_zed_hd1080_15000.ply \
  --axis-size 1.0 \
  --flip-axis x
```

The normal VR command needs no extra orientation flag:

```bash
python streamer/Streamer_real_one_cam.py \
  --engine-dir output \
  --resolution HD1080 \
  --camera-fps 30 \
  --max-points 15000
```

Use `--no-flip-x` on the streamer only when raw ZED optical X is required.
`--flip-axis x` is always display-only.

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
little-endian PLY. `--save-first-rgb` writes camera 1's matching native RGB
frame. Streaming then continues:

```bash
python streamer/Streamer_real_one_cam.py \
  --serial <SERIAL> \
  --save-first-cloud output_clouds/first_cloud.ply \
  --save-first-rgb output_clouds/first_cloud_rgb.png
```

Coordinates are stored as floating-point output units and colors as 8-bit RGB.
At `--xyz-scale 1.5`, each coordinate is 1.5 times its original metric value.

### Save many variants for later comparison

The variant saver captures every selected resolution × point count × XYZ scale
as an exact-frame PLY/PNG pair and updates `manifest.json` in the output
directory. For example, this creates six pairs (two resolutions and three
point counts at scale 1.5):

```bash
python streamer/save_zed_pointcloud_variants.py \
  --serial <SERIAL> \
  --engine-dir output \
  --output-dir output_clouds/variants \
  --resolution VGA \
  --resolution HD1080 \
  --max-points 5000 \
  --max-points 30000 \
  --max-points 50000 \
  --xyz-scale 1.5
```

Omit all three repeatable variant options to capture VGA, HD720, HD1080, and
HD2K at 5k, 10k, 20k, 30k, 40k, and 50k points using scale 1.5. That is 24
PLY/RGB pairs. Existing pairs are protected; pass `--overwrite` only when they
should be replaced. Use `--dry-run` to inspect all commands without opening the
camera. `--capture-retries 3` is the default for transient camera-open errors.

## Benchmark

Run the complete matrix with one or two camera serial numbers:

```bash
python streamer/benchmark_zed_trt.py \
  --serial <SERIAL_1> \
  --serial <SERIAL_2> \
  --frames 180 \
  --xyz-scale 1.5 \
  --output-json streamer/TRT_BENCHMARK_RESULTS.json
```

New benchmark runs record `xyz_scale` in both the top-level document and every
case. The published benchmark tables below predate the scaling option and used
the equivalent of `--xyz-scale 1`; they remain historical timing records.

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

Run every requested point limit at every resolution:

```bash
python streamer/benchmark_zed_trt.py \
  --serial <SERIAL_1> \
  --serial <SERIAL_2> \
  --engine-dir output \
  --suite resolution-points \
  --frames 180 \
  --warmup-runs 3 \
  --xyz-scale 1.5 \
  --output-json streamer/TRT_RESOLUTION_POINT_MATRIX_RESULTS.json
```

The `resolution-points` suite evaluates 5k, 10k, 20k, 30k, 40k, and 50k at
VGA, HD720, HD1080, and HD2K. One serial runs 24 cases; two serials run those
24 one-camera cases plus 24 two-camera cases. Omit the second `--serial` for a
one-camera-only evaluation.

Measured stream FPS on the RTX 5080 test machine is shown as one camera / two
cameras. Every case used 180 frames, scale 1.5, X reflection on, segmentation
off, and no box or voxel filter:

| Points | VGA/100 | HD720/60 | HD1080/30 | HD2K/15 |
|---:|---:|---:|---:|---:|
| 5k | 95.24 / 46.31 | 60.33 / 39.74 | 29.64 / 29.13 | 15.23 / 15.09 |
| 10k | 91.05 / 44.77 | 60.30 / 37.95 | 29.34 / 29.23 | 15.07 / 15.09 |
| 20k | 85.52 / 41.04 | 59.83 / 35.62 | 29.57 / 29.35 | 15.11 / 14.91 |
| 30k | 80.38 / 38.91 | 60.09 / 33.70 | 29.78 / 28.56 | 15.13 / 15.04 |
| 40k | 76.47 / 38.13 | 60.04 / 32.99 | 29.20 / 28.77 | 15.13 / 15.02 |
| 50k | 74.71 / 36.45 | 59.89 / 31.26 | 29.78 / 27.27 | 15.18 / 14.88 |

See [`TRT_RESOLUTION_POINT_MATRIX_RESULTS.json`](TRT_RESOLUTION_POINT_MATRIX_RESULTS.json)
for inference and point-cloud mean/p95 timings for every case.

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

Run the 30k resolution and camera-count variations with:

```bash
python streamer/benchmark_zed_trt.py \
  --serial <SERIAL_1> \
  --serial <SERIAL_2> \
  --engine-dir output \
  --suite resolution-30k \
  --frames 180 \
  --warmup-runs 3 \
  --output-json streamer/TRT_30K_RESOLUTION_BENCHMARK_RESULTS.json
```

| Capture | One camera, 30k | Two cameras, 30k |
|---|---:|---:|
| VGA/100 | 71.34 FPS | 39.49 FPS |
| HD720/60 | 59.97 FPS | 34.49 FPS |
| HD1080/30 | 29.47 FPS | 29.27 FPS |
| HD2K/15 | 15.09 FPS | 13.03 FPS |

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
| `--radius-outlier-filter` | disabled | Exact CUDA 3D radius outlier removal |
| `--radius-outlier-radius` | 0.01 | Neighbor radius in scaled output metres |
| `--radius-outlier-min-neighbors` | 5 | Required neighbors, excluding the point itself |
| `--noise-filter` | disabled | Optional disparity median smoothing before 3D projection |
| `--noise-kernel-size` | 3 | Median neighborhood; `5` is stronger and slower |
| `--noise-relative-threshold` | 0.03 | Allowed local disparity deviation as a fraction |
| `--noise-absolute-threshold` | 1.0 px | Minimum allowed local disparity deviation |
| `--bbox-filter` | disabled | Enable the axis-aligned 3D box |
| `--bbox-min X Y Z` | required when enabled | Minimum output-frame corner |
| `--bbox-max X Y Z` | required when enabled | Maximum output-frame corner |
| `--table-filter` | disabled | Remove the calibrated table plane and points below it |
| `--table-plane A B C D` | calibrated rig | Output-frame plane equation; its positive side is kept |
| `--table-clearance` | 0.01 m | Extra distance above the table plane to retain |
| `--flip-x` | enabled | Reflect output X for the verified real-world/VR convention |
| `--no-flip-x` | — | Keep raw ZED optical X instead |
| `--xyz-scale` | 1.5 | Uniformly scale final XYZ and transform translations; `1` restores the original size |
| `--save-first-rgb PATH` | disabled | Save camera 1's matching native rectified RGB image |
| `--remove-invisible` | enabled | Reject correspondences outside the right image |
| `--random-seed` | 0 | Reproducible point sampling |

The optional median check runs first in disparity/image space. The bounding box
and table plane run during XYZ projection, followed by voxel filtering and the
point cap. CUDA Radius Outlier Removal runs last on the points that would
otherwise be published. Running ROR after the cap keeps its cost bounded by
`--max-points`.

On a repeatable synthetic 448x640 test using the current calibrated transform,
bounding box, table plane, and a 30K cap, this reduced median CPU point-cloud
processing time from 26.34 ms to 6.16 ms (4.28x). End-to-end FPS also includes
camera capture and TensorRT inference, so measure it on the live cameras with
`--benchmark-frames`.

Live HD1080/30 benchmarks with the 2026-10-24 calibration, default 1.5 scale,
X reflection, current bounding box, no voxel filter, and a 30K output cap:

| Cameras | Spatial filters | Stream FPS | Mean inference | Mean cloud |
|---:|---|---:|---:|---:|
| 1 | box + table | 29.70 | 10.67 ms | 3.19 ms |
| 2 | box only | 28.88 | 20.49 ms | 7.82 ms |
| 2 | box + table | 28.59 | 21.00 ms | 7.24 ms |

The earlier two-camera HD1080 box-filter implementation measured 14.51 FPS in
the same 30K evaluation series. The optimized box-only result is therefore
approximately 1.99x faster end to end and nearly reaches the requested camera
rate.

### Full resolution, density, and filter matrix

The repository also includes a 48-case two-camera matrix covering all four
capture resolutions, 30K/50K/80K point caps, and filters off/box/table/both.
Every case has its own downloadable PLY, measured FPS, and matching
resolution-level RGB reference. This existing matrix was captured before the
noise filters and therefore has both median and radius filtering disabled:

[Open the interactive filter benchmark gallery](https://raw.githack.com/EnesUlasDincer/FFS_stream/zed-trt-pointcloud-streaming/streamer/filter_benchmark_gallery/index.html).

Static comparison sheets render directly in the gallery
[`README.md`](filter_benchmark_gallery/README.md), while exact measurements are
stored in
[`benchmark_results.json`](filter_benchmark_gallery/benchmark_results.json).

Reproduce the complete benchmark and capture set with:

```bash
CALIB=/home/enes/Documents/fast_mail_fork/configs/cameras/calibration/2026-10-24.yaml

python streamer/benchmark_zed_filter_matrix.py \
  --serial 17008863 --serial 16055244 \
  --transform "${CALIB}:left_cam.extrinsics" \
  --transform "${CALIB}:right_cam.extrinsics" \
  --frames 120 \
  --overwrite
```

Use `--resume` instead of `--overwrite` to continue an interrupted matrix. Add
`--refresh-stage-counts` to update older results with the mean generated,
after-box, and after-table point counts without rerunning every density cap.
Regenerate all GitHub-visible PNG comparisons with:

```bash
python streamer/render_filter_benchmark_previews.py
```

To capture a separate Radius Outlier Removal matrix, add
`--radius-outlier-filter --output-dir streamer/radius_filter_benchmark` to the
benchmark command. Keep it separate from the existing gallery when resuming.

## ZMQ wire format

The publisher binds to `tcp://*:5555` by default and sends four multipart
frames:

1. UTF-8 topic `PointCloud`
2. little-endian `uint32` point count
3. packed `(N, 3)` little-endian `int16` XYZ at 1000 output units per metre
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
python scripts_v2/run_demo_tensorrt.py \
  --engine-dir output \
  --xyz-scale 1.5
```

All tracked demo scripts that save point clouds now accept `--xyz-scale` (or
the legacy-compatible `--xyz_scale` spelling) and default to 1.5. Use scale 1 to
reproduce their original output size. Their saved `depth_meter.npy` remains
unscaled metric depth; the scale affects only saved/visualized XYZ geometry.

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
