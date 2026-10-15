# Fast-FoundationStereo ZED TensorRT benchmark

This report measures the complete ZED-to-ZMQ RGB point-cloud loop while varying
camera rate, capture resolution, output point count, bounding-box filtering,
voxel filtering, and the number of cameras. It was recorded on 2026-10-15.

Exact machine-readable measurements are available in
[`TRT_BENCHMARK_RESULTS.json`](TRT_BENCHMARK_RESULTS.json). The separate
capture-resolution sweep is in
[`TRT_RESOLUTION_BENCHMARK_RESULTS.json`](TRT_RESOLUTION_BENCHMARK_RESULTS.json),
and the 30k sweep is in
[`TRT_30K_RESOLUTION_BENCHMARK_RESULTS.json`](TRT_30K_RESOLUTION_BENCHMARK_RESULTS.json).

## Test system

| Component | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 5080, 16 GB |
| NVIDIA driver | 595.71.05 |
| TensorRT | 10.16.1.11 |
| PyTorch | 2.11.0+cu130 |
| CUDA reported by PyTorch | 13.0 |
| TensorRT input | static batch 1, RGB `448x640` |
| Camera input | rectified ZED RGB |
| Native ZED depth | disabled |
| Segmentation | disabled |
| Transform | identity |
| ZMQ | publisher enabled; no subscriber during measurements |

TensorRT warned that the existing plans were created on a different GPU model.
They ran successfully, but rebuilding both plans on the target GPU is advised
before treating these numbers as final deployment measurements.

## Method

- Each case used three TensorRT warm-up calls.
- Timing started after camera setup, engine loading, and warm-up.
- Each result covers 180 successfully processed and published frames.
- `stream_fps` covers capture, resize, TensorRT, projection, cloud processing,
  and the non-blocking ZMQ publish call.
- `inference` covers all TensorRT calls needed for one output frame. It contains
  one call in one-camera mode and two sequential calls in two-camera mode.
- `cloud` covers disparity back-projection, transform, merge, bounding box, voxel
  filtering, and point limiting. It excludes TensorRT and ZMQ publishing.
- The p95 column is the 95th-percentile per-frame latency.
- Two-camera identity clouds were intentionally unregistered for throughput
  testing.

## Parameter matrix

| Cameras | Input | Requested FPS | Limit | Box | Voxel | Stream FPS | Inference mean / p95 | Cloud mean / p95 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | VGA | 30 | 15k | off | off | 30.12 | 8.45 / 8.59 ms | 1.96 / 2.13 ms |
| 1 | VGA | 60 | 15k | off | off | 59.94 | 8.35 / 8.51 ms | 1.87 / 2.02 ms |
| 1 | VGA | 100 | 5k | off | off | 99.96 | 8.32 / 8.40 ms | 1.29 / 1.41 ms |
| 1 | VGA | 100 | 15k | off | off | 92.79 | 8.32 / 8.42 ms | 1.91 / 2.10 ms |
| 1 | VGA | 100 | 15k | on | off | 37.76 | 8.58 / 8.68 ms | 17.15 / 18.43 ms |
| 1 | VGA | 100 | 30k | off | off | 84.80 | 8.37 / 8.46 ms | 2.73 / 2.96 ms |
| 1 | VGA | 100 | unlimited | off | off | 42.44 | 8.61 / 8.70 ms | 11.94 / 12.63 ms |
| 1 | VGA | 100 | 15k | off | 5 mm | 10.88 | 8.60 / 8.69 ms | 82.50 / 94.18 ms |
| 1 | HD720 | 60 | 15k | off | off | 59.90 | 9.53 / 10.14 ms | 2.04 / 2.44 ms |
| 2 | VGA | 100 | 5k | off | off | 48.75 | 16.70 / 16.85 ms | 2.87 / 3.03 ms |
| 2 | VGA | 100 | 15k | off | off | 45.31 | 16.80 / 16.95 ms | 4.21 / 4.50 ms |
| 2 | VGA | 100 | 30k | off | off | 41.32 | 16.86 / 17.01 ms | 6.13 / 6.38 ms |
| 2 | VGA | 100 | 15k | off | 5 mm | 5.05 | 16.98 / 17.15 ms | 179.36 / 195.42 ms |

The standalone demo pair ran one TensorRT inference in 8.43 ms, or 118.6 FPS,
without camera capture, cloud construction, or streaming.

## Capture-resolution sweep

This second run holds the output at 15,000 points, disables both spatial
filters, and uses the maximum supported camera rate for every selectable ZED
mode. Each case processes 180 frames with the existing static 448x640 TensorRT
engines.

| Cameras | Capture pixels | Requested FPS | Stream FPS | Inference mean / p95 | Cloud mean / p95 |
|---:|---:|---:|---:|---:|---:|
| 1 | VGA (672x376) | 100 | 88.62 | 8.62 / 9.20 ms | 2.04 / 2.29 ms |
| 1 | HD720 (1280x720) | 60 | 59.84 | 10.06 / 10.67 ms | 2.04 / 2.24 ms |
| 1 | HD1080 (1920x1080) | 30 | 29.66 | 10.97 / 11.97 ms | 2.08 / 2.25 ms |
| 1 | HD2K (2208x1242) | 15 | 15.11 | 12.02 / 13.54 ms | 2.09 / 2.35 ms |
| 2 | VGA (672x376) | 100 | 42.74 | 17.73 / 18.31 ms | 4.50 / 4.88 ms |
| 2 | HD720 (1280x720) | 60 | 37.15 | 19.56 / 20.46 ms | 4.69 / 5.05 ms |
| 2 | HD1080 (1920x1080) | 30 | 28.80 | 21.30 / 22.88 ms | 4.63 / 4.94 ms |
| 2 | HD2K (2208x1242) | 15 | 15.06 | 23.65 / 25.93 ms | 4.47 / 4.85 ms |

Higher capture resolution increases image transfer and resize latency, visible
in the inference-stage timing. HD1080 and HD2K are camera-rate-limited. They do
not improve disparity detail with the current engine because every input is
resized to 448x640 before inference; rebuild the engine at a larger static size
when higher model resolution is required.

## 30k capture-resolution sweep

This follow-up varies capture resolution and camera count while holding the
output at 30,000 points. It uses the corrected default X reflection, identity
camera extrinsics, segmentation off, and both bounding-box and voxel filters
off. Each case processes 180 frames after three warm-up runs.

| Cameras | Capture pixels | Requested FPS | Stream FPS | Inference mean / p95 | Cloud mean / p95 |
|---:|---:|---:|---:|---:|---:|
| 1 | VGA (672x376) | 100 | 71.34 | 9.11 / 9.95 ms | 3.84 / 5.10 ms |
| 1 | HD720 (1280x720) | 60 | 59.97 | 10.84 / 11.34 ms | 3.14 / 3.45 ms |
| 1 | HD1080 (1920x1080) | 30 | 29.47 | 10.68 / 11.49 ms | 2.96 / 3.30 ms |
| 1 | HD2K (2208x1242) | 15 | 15.09 | 11.58 / 13.01 ms | 2.97 / 3.28 ms |
| 2 | VGA (672x376) | 100 | 39.49 | 17.22 / 18.04 ms | 6.79 / 7.29 ms |
| 2 | HD720 (1280x720) | 60 | 34.49 | 19.09 / 20.13 ms | 7.06 / 7.44 ms |
| 2 | HD1080 (1920x1080) | 30 | 29.27 | 20.92 / 22.54 ms | 6.93 / 7.43 ms |
| 2 | HD2K (2208x1242) | 15 | 13.03 | 24.61 / 28.20 ms | 7.61 / 10.35 ms |

VGA remains compute/processing limited at 30k. HD720 is camera-limited with
one camera but not with two; HD1080 stays close to its 30 FPS capture ceiling in
both configurations. Two-camera HD2K falls below 15 FPS because sequential
inference plus cloud work exceeds the camera period.

## Findings

### Capture rate is an upper bound

At a 15k point limit, VGA/30 and VGA/60 deliver 30.12 and 59.94 FPS. The model
is fast enough in these cases, so camera scheduling sets the output rate. A
VGA/100 request exposes downstream costs and delivers 92.79 FPS at 15k points.

### Point count matters after inference

TensorRT remains close to 8.3-8.6 ms in all one-camera VGA cases. Changing only
the output limit produces these complete-loop rates:

| Output cloud | Stream FPS | Cloud mean |
|---:|---:|---:|
| 5k | 99.96 | 1.29 ms |
| 15k | 92.79 | 1.91 ms |
| 30k | 84.80 | 2.73 ms |
| about 255k, unlimited | 42.44 | 11.94 ms |

With voxel and bounding-box filters disabled, the implementation selects valid disparity
pixels before creating the full dense cloud. This early-sampling path should be
used for real-time streaming.

### Bounding-box filtering is explicit but costs throughput

The tested box kept `[-0.5, -0.5, 0.2]` through `[0.5, 0.5, 2.0]` metres.
Filtering requires the dense transformed cloud before applying the 15k output
cap. Cloud processing rises from 1.91 to 17.15 ms, reducing the complete loop
from 92.79 to 37.76 FPS. Leave the filter disabled unless spatial rejection is
required.

### Current CPU voxel filtering is not real-time

The 5 mm option constructs the dense cloud and performs a NumPy `unique` pass
over voxel IDs before applying the 15k limit. Cloud processing rises to 82.50
ms for one camera and 179.36 ms for two, reducing throughput to 10.88 and 5.05
FPS respectively. A GPU voxel filter or a filter after early point limiting
would be required for real-time voxelized output.

### Two-camera inference is sequential

The fixed engine batch size is 1. Reusing the contexts for two sequential calls
costs approximately 16.8 ms per output frame. With direct sampling, the final
point cap then moves throughput from 48.75 FPS at 5k points to 41.32 FPS at 30k.

### HD720 does not change the network input

HD720/60 reaches 59.90 FPS, but its images are resized to the same `448x640`
TensorRT input as VGA. It adds capture and resize work without increasing the
spatial resolution seen by these engine plans.

## Recommended configurations

| Goal | Cameras | Capture | Points | Voxel | Measured FPS |
|---|---:|---|---:|---:|---:|
| Maximum one-camera rate | 1 | VGA/100 | 5k | off | 99.96 |
| Balanced one-camera cloud | 1 | VGA/100 | 15k | off | 92.79 |
| Maximum two-camera rate | 2 | VGA/100 | 5k | off | 48.75 |
| Balanced two-camera cloud | 2 | VGA/100 | 15k | off | 45.31 |

## Reproduce the complete matrix

```bash
python streamer/benchmark_zed_trt.py \
  --serial <SERIAL_1> \
  --serial <SERIAL_2> \
  --engine-dir output \
  --frames 180 \
  --warmup-runs 3 \
  --output-json streamer/TRT_BENCHMARK_RESULTS.json
```

Providing one serial runs only the one-camera cases. Providing two serials also
runs the two-camera cases.

To reproduce only the eight capture-resolution cases above, add
`--suite resolution` and write them to a separate file:

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

Reproduce the eight 30k variations with:

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

## Run one custom case

```bash
python streamer/Streamer_real_one_cam.py \
  --serial <SERIAL> \
  --engine-dir output \
  --resolution VGA \
  --camera-fps 100 \
  --max-points 15000 \
  --voxel-size 0 \
  --benchmark-frames 180
```

The final log line starts with `BENCHMARK_RESULT` and contains JSON suitable for
automation.

## Limitations

- This benchmark measures speed, not depth accuracy.
- No live viewer or remote subscriber was connected during timing.
- The two identity-transformed clouds are not geometrically registered.
- The bounding-box result is scene- and box-dependent.
- Camera exposure and scene content may slightly affect valid point count.
- TensorRT engines should be rebuilt on the deployment GPU.
