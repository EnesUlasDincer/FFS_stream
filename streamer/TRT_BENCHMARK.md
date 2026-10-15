# Fast-FoundationStereo ZED TensorRT benchmark

This report measures the complete ZED-to-ZMQ RGB point-cloud loop while varying
camera rate, capture resolution, output point count, voxel filtering, and the
number of cameras. It was recorded on 2026-10-15.

Exact machine-readable measurements are available in
[`TRT_BENCHMARK_RESULTS.json`](TRT_BENCHMARK_RESULTS.json).

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
- `cloud` covers disparity back-projection, transform, merge, crop, voxel
  filtering, and point limiting. It excludes TensorRT and ZMQ publishing.
- The p95 column is the 95th-percentile per-frame latency.
- Two-camera identity clouds were intentionally unregistered for throughput
  testing.

## Parameter matrix

| Cameras | Input | Requested FPS | Limit | Voxel | Stream FPS | Inference mean / p95 | Cloud mean / p95 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | VGA | 30 | 15k | off | 30.20 | 8.93 / 9.52 ms | 1.97 / 2.17 ms |
| 1 | VGA | 60 | 15k | off | 60.13 | 9.04 / 9.37 ms | 1.92 / 2.12 ms |
| 1 | VGA | 100 | 5k | off | 95.47 | 8.69 / 9.26 ms | 1.33 / 1.44 ms |
| 1 | VGA | 100 | 15k | off | 88.38 | 8.74 / 9.30 ms | 1.91 / 2.06 ms |
| 1 | VGA | 100 | 30k | off | 80.90 | 8.85 / 9.37 ms | 2.72 / 2.95 ms |
| 1 | VGA | 100 | unlimited | off | 39.95 | 9.01 / 9.57 ms | 12.76 / 13.02 ms |
| 1 | VGA | 100 | 15k | 5 mm | 10.95 | 8.92 / 9.57 ms | 81.49 / 86.47 ms |
| 1 | HD720 | 60 | 15k | off | 59.90 | 9.51 / 9.95 ms | 2.05 / 2.21 ms |
| 2 | VGA | 100 | 5k | off | 46.52 | 17.55 / 17.86 ms | 2.94 / 3.13 ms |
| 2 | VGA | 100 | 15k | off | 43.04 | 17.67 / 17.97 ms | 4.42 / 4.71 ms |
| 2 | VGA | 100 | 30k | off | 39.52 | 17.59 / 17.92 ms | 6.38 / 6.80 ms |
| 2 | VGA | 100 | 15k | 5 mm | 4.62 | 17.78 / 18.20 ms | 197.14 / 201.68 ms |

The standalone demo pair ran one TensorRT inference in 8.43 ms, or 118.6 FPS,
without camera capture, cloud construction, or streaming.

## Findings

### Capture rate is an upper bound

At a 15k point limit, VGA/30 and VGA/60 deliver 30.20 and 60.13 FPS. The model
is fast enough in these cases, so camera scheduling sets the output rate. A
VGA/100 request exposes downstream costs and delivers 88.38 FPS at 15k points.

### Point count matters after inference

TensorRT remains close to 8.7-9.0 ms in all one-camera VGA cases. Changing only
the output limit produces these complete-loop rates:

| Output cloud | Stream FPS | Cloud mean |
|---:|---:|---:|
| 5k | 95.47 | 1.33 ms |
| 15k | 88.38 | 1.91 ms |
| 30k | 80.90 | 2.72 ms |
| about 270k, unlimited | 39.95 | 12.76 ms |

With voxel and crop disabled, the implementation selects valid disparity
pixels before creating the full dense cloud. This early-sampling path should be
used for real-time streaming.

### Current CPU voxel filtering is not real-time

The 5 mm option constructs the dense cloud and performs a NumPy `unique` pass
over voxel IDs before applying the 15k limit. Cloud processing rises to 81.49
ms for one camera and 197.14 ms for two, reducing throughput to 10.95 and 4.62
FPS respectively. A GPU voxel filter or a filter after early point limiting
would be required for real-time voxelized output.

### Two-camera inference is sequential

The fixed engine batch size is 1. Reusing the contexts for two sequential calls
costs approximately 17.6 ms per output frame. With direct sampling, the final
point cap then moves throughput from 46.52 FPS at 5k points to 39.52 FPS at 30k.

### HD720 does not change the network input

HD720/60 reaches 59.90 FPS, but its images are resized to the same `448x640`
TensorRT input as VGA. It adds capture and resize work without increasing the
spatial resolution seen by these engine plans.

## Recommended configurations

| Goal | Cameras | Capture | Points | Voxel | Measured FPS |
|---|---:|---|---:|---:|---:|
| Maximum one-camera rate | 1 | VGA/100 | 5k | off | 95.47 |
| Balanced one-camera cloud | 1 | VGA/100 | 15k | off | 88.38 |
| Maximum two-camera rate | 2 | VGA/100 | 5k | off | 46.52 |
| Balanced two-camera cloud | 2 | VGA/100 | 15k | off | 43.04 |

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
- Camera exposure and scene content may slightly affect valid point count.
- TensorRT engines should be rebuilt on the deployment GPU.
