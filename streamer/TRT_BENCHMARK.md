# ZED TensorRT streaming benchmark

Recorded on 2026-10-15 with Fast-FoundationStereo's two-stage TensorRT path.

## Test system

- GPU: NVIDIA GeForce RTX 5080, 16 GB
- NVIDIA driver: 595.71.05
- TensorRT: 10.16.1.11
- PyTorch: 2.11.0+cu130
- CUDA reported by PyTorch: 13.0
- Engine input: static batch 1, RGB `448x640`
- ZED capture: rectified VGA `672x376` at a requested 100 FPS
- Native ZED depth: disabled
- Segmentation: disabled
- Point-cloud limit: 15,000 RGB points
- Voxel filtering: disabled
- Transforms: identity
- ZMQ publisher: enabled, with no subscriber connected during the test

## Results

| Pipeline | Observed stream rate | TensorRT inference |
|---|---:|---:|
| Standalone stereo pair | 118.6 FPS | 8.43 ms |
| One ZED camera | about 88.5 FPS (87.6-89.8) | 8.3-9.2 ms |
| Two ZED cameras | about 42.9 FPS (42.5-43.6) | 16.8-18.4 ms total |

The two-camera engines have fixed batch size 1, so each frame pair is inferred
sequentially through the same TensorRT contexts. The two identity clouds are
not spatially registered; that configuration is intended only for throughput
testing.

`--camera-fps` requests the ZED capture rate. A value of 30 limits the complete
loop to approximately 30 FPS even when inference is faster. A VGA request of
100 FPS was used here to expose the processing limit. The flag is an upper
bound, not a guarantee: capture, inference, point-cloud construction, and
publishing still determine the achieved rate.

## Reproduction

One camera:

```bash
python streamer/Streamer_real_one_cam.py \
  --serial <SERIAL> \
  --resolution VGA \
  --camera-fps 100 \
  --engine-dir output
```

Two cameras with identity transforms:

```bash
python streamer/Streamer_real_two_cam.py \
  --serial <SERIAL_1> \
  --serial <SERIAL_2> \
  --resolution VGA \
  --camera-fps 100 \
  --engine-dir output
```

Save one processed cloud while streaming:

```bash
python streamer/Streamer_real_one_cam.py \
  --serial <SERIAL> \
  --camera-fps 100 \
  --save-first-cloud streamer/sample_one_zed_15000.ply
```

The included [`sample_one_zed_15000.ply`](sample_one_zed_15000.ply) is a
binary little-endian PLY containing 15,000 XYZ points in metres and RGB color.

TensorRT emits a warning because the existing engine plans were built on a
different GPU model. They ran successfully for these measurements, but engines
built directly on the target GPU should be used for final performance numbers.
