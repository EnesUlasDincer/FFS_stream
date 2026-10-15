# ZED + Fast-FoundationStereo point-cloud streaming

The two entry points capture rectified ZED stereo images, infer disparity with
the two-stage Fast-FoundationStereo TensorRT pipeline, create an RGB point
cloud, and publish it with the same ZMQ multipart format as
`ZMQPublishPointCloudAction`.

Segmentation is always disabled. No YOLO/segmentation client is created.

## Prerequisites

- A CUDA environment containing the dependencies from `requirements.txt`.
- The ZED SDK and its matching Python API. On Linux, install the API into the
  active Python environment with `/usr/local/zed/get_python_api.py`.
- TensorRT plus the working engine set in `output/`:
  `feature_runner.engine`, `post_runner.engine`, and `onnx.yaml`.

The local engines have static input shape `(1, 3, 448, 640)`. One camera
uses one TensorRT call per frame. Two cameras reuse the same runner and perform
two sequential calls before merging the point clouds.

## One camera

Use the first available ZED:

```bash
python3 streamer/Streamer_real_one_cam.py
```

Select a camera explicitly:

```bash
python3 streamer/Streamer_real_one_cam.py \
  --serial <SERIAL> \
  --resolution VGA \
  --engine-dir output \
  --z-min 0.10 \
  --z-max 5.0
```

The output frame is the rectified left optical frame unless `--transform` is
provided.

## Two cameras

For the current FPS test, omit `--transform`. Both cameras then use identity:

```bash
python3 streamer/Streamer_real_two_cam.py \
  --serial <SERIAL_1> \
  --serial <SERIAL_2> \
  --engine-dir output
```

The resulting clouds are not spatially registered. For correct cloud merging,
camera 1 is the reference (identity), and the supplied matrix must map camera 2
into camera 1's frame:

```bash
python3 streamer/Streamer_real_two_cam.py \
  --serial <SERIAL_1> \
  --serial <SERIAL_2> \
  --engine-dir output \
  --transform /path/to/camera2_to_camera1.txt
```

Transforms can be a whitespace-delimited text matrix, a `.npy` file, or YAML.
For YAML containing multiple matrices, append the key to the path:

```bash
--transform /path/to/extrinsics.yml:T_follower_to_main
```

## Stream format

The publisher binds to TCP port `5555` by default and sends four frames:

1. UTF-8 topic `PointCloud`
2. little-endian `uint32` point count
3. packed `(N, 3)` little-endian `int16` XYZ in millimetres
4. packed `(N, 3)` `uint8` RGB

UDP discovery remains compatible on port `5556`. Use `--help` to see crop,
voxel, point-limit, engine, resolution, and port options.

For streaming speed, voxel filtering is disabled by default and the merged
cloud is sampled directly to `--max-points 15000`. Enable it explicitly, for
example with `--voxel-size 0.005`, when spatial voxel filtering is preferred.

Use `--save-first-cloud path/to/sample.ply` to save the first processed RGB
cloud without stopping the stream. Current hardware results and a sample cloud
are provided in [`TRT_BENCHMARK.md`](TRT_BENCHMARK.md).

## TensorRT inference call

The live streamer loads `core.foundation_stereo.TrtRunner`. The actual call is
in `FastFoundationStereoTensorRTRunner._forward`:

```python
disparity = self.model.forward(left, right)
```

`TrtRunner.forward` executes `feature_runner.engine`, builds the GWC volume
with the Triton CUDA kernel, and executes `post_runner.engine`. Run the clean
standalone reference on the demo pair with:

```bash
python3 scripts_v2/run_demo_tensorrt.py --engine-dir output
```

For a longer engine-only timing run, use:

```bash
python3 scripts/profile_speed_tensorrt.py \
  --onnx_dir output \
  --warmup 15 \
  --total 30
```
