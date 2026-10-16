# Interactive RGB point-cloud samples

[Open the interactive 3D explorer](https://raw.githack.com/EnesUlasDincer/FFS_stream/zed-trt-pointcloud-streaming/streamer/pointcloud_samples/index.html)

The browser explorer reads [`manifest.json`](manifest.json) and lets you
compare every resolution and point density without editing the page. It
contains the complete requested matrix:

- ZED modes: VGA, HD720, HD1080, and HD2K
- point limits: 5k, 10k, 20k, 30k, 40k, and 50k
- additional legacy comparison: 15k
- segmentation: disabled
- X orientation: reflected to match the verified real-world/legacy VR frame
- stored XYZ scale: 1.5

The native-resolution RGB image for each mode was captured with its 30k PLY.
The 30k selection is therefore an exact-frame RGB/cloud pair; the other point
counts are separate captures of the same scene and use that mode's nearby RGB
reference. The viewer shows RGB XYZ axes, point count, camera rate, point-size
control, temporary display scaling, auto-rotation, fullscreen, and direct PLY
downloads.

The two FPS cards load the measured 180-frame, 48-case TensorRT results from
[`TRT_RESOLUTION_POINT_MATRIX_RESULTS.json`](../TRT_RESOLUTION_POINT_MATRIX_RESULTS.json).
They update with the selected resolution and density. The retained 15k samples
show “Not evaluated” because 15k is intentionally outside the requested six
point-count matrix.

| Capture mode | RGB reference | Point clouds |
|---|---|---|
| VGA | [![VGA RGB](one_zed_vga_rgb.png)](one_zed_vga_rgb.png) | [5k](one_zed_vga_5000.ply) · [10k](one_zed_vga_10000.ply) · [15k extra](one_zed_vga_15000.ply) · [20k](one_zed_vga_20000.ply) · [30k](one_zed_vga_30000.ply) · [40k](one_zed_vga_40000.ply) · [50k](one_zed_vga_50000.ply) |
| HD720 | [![HD720 RGB](one_zed_hd720_rgb.png)](one_zed_hd720_rgb.png) | [5k](one_zed_hd720_5000.ply) · [10k](one_zed_hd720_10000.ply) · [15k extra](one_zed_hd720_15000.ply) · [20k](one_zed_hd720_20000.ply) · [30k](one_zed_hd720_30000.ply) · [40k](one_zed_hd720_40000.ply) · [50k](one_zed_hd720_50000.ply) |
| HD1080 | [![HD1080 RGB](one_zed_hd1080_rgb.png)](one_zed_hd1080_rgb.png) | [5k](one_zed_hd1080_5000.ply) · [10k](one_zed_hd1080_10000.ply) · [15k extra](one_zed_hd1080_15000.ply) · [20k](one_zed_hd1080_20000.ply) · [30k](one_zed_hd1080_30000.ply) · [40k](one_zed_hd1080_40000.ply) · [50k](one_zed_hd1080_50000.ply) |
| HD2K | [![HD2K RGB](one_zed_hd2k_rgb.png)](one_zed_hd2k_rgb.png) | [5k](one_zed_hd2k_5000.ply) · [10k](one_zed_hd2k_10000.ply) · [15k extra](one_zed_hd2k_15000.ply) · [20k](one_zed_hd2k_20000.ply) · [30k](one_zed_hd2k_30000.ply) · [40k](one_zed_hd2k_40000.ply) · [50k](one_zed_hd2k_50000.ply) |

## Run the website locally

From the repository root, serve the repository so the explorer can also load
the benchmark JSON one directory above the samples:

```bash
python -m http.server 8000
```

Open <http://127.0.0.1:8000/streamer/pointcloud_samples/>. A local HTTP server is
required because browsers block the JavaScript module and PLY requests when
`index.html` is opened directly as a local file.

## Capture the full matrix again

```bash
python streamer/save_zed_pointcloud_variants.py \
  --serial <SERIAL> \
  --engine-dir output \
  --output-dir output_clouds/scale_1p5
```

The defaults capture all four modes at 5k, 10k, 20k, 30k, 40k, and 50k with
`--xyz-scale 1.5`, X reflection enabled, and segmentation disabled. Existing
files are protected; add `--overwrite` only when replacement is intentional.
Use `--dry-run` to list all 24 capture commands without opening the camera.
