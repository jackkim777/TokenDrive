# TokenDrive on NVIDIA DRIVE Pegasus

[TokenDrive](../../README.md) · [Runtime reference](RUNTIME.md) · [Source manifest](SOURCE_MANIFEST.json)

A selected source release of the **v3 real-vehicle inference pipeline**: GMSL camera acquisition, CUDA image preprocessing, ONNX Runtime inference, structured-token visualization, and vehicle-specific CAN control.

```text
GMSL camera -> rectification / crop / resize / normalization
                                      |
CAN speed + high-level command ------> TokenDrive ONNX
                                      |
                     decision tokens + 10 speed / 64 route waypoints
                                      |
                         visualization / optional control / logging
```

## Included sources

| File | Purpose |
| :--- | :--- |
| [`inference_v3.cpp`](inference_v3.cpp) | The v3 camera, inference, visualization, controller, and logging runtime. |
| [`preprocess_cuda.cu`](preprocess_cuda.cu), [`preprocess_cuda.cuh`](preprocess_cuda.cuh) | GPU image transformation and normalization. |
| [`benchmark_onnx.cpp`](benchmark_onnx.cpp) | Model-session latency measurement with synthetic inputs. |
| [`eval_metrics.py`](eval_metrics.py) | Offline intervention and task metrics from CAN/RTK logs. |
| [`can_smoothness.py`](can_smoothness.py) | Offline acceleration, jerk, yaw, and smoothness analysis. |
| `nvmedia_stubs/` | Small local header shims from the vehicle build; not an SDK implementation. |
| [`CMakeLists.txt`](CMakeLists.txt) | Build targets for v3 inference and the latency benchmark. |

This package omits older inference variants, checkpoint and ONNX binaries, model caches, recordings, GPS traces, experiment results, compiled executables, and device setup scripts. The Python training policy remains in [`tokendrive/`](../../tokendrive); this deployment runtime consumes a separately supplied ONNX export.

## Platform and dependencies

The archived target is **NVIDIA DRIVE Pegasus / aarch64 Linux**, with **CUDA 10.2**, **DriveWorks 2.2**, and the **ONNX Runtime 1.10 C++ API**. It also uses GLFW, EGL, GLESv2, X11, and Linux SocketCAN. Install the matching NVIDIA SDK and runtime libraries separately. Newer ONNX Runtime APIs may require a port; this release preserves the vehicle implementation.

Camera projection, CAN signal definitions, wheelbase, steering ratio, and control tuning are specific to the recorded vehicle setup. They are source parameters, not a portable calibration profile.

## Build

On the target Linux machine, from the TokenDrive repository root:

```bash
mkdir -p deployment/pegasus/build
(cd deployment/pegasus/build && cmake .. \
  -DORT_INCLUDE_DIR=/path/to/onnxruntime/include/onnxruntime/core/session \
  -DORT_LIBRARY=/path/to/onnxruntime/libonnxruntime.so \
  -DDW_ROOT=/usr/local/driveworks-2.2/targets/aarch64-Linux \
  -DGLFW_INCLUDE_DIR=/path/to/glfw/include)

cmake --build deployment/pegasus/build --target inference_v3 -- -j4
cmake --build deployment/pegasus/build --target benchmark_onnx -- -j4
```

`ORT_INCLUDE_DIR` must directly contain `onnxruntime_cxx_api.h`. For a packaged ONNX Runtime distribution this may simply be its `include/` directory. `ORT_LIBRARY` must point to the matching shared library. Keep its provider libraries and other dynamic dependencies available at runtime.

`USE_CUDA` defaults to `ON` and requires a matching CUDA toolchain. `USE_NVMEDIA_STUBS` defaults to `ON` to match the archived build; use `-DUSE_NVMEDIA_STUBS=OFF` with a complete compatible SDK if appropriate. The shims do not replace NVIDIA runtime libraries.

For a CPU-only ONNX benchmark without DriveWorks:

```bash
mkdir -p deployment/pegasus/build-benchmark
(cd deployment/pegasus/build-benchmark && cmake .. \
  -DBUILD_VEHICLE_RUNTIME=OFF -DUSE_CUDA=OFF \
  -DORT_INCLUDE_DIR=/path/to/onnxruntime/include \
  -DORT_LIBRARY=/path/to/onnxruntime/libonnxruntime.so)
cmake --build deployment/pegasus/build-benchmark --target benchmark_onnx -- -j4
```

## Inference v3

Supply the matching **FP32, WP64, v3, token-output-enabled ONNX export**. ONNX weights are not included, and this package does not supply an exporter. The export's input/output ordering must match the [runtime contract](RUNTIME.md#onnx-contract); renaming an incompatible model file does not convert it.

Start with waypoint prediction and visualization:

```bash
cd deployment/pegasus
./build/inference_v3 \
  --model /path/to/tokendrive_fp32_wp64_op15_v3_token_on.onnx \
  --camera-group c \
  --camera-view ftheta-rectilinear \
  --no-rtk
```

The default control mode is waypoint-only. Selecting `--pp-steer`, `--pp-full`, `--stanley-steer`, or `--stanley-full` enables a vehicle-specific control mode; full modes retain the original confirmation prompt. Review the [control behavior](RUNTIME.md#control-and-device-behavior) before using them on connected hardware.

Use `--help` for the available options. `--no-gui` disables visualization; camera acquisition and CAN reception remain part of the vehicle runtime. Recording is opt-in with `--save-frames` and `--log-can`.

## Model latency

```bash
./build/benchmark_onnx \
  --model /path/to/tokendrive_fp32_wp64_op15_v3_token_on.onnx \
  --device cuda --warmup 30 --iters 100
```

Use `--device cpu` with a CPU-only build. The benchmark uses synthetic zero-filled input tensors and reports model-session timing statistics. It does not measure GMSL acquisition, preprocessing, visualization, CAN, or end-to-end vehicle response, and is intended for the FP32 export.

## Offline analysis

These tools run separately from the vehicle binary and require no NVIDIA hardware:

```bash
python -m pip install -r deployment/pegasus/requirements-analysis.txt
python deployment/pegasus/eval_metrics.py /path/to/can_log.csv
python deployment/pegasus/can_smoothness.py /path/to/can_log.csv
```

The metrics utility reads the first eight numeric columns in this order:

```text
timestamp_us,speed_kmh,steering_angle_deg,override,hlc,rtk_lat,rtk_lon,rtk_valid
```

Smoothness analysis uses the first three columns. Use regular 20 Hz logs and at least one complete analysis segment (20 samples by default). The scripts retain the vehicle's wheelbase and steering ratio. Intervention detection counts an `override` transition from 1 to 0; interpreting these events as failures requires separate review. RTK distance depends on valid fixes. No measured vehicle results are published in this source package.

The archived `eval_metrics.py --plot` path has a `Path + str` filename construction issue; the documented console workflow avoids it. The analysis implementations have otherwise been preserved byte-for-byte.

## Source preservation and validation

The original vehicle directory was treated as read-only. The published C++ copies change project/class/window names to TokenDrive, update build/help examples to v3, and use `./save_frames` instead of a machine-specific absolute recording path. The CUDA preprocessing, CAN protocol, controllers, calibration constants, and analysis calculations are preserved. The build file and these guides are new packaging.

[`SOURCE_MANIFEST.json`](SOURCE_MANIFEST.json) records SHA-256 hashes of each selected source and published copy. This release was checked with source comparisons and synthetic offline logs. Building and running the hardware pipeline still requires validation on the target Pegasus SDK and device.
