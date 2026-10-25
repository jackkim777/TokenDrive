# TokenDrive v3 runtime reference

[Deployment guide](README.md) · [TokenDrive](../../README.md)

## ONNX contract

The runtime queries tensor names from the model but binds buffers by **position**. Export order, types, and dimensions must match exactly.

| Input position | Meaning | Shape | Type |
| :--- | :--- | :--- | :--- |
| 0 | Front-view RGB image | `[1, 1, 1, 3, 448, 448]` | float32 |
| 1 | Ego speed in m/s | `[1, 1]` | float32 |
| 2 | High-level command | `[1]` | int64 |

| Output position | Meaning | Shape | Type |
| :--- | :--- | :--- | :--- |
| 0 | Speed waypoints | `[1, 10, 2]` | float32 |
| 1 | Route waypoints | `[1, 64, 2]` | float32 |
| 2 | Lateral-action logits | `[1, 4]` | float32 |
| 3 | Longitudinal-action logits | `[1, 8]` | float32 |
| 4 | Speed-context kind logits | `[1, 4]` | float32 |
| 5 | Speed-context subtype logits | `[1, 6]` | float32 |
| 6 | Route-context kind logits | `[1, 3]` | float32 |
| 7 | Route-context subtype logits | `[1, 5]` | float32 |

The runtime's input normalization uses RGB ImageNet means `[0.485, 0.456, 0.406]` and standard deviations `[0.229, 0.224, 0.225]`. The baseline view crops the bottom `4.8/16` of the frame before resizing to 448 x 448. The default `ftheta-rectilinear` view additionally uses the archived camera projection. Its CPU and CUDA implementations are in the supplied sources.

## Main options

| Option | Behavior / default |
| :--- | :--- |
| `--model FILE`, `-m FILE` | ONNX file; default `tokendrive_fp32_wp64_op15_v3_token_on.onnx`. |
| `--camera-type TYPE` | Default `ar0231-rccb-bae-sf3324`. |
| `--camera-group GROUP` | Default `c`. |
| `--camera-view MODE` | `baseline` or `ftheta-rectilinear` (default). |
| `--hlc VALUE` | Command 1-6; default 4 (follow lane). |
| `--speed MPS` | Initial ego speed; default 0.0, updated from CAN feedback. |
| `--cpu` | Select CPU inference. |
| `--no-gui` | Disable visualization and use terminal keyboard input. |
| `--no-rtk` | Disable RTK connection. |
| `--rtk IP:PORT` | RTK TCP endpoint; archived default `192.168.1.100:3002`. |
| `--save-frames` | Record raw frames and speed. |
| `--log-can` | Record CAN/RTK data at 20 Hz. |
| `--steer-speed VALUE` | Steering speed command, clamped to 0-255. |
| `--free-camera` | Stop camera-owner processes and retry initialization. |

HLC values are 1 = left, 2 = right, 3 = straight, 4 = follow lane, 5 = change left, 6 = change right. Number keys select commands during inference. Escape or Ctrl+C exits. In a control mode, `O` toggles the software override; `T` selects a tuning parameter and arrow keys adjust it.

## Control and device behavior

| Mode | Steering | Longitudinal control |
| :--- | :--- | :--- |
| Default | Off | Off |
| `--pp-steer` | Pure Pursuit | Off |
| `--pp-full` | Pure Pursuit | Speed PID |
| `--stanley-steer` | Stanley | Off |
| `--stanley-full` | Stanley | Speed PID |

These modes contain the original vehicle's CAN definitions and calibration. Full modes send real acceleration/braking commands and keep the original `YES` prompt and countdown. A selected control mode starts the CAN sender thread; the software override starts in manual. See `can_sender_thread` for the actual frame behavior, including manual-state signaling. No control code or interlocks were changed for this release.

The receiver uses `can0`. The default RTK connection is enabled unless `--no-rtk` is passed. `--free-camera` explicitly invokes process-termination commands, including privileged commands, against camera users; it is not needed for the normal example in the deployment guide.

## Recording and tuning files

Recording uses the current working directory:

```text
save_frames/
`-- run_<timestamp>/
    |-- frames/
    |   |-- frames.csv
    |   `-- <frame images>.ppm
    `-- can/
        `-- can_log_<timestamp>.csv
```

The directory is created when recording is enabled. `tune_defaults.cfg` is read from and written to the current working directory. Device-specific tuning and recordings are ignored by Git. No tuning file is bundled, so a fresh copy starts with the compiled defaults.
