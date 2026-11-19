

https://github.com/user-attachments/assets/a953ae6d-8dcb-47b7-be00-d42561d3a31a


<img width="958" height="628" alt="cpp_benchmark png" src="https://github.com/user-attachments/assets/f2ac20e1-7aaf-4ee8-bfbc-85b5d14d1f31" />
# Multi-Camera Open-Set Road Hazard & 3D Bird's-Eye-View (BEV) Perception Pipeline

A real-time, camera-only 3D spatial perception pipeline built to detect long-tail road hazards (fallen cargo, traffic cones, construction barriers, road debris) across 6 synchronized surround cameras, project detections onto a metric Bird's-Eye-View (BEV) ground plane, track 3D velocity vectors, and execute in **C++17 + NVIDIA TensorRT 11.5**.

## System Architecture

```text
[6 Surround Cameras] ──> [RF-DETR (29 CODA Hazard Classes)] ──> [2D Contact Point (u, v)]
                                                                           │
[Split-Screen Radar] <── [3D Velocity & Occupancy Tracker] <── [Ground-Plane Ray Cast (Z=0)]
```

1. **Corner-Case 2D Hazard Detection (`RF-DETR` on CODA):** Fine-tuned for 25 epochs on the **CODA Corner-Case Dataset** (29 hazard classes) using mixed-precision (`bfloat16`) training.
2. **Multi-Camera 3D Ground-Plane Projection (`nuScenes`):** Back-projects bottom-center bounding box contact pixels (u, v) through camera intrinsic matrix K and extrinsic rotation/translation [R|t] onto the ego-vehicle ground plane (Z = 0).
3. **3D Metric Velocity Tracking:** Associates detections across frames in BEV space to compute smoothed velocity vectors (vx, vy) and separate static road hazards from moving traffic.
4. **Hardware Acceleration (`C++17` & `TensorRT 11.5`):** Exported to ONNX and compiled into a native TensorRT engine executed on a non-default CUDA stream (`enqueueV3`).

## Evaluation & Hardware Benchmarks (NVIDIA RTX 5050 8GB)

### Detection Accuracy (CODA Validation Split — 29 Classes)

| Metric | Peak Score |
| :--- | :---: |
| **mAP @ 50** | **58.55%** (`0.5855`) |
| **mAP @ 75** | **44.56%** (`0.4456`) |
| **mAP @ 50:95** | **40.81%** (`0.4081`) |

### Inference Latency & Throughput

| Runtime Backend | Single-Camera Latency | Single-Camera FPS | 6-Camera Surround Latency | Speedup |
| :--- | :---: | :---: | :---: | :---: |
| PyTorch Eager (FP32) | ~88.50 ms | 11.3 FPS | ~531.0 ms | 1.0x |
| **C++17 + TensorRT 11.5 (`enqueueV3`)** | **19.83 ms** | **50.43 FPS** | **~118.98 ms** | **4.5x** |

## Build & Run (C++17 TensorRT Inference)
```bash
g++ -O3 -std=c++17 src/cpp/main.cpp -o build/bev_perception \
  -Ithird_party/tensorrt/include \$(pkg-config --cflags --libs opencv4) \
  -lnvinfer -lcudart
./build/bev_perception models/engines/rfdetr_rtx5050.engine sample_front.jpg
```
