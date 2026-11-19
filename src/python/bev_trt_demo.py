import os
import json
import time
import cv2
import numpy as np
import torch
import tensorrt as trt
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion

# 1. Load real CODA class names from dataset metadata
with open("data/coda/train/_annotations.coco.json") as f:
    coco_meta = json.load(f)
CLASS_NAMES = {c["id"]: c["name"].upper() for c in coco_meta["categories"]}

# 2. Load compiled TensorRT 11.5 engine with a dedicated non-default CUDA stream
runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
with open("models/engines/rfdetr_rtx5050.engine", "rb") as f:
    engine = runtime.deserialize_cuda_engine(f.read())
context = engine.create_execution_context()

stream = torch.cuda.Stream()
d_input = torch.empty((1, 3, 560, 560), dtype=torch.float32, device="cuda")
d_dets = torch.empty((1, 300, 4), dtype=torch.float32, device="cuda")
d_labels = torch.empty((1, 300, 30), dtype=torch.float32, device="cuda")

context.set_tensor_address("input", d_input.data_ptr())
context.set_tensor_address("dets", d_dets.data_ptr())
context.set_tensor_address("labels", d_labels.data_ptr())

MEAN = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 3, 1, 1)

def infer_trt(frame_bgr, conf_thresh=0.40):
    H, W = frame_bgr.shape[:2]
    img_rgb = cv2.cvtColor(cv2.resize(frame_bgr, (560, 560)), cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(img_rgb).to("cuda", non_blocking=True).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    d_input.copy_((tensor - MEAN) / STD)

    context.execute_async_v3(stream.cuda_stream)
    stream.synchronize()

    probs = torch.sigmoid(d_labels[0])
    scores, cls_ids = torch.max(probs, dim=-1)
    keep = scores > conf_thresh

    boxes = d_dets[0][keep].cpu().numpy()
    scores = scores[keep].cpu().numpy()
    cls_ids = cls_ids[keep].cpu().numpy()

    results = []
    for (cx, cy, w, h), sc, cid in zip(boxes, scores, cls_ids):
        x1 = int(max(0, (cx - w / 2.0) * W))
        y1 = int(max(0, (cy - h / 2.0) * H))
        x2 = int(min(W - 1, (cx + w / 2.0) * W))
        y2 = int(min(H - 1, (cy + h / 2.0) * H))
        label = CLASS_NAMES.get(int(cid), f"HAZ-{cid}")
        results.append((x1, y1, x2, y2, float(sc), label))
    return results

nusc = NuScenes(version='v1.0-mini', dataroot='data/nuscenes-mini', verbose=False)
CAMERAS = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
active_tracks, next_tid = {}, 1

def update_tracks(dets_3d, ts, max_dist=3.5):
    global next_tid
    out, matched = [], set()
    for (xf, yl, lbl) in dets_3d:
        best_id, best_d = None, max_dist
        for tid, trk in active_tracks.items():
            if tid in matched:
                continue
            dt = max(ts - trk['t'], 1e-3)
            d = np.hypot(xf - (trk['x'] + trk['vx']*dt), yl - (trk['y'] + trk['vy']*dt))
            if d < best_d:
                best_d, best_id = d, tid
        if best_id is not None:
            trk = active_tracks[best_id]
            dt = max(ts - trk['t'], 1e-3)
            vx = 0.6 * trk['vx'] + 0.4 * ((xf - trk['x']) / dt)
            vy = 0.6 * trk['vy'] + 0.4 * ((yl - trk['y']) / dt)
            active_tracks[best_id] = {'x': xf, 'y': yl, 't': ts, 'vx': vx, 'vy': vy, 'm': 0}
            matched.add(best_id)
            out.append((xf, yl, vx, vy, lbl, best_id))
        else:
            tid = next_tid
            next_tid += 1
            active_tracks[tid] = {'x': xf, 'y': yl, 't': ts, 'vx': 0.0, 'vy': 0.0, 'm': 0}
            matched.add(tid)
            out.append((xf, yl, 0.0, 0.0, lbl, tid))
    for tid in list(active_tracks):
        if tid not in matched:
            active_tracks[tid]['m'] += 1
            if active_tracks[tid]['m'] > 3:
                del active_tracks[tid]
    return out

def pixel_to_bev(u, v, K, trans, rot):
    ray_ego = Quaternion(rot).rotation_matrix @ (np.linalg.inv(K) @ np.array([u, v, 1.0]))
    if ray_ego[2] >= -1e-3:
        return None
    pt = np.array(trans) + (-trans[2] / ray_ego[2]) * ray_ego
    if abs(pt[0]) > 55.0 or abs(pt[1]) > 55.0:
        return None
    return float(pt[0]), float(pt[1])

def draw_radar(hazards, fps, lat_ms, size=1080, max_r=50.0):
    radar = np.full((size, size, 3), (16, 18, 22), dtype=np.uint8)
    c = size // 2
    sc = (c - 60) / max_r
    for r in range(10, 60, 10):
        cv2.circle(radar, (c, c), int(r * sc), (45, 50, 58), 1)
        cv2.putText(radar, f"{r}m", (c + 6, c - int(r * sc) + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 130, 145), 1)
    cv2.line(radar, (c, 40), (c, size - 40), (35, 40, 48), 1)
    cv2.line(radar, (40, c), (size - 40, c), (35, 40, 48), 1)
    vw, vl = int(0.95 * sc), int(2.35 * sc)
    cv2.rectangle(radar, (c - vw, c - vl), (c + vw, c + vl), (0, 220, 130), -1)
    cv2.putText(radar, "EGO", (c - 16, c + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (10, 15, 20), 2)

    for (xf, yl, vx, vy, lbl, tid) in hazards:
        px, py = int(c - yl * sc), int(c - xf * sc)
        spd = np.hypot(vx, vy)
        col = (0, 70, 255) if spd > 1.5 else (0, 185, 255)
        cv2.circle(radar, (px, py), 7, col, -1)
        if spd > 0.6:
            cv2.arrowedLine(radar, (px, py), (int(px - vy * sc), int(py - vx * sc)), (0, 255, 255), 2, tipLength=0.3)
        cv2.putText(radar, f"#{tid} {lbl} {np.hypot(xf, yl):.1f}m", (px + 11, py + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (235, 240, 245), 1)

    cv2.putText(radar, "3D BEV OPEN-SET HAZARD RADAR (TENSORRT 11.5)", (25, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 220, 130), 2)
    cv2.putText(radar, f"RTX 5050 6-Cam Latency: {lat_ms:.1f} ms | Pipeline: {fps:.1f} FPS | Tracked: {len(hazards)}", (25, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (190, 200, 215), 1)
    return radar

def main():
    os.makedirs("outputs", exist_ok=True)
    writer = cv2.VideoWriter("outputs/tesla_bev_trt_raw.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 12.0, (1920, 1080))
    token = nusc.scene[0]['first_sample_token']
    idx = 0
    while token:
        t0 = time.perf_counter()
        sample = nusc.get('sample', token)
        ts = sample['timestamp'] / 1e6
        raw_3d, panels = [], []
        for cam in CAMERAS:
            sd = nusc.get('sample_data', sample['data'][cam])
            calib = nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])
            frame = cv2.imread(os.path.join("data/nuscenes-mini", sd['filename']))
            for (x1, y1, x2, y2, sc, lbl) in infer_trt(frame):
                pt = pixel_to_bev((x1 + x2) / 2.0, float(y2), np.array(calib['camera_intrinsic']), calib['translation'], calib['rotation'])
                if pt:
                    raw_3d.append((pt[0], pt[1], lbl))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 130), 2)
                    cv2.putText(frame, f"{lbl} {np.hypot(*pt):.1f}m", (x1, max(y1 - 8, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 130), 2)
            p = cv2.resize(frame, (420, 360))
            cv2.putText(p, cam, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            panels.append(p)
        hazards = update_tracks(raw_3d, ts)
        lat_ms = (time.perf_counter() - t0) * 1000.0
        left = np.vstack([
            np.hstack([panels[0], panels[1]]),
            np.hstack([panels[2], panels[3]]),
            np.hstack([panels[4], panels[5]])
        ])
        writer.write(np.hstack([left, draw_radar(hazards, 1000.0 / lat_ms, lat_ms)]))
        idx += 1
        print(f"Frame {idx:02d} | Tracked Hazards: {len(hazards):02d} | End-to-End 6-Cam Time: {lat_ms:.1f} ms")
        token = sample['next']
    writer.release()
    print("[DONE] Rendered TensorRT demo video!")

if __name__ == "__main__":
    main()
