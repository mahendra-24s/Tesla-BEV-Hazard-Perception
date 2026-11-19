import os
import time
import cv2
import numpy as np
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion
from rfdetr import RFDETRBase

# 1. Load your newly trained 29-class road hazard checkpoint
ckpt_candidates = [
    "models/checkpoints/checkpoint_best_ema.pth",
    "models/checkpoints/checkpoint_best_total.pth",
    "models/checkpoints/checkpoint.pth"
]
weights_path = next((p for p in ckpt_candidates if os.path.exists(p)), None)
print(f"[INFO] Loading model weights: {weights_path or 'default pretrained RFDETRBase'}")

model = RFDETRBase(pretrain_weights=weights_path, num_classes=29) if weights_path else RFDETRBase()
nusc = NuScenes(version='v1.0-mini', dataroot='data/nuscenes-mini', verbose=False)

CAMERAS = [
    'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
    'CAM_BACK_LEFT',  'CAM_BACK',  'CAM_BACK_RIGHT'
]

# Lightweight 3D Metric Centroid Tracker (ports cleanly to C++ in Step 7)
active_tracks = {}  # track_id -> {'x': float, 'y': float, 't': float, 'vx': float, 'vy': float, 'missed': int}
next_track_id = 1

def update_3d_tracks(detections_3d, timestamp_s, max_dist=3.5):
    """Associates new 3D BEV detections with existing tracks to compute velocity vectors."""
    global next_track_id
    updated_hazards = []
    matched_ids = set()

    for (x_fwd, y_lat, label) in detections_3d:
        best_id = None
        best_dist = max_dist

        for tid, trk in active_tracks.items():
            if tid in matched_ids:
                continue
            dt = max(timestamp_s - trk['t'], 1e-3)
            # Predict current position from previous velocity
            pred_x = trk['x'] + trk['vx'] * dt
            pred_y = trk['y'] + trk['vy'] * dt
            d = np.hypot(x_fwd - pred_x, y_lat - pred_y)
            if d < best_dist:
                best_dist = d
                best_id = tid

        if best_id is not None:
            trk = active_tracks[best_id]
            dt = max(timestamp_s - trk['t'], 1e-3)
            raw_vx = (x_fwd - trk['x']) / dt
            raw_vy = (y_lat - trk['y']) / dt
            # Exponential smoothing for clean velocity arrows
            vx = 0.6 * trk['vx'] + 0.4 * raw_vx
            vy = 0.6 * trk['vy'] + 0.4 * raw_vy
            active_tracks[best_id] = {'x': x_fwd, 'y': y_lat, 't': timestamp_s, 'vx': vx, 'vy': vy, 'missed': 0}
            matched_ids.add(best_id)
            updated_hazards.append((x_fwd, y_lat, vx, vy, label, best_id))
        else:
            tid = next_track_id
            next_track_id += 1
            active_tracks[tid] = {'x': x_fwd, 'y': y_lat, 't': timestamp_s, 'vx': 0.0, 'vy': 0.0, 'missed': 0}
            matched_ids.add(tid)
            updated_hazards.append((x_fwd, y_lat, 0.0, 0.0, label, tid))

    # Age out stale tracks not seen for 3 frames
    for tid in list(active_tracks.keys()):
        if tid not in matched_ids:
            active_tracks[tid]['missed'] += 1
            if active_tracks[tid]['missed'] > 3:
                del active_tracks[tid]

    return updated_hazards

def pixel_to_ego_bev(u, v, K, cam_trans, cam_rot):
    """Projects bottom-center bounding box pixel (u, v) onto ego ground plane (Z=0)."""
    ray_cam = np.linalg.inv(K) @ np.array([u, v, 1.0])
    R = Quaternion(cam_rot).rotation_matrix
    ray_ego = R @ ray_cam
    cam_pos = np.array(cam_trans)

    # Ignore rays pointing above the horizon
    if ray_ego[2] >= -1e-3:
        return None

    scale = -cam_pos[2] / ray_ego[2]
    pt_ego = cam_pos + scale * ray_ego
    x_fwd, y_lat = float(pt_ego[0]), float(pt_ego[1])

    if abs(x_fwd) > 60.0 or abs(y_lat) > 60.0:
        return None
    return x_fwd, y_lat

def draw_bev_radar(hazards_3d, fps, latency_ms, size=1080, max_range=50.0):
    """Renders top-down 3D BEV radar with distance rings, ego vehicle, and velocity arrows."""
    radar = np.zeros((size, size, 3), dtype=np.uint8)
    radar[:] = (18, 20, 24)
    center = size // 2
    scale = (center - 60) / max_range

    for r in range(10, 60, 10):
        radius_px = int(r * scale)
        cv2.circle(radar, (center, center), radius_px, (45, 50, 58), 1)
        cv2.putText(radar, f"{r}m", (center + 6, center - radius_px + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 130, 145), 1)

    cv2.line(radar, (center, 40), (center, size - 40), (35, 40, 48), 1)
    cv2.line(radar, (40, center), (size - 40, center), (35, 40, 48), 1)

    # Ego Vehicle footprint (~4.7m x 1.9m)
    veh_w = int((1.9 / 2.0) * scale)
    veh_l = int((4.7 / 2.0) * scale)
    cv2.rectangle(radar, (center - veh_w, center - veh_l),
                  (center + veh_w, center + veh_l), (0, 220, 130), -1)
    cv2.putText(radar, "EGO", (center - 16, center + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (10, 15, 20), 2)

    for (x_fwd, y_lat, vx, vy, label, tid) in hazards_3d:
        px = int(center - y_lat * scale)
        py = int(center - x_fwd * scale)

        speed = np.hypot(vx, vy)
        color = (0, 70, 255) if speed > 1.5 else (0, 180, 255)

        cv2.circle(radar, (px, py), 8, color, -1)
        cv2.circle(radar, (px, py), 12, color, 1)

        if speed > 0.5:
            end_px = int(px - vy * 1.0 * scale)
            end_py = int(py - vx * 1.0 * scale)
            cv2.arrowedLine(radar, (px, py), (end_px, end_py), (0, 255, 255), 2, tipLength=0.3)

        dist = np.hypot(x_fwd, y_lat)
        cv2.putText(radar, f"#{tid} {label} {dist:.1f}m | {speed:.1f}m/s",
                    (px + 14, py + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (235, 240, 245), 1)

    cv2.putText(radar, "3D BEV OCCUPANCY & VELOCITY RADAR", (30, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 220, 130), 2)
    cv2.putText(radar, f"6-Cam Inference: {latency_ms:.1f} ms | Pipeline FPS: {fps:.1f} | Hazards: {len(hazards_3d)}",
                (30, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 190, 205), 1)
    return radar

def main():
    os.makedirs("outputs", exist_ok=True)
    out_video_path = "outputs/tesla_bev_demo_py.mp4"
    writer = cv2.VideoWriter(out_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 12.0, (1920, 1080))

    scene = nusc.scene
    sample_token = scene[0]['first_sample_token']
    frame_idx = 0

    print("[INFO] Processing 6-camera nuScenes sequence into 3D BEV split-screen video...")
    while sample_token:
        t_start = time.perf_counter()
        sample = nusc.get('sample', sample_token)
        timestamp_s = sample['timestamp'] / 1e6
        raw_detections_3d = []
        cam_panels = []

        for cam in CAMERAS:
            sd = nusc.get('sample_data', sample['data'][cam])
            calib = nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])
            img_path = os.path.join("data/nuscenes-mini", sd['filename'])
            frame = cv2.imread(img_path)

            detections = model.predict(frame, threshold=0.35)

            for i, xyxy in enumerate(detections.xyxy):
                x1, y1, x2, y2 = map(int, xyxy)
                cls_id = int(detections.class_id[i]) if detections.class_id is not None else 0
                label = f"HAZ-{cls_id}"

                u = (x1 + x2) / 2.0
                v = float(y2)
                bev_pt = pixel_to_ego_bev(
                    u, v,
                    np.array(calib['camera_intrinsic']),
                    calib['translation'],
                    calib['rotation']
                )

                if bev_pt is not None:
                    x_fwd, y_lat = bev_pt
                    raw_detections_3d.append((x_fwd, y_lat, label))

                    dist = np.hypot(x_fwd, y_lat)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 130), 2)
                    cv2.putText(frame, f"{label} {dist:.1f}m", (x1, max(y1 - 8, 15)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 130), 2)

            panel = cv2.resize(frame, (420, 360))
            cv2.putText(panel, cam, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cam_panels.append(panel)

        hazards_3d = update_3d_tracks(raw_detections_3d, timestamp_s)

        latency_ms = (time.perf_counter() - t_start) * 1000.0
        fps = 1000.0 / max(latency_ms, 1e-3)

        row1 = np.hstack([cam_panels[0], cam_panels[1]])
        row2 = np.hstack([cam_panels[2], cam_panels[3]])
        row3 = np.hstack([cam_panels[4], cam_panels[5]])
        left_grid = np.vstack([row1, row2, row3])

        right_radar = draw_bev_radar(hazards_3d, fps, latency_ms, size=1080)
        combined = np.hstack([left_grid, right_radar])

        writer.write(combined)
        frame_idx += 1
        print(f"Frame {frame_idx:02d} | Hazards Projected: {len(hazards_3d):02d} | Latency: {latency_ms:.1f} ms")
        sample_token = sample['next']

    writer.release()
    print(f"[DONE] Saved 1920x1080 split-screen demo video to {out_video_path}")

if __name__ == "__main__":
    main()