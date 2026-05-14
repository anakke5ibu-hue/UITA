# ═══════════════════════════════════════════════════════════════
# deteksi_final.py — Backend Face Recognition Gate
# Integrasi: OpenVINO (YOLO+ArcFace), MediaPipe Anti-Spoofing,
# WebSocket + Toggle Anti-Spoof via frontend
# ═══════════════════════════════════════════════════════════════

import os
import sys
import cv2
import numpy as np
import pickle
import time
import base64
import json
import random
import asyncio
import threading
import queue
import requests
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Suppress MediaPipe logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GLOG_minloglevel'] = '2'
os.environ['MEDIAPIPE_DISABLE_GPU'] = '1'

import mediapipe as mp
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode

import torch
num_cpus = min(os.cpu_count() or 4, 4)
num_pytorch_cores = max(1, num_cpus // 2)
torch.set_num_threads(num_pytorch_cores)
torch.set_num_interop_threads(1)
print(f"🔧 PyTorch CPU Limit: {num_pytorch_cores} cores (out of {num_cpus})")

# ═══════════════════════════════════════════════════════════════
# KONFIGURASI
# ═══════════════════════════════════════════════════════════════
CONFIG = {
    "yolo_model_path":          "models/yolov11n-face_openvino_model",
    "arcface_model_path":       "models/buffalo_sc_rec.xml",
    "face_database_path":       "models/face_database.pkl",
    "mediapipe_model_path":     "models/face_landmarker.task",
    "yolo_threshold":           0.7,
    "recognition_threshold":    0.8,
    "camera_index":             0,
    "camera_width":             640,
    "camera_height":            480,
    "max_fps":                  15,
    "motion_threshold":         2000,
    "detection_freeze_frames":  20,
    "frame_skip_interval":      6,
    "roi_enabled":              True,
    "roi_center_ratio":         0.35,
    "roi_expansion_frames":     30,
}

# Anti-spoof thresholds
BLINK_THRESHOLD = 0.15
SMILE_THRESHOLD = 0.60
YAW_THRESH = 0.15
CHALLENGES = ['BLINK', 'HEAD', 'SMILE']

# Server backend URL (main.py)
SERVER_URL = "http://localhost:8001/identify-face"
DB_SERVER_URL = "http://localhost:8001"

# ═══════════════════════════════════════════════════════════════
# STATE GLOBAL
# ═══════════════════════════════════════════════════════════════
blocked_nims = set()
blocked_cache_time = 0
BLOCKED_CACHE_TTL = 30

camera_result = None
camera_lock = threading.Lock()
connected_clients = set()

last_denied_time = 0
last_granted_time = 0
last_granted_identity = None

# Anti-spoof toggle (dapat diubah via WebSocket)
antispof_enabled = True
antispof_lock = threading.Lock()

# Shared state untuk frame stream
_det_lock = threading.Lock()
_det_state = {
    'tracked': [],
    'liveness': {'passed': False, 'challenge': 'BLINK', 'ear': 0, 'yaw': 0.5, 'smile': 0},
    'status': 'idle',
    'user_data': None,
    'faces_data': [],
    'motion': 0.0,
}

# ═══════════════════════════════════════════════════════════════
# FUNGSI BANTU
# ═══════════════════════════════════════════════════════════════
def refresh_blocked_nims():
    global blocked_nims, blocked_cache_time
    try:
        res = requests.get(f"{DB_SERVER_URL}/blokir_user", timeout=5.0)
        if res.status_code == 200:
            data = res.json()
            blocked_nims = {row['nim'] for row in data if row.get('nim')}
            blocked_cache_time = time.time()
            print(f"🔒 Cache Blokir: {blocked_nims or 'kosong'}")
        else:
            print(f"⚠️ Gagal ambil blokir: {res.status_code}")
    except Exception as e:
        print(f"⚠️ Error fetch blokir: {e}")

def is_blocked(nim: str) -> bool:
    global blocked_cache_time
    if time.time() - blocked_cache_time > BLOCKED_CACHE_TTL:
        refresh_blocked_nims()
    return nim in blocked_nims

def get_liveness_all(lms, w, h):
    def get_pt(idx): return np.array([lms[idx].x * w, lms[idx].y * h])
    def ear(pts):
        p = [get_pt(pt) for pt in pts]
        return (np.linalg.norm(p[1]-p[5]) + np.linalg.norm(p[2]-p[4])) / (2.0 * np.linalg.norm(p[0]-p[3]) + 1e-6)
    avg_ear = (ear([33,160,158,133,153,144]) + ear([362,385,387,263,373,380])) / 2.0
    yaw = abs(lms[1].x - lms[33].x) / (abs(lms[263].x - lms[33].x) + 1e-6)
    mouth_w = abs(lms[61].x - lms[291].x)
    face_w = abs(lms[33].x - lms[263].x)
    smile_ratio = mouth_w / (face_w + 1e-6)
    return avg_ear, yaw, smile_ratio

def apply_clahe(img):
    if img is None or img.size == 0: return img
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def align_face(img, keypoints, x_offset, y_offset):
    if keypoints is None or len(keypoints) < 2: return img
    left_eye = (int(keypoints[0][0]-x_offset), int(keypoints[0][1]-y_offset))
    right_eye = (int(keypoints[1][0]-x_offset), int(keypoints[1][1]-y_offset))
    angle = np.degrees(np.arctan2(right_eye[1]-left_eye[1], right_eye[0]-left_eye[0]))
    eye_center = ((left_eye[0]+right_eye[0])//2, (left_eye[1]+right_eye[1])//2)
    M = cv2.getRotationMatrix2D(eye_center, angle, 1.0)
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_CUBIC)

def detect_motion(frame, prev_small_frame=None):
    h,w = frame.shape[:2]
    small = cv2.resize(frame, (w//4, h//4))
    if prev_small_frame is None: return True, 0, small
    diff = cv2.absdiff(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY),
                       cv2.cvtColor(prev_small_frame, cv2.COLOR_BGR2GRAY))
    return np.sum(diff) > 0, np.sum(diff), small

def boxes_overlap(box1, box2, iou_threshold=0.3):
    x1_1,y1_1,x2_1,y2_1 = box1
    x1_2,y1_2,x2_2,y2_2 = box2
    xi1, yi1 = max(x1_1,x1_2), max(y1_1,y1_2)
    xi2, yi2 = min(x2_1,x2_2), min(y2_1,y2_2)
    if xi2 < xi1 or yi2 < yi1: return False
    inter = (xi2-xi1)*(yi2-yi1)
    union = (x2_1-x1_1)*(y2_1-y1_1)+(x2_2-x1_2)*(y2_2-y1_2)-inter
    return inter/union > iou_threshold if union>0 else False

def track_detections(new_detections, frozen_detections, cfg, is_worker_response=True):
    if not is_worker_response:
        return frozen_detections['detections'] if frozen_detections['freeze_count'] > 0 else []
    if not new_detections:
        if frozen_detections['freeze_count'] > 0:
            frozen_detections['freeze_count'] -= 1
            return frozen_detections['detections'] if frozen_detections['freeze_count'] > 0 else []
        return []
    frozen_detections['detections'] = new_detections
    frozen_detections['freeze_count'] = cfg['detection_freeze_frames']
    return new_detections

def extract_embedding(face_aligned, rec_model):
    face = cv2.resize(face_aligned, (112,112))
    face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    blob = (face_rgb.astype(np.float32)-127.5)/127.5
    blob = np.expand_dims(blob, 0).transpose(0,3,1,2)
    req = rec_model.create_infer_request()
    req.infer(inputs={0: blob})
    emb = req.get_output_tensor(0).data.flatten()
    norm = np.linalg.norm(emb)
    return emb/norm if norm>1e-6 else emb

def recognize_hybrid(emb):
    try:
        res = requests.post(SERVER_URL, json={"embedding": emb.tolist()}, timeout=0.5)
        data = res.json()
        if data.get("status") == "success":
            d = data["data"]
            return {'nama': d['nama'], 'nim': d['nim'], 'confidence': d['similarity'], 'is_recognized': True}
        else:
            return {'nama': 'Unknown', 'nim': '-', 'confidence': 0, 'is_recognized': False}
    except Exception:
        return {'nama': 'Unknown', 'nim': '-', 'confidence': 0, 'is_recognized': False}

def load_models(cfg):
    from ultralytics import YOLO
    import openvino as ov
    core = ov.Core()
    print(f"\n⏳ Loading YOLO: {cfg['yolo_model_path']} ...")
    yolo = YOLO(cfg['yolo_model_path'], task='detect')
    print(f"⏳ Loading ArcFace: {cfg['arcface_model_path']} ...")
    ov_model = core.read_model(cfg['arcface_model_path'])
    rec = core.compile_model(ov_model, 'CPU')
    print(f"⏳ Loading MediaPipe: {cfg['mediapipe_model_path']} ...")
    opts = FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=cfg['mediapipe_model_path']),
        running_mode=RunningMode.VIDEO, num_faces=1,
        min_face_detection_confidence=0.5, min_tracking_confidence=0.5)
    liveness = FaceLandmarker.create_from_options(opts)
    print(f"⏳ Loading face database: {cfg['face_database_path']} ...")
    with open(cfg['face_database_path'], 'rb') as f:
        db = pickle.load(f)
    emb = db['embeddings']
    if not isinstance(emb, np.ndarray): emb = np.array(emb)
    if emb.ndim == 1: emb = emb.reshape(1,-1)
    print(f"✅ Models loaded. Database: {len(db['names'])} entries")
    return yolo, rec, liveness, {'embeddings': emb, 'names': db['names']}

def detect_and_recognize(frame, yolo, rec, face_db, cfg, prev_detections=None, roi_bounds=None):
    if roi_bounds:
        ry,rx,rh,rw = roi_bounds
        detect_frame = frame[ry:ry+rh, rx:rx+rw]
    else:
        detect_frame = frame
        roi_bounds = None
    results = yolo.predict(detect_frame, verbose=False, conf=cfg['yolo_threshold'], iou=0.4, device='cpu')
    detections = []
    for res in results:
        boxes = res.boxes.xyxy.cpu().numpy()
        kps = res.keypoints.xy.cpu().numpy() if res.keypoints is not None else None
        for i, bbox in enumerate(boxes):
            x1,y1,x2,y2 = map(int, bbox)
            pad_x, pad_y = int((x2-x1)*0.1), int((y2-y1)*0.1)
            x1p, y1p = max(0, x1-pad_x), max(0, y1-pad_y)
            x2p, y2p = min(detect_frame.shape[1], x2+pad_x), min(detect_frame.shape[0], y2+pad_y)
            face = detect_frame[y1p:y2p, x1p:x2p]
            if face.size == 0: continue
            face = apply_clahe(face)
            if kps is not None and len(kps)>i:
                face = align_face(face, kps[i], x1p, y1p)
            emb = extract_embedding(face, rec)
            rec_data = recognize_hybrid(emb)
            detections.append({
                'box': (x1,y1,x2,y2),
                'name': rec_data['nama'],
                'nim': rec_data['nim'],
                'conf': rec_data['confidence'],
                'is_recognized': rec_data['is_recognized'],
                'is_new': True
            })
    if roi_bounds:
        ry,rx,_,_ = roi_bounds
        for d in detections:
            x1,y1,x2,y2 = d['box']
            d['box'] = (x1+rx, y1+ry, x2+rx, y2+ry)
    return detections

# ═══════════════════════════════════════════════════════════════
# THREADS
# ═══════════════════════════════════════════════════════════════
def frame_capture_thread(cap, stream_q, detect_q, stop_event, cfg):
    prev_small = None
    frame_time = 1.0/cfg['max_fps']
    last = time.time()
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret: break
        elapsed = time.time()-last
        if elapsed < frame_time: time.sleep(frame_time-elapsed)
        last = time.time()
        _, motion, small = detect_motion(frame, prev_small)
        prev_small = small
        data = {'frame': frame, 'motion': motion}
        try: stream_q.get_nowait()
        except: pass
        stream_q.put(data)
        try: detect_q.get_nowait()
        except: pass
        detect_q.put(data)

def detection_worker_thread(yolo, rec, liveness, face_db, cfg, detect_q, stop_event):
    global last_granted_time, last_denied_time, last_granted_identity
    frozen = {'detections': [], 'freeze_count': 0}
    liveness_state = {'challenge': random.choice(CHALLENGES), 'passed': False, 'ear': 0, 'yaw': 0.5, 'smile': 0}
    frame_count = 0
    roi_expand = 0
    while not stop_event.is_set():
        try: data = detect_q.get(timeout=0.1)
        except: continue
        frame = data['frame']
        motion = data['motion']
        h,w = frame.shape[:2]
        frame_count += 1
        has_frozen = frozen['freeze_count']>0
        use_full = motion >= cfg['motion_threshold'] or has_frozen
        if use_full:
            roi_expand = cfg['roi_expansion_frames']
            roi = None
        else:
            if roi_expand>0:
                roi_expand -= 1
                roi = None
            else:
                if cfg['roi_enabled']:
                    rh, rw, ry, rx = calculate_roi_bounds(h, w, cfg['roi_center_ratio'])
                    roi = (ry, rx, rh, rw)
                else: roi = None
        if (frame_count % cfg['frame_skip_interval']) != 0 and not has_frozen:
            continue
        # Liveness
        try:
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            ts = int(time.time()*1000)
            mp_res = liveness.detect_for_video(mp_img, ts)
            if mp_res.face_landmarks:
                ear, yaw, smile = get_liveness_all(mp_res.face_landmarks[0], w, h)
                liveness_state.update({'ear': ear, 'yaw': yaw, 'smile': smile})
                if not liveness_state['passed']:
                    ch = liveness_state['challenge']
                    if ch=='BLINK' and ear<BLINK_THRESHOLD: liveness_state['passed']=True
                    elif ch=='HEAD' and (yaw<0.5-YAW_THRESH or yaw>0.5+YAW_THRESH): liveness_state['passed']=True
                    elif ch=='SMILE' and smile>SMILE_THRESHOLD: liveness_state['passed']=True
        except: pass
        # Detection
        raw = detect_and_recognize(frame, yolo, rec, face_db, cfg, frozen['detections'], roi)
        tracked = track_detections(raw, frozen, cfg, is_worker_response=True)
        if not tracked:
            liveness_state['passed'] = False
            liveness_state['challenge'] = random.choice(CHALLENGES)
        # Access control
        status = 'idle'
        user_data = None
        faces_data = []
        with antispof_lock:
            spoof_on = antispof_enabled
        liveness_ok = liveness_state['passed'] if spoof_on else True
        for det in tracked:
            x1,y1,x2,y2 = det['box']
            faces_data.append({
                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                'confidence': det['conf'],
                'name': det['name'],
                'nim': det['nim'],
                'is_recognized': det['is_recognized']
            })
            if user_data is None and liveness_ok:
                status = 'scanning'
                now = time.time()
                if det['is_recognized']:
                    nim = det['nim']
                    if is_blocked(nim):
                        if now-last_denied_time > 10:
                            status = 'denied'
                            last_denied_time = now
                    elif now-last_granted_time > 10 or last_granted_identity != nim:
                        status = 'granted'
                        user_data = {'nama': det['name'], 'nim': nim, 'confidence': det['conf']}
                        last_granted_time = now
                        last_granted_identity = nim
                        if spoof_on:
                            liveness_state['passed'] = False
                            liveness_state['challenge'] = random.choice(CHALLENGES)
                else:
                    if now-last_denied_time > 10:
                        status = 'denied'
                        last_denied_time = now
        with _det_lock:
            _det_state['tracked'] = tracked
            _det_state['liveness'] = dict(liveness_state)
            _det_state['status'] = status
            _det_state['user_data'] = user_data
            _det_state['faces_data'] = faces_data
            _det_state['motion'] = float(motion)

def frame_stream_thread(stream_q, stop_event, cfg):
    global camera_result
    interval = 1.0 / cfg['max_fps']
    while not stop_event.is_set():
        t0 = time.time()
        try:
            data = stream_q.get_nowait()
        except queue.Empty:
            time.sleep(max(0, interval - (time.time() - t0)))
            continue

        frame = data['frame']
        with _det_lock:
            tracked = list(_det_state['tracked'])
            liveness = dict(_det_state['liveness'])
            status = _det_state['status']
            user_data = _det_state['user_data']
            faces_data = list(_det_state['faces_data'])
            motion = _det_state['motion']

        draw = frame.copy()
        h, w = frame.shape[:2]

        # ===== BANNER HITAM DI ATAS (SELALU DIGAMBAR) =====
        cv2.rectangle(draw, (0, 0), (w, 50), (30, 30, 30), -1)

        # Tampilkan pesan liveness (selalu tampil)
        if liveness.get('passed', False):
            cv2.putText(draw, "LIVENESS OK! PROCESSING...", (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            challenge = liveness.get('challenge', 'BLINK')
            if challenge == 'BLINK':
                inst = "👁️ Kedipkan Mata"
            elif challenge == 'HEAD':
                inst = "↔️ Gelengkan Kepala"
            elif challenge == 'SMILE':
                inst = "😊 Senyum"
            else:
                inst = f"CHALLENGE: {challenge}"
            cv2.putText(draw, inst, (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Gambar bounding box wajah (hanya jika ada wajah)
        for det in tracked:
            x1, y1, x2, y2 = det['box']
            if liveness.get('passed', False):
                color = (0, 255, 0) if det.get('is_recognized', False) else (0, 165, 255)
                cv2.rectangle(draw, (x1, y1), (x2, y2), color, 3)
            else:
                cv2.rectangle(draw, (x1, y1), (x2, y2), (0, 165, 255), 2)

        # Encode dan kirim
        _, buf = cv2.imencode('.jpg', draw, [cv2.IMWRITE_JPEG_QUALITY, 60])
        b64 = base64.b64encode(buf).decode()
        with camera_lock:
            camera_result = {
                'image': f'data:image/jpeg;base64,{b64}',
                'status': status,
                'face_count': len(faces_data),
                'faces': faces_data,
                'user': user_data,
                'motion': motion,
                'liveness': {
                    'passed': liveness['passed'],
                    'challenge': liveness['challenge'],
                    'ear': round(liveness.get('ear', 0), 3),
                    'yaw': round(liveness.get('yaw', 0.5), 3),
                    'smile': round(liveness.get('smile', 0), 3)
                }
            }

        elapsed = time.time() - t0
        time.sleep(max(0, interval - elapsed))

def calculate_roi_bounds(h,w,ratio):
    rh = int(h*ratio)
    rw = int(w*ratio)
    ry = (h-rh)//2
    rx = (w-rw)//2
    return rh, rw, ry, rx

# ═══════════════════════════════════════════════════════════════
# FASTAPI APP
# ═══════════════════════════════════════════════════════════════
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_stop = threading.Event()

@app.on_event("startup")
async def startup():
    refresh_blocked_nims()
    yolo, rec, liveness, db = load_models(CONFIG)
    cap = cv2.VideoCapture(CONFIG['camera_index'], cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("❌ Kamera tidak bisa dibuka!")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG['camera_width'])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG['camera_height'])
    stream_q = queue.Queue(maxsize=1)
    detect_q = queue.Queue(maxsize=1)
    threading.Thread(target=frame_capture_thread, args=(cap, stream_q, detect_q, _stop, CONFIG), daemon=True).start()
    threading.Thread(target=frame_stream_thread, args=(stream_q, _stop, CONFIG), daemon=True).start()
    threading.Thread(target=detection_worker_thread, args=(yolo, rec, liveness, db, CONFIG, detect_q, _stop), daemon=True).start()
    print("✅ Threads started. HTTP:8000 WS:/ws/detect")

@app.on_event("shutdown")
def shutdown():
    _stop.set()

@app.get("/")
def root():
    return {"status": "Face Recognition Gate (Anti-Spoof Toggle Ready)"}

@app.websocket("/ws/detect")
async def ws_detect(websocket: WebSocket):
    await websocket.accept()
    connected_clients.add(websocket)
    print(f"🔗 Client connected ({len(connected_clients)})")
    try:
        while True:
            with camera_lock:
                res = camera_result
            if res:
                await websocket.send_text(json.dumps(res))
            try:
                msg = await asyncio.wait_for(websocket.receive_text(), timeout=0.01)
                data = json.loads(msg)
                if "antispof" in data:
                    global antispof_enabled
                    with antispof_lock:
                        antispof_enabled = data["antispof"]
                    print(f"🔄 Anti-spoof mode: {'ON' if antispof_enabled else 'OFF'}")
            except asyncio.TimeoutError:
                pass
            except json.JSONDecodeError:
                pass
            await asyncio.sleep(0.067)
    except WebSocketDisconnect:
        pass
    finally:
        connected_clients.discard(websocket)
        print(f"🔌 Client disconnected ({len(connected_clients)} left)")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)