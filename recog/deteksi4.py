# ═══════════════════════════════════════════════════════════════
# deteksifinal.py — Backend Face Recognition Gate (Optimized)
# Integrasi: OpenVINO (YOLO+ArcFace), Motion Detection, 
# Server Alif (Fallback Lokal), Supabase Blocklist, FastAPI WebSocket
# ═══════════════════════════════════════════════════════════════

import os
import sys
import cv2
import numpy as np
import pickle
import time
import base64
import json
import asyncio
import threading
import queue
import requests
import warnings
from pathlib import Path
from collections import deque
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────
# PYTORCH CPU OPTIMIZATION
# ─────────────────────────────────────────────────────────────
import torch
num_cpus = min(os.cpu_count() or 4, 4)
num_pytorch_cores = max(1, num_cpus // 2)
torch.set_num_threads(num_pytorch_cores)
torch.set_num_interop_threads(1)
print(f"🔧 PyTorch CPU Limit: {num_pytorch_cores} cores (out of {num_cpus})")

# ─────────────────────────────────────────────────────────────
# KONFIGURASI 
# ─────────────────────────────────────────────────────────────
CONFIG = {
    "yolo_model_path":        "models/yolov11n-face_openvino_model", 
    "arcface_model_path":     "models/buffalo_sc_rec.xml",           
    "face_database_path":     "models/face_database.pkl",
    "yolo_threshold":         0.4,
    "recognition_threshold":  0.45,
    "camera_index":           1,      
    "camera_width":           640,
    "camera_height":          480,
    "max_fps":                15,     
    "motion_threshold":       2000,    
    "detection_freeze_frames": 50,    
}

SERVER_ALIF_URL       = "http://100.107.234.128:8000/identify-face"
SUPABASE_URL          = "https://kcskzlwxnvmvofyscqsr.supabase.co"
SUPABASE_ANON_KEY     = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imtjc2t6bHd4bnZtdm9meXNjcXNyIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzcxNzk0ODIsImV4cCI6MjA5Mjc1NTQ4Mn0.mpmImOcJWkBFwTynGUos7LmUnSYLqGe0h_KRbYQ3tuw"
SUPABASE_HEADERS      = {
    "apikey": SUPABASE_ANON_KEY,
    "Authorization": f"Bearer {SUPABASE_ANON_KEY}",
    "Content-Type": "application/json",
}

# ─────────────────────────────────────────────────────────────
# STATE & CACHE GLOBAL
# ─────────────────────────────────────────────────────────────
blocked_nims          = set()
blocked_cache_time    = 0
BLOCKED_CACHE_TTL     = 30

camera_result         = None
camera_lock           = threading.Lock()
connected_clients     = set()

last_denied_time      = 0
last_granted_time     = 0
last_granted_identity = None

# ─────────────────────────────────────────────────────────────
# SUPABASE BLOKIR LOGIC
# ─────────────────────────────────────────────────────────────
def refresh_blocked_nims():
    global blocked_nims, blocked_cache_time
    try:
        res = requests.get(
            f"{SUPABASE_URL}/rest/v1/blokir_user?select=nim",
            headers=SUPABASE_HEADERS,
            timeout=5.0
        )
        if res.status_code == 200:
            data = res.json()
            blocked_nims = {row['nim'] for row in data if row.get('nim')}
            blocked_cache_time = time.time()
            print(f"🔒 Cache Blokir Diperbarui: {blocked_nims or 'kosong'}")
        else:
            print(f"⚠️ Gagal ambil blokir: {res.status_code}")
    except Exception as e:
        print(f"⚠️ Error fetch blokir: {e}")

def is_blocked(nim: str) -> bool:
    global blocked_cache_time
    if time.time() - blocked_cache_time > BLOCKED_CACHE_TTL:
        refresh_blocked_nims()
    return nim in blocked_nims

# ─────────────────────────────────────────────────────────────
# PRE-PROCESSING (CLAHE & ALIGNMENT)
# ─────────────────────────────────────────────────────────────
def apply_clahe(img):
    if img is None or img.size == 0: return img
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def align_face(img, keypoints, x_offset, y_offset):
    if keypoints is None or len(keypoints) < 2: return img
    left_eye = (int(keypoints[0][0] - x_offset), int(keypoints[0][1] - y_offset))
    right_eye = (int(keypoints[1][0] - x_offset), int(keypoints[1][1] - y_offset))
    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dy, dx))
    eye_center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)
    M = cv2.getRotationMatrix2D(eye_center, angle, scale=1.0)
    h, w = img.shape[:2]
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC)

# ─────────────────────────────────────────────────────────────
# MOTION DETECTION & TRACKING
# ─────────────────────────────────────────────────────────────
def detect_motion(frame, prev_small_frame=None):
    h, w = frame.shape[:2]
    small_frame = cv2.resize(frame, (w//4, h//4))
    if prev_small_frame is None: return True, 0, small_frame
    gray_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
    gray_prev = cv2.cvtColor(prev_small_frame, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray_frame, gray_prev)
    motion_sum = np.sum(diff)
    return motion_sum > 0, motion_sum, small_frame

def boxes_overlap(box1, box2, iou_threshold=0.3):
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    x1_i, y1_i = max(x1_1, x1_2), max(y1_1, y1_2)
    x2_i, y2_i = min(x2_1, x2_2), min(y2_1, y2_2)
    if x2_i < x1_i or y2_i < y1_i: return False
    inter_area = (x2_i - x1_i) * (y2_i - y1_i)
    union_area = (x2_1 - x1_1) * (y2_1 - y1_1) + (x2_2 - x1_2) * (y2_2 - y1_2) - inter_area
    iou = inter_area / union_area if union_area > 0 else 0
    return iou > iou_threshold

def track_detections(new_detections, frozen_detections, cfg, is_worker_response=True):
    if not is_worker_response:
        return frozen_detections['detections'] if frozen_detections['freeze_count'] > 0 else []
    
    if not new_detections:
        if frozen_detections['freeze_count'] > 0:
            frozen_detections['freeze_count'] -= 1
            return frozen_detections['detections'] if frozen_detections['freeze_count'] > 0 else []
        return []
    
    matched = any(boxes_overlap(n['box'], f['box']) for n in new_detections for f in frozen_detections['detections'])
    frozen_detections['detections'] = new_detections
    frozen_detections['freeze_count'] = cfg['detection_freeze_frames']
    return new_detections

# ─────────────────────────────────────────────────────────────
# LOAD MODELS (OPENVINO)
# ─────────────────────────────────────────────────────────────
def load_models(cfg):
    from ultralytics import YOLO
    import openvino as ov
    core = ov.Core()

    yolo_path = Path(cfg["yolo_model_path"])
    print(f"\n⏳ Loading YOLOv11 OpenVINO: {yolo_path} ...")
    yolo_model = YOLO(str(yolo_path), task="detect")

    arcface_path = Path(cfg["arcface_model_path"])
    print(f"⏳ Loading ArcFace OpenVINO: {arcface_path} ...")
    ov_model = core.read_model(str(arcface_path))
    rec_model = core.compile_model(ov_model, "CPU")

    db_path = Path(cfg["face_database_path"])
    print(f"⏳ Loading database: {db_path} ...")
    with open(db_path, 'rb') as f: face_database = pickle.load(f)
    if face_database['embeddings'].ndim == 1:
        face_database['embeddings'] = face_database['embeddings'].reshape(1, -1)
    
    print("✅ Model & Database Loaded!")
    return yolo_model, rec_model, face_database

# ─────────────────────────────────────────────────────────────
# HYBRID RECOGNITION (SERVER ALIF -> LOKAL)
# ─────────────────────────────────────────────────────────────
def extract_embedding(face_aligned, rec_model):
    face_resized = cv2.resize(face_aligned, (112, 112))
    face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
    blob = (face_rgb.astype(np.float32) - 127.5) / 127.5
    blob = np.expand_dims(blob, 0)
    blob = np.transpose(blob, (0, 3, 1, 2)) 
    infer_request = rec_model.create_infer_request()
    infer_request.infer(inputs={0: blob})
    emb = infer_request.get_output_tensor(0).data.flatten()
    norm = np.linalg.norm(emb)
    if norm > 1e-6: emb = emb / norm
    return emb

def recognize_hybrid(emb, face_database, threshold):
    # 1. Coba Server Alif
    try:
        res = requests.post(SERVER_ALIF_URL, json={"embedding": emb.tolist()}, timeout=0.5).json()
        if res.get("status") == "success":
            d = res["data"]
            similarity = round(float(d.get('similarity', 0)), 4)
            nama = d.get('nama', 'Unknown')
            nim = d.get('nim', '-')
            return {'nama': nama, 'nim': nim, 'program_studi': 'S1 Teknik Telekomunikasi', 
                    'confidence': similarity, 'is_recognized': True, 'source': 'Alif'}
    except Exception:
        pass # Lanjut ke lokal jika error/timeout

    # 2. Fallback Lokal (OpenVINO Database)
    sims = np.dot(face_database['embeddings'], emb)
    idx  = int(np.argmax(sims))
    sim  = float(sims[idx])
    
    if sim >= threshold:
        raw  = face_database['names'][idx]
        nim, nama = raw.split('_', 1) if '_' in raw else ('-', raw)
        return {'nama': nama, 'nim': nim, 'program_studi': '', 
                'confidence': round(sim, 4), 'is_recognized': True, 'source': 'Local'}
        
    return {'nama': 'Unknown', 'nim': '-', 'program_studi': '', 
            'confidence': round(sim, 4), 'is_recognized': False, 'source': 'Local'}

# ─────────────────────────────────────────────────────────────
# CORE DETECTION LOGIC
# ─────────────────────────────────────────────────────────────
def detect_and_recognize(frame, yolo_model, rec_model, face_database, cfg, prev_detections=None):
    results = yolo_model.predict(frame, verbose=False, conf=cfg["yolo_threshold"], iou=0.4, device='cpu')
    detections = []
    
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        kps_all = result.keypoints.xy.cpu().numpy() if result.keypoints is not None else None

        for i, bbox in enumerate(boxes):
            x1, y1, x2, y2 = map(int, bbox)
            new_box = (x1, y1, x2, y2)
            
            # Match IoU
            matched_prev = None
            if prev_detections:
                for p in prev_detections:
                    if boxes_overlap(new_box, p['box'], 0.3):
                        matched_prev = p
                        break
            
            is_new_face = matched_prev is None or matched_prev['name'] == 'Unknown'

            if is_new_face:
                h, w = frame.shape[:2]
                pad_x, pad_y = int((x2 - x1) * 0.1), int((y2 - y1) * 0.1)
                x1_p, y1_p = max(0, x1 - pad_x), max(0, y1 - pad_y)
                x2_p, y2_p = min(w, x2 + pad_x), min(h, y2 + pad_y)
                
                face = frame[y1_p:y2_p, x1_p:x2_p]
                if face.size == 0: continue
                
                face_clahe = apply_clahe(face)
                face_aligned = align_face(face_clahe, kps_all[i], x1_p, y1_p) if (kps_all is not None and len(kps_all)>i) else face_clahe
                
                emb = extract_embedding(face_aligned, rec_model)
                rec_data = recognize_hybrid(emb, face_database, cfg["recognition_threshold"])
                
                detections.append({
                    "box": new_box,
                    "name": rec_data['nama'],
                    "nim": rec_data['nim'],
                    "conf": rec_data['confidence'],
                    "is_recognized": rec_data['is_recognized'],
                    "is_new": True
                })
            else:
                detections.append({
                    "box": new_box,
                    "name": matched_prev['name'],
                    "nim": matched_prev['nim'],
                    "conf": matched_prev['conf'],
                    "is_recognized": matched_prev['is_recognized'],
                    "is_new": False
                })
                
    return detections

# ─────────────────────────────────────────────────────────────
# WORKER THREADS
# ─────────────────────────────────────────────────────────────
def frame_capture_thread(cap, frame_queue, stop_event, cfg):
    frame_time = 1.0 / cfg["max_fps"]
    prev_small_frame = None
    last_time = time.time()
    
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret: break
        
        elapsed = time.time() - last_time
        if elapsed < frame_time: time.sleep(frame_time - elapsed)
        last_time = time.time()
        
        is_motion, motion_val, small_frame = detect_motion(frame, prev_small_frame)
        prev_small_frame = small_frame
        
        try: frame_queue.get_nowait()
        except queue.Empty: pass
        
        frame_queue.put({'frame': frame, 'motion': motion_val})

def processing_state_thread(yolo_model, rec_model, face_database, cfg, frame_queue, stop_event):
    global camera_result, last_granted_time, last_denied_time, last_granted_identity
    frozen_detections = {'detections': [], 'freeze_count': 0}
    
    while not stop_event.is_set():
        try:
            data = frame_queue.get(timeout=0.1)
        except queue.Empty:
            continue
            
        frame = data['frame']
        motion = data['motion']
        
        has_frozen = frozen_detections['freeze_count'] > 0
        should_detect = motion >= cfg["motion_threshold"] or has_frozen
        
        raw_detections = []
        if should_detect and connected_clients:
            raw_detections = detect_and_recognize(frame, yolo_model, rec_model, face_database, cfg, frozen_detections['detections'])
            tracked_detections = track_detections(raw_detections, frozen_detections, cfg, is_worker_response=True)
        else:
            tracked_detections = track_detections([], frozen_detections, cfg, is_worker_response=False)
            time.sleep(0.05)
            
        # UI & State Logic (Main.py style)
        status = 'idle'
        user_data = None
        faces_data = []
        
        for det in tracked_detections:
            x1, y1, x2, y2 = det["box"]
            color = (0, 255, 0) if det["is_recognized"] else (0, 165, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            
            faces_data.append({
                'x1': int(x1), 'y1': int(y1), 'x2': int(x2), 'y2': int(y2),
                'confidence': det["conf"], 'name': det["name"], 'is_recognized': det["is_recognized"]
            })
            
            if user_data is None:
                status = 'scanning'
                now = time.time()
                
                if det["is_recognized"]:
                    if is_blocked(det["nim"]):
                        if (now - last_denied_time) > 10:
                            status = 'denied'
                            last_denied_time = now
                    elif (now - last_granted_time) > 10 or last_granted_identity != det["nim"]:
                        status = 'granted'
                        user_data = {'nama': det["name"], 'nim': det["nim"], 'confidence': det["conf"]}
                        last_granted_time = now
                        last_granted_identity = det["nim"]
                else:
                    if (now - last_denied_time) > 10:
                        status = 'denied'
                        last_denied_time = now

        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        
        # SESUDAH (Fix)
        with camera_lock:
            camera_result = {
                'image': f'data:image/jpeg;base64,{img_base64}',
                'status': status,
                'face_count': len(faces_data),
                'faces': faces_data,
                'user': user_data,
                'motion': float(motion) # Paksa jadi float atau int biasa
            }

# ─────────────────────────────────────────────────────────────
# FASTAPI SETUP
# ─────────────────────────────────────────────────────────────
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

stop_event = threading.Event()

@app.on_event("startup")
async def startup_event():
    refresh_blocked_nims()
    
    yolo_model, rec_model, face_database = load_models(CONFIG)
    
    cap = cv2.VideoCapture(CONFIG["camera_index"], cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("❌ Kamera tidak bisa dibuka!")
        sys.exit(1)
        
    frame_queue = queue.Queue(maxsize=1)
    
    t_cap = threading.Thread(target=frame_capture_thread, args=(cap, frame_queue, stop_event, CONFIG), daemon=True)
    t_proc = threading.Thread(target=processing_state_thread, args=(yolo_model, rec_model, face_database, CONFIG, frame_queue, stop_event), daemon=True)
    
    t_cap.start()
    t_proc.start()
    print("✅ Async Camera & Processing Threads Started!")

@app.on_event("shutdown")
def shutdown_event():
    stop_event.set()

@app.get("/")
def root(): return {"status": "Face Recognition Gate Hybrid Backend Running"}

@app.websocket("/ws/detect")
async def websocket_detect(websocket: WebSocket):
    await websocket.accept()
    connected_clients.add(websocket)
    print(f"🔗 Client konek ({len(connected_clients)} total)")
    try:
        while True:
            with camera_lock: result = camera_result
            if result: await websocket.send_text(json.dumps(result))
            await asyncio.sleep(0.067)
    except WebSocketDisconnect: pass
    finally:
        connected_clients.discard(websocket)
        print(f"🔌 Client putus ({len(connected_clients)} tersisa)")

if __name__ == "__main__":
    print("\n🌐 Hybrid Web Server Running:")
    print("   HTTP: http://localhost:8000")
    print("   WS:   ws://localhost:8000/ws/detect")
    print("="*60)
    uvicorn.run(app, host="0.0.0.0", port=8000)