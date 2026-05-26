# ═══════════════════════════════════════════════════════════════
# deteksi_final.py — Backend Face Recognition Gate
# Dengan Dynamic Toggle untuk setiap fitur
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
import tkinter as tk
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from deepface import DeepFace

# Suppress MediaPipe logs
_real_stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
import mediapipe as mp
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode
sys.stderr = _real_stderr

import torch
num_cpus = min(os.cpu_count() or 4, 4)
num_pytorch_cores = max(1, num_cpus // 2)
torch.set_num_threads(num_pytorch_cores)
torch.set_num_interop_threads(1)
print(f"🔧 PyTorch CPU Limit: {num_pytorch_cores} cores (out of {num_cpus})")

# ═══════════════════════════════════════════════════════════════
# KONFIGURASI DENGAN DYNAMIC TOGGLE
# ═══════════════════════════════════════════════════════════════
CONFIG = {
    # ========== PATH MODEL ==========
    "yolo_model_path":          "models/yolov11n-face_openvino_model",
    "arcface_model_path":       "models/buffalo_sc_rec.xml",
    "face_database_path":       "models/face_database.pkl",
    "mediapipe_model_path":     "models/face_landmarker.task",
    
    # ========== THRESHOLD ==========
    "yolo_threshold":           0.5,
    "recognition_threshold":    0.7,
    
    # ========== KAMERA ==========
    "camera_index":             0,
    "camera_width":             1280,
    "camera_height":            720,
    "max_fps":                  15,
    
    # ========== MOTION & ROI ==========
    "motion_threshold":         2000,
    "detection_freeze_frames":  30,
    "frame_skip_interval":      6,
    "roi_enabled":              True,
    "roi_center_ratio":         0.35,
    "roi_expansion_frames":     30,
    
    # ========== DEEPFACE (Anti-Spoof) ==========
    "deepface_enabled":         True,        # 🔥 ON/OFF DeepFace
    "deepface_throttle_frames": 60,          # Setiap N frame
    "deepface_spoof_threshold": 0.90,        # Score >= ini = SPOOF
    
    # ========== MEDIAPIPE LIVENESS ==========
    "liveness_enabled":         True,        # 🔥 ON/OFF Liveness Challenge
    "blink_threshold":          0.15,
    "smile_threshold":          0.60,
    "yaw_threshold":            0.15,
    "liveness_challenges":      ['BLINK', 'HEAD', 'SMILE'],
    
    # ========== PREPROCESSING (Gamma, CLAHE, Gray World) ==========
    "preprocessing_enabled":    True,        # 🔥 ON/OFF semua preprocessing
    "preprocessing_gamma":      1.2,         # Gamma correction (1.0 = mati)
    "preprocessing_clahe":      True,        # 🔥 ON/OFF CLAHE
    "preprocessing_gray_world": True,        # 🔥 ON/OFF Gray World
    
    # ========== CROP SETTINGS ==========
    "crop_padding_ratio":       -0.08,        # Padding crop wajah
    "crop_target_size":         (112, 112),
    
    # ========== ALIGNMENT (MediaPipe) ==========
    "alignment_enabled":        True,        # 🔥 ON/OFF Face Alignment (rotasi wajah)
    
    # ========== DEBUG ==========
    "debug_save_crops":         True,        # 🔥 Simpan crop ke folder hasil_deteksi/
    "debug_print_logs":         True,        # 🔥 Print log ke console
}

# ═══════════════════════════════════════════════════════════════
# STATE GLOBAL
# ═══════════════════════════════════════════════════════════════
blocked_nims       = set()
blocked_cache_time = 0
BLOCKED_CACHE_TTL  = 30
camera_result      = None
camera_lock        = threading.Lock()
connected_clients  = set()
last_denied_time   = 0
last_granted_time  = 0
last_granted_identity = None

_det_lock = threading.Lock()
_det_state = {
    'tracked': [],
    'liveness': {'passed': False, 'challenge': 'BLINK', 'ear': 0, 'yaw': 0.5, 'smile': 0},
    'status': 'idle',
    'user_data': None,
    'faces_data': [],
    'motion': 0.0,
}

_DEBUG_CROP_DIR = Path("hasil_deteksi")
_DEBUG_CROP_DIR.mkdir(parents=True, exist_ok=True)
_debug_crop_counter = 0
_debug_crop_lock = threading.Lock()
_align_queue = queue.Queue(maxsize=1)
_align_cache: dict = {}
_align_lock = threading.Lock()

# ═══════════════════════════════════════════════════════════════
# TKINTER DIALOG
# ═══════════════════════════════════════════════════════════════

def ask_run_mode() -> str:
    result = {"mode": "local"}
    root = tk.Tk()
    root.title("Face Recognition Gate — Pilih Mode")
    root.resizable(False, False)
    w, h = 420, 200
    sw = root.winfo_screenwidth()
    sh = root.winfo_screenheight()
    root.geometry(f"{w}x{h}+{(sw-w)//2}+{(sh-h)//2}")
    tk.Label(root, text="Pilih Mode Operasi", font=("Segoe UI", 13, "bold"), pady=14).pack()
    tk.Label(root, text="Mode Server  : kirim ke main.py, fallback lokal\n"
                        "Mode Lokal   : langsung dari face_database.pkl (testing)",
             font=("Segoe UI", 9), justify="left", padx=20).pack()
    btn_frame = tk.Frame(root, pady=16)
    btn_frame.pack()
    def pilih_server():
        result["mode"] = "server"
        root.destroy()
    def pilih_local():
        result["mode"] = "local"
        root.destroy()
    tk.Button(btn_frame, text="  Mode Server  ", font=("Segoe UI", 10), width=14,
              bg="#1a73e8", fg="white", relief="flat", cursor="hand2",
              command=pilih_server).pack(side="left", padx=10)
    tk.Button(btn_frame, text="  Mode Lokal  ", font=("Segoe UI", 10), width=14,
              bg="#34a853", fg="white", relief="flat", cursor="hand2",
              command=pilih_local).pack(side="left", padx=10)
    root.protocol("WM_DELETE_WINDOW", pilih_local)
    root.mainloop()
    print(f"[MODE] Dipilih: {result['mode'].upper()}")
    return result["mode"]

# ═══════════════════════════════════════════════════════════════
# FUNGSI PREPROCESSING DENGAN TOGGLE
# ═══════════════════════════════════════════════════════════════

def apply_gray_world(img: np.ndarray) -> np.ndarray:
    if not CONFIG.get("preprocessing_gray_world", True):
        return img
    img_float = img.astype(np.float32)
    b, g, r = cv2.split(img_float)
    b_mean, g_mean, r_mean = np.mean(b), np.mean(g), np.mean(r)
    gray_mean = (b_mean + g_mean + r_mean) / 3.0
    b = np.clip(b * (gray_mean / (b_mean + 1e-6)), 0, 255)
    g = np.clip(g * (gray_mean / (g_mean + 1e-6)), 0, 255)
    r = np.clip(r * (gray_mean / (r_mean + 1e-6)), 0, 255)
    return cv2.merge([b, g, r]).astype(np.uint8)

def apply_gamma_correction(img: np.ndarray, gamma: float = None) -> np.ndarray:
    if gamma is None:
        gamma = CONFIG.get("preprocessing_gamma", 1.2)
    if gamma == 1.0 or not CONFIG.get("preprocessing_enabled", True):
        return img
    if img is None or img.size == 0:
        return img
    inv_gamma = 1.0 / gamma
    lut = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(img, lut)

def apply_clahe(img: np.ndarray) -> np.ndarray:
    if not CONFIG.get("preprocessing_clahe", True):
        return img
    if img is None or img.size == 0:
        return img
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def apply_preprocessing(img: np.ndarray) -> np.ndarray:
    """Pipeline preprocessing dengan toggle dinamis"""
    if not CONFIG.get("preprocessing_enabled", True):
        return img
    
    img_gw = apply_gray_world(img)
    img_gamma = apply_gamma_correction(img_gw)
    img_clahe = apply_clahe(img_gamma)
    return img_clahe

# ═══════════════════════════════════════════════════════════════
# FUNGSI CROP IDENTIK
# ═══════════════════════════════════════════════════════════════

def square_crop_face_identical(img: np.ndarray, box: list, padding_ratio=None, target_size=None):
    if padding_ratio is None:
        padding_ratio = CONFIG.get("crop_padding_ratio", 0.15)
    if target_size is None:
        target_size = CONFIG.get("crop_target_size", (112, 112))
    
    x1, y1, x2, y2 = box
    h_orig, w_orig = img.shape[:2]
    w_box, h_box = x2 - x1, y2 - y1
    
    side = max(w_box, h_box)
    pad = int(padding_ratio * side)
    cx = x1 + w_box // 2
    cy = y1 + h_box // 2
    half = (side // 2) + pad
    
    x1_p = max(0, cx - half)
    y1_p = max(0, cy - half)
    x2_p = min(w_orig, cx + half)
    y2_p = min(h_orig, cy + half)
    
    face = img[y1_p:y2_p, x1_p:x2_p]
    if face.size == 0:
        return None
    
    if face.shape[:2] != target_size:
        face = cv2.resize(face, target_size, interpolation=cv2.INTER_AREA)
    
    return face

# ═══════════════════════════════════════════════════════════════
# FUNGSI LAINNYA
# ═══════════════════════════════════════════════════════════════

def refresh_blocked_nims():
    global blocked_nims, blocked_cache_time
    try:
        res = requests.get(f"{DB_SERVER_URL}/blokir_user", timeout=5.0)
        if res.status_code == 200:
            data = res.json()
            blocked_nims = {row['nim'] for row in data if row.get('nim')}
            blocked_cache_time = time.time()
            if CONFIG.get("debug_print_logs", True):
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
    avg_ear = (ear([33, 160, 158, 133, 153, 144]) + ear([362, 385, 387, 263, 373, 380])) / 2.0
    yaw = abs(lms[1].x - lms[33].x) / (abs(lms[263].x - lms[33].x) + 1e-6)
    mouth_w = abs(lms[61].x - lms[291].x)
    face_w = abs(lms[33].x - lms[263].x)
    smile_ratio = mouth_w / (face_w + 1e-6)
    return avg_ear, yaw, smile_ratio

def check_deepface_antispoof(face_img):
    if not CONFIG.get("deepface_enabled", True):
        return True, 1.0  # Skip DeepFace, anggap real
    
    try:
        face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        result = DeepFace.extract_faces(img_path=face_rgb, anti_spoofing=True, enforce_detection=False)
        if result and len(result) > 0:
            spoof_score = result[0].get('antispoof_score', 0.5)
            threshold = CONFIG.get("deepface_spoof_threshold", 0.85)
            is_real = spoof_score < threshold
            if CONFIG.get("debug_print_logs", True):
                print(f"[DEEPFACE] Score: {spoof_score:.3f} | Threshold: {threshold} | {'REAL' if is_real else 'SPOOF'}")
            return is_real, spoof_score
        return False, 0.5
    except Exception as e:
        print(f"⚠️ DeepFace error: {e}")
        return False, 0.5

def detect_motion(frame, prev_small_frame=None):
    h, w = frame.shape[:2]
    small = cv2.resize(frame, (w // 4, h // 4))
    if prev_small_frame is None:
        return True, 0, small
    diff = cv2.absdiff(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY),
                       cv2.cvtColor(prev_small_frame, cv2.COLOR_BGR2GRAY))
    return np.sum(diff) > 0, np.sum(diff), small

def extract_embedding(face_aligned, rec_model):
    target_size = CONFIG.get("crop_target_size", (112, 112))
    face_resized = cv2.resize(face_aligned, target_size)
    face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
    blob = face_rgb.astype(np.float32) / 255.0
    blob = np.transpose(blob, (2, 0, 1))
    blob = np.expand_dims(blob, axis=0)
    infer_req = rec_model.create_infer_request()
    infer_req.infer(inputs={0: blob})
    emb = infer_req.get_output_tensor(0).data.flatten()
    norm = np.linalg.norm(emb)
    return emb / norm if norm > 1e-6 else emb

def recognize_hybrid(emb, face_database, threshold):
    RUN_MODE = CONFIG.get("run_mode", "local")
    if RUN_MODE == "local":
        return _recognize_local(emb, face_database, threshold, source_label="Local")
    embedding_list = emb.tolist()
    try:
        res = requests.post(SERVER_URL, json={"embedding": embedding_list}, timeout=0.5)
        if res.status_code == 200:
            res_data = res.json()
            if res_data.get("status") == "success":
                d = res_data["data"]
                sim = round(float(d.get('similarity', 0)), 4)
                return {'nama': d.get('nama', 'Unknown'), 'nim': d.get('nim', '-'),
                        'confidence': sim, 'is_recognized': True, 'source': 'Server'}
            return {'nama': 'Unknown', 'nim': '-', 'confidence': 0.0,
                    'is_recognized': False, 'source': 'Server'}
    except Exception as e:
        print(f"[SERVER] Gagal koneksi, fallback lokal: {e}")
    return _recognize_local(emb, face_database, threshold, source_label="Local(fallback)")

def _recognize_local(emb, face_database, threshold, source_label="Local"):
    sims = np.dot(face_database['embeddings'], emb)
    idx = int(np.argmax(sims))
    sim = float(sims[idx])
    if sim >= threshold:
        raw = face_database['names'][idx]
        parts = raw.split('_', 1)
        nim, nama = (parts[0], parts[1]) if len(parts) == 2 else ('-', raw)
        nama = nama.replace('_KACAMATA', '').replace('_kacamata', '')
        return {'nama': nama, 'nim': nim, 'confidence': round(sim, 4),
                'is_recognized': True, 'source': source_label}
    return {'nama': 'Unknown', 'nim': '-', 'confidence': round(sim, 4),
            'is_recognized': False, 'source': source_label}

def calculate_roi_bounds(frame_h, frame_w, roi_ratio=0.35):
    roi_h = int(frame_h * roi_ratio)
    roi_w = int(frame_w * roi_ratio)
    roi_y = (frame_h - roi_h) // 2
    roi_x = (frame_w - roi_w) // 2
    return roi_h, roi_w, roi_y, roi_x

def boxes_overlap(box1, box2, iou_threshold=0.3):
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    x1_i = max(x1_1, x1_2); y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2); y2_i = min(y2_1, y2_2)
    if x2_i < x1_i or y2_i < y1_i:
        return False
    inter_area = (x2_i - x1_i) * (y2_i - y1_i)
    union_area = (x2_1-x1_1)*(y2_1-y1_1) + (x2_2-x1_2)*(y2_2-y1_2) - inter_area
    return (inter_area / union_area if union_area > 0 else 0) > iou_threshold

def map_detections_from_roi(detections, roi_bounds):
    roi_y, roi_x, roi_h, roi_w = roi_bounds
    for det in detections:
        x1, y1, x2, y2 = det['box']
        det['box'] = (x1 + roi_x, y1 + roi_y, x2 + roi_x, y2 + roi_y)
    return detections

def match_face_detections(new_boxes, prev_detections, iou_threshold=0.3):
    matches = {i: None for i in range(len(new_boxes))}
    if not prev_detections:
        return matches
    for i, new_box in enumerate(new_boxes):
        best_match, best_iou = None, 0
        for prev_det in prev_detections:
            if boxes_overlap(new_box, prev_det['box'], iou_threshold):
                x1_1, y1_1, x2_1, y2_1 = new_box
                x1_2, y1_2, x2_2, y2_2 = prev_det['box']
                ix1, iy1 = max(x1_1, x1_2), max(y1_1, y1_2)
                ix2, iy2 = min(x2_1, x2_2), min(y2_1, y2_2)
                if ix2 > ix1 and iy2 > iy1:
                    inter = (ix2-ix1)*(iy2-iy1)
                    union = (x2_1-x1_1)*(y2_1-y1_1) + (x2_2-x1_2)*(y2_2-y1_2) - inter
                    iou = inter / union if union > 0 else 0
                    if iou > best_iou:
                        best_iou, best_match = iou, prev_det
        matches[i] = best_match
    return matches

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

# ═══════════════════════════════════════════════════════════════
# ALIGNMENT WORKER (dengan toggle)
# ═══════════════════════════════════════════════════════════════

def alignment_worker_thread(stop_event):
    if not CONFIG.get("alignment_enabled", True):
        return  # Skip alignment
    
    sys.stderr = open(os.devnull, 'w')
    mp_path = CONFIG.get("mediapipe_model_path", "models/face_landmarker.task")
    options = FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=mp_path),
        running_mode=RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
    )
    align_landmarker = FaceLandmarker.create_from_options(options)
    sys.stderr = _real_stderr
    print("✅ Alignment MediaPipe loaded (IMAGE mode)")

    while not stop_event.is_set():
        try:
            data = _align_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        frame = data['frame']
        box = data['box']
        box_key = box
        x1, y1, x2, y2 = box

        try:
            h, w = frame.shape[:2]
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, 
                              data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            result = align_landmarker.detect(mp_img)

            if result.face_landmarks:
                lms = result.face_landmarks[0]
                left_eye = np.array([lms[33].x * w, lms[33].y * h])
                right_eye = np.array([lms[263].x * w, lms[263].y * h])

                dY = float(right_eye[1] - left_eye[1])
                dX = float(right_eye[0] - left_eye[0])
                angle = np.degrees(np.arctan2(dY, dX))

                eye_center = (int((left_eye[0] + right_eye[0]) / 2), 
                              int((left_eye[1] + right_eye[1]) / 2))
                
                M = cv2.getRotationMatrix2D(eye_center, angle, scale=1.0)
                img_rotated = cv2.warpAffine(frame, M, (w, h), 
                                             flags=cv2.INTER_CUBIC, 
                                             borderMode=cv2.BORDER_REPLICATE)

                corners = np.array([
                    [x1, y1], [x2, y1], [x1, y2], [x2, y2]
                ], dtype=np.float32)
                ones = np.ones((4, 1), dtype=np.float32)
                corners_h = np.hstack([corners, ones])
                rotated_corners = (M @ corners_h.T).T
                rx1 = int(rotated_corners[:, 0].min())
                ry1 = int(rotated_corners[:, 1].min())
                rx2 = int(rotated_corners[:, 0].max())
                ry2 = int(rotated_corners[:, 1].max())
                
                face_crop = square_crop_face_identical(
                    img_rotated, 
                    [rx1, ry1, rx2, ry2], 
                    padding_ratio=None,
                    target_size=None
                )
                
                if face_crop is not None:
                    face_enhanced = apply_preprocessing(face_crop)
                    
                    if CONFIG.get("debug_save_crops", True):
                        global _debug_crop_counter
                        with _debug_crop_lock:
                            _debug_crop_counter += 1
                            debug_filename = _DEBUG_CROP_DIR / f"crop_{_debug_crop_counter:05d}.jpg"
                        cv2.imwrite(str(debug_filename), face_enhanced)
                    
                    with _align_lock:
                        _align_cache[box_key] = face_enhanced
                else:
                    fallback_face = square_crop_face_identical(
                        frame, box, padding_ratio=None, target_size=None
                    )
                    if fallback_face is not None:
                        fallback_enhanced = apply_preprocessing(fallback_face)
                        with _align_lock:
                            _align_cache[box_key] = fallback_enhanced
            else:
                fallback_face = square_crop_face_identical(
                    frame, box, padding_ratio=None, target_size=None
                )
                if fallback_face is not None:
                    fallback_enhanced = apply_preprocessing(fallback_face)
                    with _align_lock:
                        _align_cache[box_key] = fallback_enhanced
                        
        except Exception as e:
            print(f"⚠️ Alignment worker error: {e}")
            continue

def get_aligned_face(frame, box):
    box_key = box
    with _align_lock:
        cached = _align_cache.get(box_key)
    if cached is not None:
        return cached

    if CONFIG.get("alignment_enabled", True):
        try:
            _align_queue.get_nowait()
        except queue.Empty:
            pass
        try:
            _align_queue.put_nowait({'frame': frame.copy(), 'box': box_key})
        except queue.Full:
            pass

    fallback = square_crop_face_identical(frame, box, padding_ratio=None, target_size=None)
    if fallback is not None:
        return apply_preprocessing(fallback)
    return None

# ═══════════════════════════════════════════════════════════════
# DETEKSI & REKOGNISI
# ═══════════════════════════════════════════════════════════════

def detect_and_recognize(frame, yolo_model, rec_model, face_database, cfg,
                          prev_detections=None, roi_bounds=None):
    yolo_threshold = cfg["yolo_threshold"]
    recognition_threshold = cfg["recognition_threshold"]

    if roi_bounds is not None:
        roi_y, roi_x, roi_h, roi_w = roi_bounds
        detect_frame = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
    else:
        detect_frame = frame

    results = yolo_model.predict(detect_frame, verbose=False, conf=yolo_threshold, iou=0.4, device='cpu')
    new_boxes = []
    detections = []

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        for i, bbox in enumerate(boxes):
            x1, y1, x2, y2 = map(int, bbox)
            new_boxes.append((x1, y1, x2, y2))
        matches = match_face_detections(new_boxes, prev_detections or [], iou_threshold=0.3)
        for i, bbox in enumerate(boxes):
            x1, y1, x2, y2 = map(int, bbox)
            matched_prev = matches[i]
            is_new_face = (matched_prev is None) or (matched_prev['name'] in ("Unknown", "Error"))
            box_full = (x1, y1, x2, y2)
            face_ready = get_aligned_face(frame, box_full)
            if face_ready is None:
                rec_data = {'nama': 'Unknown', 'nim': '-', 'confidence': 0.0,
                           'is_recognized': False, 'source': 'Local'}
                unknown_skip_count = 5
            else:
                emb = extract_embedding(face_ready, rec_model)
                rec_data = recognize_hybrid(emb, face_database, recognition_threshold)
                unknown_skip_count = 5 if not rec_data['is_recognized'] else 0
                if CONFIG.get("debug_print_logs", True):
                    prev_name = matched_prev.get('name', '') if matched_prev else ''
                    if rec_data['nama'] != prev_name:
                        status_icon = "✅" if rec_data['is_recognized'] else "❌"
                        print(f"{status_icon} [{rec_data['source']}] Nama: {rec_data['nama']:<30} NIM: {rec_data['nim']:<15} Confidence: {rec_data['confidence']:.4f}")
            detections.append({
                'box': (x1, y1, x2, y2),
                'name': rec_data['nama'],
                'nim': rec_data['nim'],
                'conf': rec_data['confidence'],
                'is_recognized': rec_data['is_recognized'],
                'source': rec_data.get('source', 'Local'),
                'is_new': is_new_face,
                'unknown_skip_count': unknown_skip_count,
            })
    if roi_bounds is not None and detections:
        detections = map_detections_from_roi(detections, roi_bounds)
    return detections

# ═══════════════════════════════════════════════════════════════
# LOAD MODEL
# ═══════════════════════════════════════════════════════════════

def load_models(cfg):
    from ultralytics import YOLO
    import openvino as ov
    core = ov.Core()
    print(f"\n{'='*60}")
    print(f"  🖥️  Device : CPU (OpenVINO)")
    print(f"{'='*60}")
    yolo_path = Path(cfg["yolo_model_path"])
    print(f"\n⏳ Loading YOLOv11 OpenVINO: {yolo_path} ...")
    yolo_model = YOLO(str(yolo_path), task="detect")
    print(f"✅ YOLOv11 loaded!")
    arcface_path = Path(cfg["arcface_model_path"])
    if not arcface_path.exists():
        print(f"\n❌ ArcFace XML tidak ditemukan: {arcface_path}"); sys.exit(1)
    print(f"⏳ Loading ArcFace OpenVINO: {arcface_path} ...")
    ov_model = core.read_model(str(arcface_path))
    rec_model = core.compile_model(ov_model, "CPU")
    print(f"✅ ArcFace loaded!")
    
    # Load MediaPipe Liveness (jika enabled)
    liveness_model = None
    if cfg.get("liveness_enabled", True):
        mp_path = cfg.get("mediapipe_model_path", "models/face_landmarker.task")
        print(f"⏳ Loading MediaPipe Liveness: {mp_path} ...")
        sys.stderr = open(os.devnull, 'w')
        options = FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=mp_path),
            running_mode=RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        liveness_model = FaceLandmarker.create_from_options(options)
        sys.stderr = _real_stderr
        print(f"✅ MediaPipe Liveness loaded! (VIDEO mode)")
    else:
        print(f"⚠️ MediaPipe Liveness DISABLED")
    
    db_path = Path(cfg["face_database_path"])
    if not db_path.exists():
        print(f"\n❌ Face database tidak ditemukan: {db_path}"); sys.exit(1)
    print(f"⏳ Loading face database: {db_path} ...")
    with open(db_path, 'rb') as f:
        face_database = pickle.load(f)
    embs = face_database['embeddings']
    if not isinstance(embs, np.ndarray):
        face_database['embeddings'] = np.array(embs)
    if face_database['embeddings'].ndim == 1:
        face_database['embeddings'] = face_database['embeddings'].reshape(1, -1)
    print(f"✅ Face database loaded! Embeddings: {len(face_database['names'])}")
    print(f"\n{'='*60}\n")
    return yolo_model, rec_model, liveness_model, face_database

# ═══════════════════════════════════════════════════════════════
# THREADS
# ═══════════════════════════════════════════════════════════════

class FrameCaptureWorker:
    def __init__(self, cap, stream_queue, detect_queue, stop_event, max_fps, cfg):
        self.cap = cap
        self.stream_queue = stream_queue
        self.detect_queue = detect_queue
        self.stop_event = stop_event
        self.frame_time = 1.0 / max_fps
        self.cfg = cfg
        self.prev_small_frame = None
    def run(self):
        last_time = time.time()
        while not self.stop_event.is_set():
            ret, frame = self.cap.read()
            if not ret:
                break
            elapsed = time.time() - last_time
            if elapsed < self.frame_time:
                time.sleep(self.frame_time - elapsed)
            last_time = time.time()
            _, motion_val, small_frame = detect_motion(frame, self.prev_small_frame)
            self.prev_small_frame = small_frame
            data = {'frame': frame, 'motion': motion_val, 'timestamp': time.time()}
            try: self.stream_queue.get_nowait()
            except queue.Empty: pass
            self.stream_queue.put(data)
            try: self.detect_queue.get_nowait()
            except queue.Empty: pass
            self.detect_queue.put(data)

def frame_capture_thread(cap, stream_queue, detect_queue, stop_event, max_fps, cfg):
    worker = FrameCaptureWorker(cap, stream_queue, detect_queue, stop_event, max_fps, cfg)
    worker.run()

def detection_worker_thread(yolo_model, rec_model, liveness_model, face_database, cfg,
                             detect_queue, stop_event):
    global last_granted_time, last_denied_time, last_granted_identity
    frozen_detections = {'detections': [], 'freeze_count': 0}
    
    # Liveness state
    liveness_state = {
        'challenge': random.choice(CONFIG.get("liveness_challenges", ['BLINK', 'HEAD', 'SMILE'])), 
        'passed': False, 
        'ear': 0, 
        'yaw': 0.5, 
        'smile': 0
    }
    
    frame_skip_interval = cfg["frame_skip_interval"]
    roi_enabled = cfg.get("roi_enabled", True)
    roi_expansion_count = 0
    frame_count = 0
    deepface_throttle = cfg.get("deepface_throttle_frames", 30)
    deepface_frame_count = 0
    deepface_cache = (True, 1.0)
    
    # Thresholds
    blink_threshold = cfg.get("blink_threshold", 0.15)
    smile_threshold = cfg.get("smile_threshold", 0.60)
    yaw_thresh = cfg.get("yaw_threshold", 0.15)
    liveness_enabled = cfg.get("liveness_enabled", True)
    
    while not stop_event.is_set():
        try:
            data = detect_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        frame = data['frame']
        motion = data['motion']
        h, w = frame.shape[:2]
        frame_count += 1
        has_frozen = frozen_detections['freeze_count'] > 0
        use_full = motion >= cfg["motion_threshold"] or has_frozen
        if use_full:
            roi_expansion_count = cfg["roi_expansion_frames"]
            roi_bounds = None
        else:
            if roi_expansion_count > 0:
                roi_expansion_count -= 1
                roi_bounds = None
            else:
                roi_bounds = None
                if roi_enabled:
                    roi_h, roi_w, roi_y, roi_x = calculate_roi_bounds(h, w, cfg["roi_center_ratio"])
                    roi_bounds = (roi_y, roi_x, roi_h, roi_w)
        if (frame_count % frame_skip_interval) != 0 and not has_frozen:
            continue
        
        # ── LIVENESS CHECK ────────────────────────────────────────
        if liveness_enabled and liveness_model is not None:
            try:
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                timestamp = int(time.time() * 1000)
                mp_res = liveness_model.detect_for_video(mp_img, timestamp)
                if mp_res.face_landmarks:
                    ear, yaw, smile = get_liveness_all(mp_res.face_landmarks[0], w, h)
                    liveness_state.update({'ear': ear, 'yaw': yaw, 'smile': smile})
                    if not liveness_state['passed']:
                        ch = liveness_state['challenge']
                        if ch == 'BLINK' and ear < blink_threshold:
                            liveness_state['passed'] = True
                        elif ch == 'HEAD' and (yaw < (0.5 - yaw_thresh) or yaw > (0.5 + yaw_thresh)):
                            liveness_state['passed'] = True
                        elif ch == 'SMILE' and smile > smile_threshold:
                            liveness_state['passed'] = True
            except Exception:
                pass
        else:
            # If liveness disabled, always passed
            liveness_state['passed'] = True
        
        # ── YOLO + ARCFACE ────────────────────────────────────────
        raw_detections = detect_and_recognize(
            frame, yolo_model, rec_model, face_database, cfg,
            prev_detections=frozen_detections.get('detections', []),
            roi_bounds=roi_bounds
        )
        tracked = track_detections(raw_detections, frozen_detections, cfg, is_worker_response=True)
        
        if not tracked:
            liveness_state['passed'] = False
            liveness_state['challenge'] = random.choice(CONFIG.get("liveness_challenges", ['BLINK', 'HEAD', 'SMILE']))
            with _align_lock:
                _align_cache.clear()
        
        # ── DEEPFACE ANTI-SPOOF ───────────────────────────────────
        status = 'idle'
        user_data = None
        faces_data = []
        
        if liveness_state['passed'] and tracked:
            deepface_frame_count += 1
            should_run_deepface = (deepface_frame_count % deepface_throttle) == 0
            
            for det in tracked:
                x1, y1, x2, y2 = det["box"]
                pad_w = int((x2 - x1) * 0.1)
                pad_h = int((y2 - y1) * 0.1)
                x1_c = max(0, x1 - pad_w); y1_c = max(0, y1 - pad_h)
                x2_c = min(w, x2 + pad_w); y2_c = min(h, y2 + pad_h)
                face_crop = frame[y1_c:y2_c, x1_c:x2_c]
                if face_crop.size > 0:
                    if should_run_deepface:
                        if CONFIG.get("debug_print_logs", True):
                            print("🔄 DeepFace Anti-Spoofing check...")
                        is_real, spoof_score = check_deepface_antispoof(face_crop)
                        deepface_cache = (is_real, spoof_score)
                    else:
                        is_real, spoof_score = deepface_cache
                    if not is_real:
                        if CONFIG.get("debug_print_logs", True):
                            print(f"❌ DeepFace: Spoofing/Foto Terdeteksi! Score: {spoof_score}")
                        liveness_state['passed'] = False
                        liveness_state['challenge'] = random.choice(CONFIG.get("liveness_challenges", ['BLINK', 'HEAD', 'SMILE']))
                        tracked = []
                    else:
                        if should_run_deepface and CONFIG.get("debug_print_logs", True):
                            print(f"✅ DeepFace: Wajah Asli Terverifikasi! Score: {spoof_score}")
        
        # ── ACCESS CONTROL ────────────────────────────────────────
        for det in tracked:
            x1, y1, x2, y2 = det["box"]
            faces_data.append({
                'x1': int(x1), 'y1': int(y1), 'x2': int(x2), 'y2': int(y2),
                'confidence': det["conf"],
                'name': det["name"],
                'nim': det.get("nim", "-"),
                'is_recognized': det["is_recognized"],
            })
            if user_data is None and liveness_state['passed']:
                status = 'scanning'
                now = time.time()
                if det["is_recognized"]:
                    nim = det.get("nim", "-")
                    if is_blocked(nim):
                        if (now - last_denied_time) > 10:
                            status = 'denied'
                            last_denied_time = now
                    elif (now - last_granted_time) > 10 or last_granted_identity != nim:
                        status = 'granted'
                        user_data = {'nama': det["name"], 'nim': nim, 'confidence': det["conf"]}
                        last_granted_time = now
                        last_granted_identity = nim
                        liveness_state['passed'] = False
                        liveness_state['challenge'] = random.choice(CONFIG.get("liveness_challenges", ['BLINK', 'HEAD', 'SMILE']))
                        deepface_cache = (True, 1.0)
                else:
                    if (now - last_denied_time) > 10:
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
        cv2.rectangle(draw, (0, 0), (w, 50), (30, 30, 30), -1)
        if liveness.get('passed', False):
            cv2.putText(draw, "LIVENESS OK! PROCESSING...", (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            challenge = liveness.get('challenge', 'BLINK')
            if challenge == 'BLINK':
                inst = "Kedipkan Mata [BLINK Challenge]"
            elif challenge == 'HEAD':
                inst = "Gelengkan Kepala [HEAD TURN Challenge]"
            elif challenge == 'SMILE':
                inst = "Tersenyum [SMILE Challenge]"
            else:
                inst = f"CHALLENGE: {challenge}"
            cv2.putText(draw, inst, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        for det in tracked:
            x1, y1, x2, y2 = det['box']
            if liveness.get('passed', False):
                color = (0, 255, 0) if det.get('is_recognized', False) else (0, 165, 255)
                cv2.rectangle(draw, (x1, y1), (x2, y2), color, 3)
            else:
                cv2.rectangle(draw, (x1, y1), (x2, y2), (0, 165, 255), 2)
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

# ═══════════════════════════════════════════════════════════════
# FASTAPI APP
# ═══════════════════════════════════════════════════════════════
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
_stop = threading.Event()

# Server backend URL
SERVER_URL = "http://localhost:8001/identify-face"
DB_SERVER_URL = "http://localhost:8001"

@app.on_event("startup")
async def startup():
    global RUN_MODE
    RUN_MODE = CONFIG.get("run_mode", "local")
    print(f"[STARTUP] Mode: {RUN_MODE.upper()}")
    
    # Print config summary
    print("\n" + "=" * 60)
    print("📋 KONFIGURASI YANG SEDANG BERJALAN:")
    print("=" * 60)
    print(f"   🧠 DeepFace      : {'ON' if CONFIG.get('deepface_enabled', True) else 'OFF'}")
    print(f"   👁️ Liveness      : {'ON' if CONFIG.get('liveness_enabled', True) else 'OFF'}")
    print(f"   🔄 Alignment     : {'ON' if CONFIG.get('alignment_enabled', True) else 'OFF'}")
    print(f"   🎨 Preprocessing : {'ON' if CONFIG.get('preprocessing_enabled', True) else 'OFF'}")
    print(f"      - Gray World  : {'ON' if CONFIG.get('preprocessing_gray_world', True) else 'OFF'}")
    print(f"      - Gamma       : {CONFIG.get('preprocessing_gamma', 1.2)}")
    print(f"      - CLAHE       : {'ON' if CONFIG.get('preprocessing_clahe', True) else 'OFF'}")
    print(f"   📦 Crop Padding  : {CONFIG.get('crop_padding_ratio', 0.15)}")
    print(f"   🎯 Threshold     : {CONFIG.get('recognition_threshold', 0.7)}")
    print("=" * 60 + "\n")
    
    if RUN_MODE == "server":
        refresh_blocked_nims()
    
    yolo_model, rec_model, liveness_model, face_database = load_models(CONFIG)
    cap = cv2.VideoCapture(CONFIG['camera_index'], cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("❌ Kamera tidak bisa dibuka!")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG['camera_width'])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG['camera_height'])
    stream_queue = queue.Queue(maxsize=1)
    detect_queue = queue.Queue(maxsize=1)
    threading.Thread(target=frame_capture_thread,
                     args=(cap, stream_queue, detect_queue, _stop, CONFIG['max_fps'], CONFIG),
                     daemon=True).start()
    threading.Thread(target=frame_stream_thread,
                     args=(stream_queue, _stop, CONFIG),
                     daemon=True).start()
    threading.Thread(target=detection_worker_thread,
                     args=(yolo_model, rec_model, liveness_model, face_database, CONFIG, detect_queue, _stop),
                     daemon=True).start()
    
    if CONFIG.get("alignment_enabled", True):
        threading.Thread(target=alignment_worker_thread,
                         args=(_stop,),
                         daemon=True).start()
    
    print("✅ Threads started. HTTP:8000 WS:/ws/detect")

@app.on_event("shutdown")
def shutdown():
    _stop.set()

@app.get("/")
def root():
    return {"status": "Face Recognition Gate"}

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
            await asyncio.sleep(0.067)
    except WebSocketDisconnect:
        pass
    finally:
        connected_clients.discard(websocket)
        print(f"🔌 Client disconnected ({len(connected_clients)} left)")

if __name__ == "__main__":
    RUN_MODE = ask_run_mode()
    CONFIG["run_mode"] = RUN_MODE
    uvicorn.run(app, host="0.0.0.0", port=8000)