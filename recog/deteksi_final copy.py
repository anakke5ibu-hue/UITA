# ═══════════════════════════════════════════════════════════════
# deteksi_final.py — Backend Face Recognition Gate (Full Integration)
# Integrasi: OpenVINO (YOLO+ArcFace), MediaPipe Anti-Spoofing,
# Hybrid Recognition (Server Alif + Lokal), Supabase Blocklist,
# Dynamic ROI, Adaptive Motion, FastAPI WebSocket
# ═══════════════════════════════════════════════════════════════

import os #untuk operating sistem ke windows
import sys #interaksi mesin python (interpreter python)
import cv2 #opencv memproses gammbar, bukacam, detect wajah
import numpy as np #ngolah angka dan matrix
import pickle #nyimpan dan load data python (face database)
import time #untuk waktu dan delay
import base64 #encode gambar ke base64 untuk kirim ke browser
import json #ngolah data json (kirim ke browser)
import random #random untuk generate id unik
import asyncio #untuk async programming (FastAPI async) detection dan capture di thread terpisah
import threading #untuk jalankan capture dan detection di thread terpisah 
import queue #untuk komunikasi antar thread (frame capture ke detection)
import requests #untuk HTTP request (Server + Supabase)
import warnings #untuk suppress warning yang tidak penting (OpenVINO, MediaPipe)
from pathlib import Path #untuk manipulasi path file (model, database)
from collections import deque #untuk simpan history frame untuk motion detection
from fastapi import FastAPI, WebSocket, WebSocketDisconnect #untuk buat server FastAPI dan WebSocket endpoint
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

warnings.filterwarnings('ignore') #supress OpenVINO and MediaPipe warnings for cleaner output

# ─────────────────────────────────────────────────────────────
# SUPPRESS MEDIAPIPE LOGS & INITIALIZE
# ─────────────────────────────────────────────────────────────
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #supress TensorFlow logs (MediaPipe dependency)
os.environ['GLOG_minloglevel'] = '2' #supress C++ logs from OpenVINO (jika ada)
os.environ['MEDIAPIPE_DISABLE_GPU'] = '1' #force MediaPipe pakai CPU (lebih stabil di banyak sistem)

_real_stderr = sys.stderr #(stderr standar error, jalur khusus pesan kesalaahn, warning, diagnosa) backup stderr asli untuk sementara, karena MediaPipe suka banget ngeprint ke stderr walaupun sebenarnya itu info biasa. Kita redirect sementara ke os.devnull saat import MediaPipe, lalu balikin lagi setelahnya. Jadi log kita tetap bersih tanpa noise dari MediaPipe.
sys.stderr = open(os.devnull, 'w') #redirect stderr ke null sementara untuk suppress logs dari MediaPipe saat import. MediaPipe suka banget ngeprint ke stderr walaupun itu sebenarnya info biasa, jadi kita bersihin log kita dengan cara ini. Setelah import, kita balikin lagi stderr ke normal. Jadi log kita tetap bersih tanpa noise dari MediaPipe.
import mediapipe as mp #untuk liveness detection (anti-spoofing) menggunakan landmark wajah. MediaPipe ini sangat powerful untuk deteksi landmark wajah secara real-time, dan kita akan gunakan untuk menghitung EAR (Eye Aspect Ratio) untuk deteksi kedipan, serta beberapa metrik lain untuk memastikan wajah yang terdeteksi itu nyata, bukan foto atau video. Kita import setelah redirect stderr agar log kita tetap bersih tanpa noise dari MediaPipe.
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode #import class dan options untuk FaceLandmarker dari MediaPipe. FaceLandmarker ini akan kita gunakan untuk deteksi landmark wajah secara real-time, yang sangat berguna untuk liveness detection (anti-spoofing). Kita akan hitung EAR (Eye Aspect Ratio) dari landmark mata untuk deteksi kedipan, serta beberapa metrik lain untuk memastikan wajah yang terdeteksi itu nyata, bukan foto atau video. Kita import setelah redirect stderr agar log kita tetap bersih tanpa noise dari MediaPipe.
sys.stderr = _real_stderr #balikin stderr ke normal setelah import MediaPipe. Jadi log kita tetap bersih tanpa noise dari MediaPipe, tapi kita tetap bisa lihat warning atau error penting lainnya di log kita. Kita lakukan redirect sementara saat import karena MediaPipe suka banget ngeprint ke stderr walaupun itu sebenarnya info biasa, jadi dengan cara ini kita bisa bersihin log kita dari noise tersebut.

# ─────────────────────────────────────────────────────────────
# PYTORCH CPU OPTIMIZATION
# ─────────────────────────────────────────────────────────────
import torch #untuk optimasi penggunaan CPU oleh PyTorch (OpenVINO backend bisa menggunakan PyTorch untuk beberapa operasi, jadi kita batasi agar tidak overload CPU)
num_cpus = min(os.cpu_count() or 4, 4) #batasi maksimal 4 CPU untuk PyTorch agar tidak overload sistem, terutama di mesin dengan banyak core. Kita set ke setengah dari total CPU yang tersedia, tapi minimal 1 core, untuk memberikan ruang bagi proses lain dan menjaga responsivitas sistem.
num_pytorch_cores = max(1, num_cpus // 2) #gunakan setengah dari CPU yang tersedia untuk PyTorch agar tidak overload sistem, terutama di mesin dengan banyak core. Kita set ke setengah dari total CPU yang tersedia, tapi minimal 1 core, untuk memberikan ruang bagi proses lain dan menjaga responsivitas sistem.
torch.set_num_threads(num_pytorch_cores) #batasi jumlah thread yang digunakan PyTorch untuk operasi CPU agar tidak overload sistem. Kita set ke setengah dari total CPU yang tersedia, tapi minimal 1 core, untuk memberikan ruang bagi proses lain dan menjaga responsivitas sistem.
torch.set_num_interop_threads(1) #batasi thread inter-op PyTorch ke 1 untuk mengurangi overhead pada sistem dengan banyak core. Inter-op threads digunakan untuk operasi yang bisa berjalan paralel, tapi di beberapa kasus bisa menyebabkan overhead yang membuat performa malah turun, jadi kita set ke 1 untuk menjaga stabilitas dan responsivitas sistem.
print(f"🔧 PyTorch CPU Limit: {num_pytorch_cores} cores (out of {num_cpus})") #print informasi limit CPU untuk PyTorch

# ─────────────────────────────────────────────────────────────
# KONFIGURASI
# ─────────────────────────────────────────────────────────────
CONFIG = {
    "yolo_model_path":          "models/yolov11n-face_openvino_model", #(openvino untuk proses intel lebih ringan)path ke model YOLOv11 OpenVINO yang sudah dioptimasi untuk deteksi wajah. Pastikan model ini sudah di-convert ke format OpenVINO (.xml + .bin) dan disimpan di folder models dengan nama yolov11n-face_openvino_model.xml dan yolov11n-face_openvino_model.bin. Model ini akan digunakan untuk mendeteksi wajah di frame video secara real-time.
    "arcface_model_path":       "models/buffalo_sc_rec.xml", #path ke model ArcFace OpenVINO yang sudah dioptimasi untuk ekstraksi embedding wajah. Pastikan model ini sudah di-convert ke format OpenVINO (.xml + .bin) dan disimpan di folder models dengan nama buffalo_sc_rec.xml dan buffalo_sc_rec.bin. Model ini akan
    "face_database_path":       "models/face_database.pkl", #path ke file database wajah yang berisi embeddings dan nama-nama orang yang dikenali. File ini harus berupa file pickle (.pkl) yang berisi dictionary dengan keys 'embeddings' (numpy array 2D) dan 'names' (list of strings). Database ini akan digunakan untuk mencocokkan embedding wajah yang terdeteksi dengan nama-nama yang sudah dikenal.
    "mediapipe_model_path":     "models/face_landmarker.task",#path ke model MediaPipe FaceLandmarker untuk liveness detection (anti-spoofing). Pastikan file model ini sudah disimpan di folder models dengan nama face_landmarker.task. Model ini akan digunakan untuk mendeteksi landmark wajah secara real-time, yang sangat berguna untuk menghitung EAR (Eye Aspect Ratio) untuk deteksi kedipan, serta beberapa metrik lain untuk memastikan wajah yang terdeteksi itu nyata, bukan foto atau video.
    "yolo_threshold":           0.5,
    "recognition_threshold":    0.7,
    "camera_index":             1,
    "camera_width":             640,
    "camera_height":            480,
    "max_fps":                  15,
    "motion_threshold":         2000,
    "motion_history":           2,
    "detection_freeze_frames":  30,
    "face_disappear_frames":    5,
    "frame_skip_interval":      6,
    # DYNAMIC ROI
    "roi_enabled":              True,
    "roi_center_ratio":         0.35,
    "roi_expand_ratio":         1.0,
    "roi_expansion_frames":     30,
}

# ─────────────────────────────────────────────────────────────
# ANTI-SPOOFING PARAMETER
# ─────────────────────────────────────────────────────────────
BLINK_THRESHOLD = 0.15
SMILE_THRESHOLD = 0.60
YAW_THRESH      = 0.15
CHALLENGES      = ['BLINK', 'HEAD', 'SMILE']

# ─────────────────────────────────────────────────────────────
# HYBRID RECOGNITION CONFIG
# ─────────────────────────────────────────────────────────────
SERVER_ALIF_URL   = "http://100.107.234.128:8001/identify-face"
# SUPABASE_URL      = "https://kcskzlwxnvmvofyscqsr.supabase.co"
# SUPABASE_ANON_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imtjc2t6bHd4bnZtdm9meXNjcXNyIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzcxNzk0ODIsImV4cCI6MjA5Mjc1NTQ4Mn0.mpmImOcJWkBFwTynGUos7LmUnSYLqGe0h_KRbYQ3tuw"
# SUPABASE_HEADERS  = {
#     "apikey": SUPABASE_ANON_KEY,
#     "Authorization": f"Bearer {SUPABASE_ANON_KEY}",
#     "Content-Type": "application/json",
# }

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
# SERVER BLOKIR LOGIC
# ─────────────────────────────────────────────────────────────
DB_SERVER_URL = "http://100.107.234.128:8001"

def refresh_blocked_nims():
    global blocked_nims, blocked_cache_time
    try:
        res = requests.get(
            f"{DB_SERVER_URL}/users",
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
    return nim not in blocked_nims  # ← tambah "not"


# ─────────────────────────────────────────────────────────────
# ANTI-SPOOFING: LIVENESS METRICS
# ─────────────────────────────────────────────────────────────
def get_liveness_all(lms, w, h):
    """Hitung EAR (blink), Yaw (head turn), dan Smile ratio dari landmark MediaPipe."""
    def get_pt(idx): return np.array([lms[idx].x * w, lms[idx].y * h])
    def ear(pts):
        p = [get_pt(pt) for pt in pts]
        return (np.linalg.norm(p[1]-p[5]) + np.linalg.norm(p[2]-p[4])) / (2.0 * np.linalg.norm(p[0]-p[3]) + 1e-6)

    avg_ear    = (ear([33, 160, 158, 133, 153, 144]) + ear([362, 385, 387, 263, 373, 380])) / 2.0
    yaw        = abs(lms[1].x - lms[33].x) / (abs(lms[263].x - lms[33].x) + 1e-6)
    mouth_w    = abs(lms[61].x - lms[291].x)
    face_w     = abs(lms[33].x  - lms[263].x)
    smile_ratio = mouth_w / (face_w + 1e-6)

    return avg_ear, yaw, smile_ratio

# ─────────────────────────────────────────────────────────────
# PRE-PROCESSING: CLAHE & ALIGNMENT
# ─────────────────────────────────────────────────────────────
def apply_clahe_optimized(img):
    """CLAHE pada ruang warna LAB. Hanya untuk crop wajah kecil."""
    if img is None or img.size == 0:
        return img
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def apply_clahe(img):
    return apply_clahe_optimized(img)

def align_face(img, keypoints, x_offset, y_offset):
    """Face alignment berdasarkan posisi kedua mata dari YOLO keypoints."""
    if keypoints is None or len(keypoints) < 2:
        return img
    left_eye  = (int(keypoints[0][0] - x_offset), int(keypoints[0][1] - y_offset))
    right_eye = (int(keypoints[1][0] - x_offset), int(keypoints[1][1] - y_offset))
    dy        = right_eye[1] - left_eye[1]
    dx        = right_eye[0] - left_eye[0]
    angle     = np.degrees(np.arctan2(dy, dx))
    eye_center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)
    M = cv2.getRotationMatrix2D(eye_center, angle, scale=1.0)
    h, w = img.shape[:2]
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC)

# ─────────────────────────────────────────────────────────────
# MOTION DETECTION (ADAPTIVE)
# ─────────────────────────────────────────────────────────────
def detect_motion(frame, prev_small_frame=None):
    h, w = frame.shape[:2]
    small_frame = cv2.resize(frame, (w//4, h//4))
    if prev_small_frame is None:
        return True, 0, small_frame
    gray_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
    gray_prev  = cv2.cvtColor(prev_small_frame, cv2.COLOR_BGR2GRAY)
    diff       = cv2.absdiff(gray_frame, gray_prev)
    motion_sum = np.sum(diff)
    return motion_sum > 0, motion_sum, small_frame

# ─────────────────────────────────────────────────────────────
# DYNAMIC ROI
# ─────────────────────────────────────────────────────────────
def calculate_roi_bounds(frame_h, frame_w, roi_ratio=0.35):
    roi_h = int(frame_h * roi_ratio)
    roi_w = int(frame_w * roi_ratio)
    roi_y = (frame_h - roi_h) // 2
    roi_x = (frame_w - roi_w) // 2
    return roi_h, roi_w, roi_y, roi_x

def extract_roi(frame, roi_ratio=0.35):
    frame_h, frame_w = frame.shape[:2]
    roi_h, roi_w, roi_y, roi_x = calculate_roi_bounds(frame_h, frame_w, roi_ratio)
    roi_frame = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
    return roi_frame, (roi_y, roi_x, roi_h, roi_w)

def map_detections_from_roi(detections, roi_bounds):
    roi_y, roi_x, roi_h, roi_w = roi_bounds
    for det in detections:
        x1, y1, x2, y2 = det['box']
        det['box'] = (x1 + roi_x, y1 + roi_y, x2 + roi_x, y2 + roi_y)
    return detections

# ─────────────────────────────────────────────────────────────
# FRAME CAPTURE THREAD (ASYNC)
# ─────────────────────────────────────────────────────────────
class FrameCaptureWorker:
    """
    Capture frame dan broadcast ke DUA queue sekaligus:
      - stream_queue  : dibaca frame_stream_thread untuk encode + kirim ke browser (CEPAT)
      - detect_queue  : dibaca detection_worker_thread untuk inferensi berat (LAMBAT)
    Kedua queue maxsize=1 — frame lama otomatis dibuang agar selalu fresh.
    """
    def __init__(self, cap, stream_queue, detect_queue, stop_event, max_fps, cfg):
        self.cap           = cap
        self.stream_queue  = stream_queue
        self.detect_queue  = detect_queue
        self.stop_event    = stop_event
        self.frame_time    = 1.0 / max_fps
        self.cfg           = cfg
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

            is_motion, motion_val, small_frame = detect_motion(frame, self.prev_small_frame)
            self.prev_small_frame = small_frame
            data = {'frame': frame, 'motion': motion_val, 'timestamp': time.time()}

            # Broadcast ke stream queue (selalu fresh)
            try: self.stream_queue.get_nowait()
            except queue.Empty: pass
            self.stream_queue.put(data)

            # Broadcast ke detect queue (selalu fresh)
            try: self.detect_queue.get_nowait()
            except queue.Empty: pass
            self.detect_queue.put(data)

def frame_capture_thread(cap, stream_queue, detect_queue, stop_event, max_fps, cfg):
    """Wrapper dual-queue broadcast."""
    worker = FrameCaptureWorker(cap, stream_queue, detect_queue, stop_event, max_fps, cfg)
    worker.run()

# ─────────────────────────────────────────────────────────────
# DETECTION TRACKING (FREEZE & MAINTAIN)
# ─────────────────────────────────────────────────────────────
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

def track_detections(new_detections, frozen_detections, cfg, is_worker_response=True):
    if not is_worker_response:
        return frozen_detections['detections'] if frozen_detections['freeze_count'] > 0 else []
    if not new_detections:
        if frozen_detections['freeze_count'] > 0:
            frozen_detections['freeze_count'] -= 1
            if frozen_detections['freeze_count'] <= 0:
                return []
            return frozen_detections['detections']
        return []
    frozen_detections['detections']   = new_detections
    frozen_detections['freeze_count'] = cfg['detection_freeze_frames']
    return new_detections

# ─────────────────────────────────────────────────────────────
# LOAD MODELS (OPENVINO + MEDIAPIPE)
# ─────────────────────────────────────────────────────────────
def load_models(cfg: dict):
    from ultralytics import YOLO
    import openvino as ov
    core = ov.Core()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n{'='*60}")
    print(f"  🖥️  Device : {device} (OpenVINO will auto-optimize)")
    print(f"{'='*60}")

    # 1. YOLO
    yolo_path  = Path(cfg["yolo_model_path"])
    print(f"\n⏳ Loading YOLOv11 OpenVINO: {yolo_path} ...")
    yolo_model = YOLO(str(yolo_path), task="detect")
    print(f"✅ YOLOv11 loaded!")

    # 2. ArcFace
    arcface_path = Path(cfg["arcface_model_path"])
    if not arcface_path.exists():
        print(f"\n❌ ArcFace XML tidak ditemukan: {arcface_path}"); sys.exit(1)
    print(f"⏳ Loading ArcFace OpenVINO: {arcface_path} ...")
    ov_model  = core.read_model(str(arcface_path))
    rec_model = core.compile_model(ov_model, "CPU")
    print(f"✅ ArcFace loaded!")

    # 3. MediaPipe FaceLandmarker (Anti-Spoofing)
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
    print(f"✅ MediaPipe Liveness loaded!")

    # 4. Face Database
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
    names = face_database['names']
    print(f"✅ Face database loaded! Persons: {list(set(names))} | Embeddings: {len(names)}")
    print(f"\n{'='*60}\n")

    return yolo_model, rec_model, liveness_model, face_database, device

# ─────────────────────────────────────────────────────────────
# HYBRID RECOGNITION (SERVER ALIF → LOKAL)
# ─────────────────────────────────────────────────────────────
def extract_embedding(face_aligned, rec_model):
    """Ekstrak embedding ArcFace dari crop wajah."""
    face_resized = cv2.resize(face_aligned, (112, 112))
    face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
    blob = (face_rgb.astype(np.float32) - 127.5) / 127.5
    blob = np.expand_dims(blob, 0)
    blob = np.transpose(blob, (0, 3, 1, 2))
    infer_request = rec_model.create_infer_request()
    infer_request.infer(inputs={0: blob})
    emb  = infer_request.get_output_tensor(0).data.flatten()
    norm = np.linalg.norm(emb)
    if norm > 1e-6:
        emb = emb / norm
    return emb

def recognize_hybrid(emb, face_database, threshold):
    """
    Hybrid recognition: coba Server Alif dulu (timeout 0.5s),
    fallback ke database lokal jika gagal/timeout.
    """
    # 1. Server Alif
    try:
        res = requests.post(SERVER_ALIF_URL, json={"embedding": emb.tolist()}, timeout=0.5).json()
        if res.get("status") == "success":
            d   = res["data"]
            sim = round(float(d.get('similarity', 0)), 4)
            return {
                'nama': d.get('nama', 'Unknown'),
                'nim':  d.get('nim', '-'),
                'program_studi': 'S1 Teknik Telekomunikasi',
                'confidence': sim,
                'is_recognized': True,
                'source': 'Alif'
            }
    except Exception:
        pass  # Lanjut ke lokal

    # 2. Fallback Lokal
    sims = np.dot(face_database['embeddings'], emb)
    idx  = int(np.argmax(sims))
    sim  = float(sims[idx])
    if sim >= threshold:
        raw  = face_database['names'][idx]
        nim, nama = raw.split('_', 1) if '_' in raw else ('-', raw)
        return {'nama': nama, 'nim': nim, 'program_studi': '', 'confidence': round(sim, 4), 'is_recognized': True, 'source': 'Local'}
    return {'nama': 'Unknown', 'nim': '-', 'program_studi': '', 'confidence': round(sim, 4), 'is_recognized': False, 'source': 'Local'}

# ─────────────────────────────────────────────────────────────
# MATCH FACE DETECTIONS (IoU-based tracking)
# ─────────────────────────────────────────────────────────────
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
                ix1, iy1 = max(x1_1,x1_2), max(y1_1,y1_2)
                ix2, iy2 = min(x2_1,x2_2), min(y2_1,y2_2)
                if ix2 > ix1 and iy2 > iy1:
                    inter = (ix2-ix1)*(iy2-iy1)
                    union = (x2_1-x1_1)*(y2_1-y1_1) + (x2_2-x1_2)*(y2_2-y1_2) - inter
                    iou   = inter / union if union > 0 else 0
                    if iou > best_iou:
                        best_iou, best_match = iou, prev_det
        matches[i] = best_match
    return matches

# ─────────────────────────────────────────────────────────────
# DETECT & RECOGNIZE (YOLO + Hybrid ArcFace, only new faces)
# ─────────────────────────────────────────────────────────────
def detect_and_recognize(frame, yolo_model, rec_model, face_database, cfg,
                          prev_detections=None, roi_bounds=None):
    yolo_threshold        = cfg["yolo_threshold"]
    recognition_threshold = cfg["recognition_threshold"]

    # Dynamic ROI
    if roi_bounds is not None:
        roi_y, roi_x, roi_h, roi_w = roi_bounds
        detect_frame = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
    else:
        detect_frame = frame
        roi_bounds   = None

    results    = yolo_model.predict(detect_frame, verbose=False, conf=yolo_threshold, iou=0.4, device='cpu')
    detections = []
    new_boxes  = []

    for result in results:
        boxes   = result.boxes.xyxy.cpu().numpy()
        kps_all = result.keypoints.xy.cpu().numpy() if result.keypoints is not None else None

        for i, bbox in enumerate(boxes):
            x1, y1, x2, y2 = map(int, bbox)
            new_boxes.append((x1, y1, x2, y2))

        matches = match_face_detections(new_boxes, prev_detections or [], iou_threshold=0.3)

        for i, bbox in enumerate(boxes):
            x1, y1, x2, y2 = map(int, bbox)
            matched_prev = matches[i]

            should_force_recognize = (
                matched_prev is not None and
                matched_prev['name'] in ("Unknown", "Error") and
                matched_prev.get('unknown_skip_count', 0) > 0
            )
            is_new_face = (matched_prev is None) or (
                matched_prev['name'] in ("Unknown", "Error") and not should_force_recognize
            )

            h, w   = frame.shape[:2]
            pad_x  = int((x2 - x1) * 0.1)
            pad_y  = int((y2 - y1) * 0.1)
            x1_p   = max(0, x1 - pad_x); y1_p = max(0, y1 - pad_y)
            x2_p   = min(w, x2 + pad_x); y2_p = min(h, y2 + pad_y)
            face   = frame[y1_p:y2_p, x1_p:x2_p]

            if face.size == 0:
                rec_data = {'nama': 'Unknown', 'nim': '-', 'confidence': 0.0,
                            'is_recognized': False, 'source': 'Local'}
                unknown_skip_count = 5
            elif is_new_face:
                face_clahe   = apply_clahe(face)
                face_aligned = align_face(face_clahe, kps_all[i], x1_p, y1_p) if (kps_all is not None and len(kps_all) > i) else face_clahe
                emb          = extract_embedding(face_aligned, rec_model)
                rec_data     = recognize_hybrid(emb, face_database, recognition_threshold)
                unknown_skip_count = 5 if not rec_data['is_recognized'] else 0
            else:
                rec_data = {
                    'nama': matched_prev['name'], 'nim': matched_prev.get('nim', '-'),
                    'confidence': matched_prev['conf'], 'is_recognized': matched_prev['is_recognized'],
                    'source': matched_prev.get('source', 'Local')
                }
                unknown_skip_count = max(0, matched_prev.get('unknown_skip_count', 0) - 1)

            detections.append({
                "box":               (x1, y1, x2, y2),
                "name":              rec_data['nama'],
                "nim":               rec_data['nim'],
                "conf":              rec_data['confidence'],
                "is_recognized":     rec_data['is_recognized'],
                "source":            rec_data.get('source', 'Local'),
                "is_new":            is_new_face,
                "unknown_skip_count": unknown_skip_count
            })

    if roi_bounds is not None and detections:
        detections = map_detections_from_roi(detections, roi_bounds)

    return detections

# ─────────────────────────────────────────────────────────────
# DETECTION THREAD — LIVENESS + YOLO + HYBRID ARCFACE
# ─────────────────────────────────────────────────────────────
def detection_thread(yolo_model, rec_model, liveness_model, face_database, cfg,
                     input_queue, result_queue, stop_event,
                     frozen_detections_ref=None, liveness_state_ref=None):
    """
    Thread utama: jalankan liveness check (MediaPipe) dulu,
    lalu face detection + hybrid recognition (YOLO + ArcFace).
    """
    motion_threshold   = cfg["motion_threshold"]
    frame_skip_interval = cfg["frame_skip_interval"]
    frame_count        = 0
    roi_enabled        = cfg.get("roi_enabled", True)
    roi_expansion_count = 0

    while not stop_event.is_set():
        try:
            data = input_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        frame       = data['frame']
        motion      = data['motion']
        h, w        = frame.shape[:2]
        frame_count += 1

        has_frozen_face = (frozen_detections_ref is not None and
                           frozen_detections_ref.get('freeze_count', 0) > 0)
        use_full_frame  = (motion >= motion_threshold) or has_frozen_face

        # ROI logic
        if use_full_frame:
            roi_expansion_count = cfg["roi_expansion_frames"]
            roi_bounds = None
        else:
            if roi_expansion_count > 0:
                roi_expansion_count -= 1
                roi_bounds = None
            else:
                if roi_enabled:
                    _, _, roi_y, roi_x = calculate_roi_bounds(h, w, cfg["roi_center_ratio"])
                    roi_h = int(h * cfg["roi_center_ratio"])
                    roi_w = int(w * cfg["roi_center_ratio"])
                    roi_bounds = (roi_y, roi_x, roi_h, roi_w)
                else:
                    roi_bounds = None

        # Frame skipping
        should_detect = (frame_count % frame_skip_interval) == 0

        if not should_detect:
            result_queue.put({
                'detections': [], 'motion': motion,
                'processed': False, 'roi_mode': roi_bounds is not None,
                'liveness_update': None
            })
            time.sleep(0.1)
            continue

        try:
            # ── 1. LIVENESS CHECK (MediaPipe) ──
            mp_img    = mp.Image(image_format=mp.ImageFormat.SRGB,
                                  data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            timestamp = int(time.time() * 1000)
            mp_results = liveness_model.detect_for_video(mp_img, timestamp)

            liveness_update = {
                'ear':       0,
                'yaw':       0.5,
                'smile':     0,
                'passed':    liveness_state_ref['passed'] if liveness_state_ref else False,
                'challenge': liveness_state_ref['challenge'] if liveness_state_ref else random.choice(CHALLENGES)
            }

            if mp_results.face_landmarks:
                ear, yaw, smile = get_liveness_all(mp_results.face_landmarks[0], w, h)
                liveness_update.update({'ear': ear, 'yaw': yaw, 'smile': smile})

                if liveness_state_ref and not liveness_state_ref['passed']:
                    ch = liveness_state_ref['challenge']
                    if   ch == 'BLINK' and ear < BLINK_THRESHOLD:
                        liveness_update['passed'] = True
                    elif ch == 'HEAD'  and (yaw < (0.5 - YAW_THRESH) or yaw > (0.5 + YAW_THRESH)):
                        liveness_update['passed'] = True
                    elif ch == 'SMILE' and smile > SMILE_THRESHOLD:
                        liveness_update['passed'] = True

            # ── 2. FACE DETECTION + HYBRID RECOGNITION ──
            prev_dets  = frozen_detections_ref.get('detections', []) if frozen_detections_ref else None
            detections = detect_and_recognize(
                frame, yolo_model, rec_model, face_database, cfg,
                prev_detections=prev_dets, roi_bounds=roi_bounds
            )

            # Reset liveness jika wajah hilang
            if not detections and liveness_state_ref:
                liveness_update['passed']    = False
                liveness_update['challenge'] = random.choice(CHALLENGES)

            result_queue.put({
                'detections':      detections,
                'motion':          motion,
                'processed':       True,
                'roi_mode':        roi_bounds is not None,
                'liveness_update': liveness_update
            })

        except Exception as e:
            print(f"\n[DETECTION ERROR] {e}")
            result_queue.put({
                'detections': [], 'motion': motion,
                'processed': False, 'roi_mode': roi_bounds is not None,
                'liveness_update': None
            })

        time.sleep(0.1)

# ─────────────────────────────────────────────────────────────
# SHARED DETECTION STATE (thread-safe, dibaca oleh stream thread)
# ─────────────────────────────────────────────────────────────
# State ini di-update oleh detection_worker_thread, dibaca oleh
# frame_stream_thread. Pakai lock ringan agar tidak ada race condition.
_det_lock  = threading.Lock()
_det_state = {
    'tracked':        [],
    'liveness':       {'passed': False, 'challenge': 'BLINK', 'ear': 0, 'yaw': 0.5, 'smile': 0},
    'status':         'idle',
    'user_data':      None,
    'faces_data':     [],
    'motion':         0.0,
}

# ─────────────────────────────────────────────────────────────
# DETECTION WORKER THREAD — YOLO + ArcFace + MediaPipe 
# ─────────────────────────────────────────────────────────────
def detection_worker_thread(yolo_model, rec_model, liveness_model, face_database, cfg,
                             detect_queue, stop_event):
    """
    Thread khusus inferensi berat (MediaPipe + YOLO + ArcFace).
    Hasilnya ditulis ke _det_state, dibaca oleh frame_stream_thread.
    Thread ini TIDAK pernah meng-encode frame — tugasnya hanya inferensi.
    """
    global last_granted_time, last_denied_time, last_granted_identity

    frozen_detections   = {'detections': [], 'freeze_count': 0}
    liveness_state      = {'challenge': random.choice(CHALLENGES), 'passed': False, 'ear': 0, 'yaw': 0.5, 'smile': 0}
    motion_threshold    = cfg["motion_threshold"]
    frame_skip_interval = cfg["frame_skip_interval"]
    roi_enabled         = cfg.get("roi_enabled", True)
    roi_expansion_count = 0
    frame_count         = 0

    while not stop_event.is_set():
        try:
            data = detect_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        frame       = data['frame']
        motion      = data['motion']
        h, w        = frame.shape[:2]
        frame_count += 1

        has_frozen = frozen_detections['freeze_count'] > 0
        use_full   = motion >= motion_threshold or has_frozen

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
                    _, _, roi_y, roi_x = calculate_roi_bounds(h, w, cfg["roi_center_ratio"])
                    roi_h = int(h * cfg["roi_center_ratio"])
                    roi_w = int(w * cfg["roi_center_ratio"])
                    roi_bounds = (roi_y, roi_x, roi_h, roi_w)

        # Skipping: jalankan deteksi hanya setiap N frame
        if (frame_count % frame_skip_interval) != 0 and not has_frozen:
            continue

        # ── 1. LIVENESS CHECK (MediaPipe) ──
        try:
            mp_img     = mp.Image(image_format=mp.ImageFormat.SRGB,
                                   data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            timestamp  = int(time.time() * 1000)
            mp_results = liveness_model.detect_for_video(mp_img, timestamp)

            if mp_results.face_landmarks:
                ear, yaw, smile = get_liveness_all(mp_results.face_landmarks[0], w, h)
                liveness_state.update({'ear': ear, 'yaw': yaw, 'smile': smile})
                if not liveness_state['passed']:
                    ch = liveness_state['challenge']
                    if   ch == 'BLINK' and ear < BLINK_THRESHOLD:
                        liveness_state['passed'] = True
                    elif ch == 'HEAD'  and (yaw < (0.5 - YAW_THRESH) or yaw > (0.5 + YAW_THRESH)):
                        liveness_state['passed'] = True
                    elif ch == 'SMILE' and smile > SMILE_THRESHOLD:
                        liveness_state['passed'] = True
        except Exception:
            pass

        # ── 2. YOLO + HYBRID ARCFACE ──
        raw_detections = detect_and_recognize(
            frame, yolo_model, rec_model, face_database, cfg,
            prev_detections=frozen_detections.get('detections', []),
            roi_bounds=roi_bounds
        )
        tracked = track_detections(raw_detections, frozen_detections, cfg, is_worker_response=True)

        if not tracked:
            liveness_state['passed']    = False
            liveness_state['challenge'] = random.choice(CHALLENGES)

        # ── 3. ACCESS CONTROL (hitung status, simpan user_data) ──
        status    = 'idle'
        user_data = None
        faces_data = []

        for det in tracked:
            x1, y1, x2, y2 = det["box"]
            faces_data.append({
                'x1': int(x1), 'y1': int(y1), 'x2': int(x2), 'y2': int(y2),
                'confidence':    det["conf"],
                'name':          det["name"],
                'nim':           det.get("nim", "-"),
                'is_recognized': det["is_recognized"]
            })

            if user_data is None and liveness_state['passed']:
                status = 'scanning'
                now    = time.time()
                if det["is_recognized"]:
                    nim = det.get("nim", "-")
                    if is_blocked(nim):
                        if (now - last_denied_time) > 10:
                            status           = 'denied'
                            last_denied_time = now
                    elif (now - last_granted_time) > 10 or last_granted_identity != nim:
                        status                = 'granted'
                        user_data             = {'nama': det["name"], 'nim': nim, 'confidence': det["conf"]}
                        last_granted_time     = now
                        last_granted_identity = nim
                        liveness_state['passed']    = False
                        liveness_state['challenge'] = random.choice(CHALLENGES)
                else:
                    if (now - last_denied_time) > 10:
                        status           = 'denied'
                        last_denied_time = now

        # ── 4. TULIS HASIL KE SHARED STATE (lock singkat) ──
        with _det_lock:
            _det_state['tracked']    = tracked
            _det_state['liveness']   = dict(liveness_state)
            _det_state['status']     = status
            _det_state['user_data']  = user_data
            _det_state['faces_data'] = faces_data
            _det_state['motion']     = float(motion)


# ─────────────────────────────────────────────────────────────
# FRAME STREAM THREAD — encode + kirim frame ke WebSocket (CEPAT, ~15–30 FPS)
# ─────────────────────────────────────────────────────────────
def frame_stream_thread(stream_queue, stop_event, cfg):
    """
    Thread ini berjalan terus di ~15–30 FPS.
    Setiap frame: baca _det_state (tanpa blocking), overlay bounding box,
    encode JPEG, update camera_result.
    Tidak ada inferensi berat di sini — semua sudah dihitung detection_worker_thread.
    """
    global camera_result
    target_interval = 1.0 / cfg.get("max_fps", 15)

    while not stop_event.is_set():
        t0 = time.time()

        try:
            data = stream_queue.get_nowait()
        except queue.Empty:
            elapsed = time.time() - t0
            sleep_t = max(0, target_interval - elapsed)
            time.sleep(sleep_t)
            continue

        frame = data['frame']

        # Baca hasil deteksi terakhir — non-blocking
        with _det_lock:
            tracked    = list(_det_state['tracked'])
            liveness   = dict(_det_state['liveness'])
            status     = _det_state['status']
            user_data  = _det_state['user_data']
            faces_data = list(_det_state['faces_data'])
            motion     = _det_state['motion']

        # Overlay bounding box + banner liveness (ringan, hanya cv2.rectangle + putText)
        draw_frame = frame.copy()
        h, w = frame.shape[:2]

        for det in tracked:
            x1, y1, x2, y2 = det["box"]
            if liveness['passed']:
                color = (0, 255, 0) if det.get("is_recognized") else (0, 165, 255)
                cv2.rectangle(draw_frame, (x1, y1), (x2, y2), color, 3)
            else:
                cv2.rectangle(draw_frame, (x1, y1), (x2, y2), (0, 165, 255), 2)

        if tracked:
            cv2.rectangle(draw_frame, (0, 0), (w, 50), (30, 30, 30), -1)
            if liveness['passed']:
                cv2.putText(draw_frame, "LIVENESS OK! PROCESSING...", (20, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                inst = f"CHALLENGE: PLEASE {liveness['challenge']}!"
                cv2.putText(draw_frame, inst, (20, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        _, buffer  = cv2.imencode('.jpg', draw_frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        with camera_lock:
            camera_result = {
                'image':      f'data:image/jpeg;base64,{img_base64}',
                'status':     status,
                'face_count': len(faces_data),
                'faces':      faces_data,
                'user':       user_data,
                'motion':     motion,
                'liveness': {
                    'passed':    liveness['passed'],
                    'challenge': liveness['challenge'],
                    'ear':       round(liveness.get('ear', 0), 3),
                    'yaw':       round(liveness.get('yaw', 0.5), 3),
                    'smile':     round(liveness.get('smile', 0), 3),
                }
            }

        elapsed = time.time() - t0
        sleep_t = max(0, target_interval - elapsed)
        time.sleep(sleep_t)


# ─────────────────────────────────────────────────────────────
# PROCESSING STATE THREAD — wrapper untuk backward compat (tidak dipakai lagi)
# ─────────────────────────────────────────────────────────────
def processing_state_thread(yolo_model, rec_model, liveness_model, face_database, cfg,
                             frame_queue, stop_event):
    """Deprecated — startup_event sekarang panggil detection_worker_thread + frame_stream_thread langsung."""
    pass

# ─────────────────────────────────────────────────────────────
# DRAW UI (untuk mode lokal / standalone)
# ─────────────────────────────────────────────────────────────
def draw_overlay(frame, detections, liveness_state=None, roi_bounds=None):
    annotated = frame.copy()
    h_frame, w_frame = frame.shape[:2]

    for det in detections:
        x1, y1, x2, y2 = det["box"]
        color = (0, 255, 0) if det.get("is_recognized", det["name"] not in ("Unknown", "Error")) else (0, 165, 255)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)

    if roi_bounds is not None:
        roi_y, roi_x, roi_h, roi_w = roi_bounds
        cv2.rectangle(annotated, (roi_x, roi_y), (roi_x+roi_w, roi_y+roi_h), (255, 255, 0), 2)
        cv2.putText(annotated, "ROI", (roi_x+5, roi_y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

    if liveness_state and detections:
        cv2.rectangle(annotated, (0, 0), (w_frame, 50), (30, 30, 30), -1)
        if liveness_state.get('passed'):
            cv2.putText(annotated, "LIVENESS OK! PROCESSING...", (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            inst = f"CHALLENGE: PLEASE {liveness_state.get('challenge', 'BLINK')}!"
            cv2.putText(annotated, inst, (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    return annotated

def draw_info_panel(detections, fps, motion, last_processed, cfg, frame_count,
                     panel_height=480, roi_mode=False, liveness_state=None):
    panel_width = 380
    panel = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)

    cv2.rectangle(panel, (0, 0), (panel_width, 40), (40, 40, 40), -1)
    cv2.putText(panel, "GATE INFO PANEL", (10, 28), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 255), 2)

    y_pos = 60

    # Liveness Stats
    if liveness_state:
        cv2.putText(panel, "LIVENESS SENSOR:", (15, y_pos), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 100, 100), 1); y_pos += 25
        cv2.putText(panel, f"EAR (Blink): {liveness_state.get('ear', 0):.3f}", (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1); y_pos += 20
        cv2.putText(panel, f"YAW (Head) : {liveness_state.get('yaw', 0.5):.3f}", (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1); y_pos += 20
        cv2.putText(panel, f"SMILE Ratio: {liveness_state.get('smile', 0):.3f}", (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1); y_pos += 10
        passed_color = (0, 255, 0) if liveness_state.get('passed') else (0, 165, 255)
        passed_text  = "PASSED ✓" if liveness_state.get('passed') else f"WAITING: {liveness_state.get('challenge','?')}"
        cv2.putText(panel, passed_text, (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, passed_color, 1); y_pos += 20
        cv2.line(panel, (10, y_pos), (panel_width-10, y_pos), (80, 80, 80), 1); y_pos += 15

    # System Stats
    cv2.putText(panel, f"FPS: {fps:.1f}", (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 1); y_pos += 25
    status_color = (0, 255, 0) if last_processed else (0, 165, 255)
    cv2.putText(panel, f"Status: {'PROCESSING' if last_processed else 'IDLE'}", (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 1); y_pos += 25
    roi_color = (100, 200, 255) if roi_mode else (100, 255, 100)
    cv2.putText(panel, f"View: {'ROI MODE' if roi_mode else 'FULL FRAME'}", (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, roi_color, 1); y_pos += 25
    cv2.putText(panel, f"Motion: {motion:.0f} {'✓' if motion >= cfg['motion_threshold'] else '✗'}", (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1); y_pos += 25
    cv2.putText(panel, f"Frames: {frame_count} | Skip: 1/{cfg['frame_skip_interval']}", (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1); y_pos += 15
    cv2.line(panel, (10, y_pos), (panel_width-10, y_pos), (80, 80, 80), 1); y_pos += 20

    # Detections
    cv2.putText(panel, "DETECTIONS:", (15, y_pos), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 165, 0), 1); y_pos += 28
    if detections:
        if liveness_state and not liveness_state.get('passed'):
            cv2.putText(panel, "[ Liveness Pending... ]", (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 1)
        else:
            for det in detections:
                if y_pos > panel_height - 40: break
                name  = det["name"]
                conf  = det["conf"]
                src   = det.get("source", "")
                color = (0, 255, 0) if name not in ("Unknown", "Error") else (0, 165, 255)
                symbol = "✓" if name not in ("Unknown", "Error") else "?"
                cv2.putText(panel, f"{symbol} {name} [{src}]", (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1); y_pos += 20
                cv2.putText(panel, f"   Conf: {conf:.3f}", (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1); y_pos += 24
    else:
        cv2.putText(panel, "[No detections]", (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

    return panel

# ─────────────────────────────────────────────────────────────
# FASTAPI SETUP
# ─────────────────────────────────────────────────────────────
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_stop_event = threading.Event()

@app.on_event("startup")
async def startup_event():
    refresh_blocked_nims()

    yolo_model, rec_model, liveness_model, face_database, _ = load_models(CONFIG)

    cap = cv2.VideoCapture(CONFIG["camera_index"], cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("❌ Kamera tidak bisa dibuka!"); sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CONFIG["camera_width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG["camera_height"])

    # ── Tiga queue terpisah ──────────────────────────────────────
    # stream_queue  → frame_capture_thread → frame_stream_thread  (encode + kirim, CEPAT)
    # detect_queue  → frame_capture_thread → detection_worker_thread (inferensi berat, LAMBAT)
    # Keduanya maxsize=1 agar frame lama langsung dibuang
    stream_queue = queue.Queue(maxsize=1)
    detect_queue = queue.Queue(maxsize=1)

    # Thread 1: capture kamera, broadcast ke dua queue
    threading.Thread(
        target=frame_capture_thread,
        args=(cap, stream_queue, detect_queue, _stop_event, CONFIG['max_fps'], CONFIG),
        daemon=True
    ).start()

    # Thread 2: encode + overlay + kirim ke WebSocket (~15 FPS, tanpa inferensi)
    threading.Thread(
        target=frame_stream_thread,
        args=(stream_queue, _stop_event, CONFIG),
        daemon=True
    ).start()

    # Thread 3: inferensi berat (MediaPipe + YOLO + ArcFace), hasilnya ke _det_state
    threading.Thread(
        target=detection_worker_thread,
        args=(yolo_model, rec_model, liveness_model, face_database, CONFIG, detect_queue, _stop_event),
        daemon=True
    ).start()

    print("✅ Threads Started: capture | stream | detection")
    print("   HTTP : http://localhost:8000")
    print("   WS   : ws://localhost:8000/ws/detect")

@app.on_event("shutdown")
def shutdown_event():
    _stop_event.set()

@app.get("/")
def root():
    return {"status": "Face Recognition Gate (Hybrid + Anti-Spoof) Running"}

@app.websocket("/ws/detect")
async def websocket_detect(websocket: WebSocket):
    await websocket.accept()
    connected_clients.add(websocket)
    print(f"🔗 Client konek ({len(connected_clients)} total)")
    try:
        while True:
            with camera_lock:
                result = camera_result
            if result:
                await websocket.send_text(json.dumps(result))
            await asyncio.sleep(0.067)  # ~15 fps ke client
    except WebSocketDisconnect:
        pass
    finally:
        connected_clients.discard(websocket)
        print(f"🔌 Client putus ({len(connected_clients)} tersisa)")

# ─────────────────────────────────────────────────────────────
# MAIN — Mode Standalone (OpenCV Window, tanpa FastAPI)
# ─────────────────────────────────────────────────────────────
def main():
    import csv
    from datetime import datetime

    cfg = CONFIG
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file   = open(f"laporan_deteksi_{timestamp_str}.csv", mode="w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["waktu", "nama", "nim", "confidence", "threshold", "source", "x1", "y1", "x2", "y2"])

    yolo_model, rec_model, liveness_model, face_database, device = load_models(cfg)

    cap = cv2.VideoCapture(cfg["camera_index"])
    if not cap.isOpened():
        print(f"❌ Tidak bisa membuka webcam index {cfg['camera_index']}"); sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  cfg["camera_width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg["camera_height"])

    print("🎥 Webcam aktif! Tekan Q atau ESC untuk keluar\n")

    stop_event   = threading.Event()
    frame_queue  = queue.Queue(maxsize=1)   # untuk display (cv2.imshow)
    detect_queue = queue.Queue(maxsize=1)   # untuk detection_thread
    result_queue = queue.Queue(maxsize=1)

    frozen_detections = {'detections': [], 'freeze_count': 0}
    liveness_state    = {'challenge': random.choice(CHALLENGES), 'passed': False, 'ear': 0, 'yaw': 0.5, 'smile': 0}

    threading.Thread(
        target=frame_capture_thread,
        args=(cap, frame_queue, detect_queue, stop_event, cfg['max_fps'], cfg),
        daemon=True
    ).start()

    threading.Thread(
        target=detection_thread,
        args=(yolo_model, rec_model, liveness_model, face_database, cfg,
              detect_queue, result_queue, stop_event, frozen_detections, liveness_state),
        daemon=True
    ).start()

    fps_history     = []
    last_detections = []
    frame_display   = None
    last_frame_time = time.time()
    last_motion     = 0
    last_processed  = False
    last_roi_mode   = False
    last_roi_bounds = None
    frame_count     = 0
    fps             = 0

    try:
        while True:
            # Baca frame terbaru
            try:
                frame_data    = frame_queue.get_nowait()
                frame_display = frame_data['frame']
                last_motion   = frame_data['motion']
                frame_count  += 1
            except queue.Empty:
                pass

            # Baca hasil deteksi
            try:
                result          = result_queue.get_nowait()
                raw_detections  = result['detections']
                last_processed  = result['processed']
                last_roi_mode   = result.get('roi_mode', False)
                if result.get('liveness_update'):
                    liveness_state.update(result['liveness_update'])
                last_detections = track_detections(raw_detections, frozen_detections, cfg, is_worker_response=True)
            except queue.Empty:
                last_detections = track_detections([], frozen_detections, cfg, is_worker_response=False)

            # FPS
            now = time.time()
            if now - last_frame_time > 0:
                fps = 1.0 / (now - last_frame_time)
                fps_history.append(fps)
            last_frame_time = now

            # ROI bounds untuk visualisasi
            if last_roi_mode and frame_display is not None:
                fh, fw = frame_display.shape[:2]
                rh, rw, ry, rx = calculate_roi_bounds(fh, fw, cfg["roi_center_ratio"])
                last_roi_bounds = (ry, rx, rh, rw)
            else:
                last_roi_bounds = None

            if frame_display is not None:
                camera_view = draw_overlay(frame_display, last_detections, liveness_state, last_roi_bounds)
                panel_h     = camera_view.shape[0]
                info_panel  = draw_info_panel(last_detections, fps, last_motion, last_processed,
                                               cfg, frame_count, panel_h, last_roi_mode, liveness_state)
                combined    = np.hstack([camera_view, info_panel])
                cv2.imshow("TELYU GATE: Anti-Spoof + Hybrid Recognition", combined)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q'), 27):
                break

            # CSV logging — hanya wajah BARU yang sudah lulus liveness
            if last_detections and last_processed and liveness_state['passed']:
                new_faces = [d for d in last_detections if d.get('is_new', False)]
                for det in new_faces:
                    waktu = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    x1, y1, x2, y2 = det["box"]
                    csv_writer.writerow([
                        waktu, det["name"], det.get("nim", "-"),
                        f"{det['conf']:.4f}", f"{cfg['recognition_threshold']}",
                        det.get("source", "Local"), x1, y1, x2, y2
                    ])
                    csv_file.flush()
                    print(f"\n✅ [ACCESS GRANTED] {time.strftime('%H:%M:%S')} | {det['name']} ({det.get('nim','-')}) | Conf: {det['conf']:.2f} | Src: {det.get('source','Local')}")
                    # Acak challenge berikutnya
                    liveness_state['passed']    = False
                    liveness_state['challenge'] = random.choice(CHALLENGES)
            else:
                lv_status = "OK" if liveness_state['passed'] else f"WAIT:{liveness_state['challenge']}"
                sys.stdout.write(f"\r🔍 Liveness: {lv_status} | FPS: {fps:.1f} | Motion: {last_motion:.0f} | Frames: {frame_count}    ")
                sys.stdout.flush()

    except KeyboardInterrupt:
        print("\n⛔ Interrupted by user")
    finally:
        stop_event.set()
        cap.release()
        cv2.destroyAllWindows()
        csv_file.close()
        if fps_history:
            print(f"\n📊 Selesai! Rata-rata FPS: {np.mean(fps_history):.1f} | Total frame: {len(fps_history)}")

# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="TELYU GATE — Face Recognition + Anti-Spoof")
    parser.add_argument('--mode', choices=['standalone', 'server'], default='server',
                        help="'standalone' = OpenCV window lokal | 'server' = FastAPI WebSocket (default: server)")
    args = parser.parse_args()

    if args.mode == 'standalone':
        main()
    else:
        print("\n🌐 Hybrid Web Server Running:")
        print("   HTTP: http://localhost:8000")
        print("   WS:   ws://localhost:8000/ws/detect")
        print("="*60)
        uvicorn.run(app, host="0.0.0.0", port=8000)