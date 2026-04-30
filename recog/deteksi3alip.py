import os
import sys
import cv2
import numpy as np
import pickle
import time
import warnings
import threading
import queue
from pathlib import Path
from collections import deque

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────
# PYTORCH CPU OPTIMIZATION - Limit cores untuk Windows
# ─────────────────────────────────────────────────────────────
# Batasi PyTorch menggunakan max 2 cores agar Windows+AI bisa berbagi beban
import torch
num_cpus = min(os.cpu_count() or 4, 4)  # Max 4 cores available
num_pytorch_cores = max(1, num_cpus // 2)  # Use 50% of available cores (e.g., 2 from 4)
torch.set_num_threads(num_pytorch_cores)
torch.set_num_interop_threads(1)
print(f"🔧 PyTorch CPU Limit: {num_pytorch_cores} cores (out of {num_cpus})")

# ─────────────────────────────────────────────────────────────
# KONFIGURASI 
# ─────────────────────────────────────────────────────────────
CONFIG = {
    # Arahkan ke folder hasil export YOLO OpenVINO atau file .xml YOLO
    "yolo_model_path":        "models/yolov11n-face_openvino_model", 
    # Arahkan ke file XML ArcFace buffalo_sc kamu
    "arcface_model_path":     "models/buffalo_sc_rec.xml",           
    "face_database_path":     "models/face_database.pkl",
    "yolo_threshold":         0.5,
    "recognition_threshold":  0.5,
    "camera_index":           1,      # ganti ke 1 / 2 jika webcam salah
    "camera_width":           640,
    "camera_height":          480,
    "max_fps":                15,     # Limit FPS untuk efisiensi energi
    "motion_threshold":       2000,    # Threshold motion detection (TURUN dari 1500 - lebih ringan)
    "motion_history":         2,      # Frames sebelumnya untuk comparison
    "detection_freeze_frames": 50,    # Freeze hasil detection selama X frames jika face masih terlihat
    "face_disappear_frames":   5,     # Clear detection jika face hilang selama X frames
}

# ─────────────────────────────────────────────────────────────
# FUNGSI TAMBAHAN: CLAHE & ALIGNMENT
# ─────────────────────────────────────────────────────────────
def apply_clahe_optimized(img):
    """
    Memperbaiki kontras menggunakan CLAHE pada ruang warna LAB.
    HANYA untuk crop wajah kecil (~200x200), bukan untuk frame utama (640x480).
    
    In-place strategy:
    - Split channels dilakukan on-the-fly
    - CLAHE applied hanya pada L channel (grayscale operation)
    - Output merged back dengan minimal intermediate arrays
    """
    if img is None or img.size == 0:
        return img
    
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)  # In-place assignment
    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def apply_clahe(img):
    """Wrapper untuk backward compatibility."""
    return apply_clahe_optimized(img)

def align_face(img, keypoints, x_offset, y_offset):
    """
    Melakukan face alignment menggunakan keypoints dari YOLO.
    Titik koordinat YOLO disesuaikan dengan posisi crop wajah (offset).
    """
    if keypoints is None or len(keypoints) < 2:
        return img
    
    # Ambil titik mata kiri (index 0) dan kanan (index 1) dari YOLO
    left_eye = (int(keypoints[0][0] - x_offset), int(keypoints[0][1] - y_offset))
    right_eye = (int(keypoints[1][0] - x_offset), int(keypoints[1][1] - y_offset))
    
    # Hitung sudut rotasi
    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dy, dx))
    
    # Pusat rotasi (tengah-tengah mata)
    eye_center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)
    
    # Matriks rotasi
    M = cv2.getRotationMatrix2D(eye_center, angle, scale=1.0)
    h, w = img.shape[:2]
    aligned_img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC)
    
    return aligned_img

# ─────────────────────────────────────────────────────────────
# MOTION DETECTION (ADAPTIVE)
# ─────────────────────────────────────────────────────────────
def detect_motion(frame, prev_small_frame=None):
    """
    Deteksi gerakan dengan membandingkan frame current vs previous frame.
    
    OPTIMASI: Menerima prev_small_frame yang sudah di-resize agar tidak perlu 
    resize 2x setiap frame. Mengembalikan small_frame current untuk digunakan 
    di frame berikutnya.
    
    Args:
        frame: Current frame dari kamera (full resolution)
        prev_small_frame: Previous frame yang sudah di-resize (atau None untuk frame pertama)
    Returns:
        (is_motion: bool, motion_sum: float, small_frame: ndarray) 
        - small_frame digunakan untuk iterasi berikutnya
    """
    # Resize current frame untuk performa lebih cepat (downscale 4x)
    h, w = frame.shape[:2]
    small_frame = cv2.resize(frame, (w//4, h//4))
    
    if prev_small_frame is None:
        # Frame pertama - tidak ada previous frame untuk compare
        return True, 0, small_frame
    
    # Convert ke grayscale (temporary arrays - akan di-GC)
    gray_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
    gray_prev = cv2.cvtColor(prev_small_frame, cv2.COLOR_BGR2GRAY)
    
    # Hitung frame difference
    diff = cv2.absdiff(gray_frame, gray_prev)
    motion_sum = np.sum(diff)
    
    return motion_sum > 0, motion_sum, small_frame

# ─────────────────────────────────────────────────────────────
# FRAME CAPTURE THREAD (ASYNC)
# ─────────────────────────────────────────────────────────────
class FrameCaptureWorker:
    """
    Worker untuk capture frame dengan motion detection adaptive.
    
    MEMORY OPTIMIZATION:
    - Menyimpan small_frame (resized 1/4) ke history, bukan full resolution
    - Ini menghemat ~12x memory dibanding menyimpan frame penuh (640x480 -> 160x120)
    - Tidak perlu frame.copy() karena small_frame sudah hasil resize (object baru)
    """
    def __init__(self, cap, frame_queue, stop_event, max_fps, cfg):
        self.cap = cap
        self.frame_queue = frame_queue
        self.stop_event = stop_event
        self.frame_time = 1.0 / max_fps
        self.cfg = cfg
        # OPTIMASI: hanya simpan small_frame (resized), bukan full frame
        self.prev_small_frame = None
        
    def run(self):
        last_time = time.time()
        
        while not self.stop_event.is_set():
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # FPS Limiter
            elapsed = time.time() - last_time
            if elapsed < self.frame_time:
                time.sleep(self.frame_time - elapsed)
            last_time = time.time()
            
            # Motion detection dengan cached small_frame (OPTIMASI: tidak resize 2x!)
            is_motion, motion_val, small_frame = detect_motion(frame, self.prev_small_frame)
            
            # Simpan small_frame untuk iterasi berikutnya (bukan frame full resolution!)
            self.prev_small_frame = small_frame
            
            # Buang frame lama dari queue
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                pass
            
            self.frame_queue.put({
                'frame': frame,
                'motion': motion_val,
                'timestamp': time.time()
            })

def frame_capture_thread(cap, frame_queue, stop_event, max_fps, cfg):
    """Wrapper untuk thread capture frame."""
    worker = FrameCaptureWorker(cap, frame_queue, stop_event, max_fps, cfg)
    worker.run()

# ─────────────────────────────────────────────────────────────
# DETECTION TRACKING (FREEZE & MAINTAIN)
# ─────────────────────────────────────────────────────────────
def boxes_overlap(box1, box2, iou_threshold=0.3):
    """Check apakah 2 bounding boxes overlap (IoU > threshold)."""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Calculate intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i < x1_i or y2_i < y1_i:
        return False  # No intersection
    
    inter_area = (x2_i - x1_i) * (y2_i - y1_i)
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - inter_area
    
    iou = inter_area / union_area if union_area > 0 else 0
    return iou > iou_threshold

def track_detections(new_detections, frozen_detections, cfg, is_worker_response=True):
    """
    Track detections: maintain frozen ones jika face masih terlihat.
    
    Args:
        new_detections: hasil baru dari detection_thread (bisa kosong)
        frozen_detections: {'detections': [...], 'freeze_count': X}
        cfg: config dengan detection_freeze_frames
        is_worker_response: True jika response dari worker, False jika queue.Empty
    
    Returns:
        Merged detections (frozen + new)
    """
    # PENTING: Jika queue empty (bukan response dari worker),
    # jangan kurangi freeze_count - cukup maintain frozen
    if not is_worker_response:
        if frozen_detections['freeze_count'] > 0:
            return frozen_detections['detections']
        return []
    
    # Ini adalah response dari worker
    if not new_detections:
        # Worker mengirim "no detection" - kurangi freeze_count
        if frozen_detections['freeze_count'] > 0:
            frozen_detections['freeze_count'] -= 1
            if frozen_detections['freeze_count'] <= 0:
                return []  # Clear setelah cooldown
            return frozen_detections['detections']
        return []
    
    # Ada detection baru dari worker - cek apakah match dengan frozen
    matched = False
    for new_det in new_detections:
        for frozen_det in frozen_detections['detections']:
            if boxes_overlap(new_det['box'], frozen_det['box']):
                matched = True
                break
    
    if matched:
        # Face masih ada -> freeze/maintain
        frozen_detections['detections'] = new_detections
        frozen_detections['freeze_count'] = cfg['detection_freeze_frames']
        return new_detections
    else:
        # Face baru atau hilang -> update dengan baru
        frozen_detections['detections'] = new_detections
        frozen_detections['freeze_count'] = cfg['detection_freeze_frames']
        return new_detections

# ─────────────────────────────────────────────────────────────
# DETECTION THREAD (ASYNC)
# ─────────────────────────────────────────────────────────────
def detection_thread(yolo_model, rec_model, face_database, cfg, 
                     input_queue, result_queue, stop_event, frozen_detections_ref=None):
    """Thread untuk detection/recognition. Adaptive motion + always detects if face present."""
    motion_threshold = cfg["motion_threshold"]
    
    while not stop_event.is_set():
        try:
            data = input_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        
        frame = data['frame']
        motion = data['motion']
        
        # ADAPTIVE LOGIC:
        # - Jika sudah ada face frozen (freeze_count > 0), ALWAYS deteksi
        # - Jika tidak ada face frozen, hanya deteksi kalau motion > threshold (hemat daya)
        has_frozen_face = (frozen_detections_ref is not None and 
                          frozen_detections_ref.get('freeze_count', 0) > 0)
        
        should_detect = (motion >= motion_threshold) or has_frozen_face
        
        if not should_detect:
            # Tidak ada motion & tidak ada frozen face - skip detection
            result_queue.put({
                'detections': [],
                'motion': motion,
                'processed': False
            })
            # ✨ OPTIMASI: Berikan CPU kesempatan bernafas sejenak
            time.sleep(0.1)
            continue
        
        # Ada motion atau ada face yang sedang ditrack - lakukan detection
        try:
            # Pass previous detections agar recognize hanya untuk face BARU
            prev_dets = frozen_detections_ref.get('detections', []) if frozen_detections_ref else None
            detections = detect_and_recognize(frame, yolo_model, rec_model, face_database, cfg, prev_detections=prev_dets)
            result_queue.put({
                'detections': detections,
                'motion': motion,
                'processed': True
            })
        except Exception as e:
            print(f"\n[DETECTION ERROR] {e}")
            result_queue.put({
                'detections': [],
                'motion': motion,
                'processed': False
            })
        
        # ✨ OPTIMASI: Berikan CPU kesempatan bernafas setelah selesai 1 frame
        # Ini memaksa thread detection untuk tidak consume CPU 100% terus-menerus
        time.sleep(0.1)

# ─────────────────────────────────────────────────────────────
# LOAD MODELS (OPENVINO INTEGRATION)
# ─────────────────────────────────────────────────────────────
def load_models(cfg: dict):
    import torch
    from ultralytics import YOLO
    #import openvino.runtime as ov
    import openvino as ov

    core = ov.Core()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n{'='*60}")
    print(f"  🖥️  Device : {device} (OpenVINO will auto-optimize)")
    print(f"{'='*60}")

    # ── 1. YOLO (OpenVINO) ──
    yolo_path = Path(cfg["yolo_model_path"])
    print(f"\n⏳ Loading YOLOv11: {yolo_path} ...")
    yolo_model = YOLO(str(yolo_path), task="detect")
    print(f"✅ YOLOv11 loaded!")

    # ── 2. ArcFace (OpenVINO Native) ──
    arcface_path = Path(cfg["arcface_model_path"])
    if not arcface_path.exists():
        print(f"\n❌ Model ArcFace XML tidak ditemukan: {arcface_path}")
        sys.exit(1)

    print(f"\n⏳ Loading ArcFace OpenVINO: {arcface_path} ...")

    ov_model = core.read_model(str(arcface_path))
    print(f"🔎 Model Diagnosis - Inputs: {len(ov_model.inputs)}, Outputs: {len(ov_model.outputs)}")

    rec_model = core.compile_model(ov_model, "CPU")
    print(f"✅ ArcFace loaded!")

    core = ov.Core()
    # rec_model = core.compile_model(str(arcface_path), "CPU") # Paksa CPU untuk OpenVINO
    # print(f"✅ ArcFace loaded via OpenVINO!")

    # ── 3. Face Database ──
    db_path = Path(cfg["face_database_path"])
    if not db_path.exists():
        print(f"\n❌ Face database tidak ditemukan: {db_path}")
        sys.exit(1)

    print(f"\n⏳ Loading face database: {db_path} ...")
    with open(db_path, 'rb') as f:
        face_database = pickle.load(f)

    embs = face_database['embeddings']
    if not isinstance(embs, np.ndarray):
        face_database['embeddings'] = np.array(embs)
    if face_database['embeddings'].ndim == 1:
        face_database['embeddings'] = face_database['embeddings'].reshape(1, -1)

    names = face_database['names']
    unique_names = list(set(names))
    print(f"✅ Face database loaded!")
    print(f"  Persons    : {unique_names}")
    print(f"  Embeddings : {len(names)}")

    print(f"\n{'='*60}\n")
    return yolo_model, rec_model, face_database, device

# ─────────────────────────────────────────────────────────────
# RECOGNIZE FACE (ARCFACE INFERENCE)
# ─────────────────────────────────────────────────────────────
def recognize_face(face_aligned, rec_model, face_database, recognition_threshold):
    """Lakukan ArcFace inference untuk recognize wajah."""
    try:
        # 1. Preprocessing (Standard ArcFace: 112x112, RGB, Mean Subtraction)
        face_resized = cv2.resize(face_aligned, (112, 112))
        face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
        
        # Normalisasi: Skala [-1, 1]
        blob = (face_rgb.astype(np.float32) - 127.5) / 127.5
        blob = np.expand_dims(blob, 0)        # [1, 112, 112, 3]
        blob = np.transpose(blob, (0, 3, 1, 2)) # [1, 3, 112, 112] (NCHW)

        # 2. Inferensi OpenVINO
        infer_request = rec_model.create_infer_request()
        infer_request.infer(inputs={0: blob})
        
        # Ambil output tensor pertama (Index 0)
        output_tensor = infer_request.get_output_tensor(0)
        emb = output_tensor.data.flatten()
        
        # 3. L2 Normalization
        norm = np.linalg.norm(emb)
        if norm > 1e-6:
            emb = emb / norm

        # 4. Perbandingan dengan Database
        embeddings_db = face_database['embeddings']
        
        # Hitung Similarity
        sims = np.dot(embeddings_db, emb)
        idx  = int(np.argmax(sims))
        sim  = float(sims[idx])

        if sim >= recognition_threshold:
            name = face_database['names'][idx]
            conf = sim
        else:
            name, conf = "Unknown", sim
            
    except Exception as e:
        print(f"\n[RECOGNIZE ERROR] {e}")
        name, conf = "Error", 0.0
    
    return name, conf

# ─────────────────────────────────────────────────────────────
# MATCH DETECTED FACES (IoU-based tracking)
# ─────────────────────────────────────────────────────────────
def match_face_detections(new_boxes, prev_detections, iou_threshold=0.3):
    """
    Match new detected boxes dengan previous detections menggunakan IoU.
    
    Returns:
        Dictionary mapping new box index ke previous detection (atau None jika face baru)
    """
    matches = {}
    
    if not prev_detections:
        # Tidak ada previous detections, semua face baru
        for i in range(len(new_boxes)):
            matches[i] = None
        return matches
    
    # Try match setiap new box dengan prev detection
    for i, new_box in enumerate(new_boxes):
        best_match = None
        best_iou = 0
        
        for prev_det in prev_detections:
            prev_box = prev_det['box']
            if boxes_overlap(new_box, prev_box, iou_threshold):
                # Calculate actual IoU untuk match yang paling baik
                x1_1, y1_1, x2_1, y2_1 = new_box
                x1_2, y1_2, x2_2, y2_2 = prev_box
                
                inter_x1 = max(x1_1, x1_2)
                inter_y1 = max(y1_1, y1_2)
                inter_x2 = min(x2_1, x2_2)
                inter_y2 = min(y2_1, y2_2)
                
                if inter_x2 > inter_x1 and inter_y2 > inter_y1:
                    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
                    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
                    union_area = box1_area + box2_area - inter_area
                    iou = inter_area / union_area if union_area > 0 else 0
                    
                    if iou > best_iou:
                        best_iou = iou
                        best_match = prev_det
        
        matches[i] = best_match
    
    return matches

# ─────────────────────────────────────────────────────────────
# DETECT & RECOGNIZE (OPTIMIZED - Only recognize new faces)
# ─────────────────────────────────────────────────────────────
def detect_and_recognize(frame, yolo_model, rec_model, face_database, cfg, prev_detections=None):
    """
    Deteksi YOLO + Recognize hanya untuk face BARU.
    Face yang masih ada (overlap dengan frame sebelumnya) reuse recognition result lama.
    
    MEMORY OPTIMIZATION:
    - CLAHE HANYA pada crop wajah kecil (avg 192x212 pixels)
    - Frame utama (640x480) tetap ORIGINAL untuk display
    - Temporary arrays (diff, small_frames) langsung di-GC
    - In-place operations pada CLAHE L channel
    - Crop wajah di-allocate once, reuse untuk CLAHE & alignment
    """
    yolo_threshold        = cfg["yolo_threshold"]
    recognition_threshold = cfg["recognition_threshold"]

    results = yolo_model.predict(frame, verbose=False, conf=yolo_threshold, iou=0.4, device='cpu')
    detections = [] 
    new_boxes = []

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        kps_all = result.keypoints.xy.cpu().numpy() if result.keypoints is not None else None

        # Collect all boxes dulu
        for i, bbox in enumerate(boxes):
            x1, y1, x2, y2 = map(int, bbox)
            new_boxes.append((x1, y1, x2, y2))

        # Match dengan previous detections
        matches = match_face_detections(new_boxes, prev_detections or [], iou_threshold=0.3)

        for i, bbox in enumerate(boxes):
            x1, y1, x2, y2 = map(int, bbox)
            matched_prev = matches[i]
            # Jika prev detection adalah "Unknown", treat sebagai face BARU (selalu recognize ulang)
            # TAPI: Tambah cooldown - recognize Unknown hanya setiap 5 frame (bukan setiap frame)
            # Hanya reuse jika prev adalah KNOWN PERSON
            should_force_recognize = (matched_prev is not None and 
                                     matched_prev['name'] in ("Unknown", "Error") and
                                     matched_prev.get('unknown_skip_count', 0) > 0)
            
            is_new_face = (matched_prev is None) or (matched_prev['name'] in ("Unknown", "Error")) and not should_force_recognize
            
            # Tambahkan padding untuk hasil crop wajah yang lebih baik
            h, w = frame.shape[:2]
            pad_x = int((x2 - x1) * 0.1)
            pad_y = int((y2 - y1) * 0.1)
            x1_p = max(0, x1 - pad_x)
            y1_p = max(0, y1 - pad_y)
            x2_p = min(w, x2 + pad_x)
            y2_p = min(h, y2 + pad_y)

            face = frame[y1_p:y2_p, x1_p:x2_p]
            
            if face.size == 0:
                name, conf = "Unknown", 0.0
                unknown_skip_count = 5
            elif is_new_face:
                # FACE BARU atau UNKNOWN sebelumnya - lakukan recognize
                face_clahe = apply_clahe(face)
                
                if kps_all is not None and len(kps_all) > i:
                    face_aligned = align_face(face_clahe, kps_all[i], x1_p, y1_p)
                else:
                    face_aligned = face_clahe

                name, conf = recognize_face(face_aligned, rec_model, face_database, recognition_threshold)
                unknown_skip_count = 5 if name == "Unknown" else 0
                
            else:
                # FACE LAMA yang KNOWN PERSON atau Unknown dengan cooldown - reuse hasil lama
                name = matched_prev['name']
                conf = matched_prev['conf']
                # Decrease cooldown counter untuk Unknown
                unknown_skip_count = max(0, matched_prev.get('unknown_skip_count', 0) - 1)

            detections.append({
                "box": (x1, y1, x2, y2),
                "name": name,
                "conf": conf,
                "is_new": is_new_face,  # Flag: face ini baru atau reuse?
                "unknown_skip_count": unknown_skip_count  # Cooldown untuk Unknown
            })

    return detections

# ─────────────────────────────────────────────────────────────
# DRAW OVERLAY (CLEAN - HANYA BOUNDING BOX)
# ─────────────────────────────────────────────────────────────
def draw_overlay(frame, detections):
    """Draw hanya bounding box, clean view."""
    annotated = frame.copy()

    for det in detections:
        x1, y1, x2, y2 = det["box"]
        name           = det["name"]
        conf           = det["conf"]

        color = (0, 255, 0) if name not in ("Unknown", "Error") else (0, 165, 255)
        
        # Draw thick bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)

    return annotated

# ─────────────────────────────────────────────────────────────
# DRAW INFO PANEL (PANEL TERPISAH)
# ─────────────────────────────────────────────────────────────
def draw_info_panel(detections, fps, motion, last_processed, cfg, frame_count, panel_height=480):
    """Bikin panel info dengan background hitam."""
    panel_width = 380
    
    # Buat panel dengan height = camera height
    panel = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)
    
    # Header
    cv2.rectangle(panel, (0, 0), (panel_width, 40), (40, 40, 40), -1)
    cv2.putText(panel, "DETECTION INFO", (10, 28),
                cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 255), 2)
    
    y_pos = 60
    line_height = 30
    
    # FPS & Status
    cv2.putText(panel, f"FPS: {fps:.1f}", (15, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 1)
    y_pos += line_height
    
    status_color = (0, 255, 0) if last_processed else (0, 165, 255)
    status_text = "PROCESSING" if last_processed else "IDLE"
    cv2.putText(panel, f"Status: {status_text}", (15, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 1)
    y_pos += line_height
    
    # Motion & Threshold
    threshold_status = "✓" if motion >= cfg["motion_threshold"] else "✗"
    cv2.putText(panel, f"Motion: {motion:.0f} {threshold_status}", (15, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
    y_pos += line_height
    
    cv2.putText(panel, f"Threshold: {cfg['motion_threshold']}", (15, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
    y_pos += line_height
    
    # Frame count
    cv2.putText(panel, f"Frames: {frame_count}", (15, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
    y_pos += 15
    
    # Separator
    cv2.line(panel, (10, y_pos), (panel_width - 10, y_pos), (80, 80, 80), 1)
    y_pos += 20
    
    # Deteksi Results
    cv2.putText(panel, "DETECTIONS:", (15, y_pos),
                cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 165, 0), 1)
    y_pos += 28
    
    if detections:
        for i, det in enumerate(detections):
            if y_pos > panel_height - 40:
                cv2.putText(panel, "...", (15, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
                break
            
            name = det["name"]
            conf = det["conf"]
            
            # Color: hijau jika known, orange jika unknown
            if name not in ("Unknown", "Error"):
                color = (0, 255, 0)
                symbol = "✓"
            else:
                color = (0, 165, 255)
                symbol = "?"
            
            text = f"{symbol} {name}"
            cv2.putText(panel, text, (15, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
            y_pos += 20
            
            conf_text = f"   Conf: {conf:.3f}"
            cv2.putText(panel, conf_text, (15, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)
            y_pos += 24
    else:
        cv2.putText(panel, "[No detections]", (15, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    
    return panel

def main():
    import csv
    from datetime import datetime

    cfg = CONFIG
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"laporan_deteksi_{timestamp_str}.csv"

    csv_header = ["waktu", "nama", "confidence", "threshold", "x1", "y1", "x2", "y2"]
    csv_file = open(csv_filename, mode="w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(csv_header)

    yolo_model, rec_model, face_database, device = load_models(cfg)

    cap = cv2.VideoCapture(cfg["camera_index"])
    if not cap.isOpened():
        print(f"❌ Tidak bisa membuka webcam index {cfg['camera_index']}")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  cfg["camera_width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg["camera_height"])

    print("🎥 Webcam aktif!")
    print("  Mode: ASYNC + ADAPTIVE (motion-based processing)")
    print(f"  FPS Limit: {cfg['max_fps']}")
    print(f"  Motion Threshold: {cfg['motion_threshold']}")
    print("  Tekan  Q  atau  ESC  untuk keluar\n")

    # Setup threading
    stop_event = threading.Event()
    frame_queue = queue.Queue(maxsize=1)
    result_queue = queue.Queue(maxsize=1)
    
    # Start threads
    cap_thread = threading.Thread(
        target=frame_capture_thread,
        args=(cap, frame_queue, stop_event, cfg['max_fps'], cfg),
        daemon=True
    )
    cap_thread.start()

    # Create frozen_detections dict sebagai reference untuk detection thread
    # (agar detection thread bisa check apakah ada face yang sedang ditrack)
    frozen_detections = {
        'detections': [],
        'freeze_count': 0
    }
    
    det_thread = threading.Thread(
        target=detection_thread,
        args=(yolo_model, rec_model, face_database, cfg, frame_queue, result_queue, stop_event, frozen_detections),
        daemon=True
    )
    det_thread.start()

    # Display loop (main thread)
    fps_history = []
    last_detections = []
    frame_display = None
    last_frame_time = time.time()
    last_motion = 0
    last_processed = False
    frame_count = 0
    fps=0

    try:
        while True:
            # Baca frame dari capture thread
            try:
                frame_data = frame_queue.get_nowait()
                frame_display = frame_data['frame']
                last_motion = frame_data['motion']
                frame_count += 1
            except queue.Empty:
                pass

            # Baca result dari detection thread
            has_worker_result = False
            try:
                result = result_queue.get_nowait()
                raw_detections = result['detections']
                last_processed = result['processed']
                has_worker_result = True
                
                # TRACKING: maintain detections jika face masih ada
                # is_worker_response=True karena ini adalah response dari worker
                last_detections = track_detections(raw_detections, frozen_detections, cfg, is_worker_response=True)
            except queue.Empty:
                # Queue kosong - jangan kurangi freeze_count!
                # is_worker_response=False agar track_detections cukup maintain frozen
                last_detections = track_detections([], frozen_detections, cfg, is_worker_response=False)

            # Hitung FPS berdasarkan display
            now = time.time()
            elapsed_display = now - last_frame_time
            if elapsed_display > 0:
                fps = 1.0 / elapsed_display
                fps_history.append(fps)
            last_frame_time = now

            # Draw UI jika ada frame
            if frame_display is not None:
                # 1. Draw clean camera view (bounding box saja)
                camera_view = draw_overlay(frame_display, last_detections)
                
                # 2. Draw info panel (sama height dengan camera)
                panel_h = camera_view.shape[0]
                info_panel = draw_info_panel(last_detections, fps if elapsed_display > 0 else 0, 
                                            last_motion, last_processed, cfg, frame_count, panel_h)
                
                # 3. Combine: camera + panel side by side
                combined = np.hstack([camera_view, info_panel])
                
                cv2.imshow("YOLOv11 + ArcFace [ASYNC+ADAPTIVE] - Clean UI", combined)

            # Cek keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q'), 27):
                break

            # Write to CSV if detection found
            # ONLY untuk face BARU (is_new=True) - tidak spam untuk face yang sama
            if last_detections and last_processed:
                new_faces = [det for det in last_detections if det.get('is_new', False)]
                
                if new_faces:
                    # Ada face BARU - print ke terminal + CSV
                    for det in new_faces:
                        waktu = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        x1, y1, x2, y2 = det["box"]
                        csv_writer.writerow([
                            waktu, det["name"], f"{det['conf']:.4f}", f"{cfg['recognition_threshold']}", x1, y1, x2, y2
                        ])
                        csv_file.flush()
                    
                    # Print report di terminal (hanya untuk face baru)
                    print(f"\n[NEW FACE DETECTED] Time: {time.strftime('%H:%M:%S')} | Motion: {last_motion:.0f} | FPS: {fps:.1f}")
                    print(f"{'-'*60}")
                    print(f"{'Subject Name':<25} | {'Conf':<8} | {'Thresh':<6}")
                    print(f"{'-'*60}")
                    for det in new_faces:
                        print(f"{det['name']:<25} | {det['conf']:<8.2f} | {cfg['recognition_threshold']:<6}")
                else:
                    # Face lama (reused recognition) - cukup status bar
                    status = "[Processing]" if last_processed else "[Idle]"
                    sys.stdout.write(f"\r🔍 Motion: {last_motion:.0f} | FPS: {fps:.1f} {status} | Frames: {frame_count}    ")
                    sys.stdout.flush()
            else:
                status = "[Processing]" if last_processed else "[Idle]"
                sys.stdout.write(f"\r🔍 Motion: {last_motion:.0f} | FPS: {fps:.1f} {status} | Frames: {frame_count}    ")
                sys.stdout.flush()

    except KeyboardInterrupt:
        print("\n⛔ Interrupted by user")
    finally:
        stop_event.set()
        cap.release()
        cv2.destroyAllWindows()
        csv_file.close()

        if fps_history:
            print(f"\n📊 Selesai!")
            print(f"  Rata-rata FPS : {np.mean(fps_history):.1f}")
            print(f"  Total frame   : {len(fps_history)}")

if __name__ == "__main__":
    main()