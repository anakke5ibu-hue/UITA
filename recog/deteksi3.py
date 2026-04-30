import os
import sys
import cv2
import numpy as np
import pickle
import time
import warnings
import threading
import queue
import requests
from pathlib import Path
from collections import deque

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────
# KONFIGURASI 
# ─────────────────────────────────────────────────────────────
CONFIG = {
    # Arahkan ke folder hasil export YOLO OpenVINO atau file .xml YOLO
    "yolo_model_path":        "models/yolov11n-face_openvino_model", 
    # Arahkan ke file XML ArcFace buffalo_sc kamu
    "arcface_model_path":     "models/buffalo_sc_rec.xml",           
    # Server untuk recognition (ganti IP sesuai server kamu)
    "server_url":             "http://100.107.234.128:8000/identify-face",
    "yolo_threshold":         0.25,
    "recognition_threshold":  0.5,
    "camera_index":           0,      # ganti ke 1 / 2 jika webcam salah
    "camera_width":           640,
    "camera_height":          480,
    "max_fps":                30,     # Limit FPS untuk efisiensi energi
    "motion_threshold":       1500,   # Threshold motion detection (sum of absolute diff)
    "motion_history":         2,      # Frames sebelumnya untuk comparison
    "detection_freeze_frames": 3,     # Freeze hasil detection selama X frames jika face masih terlihat
    "face_disappear_frames":   5,     # Clear detection jika face hilang selama X frames
    "server_timeout":         2.0,    # Timeout untuk server request (detik)
}

# ─────────────────────────────────────────────────────────────
# FUNGSI TAMBAHAN: CLAHE & ALIGNMENT
# ─────────────────────────────────────────────────────────────
def apply_clahe(img):
    """Memperbaiki kontras menggunakan ruang warna LAB."""
    if img is None or img.size == 0: return img
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

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
# FUNGSI KIRIM KE SERVER
# ─────────────────────────────────────────────────────────────
def check_with_server(embedding, cfg):
    """
    Mengirim embedding ke server untuk recognition.
    
    Args:
        embedding: array embedding 512-dimensi (normalized)
        cfg: config dengan server_url dan timeout
        
    Returns:
        dict dengan hasil recognition atau None jika error
    """
    try:
        payload = {"embedding": embedding.tolist()}
        response = requests.post(cfg["server_url"], json=payload, timeout=cfg["server_timeout"])
        
        if response.status_code == 200:
            result = response.json()
            return result
        else:
            print(f"[SERVER ERROR] Status: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"[SERVER ERROR] Koneksi gagal: {e}")
        return None

# ─────────────────────────────────────────────────────────────
# MOTION DETECTION (ADAPTIVE)
# ─────────────────────────────────────────────────────────────
def detect_motion(frame, prev_frame):
    """Deteksi gerakan dengan membandingkan frame current vs previous."""
    if prev_frame is None:
        return True, 0  # Frame pertama selalu process, motion=0
    
    # Resize untuk performa lebih cepat
    h, w = frame.shape[:2]
    small_frame = cv2.resize(frame, (w//4, h//4))
    small_prev = cv2.resize(prev_frame, (w//4, h//4))
    
    # Convert ke grayscale untuk lebih cepat
    gray_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
    gray_prev = cv2.cvtColor(small_prev, cv2.COLOR_BGR2GRAY)
    
    # Hitung frame difference
    diff = cv2.absdiff(gray_frame, gray_prev)
    motion_sum = np.sum(diff)
    
    return motion_sum > 0, motion_sum  # Return (is_motion, motion_value)

# ─────────────────────────────────────────────────────────────
# FRAME CAPTURE THREAD (ASYNC)
# ─────────────────────────────────────────────────────────────
class FrameCaptureWorker:
    """Worker untuk capture frame dengan motion detection."""
    def __init__(self, cap, frame_queue, stop_event, max_fps, cfg):
        self.cap = cap
        self.frame_queue = frame_queue
        self.stop_event = stop_event
        self.frame_time = 1.0 / max_fps
        self.cfg = cfg
        self.prev_frame = None
        
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
            
            # Motion detection
            is_motion, motion_val = detect_motion(frame, self.prev_frame)
            self.prev_frame = frame.copy()
            
            # Masukkan ke queue
            try:
                self.frame_queue.get_nowait()  # Buang frame lama
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

def track_detections(new_detections, frozen_detections, cfg):
    """
    Track detections: maintain frozen ones jika face masih terlihat.
    
    Args:
        new_detections: hasil baru dari detection_thread
        frozen_detections: {'detections': [...], 'freeze_count': X}
        cfg: config dengan detection_freeze_frames
    
    Returns:
        Merged detections (frozen + new)
    """
    if not new_detections:
        # Tidak ada detection baru
        if frozen_detections['freeze_count'] > 0:
            frozen_detections['freeze_count'] -= 1
            if frozen_detections['freeze_count'] <= 0:
                return []  # Clear setelah cooldown
            return frozen_detections['detections']
        return []
    
    # Ada detection baru - cek apakah match dengan frozen
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
def detection_thread(yolo_model, rec_model, cfg, 
                     input_queue, result_queue, stop_event):
    """Thread untuk detection/recognition. Hanya process jika ada motion."""
    motion_threshold = cfg["motion_threshold"]
    
    while not stop_event.is_set():
        try:
            data = input_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        
        frame = data['frame']
        motion = data['motion']
        
        # ADAPTIVE: Hanya process kalau motion > threshold
        if motion < motion_threshold:
            # Tidak ada motion, skip detection, tapi kirim "no detection"
            result_queue.put({
                'detections': [],
                'motion': motion,
                'processed': False
            })
            continue
        
        # Ada motion, lakukan detection
        try:
            detections = detect_and_recognize(frame, yolo_model, rec_model, cfg)
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

# ─────────────────────────────────────────────────────────────
# LOAD MODELS (OPENVINO INTEGRATION)
# ─────────────────────────────────────────────────────────────
def load_models(cfg: dict):
    import torch
    from ultralytics import YOLO
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
    
    # ── 3. Server Connection ──
    print(f"\n⏳ Checking Server Connection: {cfg['server_url']} ...")
    try:
        test_response = requests.get(cfg["server_url"], timeout=1.0)
        print(f"✅ Server reachable!")
    except Exception as e:
        print(f"⚠️  Server mungkin offline: {e}")
        print(f"   Akan menggunakan mode offline (server-only saat diperlukan)")

    print(f"\n{'='*60}\n")
    return yolo_model, rec_model, device

# ─────────────────────────────────────────────────────────────
# DETECT & RECOGNIZE (WITH SERVER)
# ─────────────────────────────────────────────────────────────
def detect_and_recognize(frame, yolo_model, rec_model, cfg):
    yolo_threshold        = cfg["yolo_threshold"]
    recognition_threshold = cfg["recognition_threshold"]

    # Menambahkan device='cpu' di sini
    results = yolo_model.predict(frame, verbose=False, conf=yolo_threshold, iou=0.4, device='cpu')
    detections = [] 

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        # Ambil keypoints dari YOLO jika modelnya men-support (yolov11n-face)
        kps_all = result.keypoints.xy.cpu().numpy() if result.keypoints is not None else None

        for i, bbox in enumerate(boxes):
            x1, y1, x2, y2 = map(int, bbox)
            
            # Tambahkan padding untuk hasil crop wajah yang lebih baik
            h, w = frame.shape[:2]
            pad_x = int((x2 - x1) * 0.1)
            pad_y = int((y2 - y1) * 0.1)
            x1_p = max(0, x1 - pad_x)
            y1_p = max(0, y1 - pad_y)
            x2_p = min(w, x2 + pad_x)
            y2_p = min(h, y2 + pad_y)

            face = frame[y1_p:y2_p, x1_p:x2_p]
            name, conf = "Unknown", 0.0

            if face.size > 0:
                # 1. Terapkan CLAHE
                face_clahe = apply_clahe(face)
                
                # 2. Terapkan Alignment menggunakan Keypoints YOLO
                if kps_all is not None and len(kps_all) > i:
                    face_aligned = align_face(face_clahe, kps_all[i], x1_p, y1_p)
                else:
                    face_aligned = face_clahe

                # 3. Resize & Preprocessing untuk OpenVINO ArcFace
                face_resized = cv2.resize(face_aligned, (112, 112))
                face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
                
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
                    
                    # 3. L2 Normalization (WAJIB agar np.dot memberikan hasil Cosine Similarity)
                    norm = np.linalg.norm(emb)
                    if norm > 1e-6:
                        emb = emb / norm

                    # 4. Kirim ke Server untuk Recognition
                    server_result = check_with_server(emb, cfg)
                    
                    if server_result and server_result.get("status") == "success":
                        name = server_result["data"]["nama"]
                        conf = server_result["data"]["similarity"]
                    elif server_result:
                        # Server respond tapi tidak match
                        name = "Unknown"
                        conf = 0.0
                    else:
                        # Server error
                        name = "Error"
                        conf = 0.0
                        
                except Exception as e:
                    print(f"\n[DEBUG ERROR] Detail: {e}")
                    name, conf = "Error", 0.0
            detections.append({
                "box": (x1, y1, x2, y2),
                "name": name,
                "conf": conf
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

    yolo_model, rec_model, device = load_models(cfg)

    cap = cv2.VideoCapture(cfg["camera_index"])
    if not cap.isOpened():
        print(f"❌ Tidak bisa membuka webcam index {cfg['camera_index']}")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  cfg["camera_width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg["camera_height"])

    print("🎥 Webcam aktif!")
    print("  Mode: ASYNC + ADAPTIVE (motion-based processing)")
    print("  Recognition: SERVER-BASED")
    print(f"  Server: {cfg['server_url']}")
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

    det_thread = threading.Thread(
        target=detection_thread,
        args=(yolo_model, rec_model, cfg, frame_queue, result_queue, stop_event),
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
    fps = 0.0
    
    # Tracking state
    frozen_detections = {
        'detections': [],
        'freeze_count': 0
    }

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
            try:
                result = result_queue.get_nowait()
                raw_detections = result['detections']
                last_processed = result['processed']
                
                # TRACKING: maintain detections jika face masih ada
                last_detections = track_detections(raw_detections, frozen_detections, cfg)
            except queue.Empty:
                # Maintain frozen detections
                last_detections = track_detections([], frozen_detections, cfg)

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
                
                cv2.imshow("YOLOv11 + ArcFace [ASYNC+ADAPTIVE] + SERVER", combined)

            # Cek keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q'), 27):
                break

            # Write to CSV if detection found
            if last_detections and last_processed:
                for det in last_detections:
                    waktu = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    x1, y1, x2, y2 = det["box"]
                    csv_writer.writerow([
                        waktu, det["name"], f"{det['conf']:.4f}", f"{cfg['recognition_threshold']}", x1, y1, x2, y2
                    ])
                    csv_file.flush()
                    
                # Print report di terminal
                print(f"\n[REPORT] Time: {time.strftime('%H:%M:%S')} | Motion: {last_motion:.0f} | FPS: {fps:.1f}")
                print(f"{'-'*60}")
                print(f"{'Subject Name':<25} | {'Conf':<8} | {'Thresh':<6}")
                print(f"{'-'*60}")
                for det in last_detections:
                    print(f"{det['name']:<25} | {det['conf']:<8.2f} | {cfg['recognition_threshold']:<6}")
                
                last_detections = []  # Reset untuk next detection
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