# ═══════════════════════════════════════════════════════════════
# main.py — Backend Face Recognition Gate
# - 1 kamera dibagi ke semua client WebSocket (tidak rebutan)
# - Kalau server Alif online → pakai Alif
# - Kalau server Alif offline → langsung pakai database lokal
# ═══════════════════════════════════════════════════════════════

import cv2
import numpy as np
import pickle
import base64
import asyncio
import json
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import requests
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# ─── Konfigurasi ─────────────────────────────────────────────
SERVER_ALIF_URL       = "http://100.107.234.128:8000/identify-face"
YOLO_THRESHOLD        = 0.4
RECOGNITION_THRESHOLD = 0.60
CAMERA_INDEX          = 0

print("="*60)
print("  🚀 Face Recognition Gate — Backend")
print("="*60)

# ─── Load YOLO ────────────────────────────────────────────────
print("\n📦 Loading YOLO...")
from ultralytics import YOLO
yolo_model = YOLO('yolov11n-face.pt')
print("   ✅ YOLO loaded")

# ─── Load ArcFace ─────────────────────────────────────────────
print("\n📦 Loading ArcFace...")
import insightface
from insightface.app import FaceAnalysis
face_app = FaceAnalysis(
    name='buffalo_l',
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
)
face_app.prepare(ctx_id=0, det_size=(640, 640))
rec_model = face_app.models['recognition']
print("   ✅ ArcFace loaded")

# ─── Load Database Lokal ──────────────────────────────────────
print("\n📦 Loading face database...")
DB_PATH = Path(__file__).parent / 'face_database.pkl'
with open(DB_PATH, 'rb') as f:
    face_database = pickle.load(f)
db_embeddings = np.array(face_database['embeddings'])
print(f"   ✅ Database: {len(face_database['names'])} data")
for n in set(face_database['names']):
    print(f"      - {n}")

print("\n" + "="*60)

# ─── State Kamera (1 kamera dibagi semua client) ──────────────
camera_frame   = None      # frame terakhir dari kamera
camera_result  = None      # hasil deteksi terakhir
camera_lock    = threading.Lock()
connected_clients = set() 
last_denied_time  = 0 # set WebSocket yang sedang konek


# ─── Fungsi Preprocessing (CLAHE) ─────────────────────────────
def apply_clahe(img):
    """
    Memperbaiki kontras wajah menggunakan CLAHE pada ruang warna LAB.
    Sangat berguna untuk menerangkan area mata yang tertutup bayangan (helm/kacamata).
    """
    # Ubah ke LAB agar bisa memproses pencahayaan (L) saja tanpa merusak warna
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Terapkan CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)

    # Gabungkan kembali dan kembalikan ke format BGR
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)


# ─── Fungsi Rekognisi Lokal ───────────────────────────────────
def recognize_local(embedding):
    similarities = np.dot(db_embeddings, embedding)
    best_idx     = int(np.argmax(similarities))
    best_sim     = float(similarities[best_idx])

    print(f"🔍 SIMILARITY: best_sim={best_sim:.4f}, threshold={RECOGNITION_THRESHOLD}")
    print(f"🔍 Best match: {face_database['names'][best_idx]}")

    if best_sim >= RECOGNITION_THRESHOLD:
        raw  = face_database['names'][best_idx]
        nim, nama = raw.split('_', 1) if '_' in raw else ('-', raw)
        return {
            'nama': nama, 'nim': nim,
            'program_studi': 'S1 Teknik Telekomunikasi',
            'confidence': round(best_sim, 4),
            'is_recognized': True,
        }
    return {
        'nama': 'Unknown', 'nim': '-',
        'program_studi': '', 'confidence': round(best_sim, 4),
        'is_recognized': False,
    }

# ─── Fungsi Rekognisi: coba Alif dulu, fallback lokal ─────────
executor = ThreadPoolExecutor(max_workers=2)

def recognize_face(face_img_bgr):
    try:
        # === TERAPKAN CLAHE DI SINI ===
        # Perbaiki kontras potongan wajah sebelum diekstrak fiturnya
        face_enhanced = apply_clahe(face_img_bgr)
        # ==============================

        # Gunakan gambar yang sudah di-enhance (face_enhanced) untuk proses selanjutnya
        face_resized = cv2.resize(face_enhanced, (112, 112))
        face_rgb     = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
        embedding    = rec_model.get_feat(face_rgb).flatten()
        embedding    = embedding / (np.linalg.norm(embedding) + 1e-10)

        # Coba kirim ke server Alif (timeout pendek = 0.5 detik)
        try:
            res = requests.post(
                SERVER_ALIF_URL,
                json={"embedding": embedding.tolist()},
                timeout=5.0
            ).json()

            if res.get("status") == "success":
                print("✅ Terhubung ke Server Alif")
                d = res["data"]
                return {
                    'nama': d.get('nama', 'Unknown'),
                    'nim': d.get('nim', '-'),
                    'program_studi': 'S1 Teknik Telekomunikasi',
                    'confidence': round(float(d.get('similarity', 0)), 4),
                    'is_recognized': True,
                }
        except Exception:
            print("⚠️ Menggunakan database lokal")  # Alif offline → lanjut ke lokal

        # Fallback: database lokal
        return recognize_local(embedding)

    except Exception as e:
        print(f"❌ recognize_face error: {e}")
        return {'nama': 'Error', 'nim': '-', 'program_studi': '',
                'confidence': 0.0, 'is_recognized': False}

# ─── Fungsi Proses 1 Frame ────────────────────────────────────
def process_frame(frame):
    h, w       = frame.shape[:2]
    results    = yolo_model.predict(frame, verbose=False, conf=YOLO_THRESHOLD, iou=0.4)
    faces_data = []
    status     = 'idle'
    user_data  = None

    for result in results:
        for bbox in result.boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = map(int, bbox)
            conf_det = float(result.boxes.conf[0].cpu().numpy())

            pad_x = int((x2 - x1) * 0.1)
            pad_y = int((y2 - y1) * 0.1)
            face_crop = frame[
                max(0, y1-pad_y):min(h, y2+pad_y),
                max(0, x1-pad_x):min(w, x2+pad_x)
            ]
            if face_crop.size == 0:
                continue

            status = 'scanning'
            rec    = recognize_face(face_crop)

            faces_data.append({
                'x1': int(x1), 'y1': int(y1),
                'x2': int(x2), 'y2': int(y2),
                'confidence': round(conf_det, 3),
                'name': rec['nama'],
                'is_recognized': rec['is_recognized'],
                'frameWidth': int(w),
                'frameHeight': int(h),
            })

            if user_data is None:
                if rec['is_recognized']:
                    status    = 'granted'
                    user_data = {
                        'nama': rec['nama'],
                        'nim': rec['nim'],
                        'program_studi': rec['program_studi'],
                    }
                else:
                    status = 'denied'

    _, buffer  = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    return {
        'image'     : f'data:image/jpeg;base64,{img_base64}',
        'status'    : status,
        'face_count': len(faces_data),
        'faces'     : faces_data,
        'user'      : user_data,
    }

# ─── Thread Kamera (jalan terus di background) ────────────────
def camera_thread():
    global camera_result
    print("\n📹 Camera thread starting...")

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("❌ Kamera tidak bisa dibuka!")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    print("📹 Kamera aktif!")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Gagal baca frame")
            break

        # Proses frame hanya kalau ada client yang konek
        if connected_clients:
            result = process_frame(frame)
            with camera_lock:
                camera_result = result
        else:
            # Tidak ada client, kirim frame kosong aja
            _, buffer  = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            with camera_lock:
                camera_result = {
                    'image'     : f'data:image/jpeg;base64,{img_base64}',
                    'status'    : 'idle',
                    'face_count': 0,
                    'faces'     : [],
                    'user'      : None,
                }

    cap.release()
    print("📹 Kamera dimatikan")

# ─── FastAPI ──────────────────────────────────────────────────
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    # Jalankan thread kamera saat server start
    t = threading.Thread(target=camera_thread, daemon=True)
    t.start()
    print("✅ Camera thread started")

@app.get("/")
def root():
    return {"status": "Face Recognition Gate Backend Running"}

@app.websocket("/ws/detect")
async def websocket_detect(websocket: WebSocket):
    await websocket.accept()
    connected_clients.add(websocket)
    print(f"🔗 Client konek ({len(connected_clients)} total)")

    try:
        while True:
            # Ambil hasil kamera terbaru
            with camera_lock:
                result = camera_result

            if result:
                await websocket.send_text(json.dumps(result))

            # ~15 FPS
            await asyncio.sleep(0.067)

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"❌ WebSocket error: {e}")
    finally:
        connected_clients.discard(websocket)
        print(f"🔌 Client putus ({len(connected_clients)} tersisa)")

# ─── Main ─────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n🌐 Server:")
    print("   http://localhost:8000")
    print("   ws://localhost:8000/ws/detect")
    print("\n   Buka: http://localhost:5173/gate")
    print("   atau: http://localhost:5173/dashboard")
    print("\n   Ctrl+C untuk berhenti")
    print("="*60)

    uvicorn.run(app, host="0.0.0.0", port=8000)