import cv2
import time
from deepface import DeepFace

# ==========================================
# 1. FUNGSI WAJIB UNTUK TRACKBAR
# ==========================================
def nothing(x):
    pass

def run_gate_fast():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # ==========================================
    # 2. SETUP GUI WINDOW & TRACKBAR
    # ==========================================
    window_name = "Telyu Gate - High FPS"
    cv2.namedWindow(window_name)
    # Membuat Trackbar: (Nama_Trackbar, Nama_Window, Nilai_Awal, Nilai_Maksimal, Callback)
    cv2.createTrackbar("Anti-Spoof", window_name, 1, 1, nothing)

    FRAME_SKIP = 5
    frame_count = 0
    last_faces = []
    prev_time = 0

    print("Sistem Telyu Gate (High FPS Mode) Aktif...")
    print("Geser slider 'Anti-Spoof' di jendela kamera untuk ON(1) / OFF(0)")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        frame_count += 1
        
        # --- PERHITUNGAN FPS ---
        curr_time = time.time()
        selisih_waktu = curr_time - prev_time
        fps = 1 / selisih_waktu if selisih_waktu > 0 else 0
        prev_time = curr_time

        # --- BACA STATUS SWITCH DARI TRACKBAR ---
        switch_state = cv2.getTrackbarPos("Anti-Spoof", window_name)
        anti_spoofing_enabled = (switch_state == 1) # True jika 1, False jika 0

        # Hanya tembak ke DeepFace setiap 5 frame
        if frame_count % FRAME_SKIP == 0:
            try:
                last_faces = DeepFace.extract_faces(
                    img_path=frame, 
                    enforce_detection=False, 
                    anti_spoofing=anti_spoofing_enabled, # <-- Parameter dinamis mengikuti Switch
                    detector_backend="opencv"
                )
            except Exception as e:
                last_faces = []

        # --- PROSES HASIL DETEKSI ---
        for face in last_faces:
            if face.get('confidence', 0) > 0:
                area = face['facial_area']
                x, y, w, h = area['x'], area['y'], area['w'], area['h']
                
                # Jika Switch Anti-Spoofing ON
                if anti_spoofing_enabled:
                    is_real = face.get('is_real', False)
                    spoof_score = face.get('antispoof_score', 0.0)

                    if is_real:
                        warna = (0, 255, 0) # Hijau
                        teks = f"REAL ({spoof_score:.2f}) - BUKA GATE"
                    else:
                        warna = (0, 0, 255) # Merah
                        teks = f"SPOOF ({spoof_score:.2f}) - GATE DITUTUP"
                
                # Jika Switch Anti-Spoofing OFF
                else:
                    warna = (0, 255, 255) # Kuning
                    teks = "WAJAH TERDETEKSI (Cek Liveness OFF)"

                # Gambar kotak wajah & Status
                cv2.rectangle(frame, (x, y), (x+w, y+h), warna, 2)
                cv2.putText(frame, teks, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, warna, 2)

        # Tampilkan teks FPS
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        cv2.imshow(window_name, frame)
        
        # Keluar dengan menekan tombol 'q' atau 'ESC' (kode 27)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27: 
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_gate_fast()