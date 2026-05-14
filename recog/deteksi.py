# import os
# import cv2
# import numpy as np
# import asyncio
# import json
# import base64
# import requests
# import time
# from fastapi import FastAPI, WebSocket, WebSocketDisconnect
# from fastapi.middleware.cors import CORSMiddleware
# import uvicorn
# from ultralytics import YOLO
# from insightface.app import FaceAnalysis

# # --- KONFIGURASI GABUNGAN (ALIF + WEBSOCKET) ---
# CONFIG = {
#     "yolo_model_path": "yolov11n-face.pt",
#     "server_url": "http://100.107.234.128:8000/identify-face", # IP Server Alif
#     "yolo_threshold": 0.35,
#     "camera_index": 0,
# }

# app = FastAPI()
# app.add_middleware(
#     CORSMiddleware, 
#     allow_origins=["*"], 
#     allow_methods=["*"], 
#     allow_headers=["*"]
# )

# # --- LOAD MODELS (LOGIKA ALIF) ---
# print("🚀 Loading Models...")
# yolo_model = YOLO(CONFIG["yolo_model_path"])
# # Pakai CPUProvider agar tidak error jika tidak ada CUDA
# face_app = FaceAnalysis(name='buffalo_sc', providers=['CPUExecutionProvider'])
# face_app.prepare(ctx_id=-1, det_size=(640, 640))
# rec_model = face_app.models['recognition']
# print("✅ Models Ready!")

# @app.websocket("/ws/detect")
# async def websocket_detect(websocket: WebSocket):
#     await websocket.accept()
#     print("🔗 Web Connected!")
    
#     cap = cv2.VideoCapture(CONFIG["camera_index"])
#     if not cap.isOpened():
#         print("❌ Kamera tidak bisa dibuka!")
#         return

#     try:
#         while True:
#             ret, frame = cap.read()
#             if not ret: break

#             # 1. DETEKSI WAJAH (LOGIKA ALIF)
#             results = yolo_model.predict(frame, verbose=False, conf=CONFIG["yolo_threshold"])
#             faces_list = []
#             status = "idle"
#             user_data = None

#             for result in results:
#                 for bbox in result.boxes.xyxy.cpu().numpy():
#                     x1, y1, x2, y2 = map(int, bbox)
#                     face_crop = frame[y1:y2, x1:x2]
                    
#                     if face_crop.size > 0:
#                         # Identifikasi via Server Alif
#                         face_resized = cv2.resize(face_crop, (112, 112))
#                         face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
#                         emb = rec_model.get_feat(face_rgb)
#                         emb = emb.flatten() / np.linalg.norm(emb)
                        
#                         try:
#                             # Kirim embedding ke IP Tailscale Alif
#                             res = requests.post(CONFIG["server_url"], json={"embedding": emb.tolist()}, timeout=0.8).json()
#                             if res.get("status") == "success":
#                                 status = "granted"
#                                 user_data = {
#                                     "nama": res["data"]["nama"],
#                                     "nim": "Student",
#                                     "program_studi": "Teknik Telekomunikasi"
#                                 }
#                             else:
#                                 status = "denied"
#                         except:
#                             status = "scanning"

#                     faces_list.append({'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2})

#             # 2. ENCODE GAMBAR UNTUK WEB
#             _, buffer = cv2.imencode('.jpg', frame)
#             img_base64 = base64.b64encode(buffer).decode('utf-8')
            
#             # 3. KIRIM DATA KE REACT
#             payload = {
#                 "image": f"data:image/jpeg;base64,{img_base64}",
#                 "status": status,
#                 "face_count": len(faces_list),
#                 "faces": faces_list,
#                 "user": user_data
#             }
            
#             await websocket.send_text(json.dumps(payload))
#             await asyncio.sleep(0.05) # Menjaga FPS agar stabil

#     except WebSocketDisconnect:
#         print("🔌 Web Disconnected")
#     finally:
#         cap.release()

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)