# test_sync.py
import cv2
import numpy as np
from pathlib import Path

# Copy fungsi dari kedua file
from crop_faces import apply_gamma_clahe as crop_preprocess
from deteksi_final import apply_gamma_clahe as gate_preprocess

# Test dengan sample face
test_face = cv2.imread("test_img.jpg")
crop_result = crop_preprocess(test_face)
gate_result = gate_preprocess(test_face)

diff = np.abs(crop_result.astype(float) - gate_result.astype(float)).mean()
print(f"Mean difference: {diff:.2f}")  # Harus 0.00
assert np.allclose(crop_result, gate_result), "Preprocessing MISMATCH!"
print("✅ SINKRON!")    