# test_alif.py
import requests
import numpy as np

# Buat embedding dummy (random)
dummy_embedding = np.random.randn(512).tolist()

try:
    res = requests.post(
        "http://100.107.234.128:8000/identify-face",
        json={"embedding": dummy_embedding},
        timeout=5  # ← timeout HARUS di luar json!
    )
    print(f"Status Code: {res.status_code}")
    print(f"Response: {res.json()}")
except requests.exceptions.ConnectionError:
    print("❌ Gagal konek ke server Alif! Cek:")
    print("   1. Apakah server Alif sedang running?")
    print("   2. Apakah IP 100.107.234.128 benar?")
    print("   3. Coba ping: ping 100.107.234.128")
except requests.exceptions.Timeout:
    print("❌ Timeout - Server Alif tidak merespon dalam 5 detik")
except Exception as e:
    print(f"❌ Error: {e}")