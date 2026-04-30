# cek_db.py
import pickle

with open('face_database.pkl', 'rb') as f:
    db = pickle.load(f)

print(f"Jumlah data: {len(db['names'])}")
print("\nDaftar pengguna:")
for i, name in enumerate(db['names']):
    print(f"  {i+1}. {name}")