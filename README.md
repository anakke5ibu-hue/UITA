# Face Recognition for Parking Gate - UI 

Sistem autentikasi akses gate parkir motor menggunakan pengenalan wajah (Face Recognition) secara stand-alone untuk Tugas Akhir di Telkom University.

## 🚀 Struktur Proyek

Proyek ini terbagi menjadi dua bagian:
1.  **Frontend**: Web dashboard menggunakan Vite + React.
2.  **Recog Engine**: Pemrosesan AI menggunakan YOLOv11 dan ArcFace untuk deteksi dan pengenalan wajah.

## 💡 What I Built

**UITA** is a complete **face-recognition access control system** built for Telkom University's parking gate. Here's what it does:

✅ **Intelligent Dashboard** — Real-time analytics with total users, access statistics, and a searchable authentication log that can be filtered by date range and exported to Excel.

✅ **User Management** — Full CRUD operations with blocking/deletion capabilities and smart search functionality.

✅ **Facial Registration** — Automated face capture powered by **MediaPipe Face Landmarker**, ensuring consistent and high-quality biometric data.

✅ **Gate Terminal** — Real-time facial verification integrated with backend and database, delivering access decisions in under 6 seconds.

## 🛠️ Persyaratan Sistem

*   **Node.js**: Versi 18 atau lebih baru.
*   **Python**: Versi 3.10 atau 3.11 (Sangat disarankan).
*   **Hardware**: Kamera (Webcam) untuk input stream video.

## 👥 Collaborators

| Name | Role | GitHub |
|------|------|--------|
| Muhammad Alif Syawaliana | Project Lead, Backend, Computer Vision Engineer & Database Engineer| [@anakkeSibu-hue](https://github.com/anakkeSibu-hue) |
| Rheira Nisrina Abiyah | Frontend Developer | [@rheira](https://github.com/rheira) |
| Bilal Brilyawan | Anti-Spoofing & Liveness Detection Engineer | [@username3](https://github.com/username3) |

## 🔧 Panduan Instalasi & Penggunaan

### 1. Setup Frontend (Vite + React)
```bash
# Instalasi library frontend
npm install

# Menjalankan server development
npm run dev

# Masuk ke folder backend
cd recog

# Membuat virtual environment baru (Windows)
python -m venv venv_recog

# Mengaktifkan virtual environment (Windows)
.\venv_recog\Scripts\activate

# Update pip ke versi terbaru
python -m pip install --upgrade pip

# Masuk ke folder recog
# Buat venv
# Masuk ke folder venv
# Instalasi semua library yang dibutuhkan
pip install -r requirements.txt



# React + Vite

This template provides a minimal setup to get React working in Vite with HMR and some ESLint rules.

Currently, two official plugins are available:

- [@vitejs/plugin-react](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react) uses [Oxc](https://oxc.rs)
- [@vitejs/plugin-react-swc](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react-swc) uses [SWC](https://swc.rs/)

## React Compiler

The React Compiler is not enabled on this template because of its impact on dev & build performances. To add it, see [this documentation](https://react.dev/learn/react-compiler/installation).

## Expanding the ESLint configuration

If you are developing a production application, we recommend using TypeScript with type-aware lint rules enabled. Check out the [TS template](https://github.com/vitejs/vite/tree/main/packages/create-vite/template-react-ts) for information on how to integrate TypeScript and [`typescript-eslint`](https://typescript-eslint.io) in your project.
