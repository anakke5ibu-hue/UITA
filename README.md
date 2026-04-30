# UI TA - Stand-alone Face Recognition System

Sistem autentikasi akses gate parkir motor menggunakan pengenalan wajah (Face Recognition) secara stand-alone untuk Tugas Akhir di Telkom University.

## 🚀 Struktur Proyek

Proyek ini terbagi menjadi dua bagian:
1.  **Frontend**: Web dashboard menggunakan Vite + React.
2.  **Recog Engine**: Pemrosesan AI menggunakan YOLOv11 dan ArcFace untuk deteksi dan pengenalan wajah.

## 🛠️ Persyaratan Sistem

*   **Node.js**: Versi 18 atau lebih baru.
*   **Python**: Versi 3.10 atau 3.11 (Sangat disarankan).
*   **Hardware**: Kamera (Webcam) untuk input stream video.

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
