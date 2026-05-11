import { useState, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import logo from '../assets/logo-telkom.png'

// ═══════════════════════════════════════════════════════════════
// KONFIGURASI SERVER
// ═══════════════════════════════════════════════════════════════
const SERVER_URL = 'http://100.89.141.47:8001'

// ─── Kirim foto ke server untuk diproses embedding ───────────
const registerUserWithPhotos = async (nama, nim, photoBlobs) => {
  const formData = new FormData()
  formData.append('nama', nama)
  formData.append('nim', nim)
  photoBlobs.forEach((blob, i) => {
    formData.append('photos', blob, `photo_${i}.jpg`)
  })
  try {
    const res = await fetch(`${SERVER_URL}/register_from_photos`, {
      method: 'POST',
      body: formData,
    })
    const data = await res.json()
    return { ok: res.ok, message: data.detail || data.message || 'Berhasil' }
  } catch (e) {
    console.error('Gagal registrasi:', e)
    return { ok: false, message: 'Gagal terhubung ke server. Pastikan server berjalan di localhost:8001' }
  }
}

// ═══════════════════════════════════════════════════════════════
// MAIN COMPONENT
// ═══════════════════════════════════════════════════════════════
function RegistrasiPage() {
  const navigate = useNavigate()

  // Mode: 'select' | 'kamera' | 'upload'
  const [mode, setMode] = useState('select')

  const [nama, setNama] = useState('')
  const [nim, setNim] = useState('')
  const [photos, setPhotos] = useState([]) // { blob, dataUrl }[]
  const [cameraOpen, setCameraOpen] = useState(false)
  const [capturing, setCapturing] = useState(false)
  const [countdown, setCountdown] = useState(null)
  const [status, setStatus] = useState(null) // { type: 'success'|'error'|'loading'|'info', msg }

  const videoRef = useRef(null)
  const streamRef = useRef(null)
  const captureIntervalRef = useRef(null)
  const cdIntervalRef = useRef(null)
  const fileInputRef = useRef(null)

  // ─── Reset semua state ────────────────────────────────────
  const resetAll = () => {
    closeCamera()
    setNama('')
    setNim('')
    setPhotos([])
    setStatus(null)
  }

  const handleModeSelect = (selectedMode) => {
    resetAll()
    setMode(selectedMode)
  }

  // ─── Buka Kamera ──────────────────────────────────────────
  const openCamera = async () => {
    if (!nama.trim() || !nim.trim()) {
      setStatus({ type: 'error', msg: 'Nama dan NIM harus diisi terlebih dahulu!' })
      return
    }
    setStatus(null)
    setPhotos([])
    setCameraOpen(true)
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true })
      streamRef.current = stream
      setTimeout(() => {
        if (videoRef.current) videoRef.current.srcObject = stream
      }, 100)
    } catch (e) {
      setCameraOpen(false)
      setStatus({ type: 'error', msg: 'Gagal membuka kamera: ' + e.message })
    }
  }

  // ─── Tutup Kamera ─────────────────────────────────────────
  const closeCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(t => t.stop())
      streamRef.current = null
    }
    if (captureIntervalRef.current) {
      clearInterval(captureIntervalRef.current)
      captureIntervalRef.current = null
    }
    if (cdIntervalRef.current) {
      clearInterval(cdIntervalRef.current)
      cdIntervalRef.current = null
    }
    setCameraOpen(false)
    setCapturing(false)
    setCountdown(null)
  }

  // ─── Auto Capture 10 Foto (interval 3 detik) ──────────────
  const startAutoCapture = () => {
    if (!videoRef.current) return
    setCapturing(true)
    setPhotos([])

    let cd = 3
    setCountdown(cd)

    cdIntervalRef.current = setInterval(() => {
      cd -= 1
      if (cd > 0) {
        setCountdown(cd)
      } else {
        clearInterval(cdIntervalRef.current)
        cdIntervalRef.current = null
        setCountdown('📸')

        let count = 0
        captureIntervalRef.current = setInterval(() => {
          if (!videoRef.current) return

          const canvas = document.createElement('canvas')
          canvas.width = videoRef.current.videoWidth
          canvas.height = videoRef.current.videoHeight
          canvas.getContext('2d').drawImage(videoRef.current, 0, 0)
          const dataUrl = canvas.toDataURL('image/jpeg', 0.9)

          canvas.toBlob((blob) => {
            count += 1
            const currentCount = count
            setPhotos(prev => {
              const next = [...prev, { blob, dataUrl }]
              if (currentCount >= 10) {
                clearInterval(captureIntervalRef.current)
                captureIntervalRef.current = null
                setCapturing(false)
                setCountdown(null)
                closeCamera()
                setStatus({ type: 'info', msg: '10 foto berhasil diambil! Klik "Proses & Simpan" untuk mendaftar.' })
              }
              return next
            })
          }, 'image/jpeg', 0.9)
        }, 3000)
      }
    }, 1000)
  }

  // ─── Upload Foto dari File ────────────────────────────────
  const handleFileUpload = (e) => {
    const files = Array.from(e.target.files)
    if (files.length === 0) return

    if (files.length < 2) {
      setStatus({ type: 'error', msg: 'Upload minimal 5 foto untuk hasil embedding yang baik.' })
      return
    }
    if (files.length > 10) {
      setStatus({ type: 'error', msg: 'Maksimal 10 foto yang bisa diupload sekaligus.' })
      return
    }

    setStatus({ type: 'info', msg: `Memproses ${files.length} foto...` })
    setPhotos([])

    const loaded = []
    files.forEach((file, i) => {
      const reader = new FileReader()
      reader.onload = (ev) => {
        const dataUrl = ev.target.result
        // Convert dataUrl ke blob
        fetch(dataUrl)
          .then(r => r.blob())
          .then(blob => {
            loaded.push({ blob, dataUrl, index: i })
            if (loaded.length === files.length) {
              // Sort by original index
              loaded.sort((a, b) => a.index - b.index)
              setPhotos(loaded.map(({ blob, dataUrl }) => ({ blob, dataUrl })))
              setStatus({ type: 'info', msg: `${files.length} foto siap! Klik "Proses & Simpan" untuk mendaftar.` })
            }
          })
      }
      reader.readAsDataURL(file)
    })

    // Reset input biar bisa upload ulang file yang sama
    e.target.value = ''
  }

  // ─── Submit Registrasi ─────────────────────────────────────
  const handleSubmit = async () => {
    if (!nama.trim() || !nim.trim()) {
      setStatus({ type: 'error', msg: 'Nama dan NIM harus diisi!' })
      return
    }
    if (photos.length === 0) {
      setStatus({ type: 'error', msg: mode === 'kamera' ? 'Ambil foto wajah terlebih dahulu!' : 'Upload foto wajah terlebih dahulu!' })
      return
    }
    setStatus({ type: 'loading', msg: 'Mengirim data ke server & memproses embedding wajah...' })
    const blobs = photos.map(p => p.blob)
    const result = await registerUserWithPhotos(nama.trim(), nim.trim(), blobs)
    if (result.ok) {
      setStatus({ type: 'success', msg: `✅ Pendaftaran berhasil! Data wajah ${nama} tersimpan di server.` })
      setNama('')
      setNim('')
      setPhotos([])
    } else {
      setStatus({ type: 'error', msg: result.message })
    }
  }

  // ─── RENDER: Halaman Pilihan Mode ─────────────────────────
  const renderSelectMode = () => (
    <div className="flex flex-col gap-5">
      {/* Header */}
      <div className="bg-gray-900 border border-gray-800 rounded-2xl p-5">
        <p className="font-semibold text-white">Registrasi Pengguna Baru</p>
        <p className="text-gray-500 text-xs mt-0.5">Daftarkan wajah user baru ke sistem Face Recognition Gate</p>
      </div>

      {/* Pilihan Mode */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        {/* Mode: Foto Langsung */}
        <button
          onClick={() => handleModeSelect('kamera')}
          className="group bg-gray-900 border border-gray-800 hover:border-red-500/50 rounded-2xl p-6 flex flex-col items-center gap-4 transition-all duration-300 hover:bg-gray-800/50 text-left"
        >
          <div className="w-16 h-16 rounded-2xl bg-red-900/30 border border-red-500/30 flex items-center justify-center text-3xl group-hover:scale-110 transition-transform duration-300">
            📷
          </div>
          <div className="text-center">
            <p className="text-white font-semibold text-sm">Foto Langsung</p>
            <p className="text-gray-500 text-xs mt-1.5 leading-relaxed">
              Ambil 10 foto secara otomatis menggunakan kamera secara langsung (interval 3 detik)
            </p>
          </div>
          <span className="text-xs font-medium px-3 py-1 rounded-full bg-red-900/40 text-red-400 border border-red-500/30">
            Gunakan Kamera →
          </span>
        </button>

        {/* Mode: Upload Foto */}
        <button
          onClick={() => handleModeSelect('upload')}
          className="group bg-gray-900 border border-gray-800 hover:border-blue-500/50 rounded-2xl p-6 flex flex-col items-center gap-4 transition-all duration-300 hover:bg-gray-800/50 text-left"
        >
          <div className="w-16 h-16 rounded-2xl bg-blue-900/30 border border-blue-500/30 flex items-center justify-center text-3xl group-hover:scale-110 transition-transform duration-300">
            📁
          </div>
          <div className="text-center">
            <p className="text-white font-semibold text-sm">Upload Foto</p>
            <p className="text-gray-500 text-xs mt-1.5 leading-relaxed">
              Upload 5–10 foto wajah dari file yang sudah tersimpan di perangkat
            </p>
          </div>
          <span className="text-xs font-medium px-3 py-1 rounded-full bg-blue-900/40 text-blue-400 border border-blue-500/30">
            Pilih dari File →
          </span>
        </button>
      </div>
    </div>
  )

  // ─── RENDER: Form Input (dipakai kedua mode) ──────────────
  const renderFormInput = () => (
    <div className="bg-gray-900 border border-gray-800 rounded-2xl p-5 flex flex-col gap-4">
      <div>
        <label className="text-gray-400 text-xs uppercase tracking-wider mb-1.5 block">Nama Lengkap</label>
        <input
          type="text"
          value={nama}
          onChange={e => setNama(e.target.value)}
          placeholder="Masukkan nama lengkap..."
          disabled={capturing}
          className="w-full bg-gray-800 border border-gray-700 text-white text-sm rounded-lg px-4 py-3 outline-none focus:ring-2 focus:ring-red-600/50 placeholder-gray-600 disabled:opacity-50"
        />
      </div>
      <div>
        <label className="text-gray-400 text-xs uppercase tracking-wider mb-1.5 block">NIM</label>
        <input
          type="text"
          value={nim}
          onChange={e => setNim(e.target.value)}
          placeholder="Masukkan NIM..."
          disabled={capturing}
          className="w-full bg-gray-800 border border-gray-700 text-white text-sm rounded-lg px-4 py-3 outline-none focus:ring-2 focus:ring-red-600/50 placeholder-gray-600 disabled:opacity-50"
        />
      </div>
    </div>
  )

  // ─── RENDER: Thumbnail foto ────────────────────────────────
  const renderThumbnails = () => (
    photos.length > 0 && (
      <div className="grid grid-cols-5 gap-2">
        {photos.map((p, i) => (
          <div key={i} className="relative rounded-lg overflow-hidden aspect-square bg-gray-800">
            <img src={p.dataUrl} alt={`foto-${i + 1}`} className="w-full h-full object-cover" />
            <div className="absolute bottom-0 left-0 right-0 bg-black/60 text-center text-[10px] text-gray-300 py-0.5">
              #{i + 1}
            </div>
          </div>
        ))}
      </div>
    )
  )

  // ─── RENDER: Mode Kamera ───────────────────────────────────
  const renderModeKamera = () => (
    <div className="flex flex-col gap-5">
      {/* Header + back */}
      <div className="bg-gray-900 border border-gray-800 rounded-2xl p-5">
        <div className="flex items-center justify-between">
          <div>
            <p className="font-semibold text-white">📷 Registrasi — Foto Langsung</p>
            <p className="text-gray-500 text-xs mt-0.5">Isi data lalu ambil 10 foto otomatis via kamera</p>
          </div>
          <button
            onClick={() => { resetAll(); setMode('select') }}
            className="text-gray-500 hover:text-white text-xs px-3 py-1.5 rounded-lg bg-gray-800 border border-gray-700 transition-all"
          >
            ← Ganti Metode
          </button>
        </div>
      </div>

      {/* Form */}
      {renderFormInput()}

      {/* Kamera Section */}
      <div className="bg-gray-900 border border-gray-800 rounded-2xl p-5 flex flex-col gap-4">
        <div className="flex items-center justify-between">
          <div>
            <p className="text-white text-sm font-medium">Pengambilan Foto Wajah</p>
            <p className="text-gray-500 text-xs mt-0.5">
              {photos.length > 0
                ? `${photos.length} / 10 foto terambil`
                : 'Diperlukan 10 foto otomatis (interval 3 detik)'}
            </p>
          </div>
          {photos.length > 0 && (
            <span className="text-xs font-bold px-3 py-1 rounded-full bg-green-900/40 text-green-400 border border-green-500/30">
              ✓ {photos.length} foto
            </span>
          )}
        </div>

        {/* Video Preview */}
        {cameraOpen && (
          <div className="relative rounded-xl overflow-hidden bg-black">
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              className="w-full rounded-xl"
              style={{ maxHeight: '360px', objectFit: 'cover' }}
            />
            {countdown !== null && (
              <div className="absolute inset-0 flex items-end justify-center pb-6 pointer-events-none">
                <div className={`text-2xl font-bold px-5 py-2 rounded-2xl ${
                  countdown === '📸'
                    ? 'bg-green-900/80 text-green-300'
                    : 'bg-black/70 text-yellow-300'
                }`}>
                  {countdown === '📸' ? `📸 Mengambil foto... (${photos.length}/10)` : `Bersiap: ${countdown}s`}
                </div>
              </div>
            )}
            {capturing && (
              <div className="absolute bottom-0 left-0 right-0 h-1 bg-gray-800">
                <div
                  className="h-1 bg-red-500 transition-all duration-500"
                  style={{ width: `${(photos.length / 10) * 100}%` }}
                />
              </div>
            )}
          </div>
        )}

        {renderThumbnails()}

        {/* Tombol Kamera */}
        <div className="flex gap-3">
          {!cameraOpen ? (
            <button
              onClick={openCamera}
              disabled={capturing}
              className="flex-1 flex items-center justify-center gap-2 bg-gray-800 hover:bg-gray-700 border border-gray-700 text-white text-sm font-medium px-4 py-3 rounded-xl transition-all"
            >
              📷 Buka Kamera
            </button>
          ) : (
            <>
              {!capturing && (
                <button
                  onClick={startAutoCapture}
                  className="flex-1 flex items-center justify-center gap-2 bg-red-700 hover:bg-red-600 text-white text-sm font-semibold px-4 py-3 rounded-xl transition-all"
                >
                  ▶ Mulai Auto-Capture (10 foto)
                </button>
              )}
              <button
                onClick={closeCamera}
                disabled={capturing}
                className="flex items-center justify-center gap-2 bg-gray-800 hover:bg-gray-700 border border-gray-700 text-gray-400 text-sm px-4 py-3 rounded-xl transition-all disabled:opacity-40"
              >
                ✕ Tutup
              </button>
            </>
          )}
        </div>
      </div>

      {renderStatusAndActions()}
    </div>
  )

  // ─── RENDER: Mode Upload ───────────────────────────────────
  const renderModeUpload = () => (
    <div className="flex flex-col gap-5">
      {/* Header + back */}
      <div className="bg-gray-900 border border-gray-800 rounded-2xl p-5">
        <div className="flex items-center justify-between">
          <div>
            <p className="font-semibold text-white">📁 Registrasi — Upload Foto</p>
            <p className="text-gray-500 text-xs mt-0.5">Isi data lalu upload 5–10 foto wajah dari file</p>
          </div>
          <button
            onClick={() => { resetAll(); setMode('select') }}
            className="text-gray-500 hover:text-white text-xs px-3 py-1.5 rounded-lg bg-gray-800 border border-gray-700 transition-all"
          >
            ← Ganti Metode
          </button>
        </div>
      </div>

      {/* Form */}
      {renderFormInput()}

      {/* Upload Section */}
      <div className="bg-gray-900 border border-gray-800 rounded-2xl p-5 flex flex-col gap-4">
        <div className="flex items-center justify-between">
          <div>
            <p className="text-white text-sm font-medium">Upload Foto Wajah</p>
            <p className="text-gray-500 text-xs mt-0.5">
              {photos.length > 0
                ? `${photos.length} foto siap dikirim`
                : 'Pilih 5–10 foto wajah (JPG/PNG) dari perangkat'}
            </p>
          </div>
          {photos.length > 0 && (
            <span className="text-xs font-bold px-3 py-1 rounded-full bg-green-900/40 text-green-400 border border-green-500/30">
              ✓ {photos.length} foto
            </span>
          )}
        </div>

        {/* Drop Zone / Upload Button */}
        {photos.length === 0 ? (
          <div
            onClick={() => fileInputRef.current?.click()}
            className="border-2 border-dashed border-gray-700 hover:border-blue-500/50 rounded-xl p-10 flex flex-col items-center gap-3 cursor-pointer transition-all duration-300 hover:bg-gray-800/30 group"
          >
            <div className="text-4xl group-hover:scale-110 transition-transform duration-300">📂</div>
            <div className="text-center">
              <p className="text-white text-sm font-medium">Klik untuk pilih foto</p>
              <p className="text-gray-500 text-xs mt-1">JPG, JPEG, PNG — Pilih 5 sampai 10 foto sekaligus</p>
            </div>
            <span className="text-xs font-medium px-4 py-1.5 rounded-full bg-blue-900/40 text-blue-400 border border-blue-500/30">
              Pilih File
            </span>
          </div>
        ) : (
          <div className="flex gap-2">
            <button
              onClick={() => fileInputRef.current?.click()}
              className="flex-1 flex items-center justify-center gap-2 bg-gray-800 hover:bg-gray-700 border border-gray-700 text-white text-sm font-medium px-4 py-3 rounded-xl transition-all"
            >
              📂 Ganti Foto
            </button>
            <button
              onClick={() => { setPhotos([]); setStatus(null) }}
              className="flex items-center justify-center gap-2 bg-gray-800 hover:bg-gray-700 border border-gray-700 text-gray-400 text-sm px-4 py-3 rounded-xl transition-all"
            >
              ✕ Hapus
            </button>
          </div>
        )}

        {/* Hidden file input */}
        <input
          ref={fileInputRef}
          type="file"
          accept="image/jpeg,image/jpg,image/png"
          multiple
          onChange={handleFileUpload}
          className="hidden"
        />

        {renderThumbnails()}
      </div>

      {renderStatusAndActions()}
    </div>
  )

  // ─── RENDER: Status Alert + Tombol Aksi ───────────────────
  const renderStatusAndActions = () => (
    <>
      {status && (
        <div className={`rounded-xl px-4 py-3 text-sm border ${
          status.type === 'success' ? 'bg-green-900/30 border-green-500/40 text-green-400' :
          status.type === 'error'   ? 'bg-red-900/30 border-red-500/40 text-red-400' :
          status.type === 'loading' ? 'bg-blue-900/30 border-blue-500/40 text-blue-400' :
                                      'bg-gray-800 border-gray-700 text-gray-300'
        }`}>
          {status.type === 'loading' && (
            <span className="inline-block w-3 h-3 border border-blue-400/30 border-t-blue-400 rounded-full animate-spin mr-2 align-middle" />
          )}
          {status.msg}
        </div>
      )}

      <div className="flex gap-3">
        <button
          onClick={handleSubmit}
          disabled={photos.length === 0 || capturing || status?.type === 'loading'}
          className={`flex-1 flex items-center justify-center gap-2 text-sm font-semibold px-4 py-3.5 rounded-xl transition-all ${
            photos.length === 0 || capturing || status?.type === 'loading'
              ? 'bg-gray-800 text-gray-600 border border-gray-700 cursor-not-allowed'
              : 'bg-red-600 hover:bg-red-500 text-white shadow-lg shadow-red-900/30'
          }`}
        >
          {status?.type === 'loading'
            ? <><span className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" /> Memproses...</>
            : '💾 Proses & Simpan ke Server'
          }
        </button>
        <button
          onClick={resetAll}
          disabled={status?.type === 'loading'}
          className="flex items-center gap-2 bg-gray-800 hover:bg-gray-700 border border-gray-700 text-gray-400 hover:text-white text-sm px-4 py-3.5 rounded-xl transition-all disabled:opacity-40"
        >
          🔄 Reset
        </button>
      </div>
    </>
  )

  // ─── RENDER UTAMA ──────────────────────────────────────────
  return (
    <div className="min-h-screen bg-gray-950 text-white" style={{ fontFamily: "'DM Sans', 'Segoe UI', sans-serif" }}>

      {/* NAVBAR */}
      <nav className="bg-gray-900/95 backdrop-blur border-b border-gray-800 px-6 py-3.5 flex items-center justify-between sticky top-0 z-50">
        <div className="flex items-center gap-3">
          <img src={logo} alt="logo" className="w-8 h-8 object-contain" />
          <div>
            <p className="text-white font-bold text-sm leading-tight">Face Recognition Gate</p>
            <p className="text-gray-500 text-[11px]">Telkom University — Registrasi User</p>
          </div>
        </div>
        <button
          onClick={() => navigate('/dashboard')}
          className="text-gray-400 hover:text-red-400 text-sm transition-all"
        >
          ← Kembali ke Dashboard
        </button>
      </nav>

      {/* KONTEN */}
      <div className="p-6 max-w-2xl mx-auto flex flex-col gap-5">
        {mode === 'select' && renderSelectMode()}
        {mode === 'kamera' && renderModeKamera()}
        {mode === 'upload' && renderModeUpload()}
      </div>
    </div>
  )
}

export default RegistrasiPage