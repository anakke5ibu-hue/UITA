import { useState, useRef, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import logo from '../assets/logo-telkom.png'

// ═══════════════════════════════════════════════════════════════
// KONFIGURASI SERVER
// ═══════════════════════════════════════════════════════════════
const SERVER_URL = 'http://100.89.141.47:8001'

// 7 POSE WAJAH
const POSES = [
  { id: 1, name: 'POSE 1', instruction: 'Arahkan wajah ke depan kamera', hint: 'Pastikan dahi, telinga, dagu masuk di dalam kotak' },
  { id: 2, name: 'POSE 2', instruction: 'Tundukkan sedikit kepala', hint: 'Pastikan dahi, telinga, dagu masuk di dalam kotak' },
  { id: 3, name: 'POSE 3', instruction: 'Arahkan kepala menghadap sedikit ke kanan', hint: 'Pastikan dahi, telinga, dagu masuk di dalam kotak' },
  { id: 4, name: 'POSE 4', instruction: 'Arahkan kepala menghadap sedikit ke atas', hint: 'Pastikan dahi, telinga, dagu masuk di dalam kotak' },
  { id: 5, name: 'POSE 5', instruction: 'Arahkan kepala menghadap sedikit ke kiri', hint: 'Pastikan dahi, telinga, dagu masuk di dalam kotak' },
  { id: 6, name: 'POSE 6', instruction: 'Miringkan kepala sedikit ke kanan', hint: 'Pastikan dahi, telinga, dagu masuk di dalam kotak' },
  { id: 7, name: 'POSE 7', instruction: 'Miringkan kepala sedikit ke kiri', hint: 'Pastikan dahi, telinga, dagu masuk di dalam kotak' },
]

// ═══════════════════════════════════════════════════════════════
// MAIN COMPONENT
// ═══════════════════════════════════════════════════════════════
function RegistrasiPage() {
  const navigate = useNavigate()

  // Mode: 'select' | 'kamera' | 'upload'
  const [mode, setMode] = useState('select')

  // Form data
  const [nama, setNama] = useState('')
  const [nim, setNim] = useState('')

  // Upload mode state
  const [uploadPhotos, setUploadPhotos] = useState([]) // { blob, dataUrl }[]
  const fileInputRef = useRef(null)

  // Camera mode state
  const [cameraOpen, setCameraOpen] = useState(false)
  const [showGuidePopup, setShowGuidePopup] = useState(false)
  const [guideStep, setGuideStep] = useState('welcome') // 'welcome', 'pose', 'kacamata_question'
  const [currentPoseIndex, setCurrentPoseIndex] = useState(0)
  const [isKacamataMode, setIsKacamataMode] = useState(false)
  const [kacamataAnswer, setKacamataAnswer] = useState(null) // null, 'yes', 'no'
  
  // Foto collection (blob saja)
  const [photos, setPhotos] = useState({
    normal: {
      pose1: [],
      pose2: [],
      pose3: [],
      pose4: [],
      pose5: [],
      pose6: [],
      pose7: [],
    },
    kacamata: {
      pose1: [],
      pose2: [],
      pose3: [],
      pose4: [],
      pose5: [],
      pose6: [],
      pose7: [],
    }
  })
  
  // Metadata collection
  const [photoMetadata, setPhotoMetadata] = useState([])

  // Camera capture state
  const [countdown, setCountdown] = useState(null)
  const [isCapturing, setIsCapturing] = useState(false)
  const [captureMessage, setCaptureMessage] = useState('')
  const [flashEffect, setFlashEffect] = useState(false)
  
  // Refs
  const videoRef = useRef(null)
  const streamRef = useRef(null)
  const countdownIntervalRef = useRef(null)
  const canvasRef = useRef(null)
  
  // Status
  const [status, setStatus] = useState(null)

  // Helper: get current pose key
  const getCurrentPoseKey = () => {
    const poseNum = currentPoseIndex + 1
    return `pose${poseNum}`
  }

  // Helper: get current photos array
  const getCurrentPhotosArray = () => {
    const modeKey = isKacamataMode ? 'kacamata' : 'normal'
    const poseKey = getCurrentPoseKey()
    return photos[modeKey][poseKey] || []
  }

  // Helper: get total photos count
  const getTotalPhotosCount = () => {
    let total = 0
    Object.values(photos.normal).forEach(arr => total += arr.length)
    if (kacamataAnswer === 'yes') {
      Object.values(photos.kacamata).forEach(arr => total += arr.length)
    }
    return total
  }

  // Helper: get target total photos
  const getTargetTotalPhotos = () => {
    const normalCount = 14
    const kacamataCount = kacamataAnswer === 'yes' ? 14 : 0
    return normalCount + kacamataCount
  }

  // Helper: check if current pose is complete
  const isCurrentPoseComplete = () => {
    return getCurrentPhotosArray().length >= 2
  }

  // Helper: check if all normal poses are complete
  const areAllNormalPosesComplete = () => {
    for (let i = 1; i <= 7; i++) {
      if (photos.normal[`pose${i}`].length < 2) return false
    }
    return true
  }

  // Helper: check if all kacamata poses are complete
  const areAllKacamataPosesComplete = () => {
    if (kacamataAnswer !== 'yes') return true
    for (let i = 1; i <= 7; i++) {
      if (photos.kacamata[`pose${i}`].length < 2) return false
    }
    return true
  }

  // Check if all requirements are met
  const isRegistrationReady = () => {
    if (!nama.trim() || !nim.trim()) return false
    if (!areAllNormalPosesComplete()) return false
    if (kacamataAnswer === 'yes' && !areAllKacamataPosesComplete()) return false
    return true
  }

  // ─── RESET ALL ────────────────────────────────────────────────
  const resetAll = () => {
    closeCamera()
    setNama('')
    setNim('')
    setUploadPhotos([])
    setPhotos({
      normal: { pose1: [], pose2: [], pose3: [], pose4: [], pose5: [], pose6: [], pose7: [] },
      kacamata: { pose1: [], pose2: [], pose3: [], pose4: [], pose5: [], pose6: [], pose7: [] }
    })
    setPhotoMetadata([])
    setKacamataAnswer(null)
    setIsKacamataMode(false)
    setCurrentPoseIndex(0)
    setStatus(null)
  }

  // ─── CAMERA FUNCTIONS ─────────────────────────────────────────
  const openCamera = async () => {
    if (!nama.trim() || !nim.trim()) {
      setStatus({ type: 'error', msg: 'Nama dan NIM harus diisi terlebih dahulu!' })
      return
    }
    setStatus(null)
    setCameraOpen(true)
    setShowGuidePopup(true)
    setGuideStep('welcome')
    
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true })
      streamRef.current = stream
      if (videoRef.current) {
        videoRef.current.srcObject = stream
      }
    } catch (e) {
      setCameraOpen(false)
      setShowGuidePopup(false)
      setStatus({ type: 'error', msg: 'Gagal membuka kamera: ' + e.message })
    }
  }

  const closeCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(t => t.stop())
      streamRef.current = null
    }
    if (countdownIntervalRef.current) {
      clearInterval(countdownIntervalRef.current)
      countdownIntervalRef.current = null
    }
    setCameraOpen(false)
    setShowGuidePopup(false)
    setIsCapturing(false)
    setCountdown(null)
    setCaptureMessage('')
    setCurrentPoseIndex(0)
    setIsKacamataMode(false)
    setGuideStep('welcome')
  }

  // Capture photo from video
  const capturePhoto = () => {
    return new Promise((resolve) => {
      const canvas = canvasRef.current
      const video = videoRef.current
      if (!canvas || !video) return resolve(null)
      
      canvas.width = video.videoWidth
      canvas.height = video.videoHeight
      const ctx = canvas.getContext('2d')
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height)
      
      // Add flash effect
      setFlashEffect(true)
      setTimeout(() => setFlashEffect(false), 150)
      
      canvas.toBlob((blob) => {
        resolve(blob)
      }, 'image/jpeg', 0.9)
    })
  }

  // Start capturing for current pose (2 photos)
  const startPoseCapture = async () => {
    if (isCapturing) return
    setIsCapturing(true)
    
    const currentPhotosArr = getCurrentPhotosArray()
    const remaining = 2 - currentPhotosArr.length
    const currentPoseNumber = currentPoseIndex + 1
    const modeType = isKacamataMode ? 'kacamata' : 'normal'
    
    for (let i = 0; i < remaining; i++) {
      const photoIndexInPose = currentPhotosArr.length + i + 1
      
      // Countdown 3 detik
      for (let cd = 3; cd > 0; cd--) {
        setCountdown(cd)
        setCaptureMessage(`Jepretan ke-${photoIndexInPose}/2 - Bersiap: ${cd}`)
        await new Promise(r => setTimeout(r, 1000))
      }
      
      setCountdown('📸')
      setCaptureMessage(`Mengambil foto ${photoIndexInPose}/2...`)
      
      // Capture photo
      const blob = await capturePhoto()
      
      if (blob) {
        const modeKey = isKacamataMode ? 'kacamata' : 'normal'
        const poseKey = getCurrentPoseKey()
        
        // Simpan blob foto
        setPhotos(prev => ({
          ...prev,
          [modeKey]: {
            ...prev[modeKey],
            [poseKey]: [...prev[modeKey][poseKey], blob]
          }
        }))
        
        // Simpan metadata
        const metadata = {
          pose: currentPoseNumber,
          type: modeType,
          index_in_pose: photoIndexInPose,
          timestamp: Date.now()
        }
        setPhotoMetadata(prev => [...prev, metadata])
        
        setCaptureMessage(`✅ Foto ${photoIndexInPose}/2 berhasil!`)
      } else {
        setCaptureMessage(`❌ Gagal mengambil foto, coba lagi`)
      }
      
      await new Promise(r => setTimeout(r, 500))
    }
    
    setCountdown(null)
    setCaptureMessage('')
    setIsCapturing(false)
  }

  // Reset current pose
  const resetCurrentPose = () => {
    if (window.confirm('Yakin ingin mengulang pose ini? Semua foto untuk pose ini akan dihapus.')) {
      const modeKey = isKacamataMode ? 'kacamata' : 'normal'
      const poseKey = getCurrentPoseKey()
      
      // Hitung berapa foto yang akan dihapus
      const photosToRemove = photos[modeKey][poseKey].length
      
      // Hapus foto dari state
      setPhotos(prev => ({
        ...prev,
        [modeKey]: {
          ...prev[modeKey],
          [poseKey]: []
        }
      }))
      
      // Hapus metadata yang sesuai
      setPhotoMetadata(prev => {
        const currentPoseNumber = currentPoseIndex + 1
        return prev.filter(meta => 
          !(meta.pose === currentPoseNumber && meta.type === modeKey)
        )
      })
      
      setCaptureMessage(`Pose direset, ${photosToRemove} foto dihapus. Silakan ulang pengambilan foto`)
      setStatus({ type: 'info', msg: 'Pose direset, klik "Mulai Ambil Foto" lagi' })
    }
  }

  // Next pose
  const nextPose = () => {
    if (!isCurrentPoseComplete()) {
      setStatus({ type: 'error', msg: 'Harap selesaikan 2 foto untuk pose ini terlebih dahulu!' })
      return
    }
    
    if (currentPoseIndex + 1 < POSES.length) {
      setCurrentPoseIndex(currentPoseIndex + 1)
      setCaptureMessage('')
      setStatus({ type: 'info', msg: `Masuk ke ${POSES[currentPoseIndex + 1].name}` })
    } else {
      // All poses completed for current mode
      if (!isKacamataMode) {
        // Normal mode completed, ask for glasses
        setGuideStep('kacamata_question')
      } else {
        // Kacamata mode completed, close camera
        setGuideStep(null)
        setShowGuidePopup(false)
        closeCamera()
        setStatus({ type: 'success', msg: 'Semua foto berhasil diambil! Silakan simpan data ke server.' })
      }
    }
  }

  // Handle glasses answer
  const handleKacamataAnswer = (answer) => {
    setKacamataAnswer(answer)
    if (answer === 'yes') {
      // Reset for kacamata mode
      setIsKacamataMode(true)
      setCurrentPoseIndex(0)
      setGuideStep('pose')
      setStatus({ type: 'info', msg: 'Mode KAMERA DENGAN KACAMATA aktif. Silakan ulangi 7 pose dengan kacamata.' })
    } else {
      // No glasses, close camera
      setGuideStep(null)
      setShowGuidePopup(false)
      closeCamera()
      setStatus({ type: 'success', msg: 'Semua foto berhasil diambil! Silakan simpan data ke server.' })
    }
  }

  // Welcome guide next
  const welcomeNext = () => {
    setGuideStep('pose')
  }

  // ─── UPLOAD MODE FUNCTIONS ───────────────────────────────────
  const handleFileUpload = (e) => {
    const files = Array.from(e.target.files)
    if (files.length === 0) return
    
    if (files.length < 5) {
      setStatus({ type: 'error', msg: 'Upload minimal 5 foto untuk hasil embedding yang baik.' })
      return
    }
    if (files.length > 15) {
      setStatus({ type: 'error', msg: 'Maksimal 15 foto yang bisa diupload sekaligus.' })
      return
    }
    
    setStatus({ type: 'info', msg: `Memproses ${files.length} foto...` })
    setUploadPhotos([])
    setPhotoMetadata([])
    
    const loaded = []
    files.forEach((file, i) => {
      const reader = new FileReader()
      reader.onload = (ev) => {
        const dataUrl = ev.target.result
        fetch(dataUrl)
          .then(r => r.blob())
          .then(blob => {
            // Untuk upload mode, buat metadata sederhana
            const metadata = {
              type: 'upload',
              original_name: file.name,
              index: i,
              timestamp: Date.now()
            }
            loaded.push({ blob, dataUrl, metadata, index: i })
            if (loaded.length === files.length) {
              loaded.sort((a, b) => a.index - b.index)
              setUploadPhotos(loaded.map(({ blob, dataUrl }) => ({ blob, dataUrl })))
              setPhotoMetadata(loaded.map(({ metadata }) => metadata))
              setStatus({ type: 'info', msg: `${files.length} foto siap! Klik "Proses & Simpan" untuk mendaftar.` })
            }
          })
      }
      reader.readAsDataURL(file)
    })
    e.target.value = ''
  }

  // ─── SUBMIT REGISTRATION ──────────────────────────────────────
  const handleSubmit = async () => {
    if (!isRegistrationReady()) {
      setStatus({ type: 'error', msg: 'Lengkapi semua data dan foto terlebih dahulu!' })
      return
    }
    
    setStatus({ type: 'loading', msg: 'Mengirim data ke server & memproses embedding wajah...' })
    
    let allBlobs = []
    let allMetadata = []
    
    if (mode === 'kamera') {
      // Collect all blobs and metadata from camera mode
      // Normal photos
      for (let i = 1; i <= 7; i++) {
        const poseBlobs = photos.normal[`pose${i}`]
        allBlobs.push(...poseBlobs)
      }
      // Kacamata photos if answered yes
      if (kacamataAnswer === 'yes') {
        for (let i = 1; i <= 7; i++) {
          const poseBlobs = photos.kacamata[`pose${i}`]
          allBlobs.push(...poseBlobs)
        }
      }
      allMetadata = photoMetadata
    } else {
      // Upload mode
      allBlobs = uploadPhotos.map(p => p.blob)
      allMetadata = photoMetadata
    }
    
    if (allBlobs.length === 0) {
      setStatus({ type: 'error', msg: 'Tidak ada foto yang dikirim!' })
      return
    }
    
    const formData = new FormData()
    formData.append('nama', nama.trim())
    formData.append('nim', nim.trim())
    formData.append('kacamata', (kacamataAnswer === 'yes') ? 'true' : 'false')
    formData.append('mode', mode)
    formData.append('total_foto', allBlobs.length.toString())
    
    // Kirim metadata per foto (sebagai JSON string)
    allMetadata.forEach((meta, idx) => {
      formData.append(`meta_${idx}`, JSON.stringify(meta))
    })
    
    // Kirim foto
    allBlobs.forEach((blob, i) => {
      formData.append('photos', blob, `photo_${i}.jpg`)
    })
    
    try {
      const res = await fetch(`${SERVER_URL}/register_from_photos`, {
        method: 'POST',
        body: formData,
      })
      const data = await res.json()
      if (res.ok) {
        setStatus({ type: 'success', msg: `✅ Pendaftaran berhasil! Data wajah ${nama} tersimpan di server.` })
        setTimeout(() => {
          resetAll()
          setMode('select')
        }, 2000)
      } else {
        setStatus({ type: 'error', msg: data.detail || data.message || 'Gagal registrasi' })
      }
    } catch (e) {
      console.error('Gagal registrasi:', e)
      setStatus({ type: 'error', msg: 'Gagal terhubung ke server. Pastikan server berjalan.' })
    }
  }

  // ─── RENDER: Mode Select ─────────────────────────────────────
  const renderSelectMode = () => (
    <div className="flex flex-col gap-5">
      <div className="bg-gray-900 border border-gray-800 rounded-2xl p-5">
        <p className="font-semibold text-white">Registrasi Pengguna Baru</p>
        <p className="text-gray-500 text-xs mt-0.5">Daftarkan wajah user baru ke sistem Face Recognition Gate</p>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        <button
          onClick={() => setMode('kamera')}
          className="group bg-gray-900 border border-gray-800 hover:border-red-500/50 rounded-2xl p-6 flex flex-col items-center gap-4 transition-all duration-300 hover:bg-gray-800/50"
        >
          <div className="w-16 h-16 rounded-2xl bg-red-900/30 border border-red-500/30 flex items-center justify-center text-3xl group-hover:scale-110 transition-transform duration-300">
            📷
          </div>
          <div className="text-center">
            <p className="text-white font-semibold text-sm">Foto Langsung</p>
            <p className="text-gray-500 text-xs mt-1.5 leading-relaxed">
              Ambil 14 foto dengan 7 pose berbeda (2 foto per pose)
            </p>
          </div>
          <span className="text-xs font-medium px-3 py-1 rounded-full bg-red-900/40 text-red-400 border border-red-500/30">
            Gunakan Kamera →
          </span>
        </button>

        <button
          onClick={() => setMode('upload')}
          className="group bg-gray-900 border border-gray-800 hover:border-blue-500/50 rounded-2xl p-6 flex flex-col items-center gap-4 transition-all duration-300 hover:bg-gray-800/50"
        >
          <div className="w-16 h-16 rounded-2xl bg-blue-900/30 border border-blue-500/30 flex items-center justify-center text-3xl group-hover:scale-110 transition-transform duration-300">
            📁
          </div>
          <div className="text-center">
            <p className="text-white font-semibold text-sm">Upload Foto</p>
            <p className="text-gray-500 text-xs mt-1.5 leading-relaxed">
              Upload 5–15 foto wajah dari file yang sudah tersimpan di perangkat
            </p>
          </div>
          <span className="text-xs font-medium px-3 py-1 rounded-full bg-blue-900/40 text-blue-400 border border-blue-500/30">
            Pilih dari File →
          </span>
        </button>
      </div>
    </div>
  )

  // ─── RENDER: Form Input ──────────────────────────────────────
  const renderFormInput = () => (
    <div className="bg-gray-900 border border-gray-800 rounded-2xl p-5 flex flex-col gap-4">
      <div>
        <label className="text-gray-400 text-xs uppercase tracking-wider mb-1.5 block">Nama Lengkap</label>
        <input
          type="text"
          value={nama}
          onChange={e => setNama(e.target.value)}
          placeholder="Masukkan nama lengkap..."
          className="w-full bg-gray-800 border border-gray-700 text-white text-sm rounded-lg px-4 py-3 outline-none focus:ring-2 focus:ring-red-600/50 placeholder-gray-600"
        />
      </div>
      <div>
        <label className="text-gray-400 text-xs uppercase tracking-wider mb-1.5 block">NIM</label>
        <input
          type="text"
          value={nim}
          onChange={e => setNim(e.target.value)}
          placeholder="Masukkan NIM..."
          className="w-full bg-gray-800 border border-gray-700 text-white text-sm rounded-lg px-4 py-3 outline-none focus:ring-2 focus:ring-red-600/50 placeholder-gray-600"
        />
      </div>
    </div>
  )

  // ─── RENDER: Upload Mode ─────────────────────────────────────
  const renderModeUpload = () => (
    <div className="flex flex-col gap-5">
      <div className="bg-gray-900 border border-gray-800 rounded-2xl p-5">
        <div className="flex items-center justify-between">
          <div>
            <p className="font-semibold text-white">📁 Registrasi — Upload Foto</p>
            <p className="text-gray-500 text-xs mt-0.5">Isi data lalu upload 5–15 foto wajah dari file</p>
          </div>
          <button
            onClick={() => { resetAll(); setMode('select') }}
            className="text-gray-500 hover:text-white text-xs px-3 py-1.5 rounded-lg bg-gray-800 border border-gray-700 transition-all"
          >
            ← Ganti Metode
          </button>
        </div>
      </div>

      {renderFormInput()}

      <div className="bg-gray-900 border border-gray-800 rounded-2xl p-5 flex flex-col gap-4">
        <div className="flex items-center justify-between">
          <div>
            <p className="text-white text-sm font-medium">Upload Foto Wajah</p>
            <p className="text-gray-500 text-xs mt-0.5">
              {uploadPhotos.length > 0
                ? `${uploadPhotos.length} foto siap dikirim`
                : 'Pilih 5–15 foto wajah (JPG/PNG) dari perangkat'}
            </p>
          </div>
          {uploadPhotos.length > 0 && (
            <span className="text-xs font-bold px-3 py-1 rounded-full bg-green-900/40 text-green-400 border border-green-500/30">
              ✓ {uploadPhotos.length} foto
            </span>
          )}
        </div>

        {uploadPhotos.length === 0 ? (
          <div
            onClick={() => fileInputRef.current?.click()}
            className="border-2 border-dashed border-gray-700 hover:border-blue-500/50 rounded-xl p-10 flex flex-col items-center gap-3 cursor-pointer transition-all duration-300 hover:bg-gray-800/30 group"
          >
            <div className="text-4xl group-hover:scale-110 transition-transform duration-300">📂</div>
            <div className="text-center">
              <p className="text-white text-sm font-medium">Klik untuk pilih foto</p>
              <p className="text-gray-500 text-xs mt-1">JPG, JPEG, PNG — Pilih 5 sampai 15 foto sekaligus</p>
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
              onClick={() => { setUploadPhotos([]); setPhotoMetadata([]); setStatus(null) }}
              className="flex items-center justify-center gap-2 bg-gray-800 hover:bg-gray-700 border border-gray-700 text-gray-400 text-sm px-4 py-3 rounded-xl transition-all"
            >
              ✕ Hapus
            </button>
          </div>
        )}

        <input
          ref={fileInputRef}
          type="file"
          accept="image/jpeg,image/jpg,image/png"
          multiple
          onChange={handleFileUpload}
          className="hidden"
        />

        {uploadPhotos.length > 0 && (
          <div className="grid grid-cols-5 gap-2 max-h-32 overflow-y-auto p-2 bg-gray-800 rounded-xl">
            {uploadPhotos.slice(0, 15).map((p, i) => (
              <div key={i} className="relative rounded-lg overflow-hidden aspect-square bg-gray-700">
                <img src={p.dataUrl} alt={`foto-${i+1}`} className="w-full h-full object-cover" />
                <div className="absolute bottom-0 left-0 right-0 bg-black/60 text-center text-[10px] text-gray-300 py-0.5">
                  #{i+1}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {renderStatusAndActions()}
    </div>
  )

  // ─── RENDER: Guide Popup ─────────────────────────────────────
  const renderGuidePopup = () => {
    if (!showGuidePopup) return null
    
    if (guideStep === 'welcome') {
      return (
        <div className="fixed inset-0 bg-black/80 flex items-center justify-center z-50 p-4">
          <div className="bg-gray-900 border border-gray-700 rounded-2xl max-w-md w-full p-6">
            <div className="text-center">
              <div className="text-5xl mb-4">📷</div>
              <h3 className="text-white font-bold text-lg mb-2">Persiapan Kamera</h3>
              <p className="text-gray-400 text-sm mb-4">Pastikan hal-hal berikut sebelum melanjutkan:</p>
              <ul className="text-left text-gray-300 text-sm space-y-2 mb-6">
                <li>✓ Perangkat memiliki kamera dan aktif</li>
                <li>✓ Posisikan mata sejajar dengan kamera</li>
                <li>✓ Pastikan pencahayaan cukup</li>
                <li>✓ Wajah menghadap langsung ke kamera</li>
              </ul>
              <button
                onClick={welcomeNext}
                className="w-full bg-red-600 hover:bg-red-500 text-white font-semibold py-3 rounded-xl transition-all"
              >
                Lanjutkan →
              </button>
            </div>
          </div>
        </div>
      )
    }
    
    if (guideStep === 'kacamata_question') {
      return (
        <div className="fixed inset-0 bg-black/80 flex items-center justify-center z-50 p-4">
          <div className="bg-gray-900 border border-gray-700 rounded-2xl max-w-md w-full p-6">
            <div className="text-center">
              <div className="text-5xl mb-4">👓</div>
              <h3 className="text-white font-bold text-lg mb-2">Apakah Anda pengguna kacamata?</h3>
              <p className="text-gray-400 text-sm mb-6">
                Jika iya, Anda akan diminta mengambil 14 foto tambahan dengan menggunakan kacamata.
              </p>
              <div className="flex gap-3">
                <button
                  onClick={() => handleKacamataAnswer('yes')}
                  className="flex-1 bg-blue-600 hover:bg-blue-500 text-white font-semibold py-3 rounded-xl transition-all"
                >
                  Ya, pakai kacamata
                </button>
                <button
                  onClick={() => handleKacamataAnswer('no')}
                  className="flex-1 bg-gray-700 hover:bg-gray-600 text-white font-semibold py-3 rounded-xl transition-all"
                >
                  Tidak
                </button>
              </div>
            </div>
          </div>
        </div>
      )
    }
    
    return null
  }

  // ─── RENDER: Camera Popup ────────────────────────────────────
  const renderCameraPopup = () => {
    if (!cameraOpen) return null
    
    const currentPose = POSES[currentPoseIndex]
    const currentPhotosArr = getCurrentPhotosArray()
    const isComplete = currentPhotosArr.length >= 2
    const totalPhotos = getTotalPhotosCount()
    const targetPhotos = getTargetTotalPhotos()
    const currentPoseNumber = currentPoseIndex + 1
    
    return (
      <div className="fixed inset-0 bg-black flex flex-col z-50">
        {/* Header */}
        <div className="bg-gray-900 px-4 py-3 flex items-center justify-between border-b border-gray-800">
          <div>
            <p className="text-white font-semibold">
              {isKacamataMode ? '👓 Mode Kacamata' : '📷 Registrasi Wajah'}
            </p>
            <p className="text-gray-500 text-xs">
              Progress: {totalPhotos}/{targetPhotos} foto | Pose {currentPoseNumber}/7
            </p>
          </div>
          <button
            onClick={closeCamera}
            className="text-gray-400 hover:text-white text-2xl"
          >
            ✕
          </button>
        </div>
        
        {/* Video dengan bounding box */}
        <div className="relative flex-1 bg-black flex items-center justify-center">
          <div className="relative">
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              className="max-h-[70vh] rounded-xl"
              style={{ maxWidth: '100%', objectFit: 'cover' }}
            />
            {/* Flash effect overlay */}
            {flashEffect && (
              <div className="absolute inset-0 bg-white rounded-xl pointer-events-none" style={{ opacity: 0.7 }} />
            )}
            {/* Bounding box tetap */}
            <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2">
              <div className="w-90 h-90 border-4 border-red-500 rounded-1xl shadow-lg"></div>
            </div>
          </div>
        </div>
        
        {/* Guide and controls */}
        <div className="bg-gray-900 p-5 border-t border-gray-800">
          <div className="mb-4">
            <div className="flex items-center justify-between mb-2">
              <p className="text-red-400 font-bold text-sm">{currentPose.name}</p>
              <p className="text-gray-400 text-xs">Foto: {currentPhotosArr.length}/2</p>
            </div>
            <p className="text-white font-medium">{currentPose.instruction}</p>
            <p className="text-yellow-400 text-xs mt-1">⚠️ {currentPose.hint}</p>
          </div>
          
          {captureMessage && (
            <div className="mb-3 text-center text-sm text-blue-400 bg-gray-800 py-2 rounded-lg">
              {captureMessage}
            </div>
          )}
          
          {countdown !== null && (
            <div className="mb-3 text-center">
              <span className="text-3xl font-bold text-yellow-400">{countdown}</span>
            </div>
          )}
          
          <div className="flex gap-3">
            {!isCapturing && !isComplete && (
              <button
                onClick={startPoseCapture}
                className="flex-1 bg-green-600 hover:bg-green-500 text-white font-semibold py-3 rounded-xl transition-all"
              >
                ▶ Mulai Ambil 2 Foto
              </button>
            )}
            
            {isCapturing && (
              <button
                disabled
                className="flex-1 bg-gray-700 text-gray-400 font-semibold py-3 rounded-xl cursor-not-allowed"
              >
                ⏳ Mengambil foto...
              </button>
            )}
            
            {!isCapturing && currentPhotosArr.length > 0 && currentPhotosArr.length < 2 && (
              <button
                onClick={startPoseCapture}
                className="flex-1 bg-yellow-600 hover:bg-yellow-500 text-white font-semibold py-3 rounded-xl transition-all"
              >
                📸 Lanjutkan ({currentPhotosArr.length}/2)
              </button>
            )}
            
            <button
              onClick={resetCurrentPose}
              disabled={isCapturing || currentPhotosArr.length === 0}
              className="px-4 bg-gray-800 hover:bg-gray-700 text-gray-400 rounded-xl transition-all disabled:opacity-50"
            >
              🔄 Ulang
            </button>
            
            <button
              onClick={nextPose}
              disabled={!isComplete || isCapturing}
              className={`px-6 font-semibold py-3 rounded-xl transition-all ${
                isComplete && !isCapturing
                  ? 'bg-red-600 hover:bg-red-500 text-white'
                  : 'bg-gray-800 text-gray-600 cursor-not-allowed'
              }`}
            >
              {currentPoseIndex + 1 >= POSES.length ? 'Selesai →' : 'Lanjut →'}
            </button>
          </div>
        </div>
        
        <canvas ref={canvasRef} className="hidden" />
      </div>
    )
  }

  // ─── RENDER: Status & Actions ────────────────────────────────
  const renderStatusAndActions = () => {
    const isReady = isRegistrationReady()
    const totalPhotos = mode === 'kamera' ? getTotalPhotosCount() : uploadPhotos.length
    const targetPhotos = mode === 'kamera' ? getTargetTotalPhotos() : (uploadPhotos.length >= 5 ? uploadPhotos.length : 5)
    
    return (
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
        
        {mode === 'kamera' && (
          <div className="bg-gray-900/50 rounded-xl p-3 text-center">
            <p className="text-xs text-gray-500">
              📸 Total foto terkumpul: <span className="text-white font-bold">{totalPhotos}</span> / {targetPhotos}
            </p>
            {!isReady && totalPhotos < targetPhotos && (
              <p className="text-yellow-500 text-xs mt-1">
                ⚠️ Masih ada {targetPhotos - totalPhotos} foto yang harus diambil
              </p>
            )}
            {isReady && (
              <p className="text-green-400 text-xs mt-1">
                ✅ Semua foto lengkap! Silakan klik Simpan.
              </p>
            )}
          </div>
        )}
        
        <div className="flex gap-3">
          <button
            onClick={handleSubmit}
            disabled={!isReady}
            className={`flex-1 flex items-center justify-center gap-2 text-sm font-semibold px-4 py-3.5 rounded-xl transition-all ${
              isReady
                ? 'bg-green-600 hover:bg-green-500 text-white shadow-lg shadow-green-900/30'
                : 'bg-gray-800 text-gray-600 border border-gray-700 cursor-not-allowed'
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
  }

  // ─── RENDER UTAMA ────────────────────────────────────────────
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
        {mode === 'kamera' && (
          <>
            {renderFormInput()}
            <div className="bg-gray-900 border border-gray-800 rounded-2xl p-5">
              <button
                onClick={openCamera}
                className="w-full flex items-center justify-center gap-2 bg-red-600 hover:bg-red-500 text-white font-semibold py-3 rounded-xl transition-all"
              >
                📷 Buka Kamera & Ambil Foto
              </button>
              {renderStatusAndActions()}
            </div>
          </>
        )}
        {mode === 'upload' && renderModeUpload()}
      </div>

      {/* POPUPS */}
      {renderGuidePopup()}
      {renderCameraPopup()}
    </div>
  )
}

export default RegistrasiPage