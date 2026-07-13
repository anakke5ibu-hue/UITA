import { useState, useRef, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import logo from '../assets/logo-telkom.png'

// ═══════════════════════════════════════════════════════════════
// SERVER CONFIGURATION
// ═══════════════════════════════════════════════════════════════
const SERVER_URL = 'http://100.89.141.47:8001'

// ═══════════════════════════════════════════════════════════════
// 7 POSE WAJAH (LENGKAP)
// ═══════════════════════════════════════════════════════════════
const FACE_ID_STEPS = [
  { key: 'DEPAN', label: 'POSE 1', instruction: 'Arahkan wajah ke depan kamera' },
  { key: 'KIRI', label: 'POSE 2', instruction: 'Arahkan kepala menghadap sedikit ke kiri' },
  { key: 'KANAN', label: 'POSE 3', instruction: 'Arahkan kepala menghadap sedikit ke kanan' },
  { key: 'ATAS', label: 'POSE 4', instruction: 'Arahkan kepala menghadap sedikit ke atas' },
  { key: 'BAWAH', label: 'POSE 5', instruction: 'Tundukkan sedikit kepala' },
  { key: 'KIRI_SERONG', label: 'POSE 6', instruction: 'Miringkan kepala sedikit ke kiri' },
  { key: 'KANAN_SERONG', label: 'POSE 7', instruction: 'Miringkan kepala sedikit ke kanan' },
]

// ═══════════════════════════════════════════════════════════════
// MAIN COMPONENT
// ═══════════════════════════════════════════════════════════════
export default function RegistrasiPage() {
  const navigate = useNavigate()

  // ─── FORM DATA ──────────────────────────────────────────────
  const [nama, setNama] = useState('')
  const [nim, setNim] = useState('')
  const [visible, setVisible] = useState(false)

  // ─── CAMERA & MEDIAPIPE STATE ──────────────────────────────
  const [cameraOpen, setCameraOpen] = useState(false)
  const [scriptsLoaded, setScriptsLoaded] = useState(false)
  const [currentStepIndex, setCurrentStepIndex] = useState(0)
  const [capturedBlobs, setCapturedBlobs] = useState([])
  const [photoMetadata, setPhotoMetadata] = useState([])
  const [status, setStatus] = useState(null)

  // ─── GLASSES STATE ──────────────────────────────────────────
  const [isKacamataMode, setIsKacamataMode] = useState(false)
  const [showKacamataQuestion, setShowKacamataQuestion] = useState(false)
  const [showPutOnGlassesInstruction, setShowPutOnGlassesInstruction] = useState(false)
  const [showRulesOverlay, setShowRulesOverlay] = useState(false)
  const [kacamataAnswer, setKacamataAnswer] = useState(null)

  // ─── GRID PREVIEW STATE ─────────────────────────────────────
  const [showGridPreview, setShowGridPreview] = useState(false)
  const [selectedPhotoIndex, setSelectedPhotoIndex] = useState(null)

  // ─── REAL-TIME FEEDBACK ─────────────────────────────────────
  const [faceStatusText, setFaceStatusText] = useState('Menunggu kamera...')
  const [isFaceValid, setIsFaceValid] = useState(false)
  const [photosCountInCurrentStep, setPhotosCountInCurrentStep] = useState(0)
  const [feedbackType, setFeedbackType] = useState('info')

  // ─── COUNTDOWN STATE ────────────────────────────────────────
  const [countdown, setCountdown] = useState(null)
  const [isCountingDown, setIsCountingDown] = useState(false)
  const [flashEffect, setFlashEffect] = useState(false)

  // ─── REFS (PENTING UNTUK MEDIAPIPE CALLBACK) ────────────────
  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const streamRef = useRef(null)
  const activeCameraRef = useRef(null)
  
  const stepIndexRef = useRef(0)
  const photoCountRef = useRef(0)
  const isCapturingLockRef = useRef(false)
  const blobsAccumulatorRef = useRef([])
  const metadataAccumulatorRef = useRef([])
  const isKacamataModeRef = useRef(false)
  const kacamataAnswerRef = useRef(null)
  
  // Refs untuk sinkronisasi state UI dengan callback MediaPipe yang berjalan cepat
  const showRulesOverlayRef = useRef(false)
  const showKacamataQuestionRef = useRef(false)
  const showPutOnGlassesInstructionRef = useRef(false)
  const showGridPreviewRef = useRef(false)
  const isCountingDownRef = useRef(false)
  const countdownTimerRef = useRef(null)
  const countdownValueRef = useRef(3)
  const validPoseFrameCountRef = useRef(0)
  const invalidPoseFrameCountRef = useRef(0)

  // ─── EFFECTS ────────────────────────────────────────────────
  useEffect(() => {
    setTimeout(() => setVisible(true), 100)
  }, [])

  // Load MediaPipe scripts
  useEffect(() => {
    const loadScripts = async () => {
      if (window.FaceMesh && window.Camera) {
        setScriptsLoaded(true)
        return
      }

      const cameraScript = document.createElement('script')
      cameraScript.src = 'https://cdn.jsdelivr.net/npm/@mediapipe/camera_utils/camera_utils.js'
      cameraScript.async = true
      document.body.appendChild(cameraScript)

      const faceMeshScript = document.createElement('script')
      faceMeshScript.src = 'https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/face_mesh.js'
      faceMeshScript.async = true
      document.body.appendChild(faceMeshScript)

      faceMeshScript.onload = () => setScriptsLoaded(true)
    }
    loadScripts()

    return () => {
      closeCamera()
      if (countdownTimerRef.current) {
        clearInterval(countdownTimerRef.current)
      }
    }
  }, [])

  // ─── HELPERS ──────────────────────────────────────────────────
  const getTargetTotalPhotos = () => {
    const normalCount = 14 // 7 poses × 2 photos
    const kacamataCount = kacamataAnswer === 'yes' ? 14 : 0
    return normalCount + kacamataCount
  }

  const isAllPosesComplete = () => {
    const total = capturedBlobs.length
    const target = getTargetTotalPhotos()
    return total >= target
  }

  // Helper untuk update state dan ref bersamaan
  const updateOverlayState = (setter, refVal, value) => {
    setter(value)
    refVal.current = value
  }

  // ─── CAMERA FUNCTIONS ────────────────────────────────────────
  const openCamera = async () => {
    if (!nama.trim() || !nim.trim()) {
      setStatus({ type: 'error', msg: 'Nama dan NIM harus diisi terlebih dahulu!' })
      return
    }
    if (!scriptsLoaded) {
      setStatus({ type: 'error', msg: 'Memuat library, harap tunggu...' })
      return
    }

    setStatus(null)
    setCameraOpen(true)
    
    updateOverlayState(setShowRulesOverlay, showRulesOverlayRef, true)
    updateOverlayState(setShowKacamataQuestion, showKacamataQuestionRef, false)
    updateOverlayState(setShowPutOnGlassesInstruction, showPutOnGlassesInstructionRef, false)
    updateOverlayState(setShowGridPreview, showGridPreviewRef, false)
    
    setCurrentStepIndex(0)
    setCapturedBlobs([])
    setPhotoMetadata([])
    setIsKacamataMode(false)
    setKacamataAnswer(null)
    setSelectedPhotoIndex(null)
    setCountdown(null)
    setIsCountingDown(false)
    setIsFaceValid(false)
    setFeedbackType('info')
    setFaceStatusText('👀 Siap scan: posisikan wajah di tengah lingkaran.')

    stepIndexRef.current = 0
    photoCountRef.current = 0
    blobsAccumulatorRef.current = []
    metadataAccumulatorRef.current = []
    isCapturingLockRef.current = true
    isKacamataModeRef.current = false
    kacamataAnswerRef.current = null
    countdownValueRef.current = 3
    validPoseFrameCountRef.current = 0
    invalidPoseFrameCountRef.current = 0
    isCountingDownRef.current = false

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, frameRate: { ideal: 30 } },
      })
      streamRef.current = stream
      if (videoRef.current) {
        videoRef.current.srcObject = stream
      }

      const faceMesh = new window.FaceMesh({
        locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`,
      })
      faceMesh.setOptions({
        maxNumFaces: 1,
        refineLandmarks: false,
        minDetectionConfidence: 0.6,
        minTrackingConfidence: 0.6,
      })
      faceMesh.onResults(onFaceMeshResults)

      const cameraUtils = new window.Camera(videoRef.current, {
        onFrame: async () => {
          if (videoRef.current && streamRef.current) {
            await faceMesh.send({ image: videoRef.current })
          }
        },
        width: 640,
        height: 480,
      })

      activeCameraRef.current = cameraUtils
      cameraUtils.start()
    } catch (err) {
      setCameraOpen(false)
      setStatus({ type: 'error', msg: 'Gagal mengakses kamera: ' + err.message })
    }
  }

  const closeCamera = () => {
    if (activeCameraRef.current) {
      try { activeCameraRef.current.stop() } catch (e) {}
      activeCameraRef.current = null
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop())
      streamRef.current = null
    }
    if (countdownTimerRef.current) {
      clearInterval(countdownTimerRef.current)
      countdownTimerRef.current = null
    }
    isCapturingLockRef.current = false
    isCountingDownRef.current = false
    setIsCountingDown(false)
    setCountdown(null)
    setCameraOpen(false)
    
    updateOverlayState(setShowRulesOverlay, showRulesOverlayRef, false)
    updateOverlayState(setShowKacamataQuestion, showKacamataQuestionRef, false)
    updateOverlayState(setShowPutOnGlassesInstruction, showPutOnGlassesInstructionRef, false)
    updateOverlayState(setShowGridPreview, showGridPreviewRef, false)
    
    setSelectedPhotoIndex(null)
  }

  // ─── CAPTURE PHOTO ──────────────────────────────────────────
  const capturePhoto = () => {
    return new Promise((resolve) => {
      const canvas = canvasRef.current
      const video = videoRef.current
      if (!canvas || !video) return resolve(null)

      canvas.width = video.videoWidth
      canvas.height = video.videoHeight
      const ctx = canvas.getContext('2d')
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height)

      setFlashEffect(true)
      setTimeout(() => setFlashEffect(false), 150)

      canvas.toBlob((blob) => {
        resolve(blob)
      }, 'image/jpeg', 0.9)
    })
  }

  // ─── DETECT POSE ──────────────────────────────────────────
  const detectPose = (landmarks) => {
    const nose = landmarks[4]
    const leftEyeOuter = landmarks[130]
    const rightEyeOuter = landmarks[359]
    const forehead = landmarks[10]
    const chin = landmarks[152]
    const leftEye = landmarks[33]
    const rightEye = landmarks[263]

    const distToLeftEye = Math.abs(nose.x - leftEyeOuter.x)
    const distToRightEye = Math.abs(nose.x - rightEyeOuter.x)
    const yawRatio = distToLeftEye / distToRightEye

    const distToForehead = Math.abs(nose.y - forehead.y)
    const distToChin = Math.abs(nose.y - chin.y)
    const pitchRatio = distToForehead / distToChin

    const dx = rightEye.x - leftEye.x
    const dy = rightEye.y - leftEye.y
    const rollAngle = Math.atan2(dy, dx) * (180 / Math.PI)

    if (yawRatio < 0.55) return 'KANAN'
    if (yawRatio > 1.75) return 'KIRI'
    if (pitchRatio < 0.75) return 'ATAS'
    if (pitchRatio > 1.45) return 'BAWAH'
    if (rollAngle > 8) return 'KIRI_SERONG'
    if (rollAngle < -8) return 'KANAN_SERONG'
    return 'DEPAN'
  }

  // ─── START COUNTDOWN ────────────────────────────────────────
  const startCountdown = () => {
    if (isCountingDownRef.current || isCapturingLockRef.current) return

    isCountingDownRef.current = true
    setIsCountingDown(true)
    countdownValueRef.current = 3
    setCountdown(3)

    if (countdownTimerRef.current) clearInterval(countdownTimerRef.current)

    countdownTimerRef.current = setInterval(() => {
      countdownValueRef.current -= 1
      setCountdown(countdownValueRef.current)

      if (countdownValueRef.current <= 0) {
        clearInterval(countdownTimerRef.current)
        countdownTimerRef.current = null
        performCapture() // Jalankan capture ketika habis
      }
    }, 1000)
  }

  const stopCountdown = () => {
    if (countdownTimerRef.current) {
      clearInterval(countdownTimerRef.current)
      countdownTimerRef.current = null
    }
    isCountingDownRef.current = false
    setIsCountingDown(false)
    setCountdown(null)
  }

  // ─── PERFORM CAPTURE ──────────────────────────────────────
  const performCapture = async () => {
    if (isCapturingLockRef.current) return
    isCapturingLockRef.current = true
    
    // Matikan state countdown
    isCountingDownRef.current = false
    setIsCountingDown(false)
    setCountdown(null)

    const blob = await capturePhoto()
    
    if (blob) {
      const currentStepNumber = stepIndexRef.current + 1
      const photoIndexInPose = photoCountRef.current + 1
      const modeType = isKacamataModeRef.current ? 'kacamata' : 'normal'

      blobsAccumulatorRef.current.push(blob)
      setCapturedBlobs([...blobsAccumulatorRef.current])

      const metadata = {
        pose: currentStepNumber,
        type: modeType,
        index_in_pose: photoIndexInPose,
        timestamp: Date.now(),
      }
      metadataAccumulatorRef.current.push(metadata)
      setPhotoMetadata([...metadataAccumulatorRef.current])

      photoCountRef.current += 1
      setPhotosCountInCurrentStep(photoCountRef.current)

      setFaceStatusText(`📸 Foto ${photoIndexInPose}/2 berhasil!`)
      setFeedbackType('success')
      
      // Reset valid frame supaya user harus menahan pose lagi untuk capture berikutnya
      validPoseFrameCountRef.current = 0 

      setTimeout(() => {
        if (photoCountRef.current >= 2) {
          photoCountRef.current = 0
          setPhotosCountInCurrentStep(0)
          stepIndexRef.current += 1
          setCurrentStepIndex(stepIndexRef.current)

          if (stepIndexRef.current >= FACE_ID_STEPS.length) {
            if (!isKacamataModeRef.current && kacamataAnswerRef.current === null) {
              setFaceStatusText('✅ Semua pose normal selesai!')
              setFeedbackType('success')
              setTimeout(() => {
                updateOverlayState(setShowKacamataQuestion, showKacamataQuestionRef, true)
              }, 500)
            } else {
              setFaceStatusText('✅ Semua foto berhasil diambil!')
              setFeedbackType('success')
              setTimeout(() => {
                updateOverlayState(setShowGridPreview, showGridPreviewRef, true)
                isCapturingLockRef.current = true
              }, 500)
            }
          } else {
            setFaceStatusText(`Pose ${stepIndexRef.current} selesai! Lanjut ke pose ${stepIndexRef.current + 1}.`)
            setFeedbackType('info')
          }
        }
        isCapturingLockRef.current = false
      }, 800)
    } else {
      setFaceStatusText('❌ Gagal mengambil foto, coba lagi')
      setFeedbackType('error')
      isCapturingLockRef.current = false
    }
  }

  // ─── MEDIAPIPE ONRESULTS ──────────────────────────────────
  const onFaceMeshResults = (results) => {
    if (!streamRef.current) return
    
    // Ganti pengecekan menggunakan REFS, bukan state langsung!
    if (showKacamataQuestionRef.current || showPutOnGlassesInstructionRef.current || showRulesOverlayRef.current || showGridPreviewRef.current) {
      return
    }

    if (isCapturingLockRef.current) return

    const currentStep = FACE_ID_STEPS[stepIndexRef.current]
    if (!currentStep) return

    if (!results.multiFaceLandmarks || results.multiFaceLandmarks.length === 0) {
      validPoseFrameCountRef.current = 0
      invalidPoseFrameCountRef.current += 1
      if (invalidPoseFrameCountRef.current >= 2 && isCountingDownRef.current) stopCountdown()
      setIsFaceValid(false)
      setFaceStatusText('⚠️ Wajah tidak terdeteksi! Masuk ke frame.')
      setFeedbackType('error')
      return
    }

    const landmarks = results.multiFaceLandmarks[0]
    const nose = landmarks[4]

    const distanceFromCenter = Math.sqrt(Math.pow(nose.x - 0.5, 2) + Math.pow(nose.y - 0.5, 2))
    const isInsideCircle = distanceFromCenter < 0.20

    if (!isInsideCircle) {
      validPoseFrameCountRef.current = 0
      invalidPoseFrameCountRef.current += 1
      if (invalidPoseFrameCountRef.current >= 2 && isCountingDownRef.current) stopCountdown()
      setIsFaceValid(false)
      setFaceStatusText('⚠️ Posisikan wajah di TENGAH lingkaran!')
      setFeedbackType('error')
      return
    }

    const detectedOrientation = detectPose(landmarks)

    if (detectedOrientation === currentStep.key) {
      invalidPoseFrameCountRef.current = 0
      validPoseFrameCountRef.current += 1
      
      setIsFaceValid(true)
      setFaceStatusText(`✅ Pose ${stepIndexRef.current + 1} terdeteksi!`)
      setFeedbackType('success')
      
      if (!isCountingDownRef.current && photoCountRef.current < 2) {
        startCountdown()
      }
    } else {
      validPoseFrameCountRef.current = 0
      invalidPoseFrameCountRef.current += 1
      
      if (invalidPoseFrameCountRef.current >= 2 && isCountingDownRef.current) stopCountdown()
      
      setIsFaceValid(false)
      setFaceStatusText(`❌ Harap: ${currentStep.instruction}`)
      setFeedbackType('warning')
    }
  }

  // ─── KACAMATA HANDLERS ────────────────────────────────────
  const handleKacamataAnswer = (answer) => {
    setKacamataAnswer(answer)
    kacamataAnswerRef.current = answer
    updateOverlayState(setShowKacamataQuestion, showKacamataQuestionRef, false)

    if (answer === 'yes') {
      updateOverlayState(setShowPutOnGlassesInstruction, showPutOnGlassesInstructionRef, true)
    } else {
      setFaceStatusText('✅ Registrasi selesai!')
      setIsFaceValid(true)
      setTimeout(() => {
        updateOverlayState(setShowGridPreview, showGridPreviewRef, true)
        isCapturingLockRef.current = true
      }, 500)
    }
  }

  const handleStartGlassesScanningPhase = () => {
    updateOverlayState(setShowPutOnGlassesInstruction, showPutOnGlassesInstructionRef, false)
    setIsKacamataMode(true)
    isKacamataModeRef.current = true

    stepIndexRef.current = 0
    setCurrentStepIndex(0)
    photoCountRef.current = 0
    setPhotosCountInCurrentStep(0)
    
    stopCountdown()
    isCapturingLockRef.current = false
    setFaceStatusText('👓 Mode Kacamata dimulai! Lakukan 7 pose dengan kacamata.')
    setFeedbackType('info')
  }

  // ─── SUBMIT REGISTRATION ──────────────────────────────────
  const handleSubmitRegistration = async () => {
    if (!isAllPosesComplete()) {
      setStatus({ type: 'error', msg: 'Foto belum lengkap!' })
      return
    }

    setStatus({ type: 'loading', msg: 'Mengirim data ke server...' })

    const formData = new FormData()
    formData.append('nama', nama.trim())
    formData.append('nim', nim.trim())
    formData.append('mode', 'kamera_face_id_auto')
    formData.append('kacamata', (kacamataAnswer === 'yes').toString())
    formData.append('total_foto', capturedBlobs.length.toString())

    const metadataJson = JSON.stringify(photoMetadata)
    formData.append('metadata_json', metadataJson)

    photoMetadata.forEach((meta, idx) => {
      formData.append(`meta_${idx}`, JSON.stringify(meta))
    })

    capturedBlobs.forEach((blob, idx) => {
      formData.append('photos', blob, `photo_${idx}.jpg`)
    })

    try {
      const response = await fetch(`${SERVER_URL}/register_from_photos`, {
        method: 'POST',
        body: formData,
      })
      const result = await response.json()

      if (response.ok) {
        setStatus({ type: 'success', msg: `✅ Registrasi berhasil!` })
        setTimeout(() => {
          setNama('')
          setNim('')
          setCapturedBlobs([])
          setPhotoMetadata([])
          setKacamataAnswer(null)
          setIsKacamataMode(false)
          updateOverlayState(setShowGridPreview, showGridPreviewRef, false)
          setSelectedPhotoIndex(null)
          setStatus(null)
          navigate('/dashboard')
        }, 3000)
      } else {
        setStatus({ type: 'error', msg: result.detail || result.message || 'Gagal registrasi' })
      }
    } catch (err) {
      setStatus({ type: 'error', msg: 'Gagal terhubung ke server.' })
    }
  }

  // ─── RENDER: GRID PREVIEW ─────────────────────────────────
  const renderGridPreview = () => {
    if (!showGridPreview) return null

    const totalPhotos = capturedBlobs.length
    const targetPhotos = getTargetTotalPhotos()
    const isComplete = totalPhotos >= targetPhotos

    const gridItems = []
    for (let i = 0; i < 16; i++) {
      if (i < totalPhotos) {
        gridItems.push({
          blob: capturedBlobs[i],
          metadata: photoMetadata[i],
          index: i,
        })
      } else {
        gridItems.push(null)
      }
    }

    return (
      <div className="fixed inset-0 bg-black/95 backdrop-blur-sm flex flex-col items-center justify-center z-50 p-6 animate-fade-in">
        <div className="bg-gray-900/90 border border-gray-700 rounded-2xl max-w-4xl w-full p-6 max-h-[90vh] overflow-y-auto">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h3 className="text-white font-bold text-lg">📸 Semua Foto ({totalPhotos}/{targetPhotos})</h3>
              <p className="text-gray-400 text-xs mt-0.5">Klik foto untuk melihat lebih besar</p>
            </div>
            <div className="flex gap-3">
              {isComplete && (
                <button
                  onClick={handleSubmitRegistration}
                  className="bg-emerald-600 hover:bg-emerald-500 text-white font-semibold px-6 py-2 rounded-xl transition-all text-sm"
                >
                  💾 Register Now
                </button>
              )}
              <button
                onClick={() => {
                  updateOverlayState(setShowGridPreview, showGridPreviewRef, false)
                  closeCamera()
                }}
                className="bg-gray-700 hover:bg-gray-600 text-white px-4 py-2 rounded-xl transition-all text-sm"
              >
                ✕ Tutup
              </button>
            </div>
          </div>

          <div className="grid grid-cols-4 gap-3">
            {gridItems.map((item, idx) => (
              <div
                key={idx}
                onClick={() => item && setSelectedPhotoIndex(idx)}
                className={`aspect-square rounded-xl overflow-hidden bg-gray-800 border-2 ${
                  item ? 'border-gray-600 hover:border-blue-500 cursor-pointer' : 'border-gray-700'
                } transition-all duration-200 relative group`}
              >
                {item ? (
                  <>
                    <img
                      src={URL.createObjectURL(item.blob)}
                      alt={`Foto ${idx + 1}`}
                      className="w-full h-full object-cover"
                    />
                    <div className="absolute bottom-0 left-0 right-0 bg-black/70 text-center text-[10px] text-gray-300 py-1">
                      {item.metadata ? `POSE ${item.metadata.pose} (${item.metadata.index_in_pose}/2)` : `Foto ${idx + 1}`}
                    </div>
                    <div className="absolute inset-0 bg-blue-500/0 group-hover:bg-blue-500/10 transition-all duration-200" />
                  </>
                ) : (
                  <div className="w-full h-full flex items-center justify-center text-gray-600 text-2xl">
                    +
                  </div>
                )}
              </div>
            ))}
          </div>

          {!isComplete && (
            <div className="mt-4 text-center text-yellow-400 text-sm">
              ⚠️ Masih ada {targetPhotos - totalPhotos} foto yang harus diambil
            </div>
          )}
        </div>

        {selectedPhotoIndex !== null && capturedBlobs[selectedPhotoIndex] && (
          <div
            className="fixed inset-0 bg-black/90 flex items-center justify-center z-[60] p-6 animate-fade-in"
            onClick={() => setSelectedPhotoIndex(null)}
          >
            <div className="relative max-w-2xl w-full" onClick={(e) => e.stopPropagation()}>
              <img
                src={URL.createObjectURL(capturedBlobs[selectedPhotoIndex])}
                alt={`Foto ${selectedPhotoIndex + 1}`}
                className="w-full rounded-2xl shadow-2xl"
              />
              <button
                onClick={() => setSelectedPhotoIndex(null)}
                className="absolute top-4 right-4 bg-black/70 hover:bg-black/90 text-white w-10 h-10 rounded-full flex items-center justify-center text-xl transition-all"
              >
                ✕
              </button>
              <div className="absolute bottom-4 left-1/2 transform -translate-x-1/2 bg-black/70 px-4 py-2 rounded-lg text-white text-sm">
                {photoMetadata[selectedPhotoIndex] ? (
                  `POSE ${photoMetadata[selectedPhotoIndex].pose} - Foto ${photoMetadata[selectedPhotoIndex].index_in_pose}/2`
                ) : (
                  `Foto ${selectedPhotoIndex + 1}`
                )}
              </div>
            </div>
          </div>
        )}
      </div>
    )
  }

  // ─── RENDER: MAIN UI ──────────────────────────────────────
  const targetTotalPhotos = getTargetTotalPhotos()
  const isFormValidToSubmit = nama.trim() !== '' && nim.trim() !== '' && capturedBlobs.length >= targetTotalPhotos

  return (
    <div className="min-h-screen bg-gradient-to-br from-black via-red-950 to-black text-white selection:bg-red-600 selection:text-white pb-12">
      
      <nav className="bg-white backdrop-blur px-6 py-3.5 flex items-center justify-between sticky top-0 z-50">
        <div className="flex items-center gap-3">
          <img src={logo} alt="logo" className="w-8 h-8 object-contain" />
          <div>
            <p className="text-black font-bold text-sm leading-tight">Face Recognition Gate</p>
            <p className="text-gray-900 text-[11px]">Biometric Registration</p>
          </div>
        </div>
        <button
          onClick={() => { closeCamera(); navigate('/dashboard') }}
          className="text-gray-700 hover:text-red-500 text-sm font-semibold transition-all duration-200 hover:scale-105"
        >
          ⮜ Back
        </button>
      </nav>

      <div className={`p-6 max-w-2xl mx-auto flex flex-col gap-5 mt-4 transition-all duration-700 ${
        visible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-10'
      }`}>
        
        <div className="bg-gray-500/80 backdrop-blur-sm rounded-2xl border border-gray-800 p-5 shadow-xl">
          <p className="font-bold text-white text-base tracking-wide">Register New User</p>
        </div>

        <div className="bg-gray-500/80 backdrop-blur-sm rounded-2xl border border-gray-800 p-5 flex flex-col gap-4 shadow-xl">
          <div>
            <label className="text-white text-xs uppercase tracking-wider mb-1.5 block font-medium">Full Name</label>
            <input
              type="text"
              disabled={cameraOpen}
              value={nama}
              onChange={e => setNama(e.target.value)}
              placeholder="Enter full name..."
              className="w-full bg-black/30 border border-gray-600 text-white text-sm rounded-xl px-4 py-3 outline-none focus:ring-2 focus:ring-red-500/80 placeholder-gray-400 disabled:opacity-50 transition-all"
            />
          </div>
          <div>
            <label className="text-white text-xs uppercase tracking-wider mb-1.5 block font-medium">NIM</label>
            <input
              type="text"
              disabled={cameraOpen}
              value={nim}
              onChange={e => setNim(e.target.value)}
              placeholder="Enter NIM..."
              className="w-full bg-black/30 border border-gray-600 text-white text-sm rounded-xl px-4 py-3 outline-none focus:ring-2 focus:ring-red-500/80 placeholder-gray-400 disabled:opacity-50 transition-all"
            />
          </div>
        </div>

        <div className="bg-gray-500/80 backdrop-blur-sm rounded-2xl border border-gray-800 p-5 flex flex-col gap-4 shadow-xl">
          <div className="flex justify-between items-center">
            <div>
              <p className="text-white text-sm font-semibold">Biometric Scan</p>
              <p className="text-gray-300 text-xs mt-0.5">
                {capturedBlobs.length > 0 ? `Captured: ${capturedBlobs.length}/${targetTotalPhotos} face samples` : 'No face samples captured yet.'}
              </p>
            </div>
            {capturedBlobs.length >= targetTotalPhotos && (
              <span className="text-xs font-bold px-3 py-1 rounded-full bg-green-900/60 text-green-400 border border-green-500/40">
                ✓ Samples Complete
              </span>
            )}
          </div>

          <button
            type="button"
            onClick={openCamera}
            disabled={cameraOpen}
            className="w-full flex items-center justify-center gap-2 bg-red-700 hover:bg-red-600 text-white font-semibold py-3.5 rounded-xl transition-all duration-200 disabled:bg-gray-700 disabled:text-gray-500 shadow-lg shadow-red-900/30"
          >
            {capturedBlobs.length > 0 ? 'Rescan Face' : 'Start Camera & Scan Face'}
          </button>
        </div>

        {status && (
          <div className={`p-4 rounded-xl border text-sm font-medium ${
            status.type === 'loading' ? 'bg-blue-900/20 border-blue-800/50 text-blue-400' :
            status.type === 'success' ? 'bg-green-900/20 border-green-800/50 text-green-400' :
            'bg-red-900/20 border-red-800/50 text-red-400'
          }`}>
            {status.msg}
          </div>
        )}

        {capturedBlobs.length >= targetTotalPhotos && (
          <button
            onClick={handleSubmitRegistration}
            disabled={!isFormValidToSubmit || status?.type === 'loading'}
            className="w-full bg-emerald-700 hover:bg-emerald-600 text-white font-bold py-4 rounded-xl transition-all duration-200 tracking-wide disabled:bg-gray-700 disabled:text-gray-500 shadow-lg shadow-emerald-900/30"
          >
            REGISTER NOW
          </button>
        )}
      </div>

      {renderGridPreview()}

      {cameraOpen && (
        <div className="fixed inset-0 bg-black flex flex-col z-50 animate-fade-in">
          
          <div className="bg-white backdrop-blur px-6 py-3.5 flex items-center justify-between border-b border-gray-200 shadow-md z-20">
            <div className="flex items-center gap-3">
              <img src={logo} alt="logo" className="w-8 h-8 object-contain" />
              <div>
                <p className="text-black font-bold text-sm leading-tight">Face Recognition Gate</p>
                <p className="text-gray-900 text-[11px]">
                  Biometric Scan Registration {isKacamataMode && <span className="text-blue-600 text-xs font-mono bg-blue-100 border border-blue-200 px-2 py-0.5 rounded-md ml-1">👓 Mode</span>}
                </p>
              </div>
            </div>
            <div className="flex items-center gap-3">
              <button
                onClick={closeCamera}
                className="text-gray-700 hover:text-red-500 text-sm font-semibold transition-all duration-200 hover:scale-105"
              >
                ✕ Close
              </button>
            </div>
          </div>

          <div className="flex-1 flex flex-col lg:flex-row overflow-hidden relative">
            
            <div className="w-full lg:w-[320px] bg-gray-500/80 backdrop-blur-sm border-b lg:border-b-0 lg:border-r border-gray-700 p-6 flex flex-col gap-5 overflow-y-auto z-10 shadow-xl">
              <h2 className="text-gray-300 font-bold text-xs tracking-widest uppercase border-b border-gray-700 pb-3 mb-2">Scan Status</h2>
              
              <div className="flex flex-col gap-4">
                <div className="bg-black/30 border border-gray-700 rounded-xl px-4 py-3 flex items-center justify-between shadow-inner">
                  <span className="text-gray-300 font-medium text-xs">Images Cached</span>
                  <span className="text-blue-400 font-bold bg-blue-900/60 px-3 py-1 rounded-md border border-blue-500/30 text-sm">
                    {capturedBlobs.length} / {targetTotalPhotos}
                  </span>
                </div>
                
                <div className="bg-black/30 border border-gray-700 rounded-xl px-4 py-3 flex items-center justify-between shadow-inner">
                  <span className="text-gray-300 font-medium text-xs">Session Stage</span>
                  <span className="text-purple-400 font-bold bg-purple-900/60 px-3 py-1 rounded-md border border-purple-500/30 text-xs uppercase tracking-wider">
                    {isKacamataMode ? 'Glasses (2/2)' : 'Normal (1/2)'}
                  </span>
                </div>
              </div>

              {FACE_ID_STEPS[currentStepIndex] && (
                <div className="mt-2 flex flex-col gap-3 bg-black/30 border border-gray-700 rounded-xl p-4 shadow-lg relative overflow-hidden">
                  <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-red-500 to-red-700" />
                  <span className="text-[10px] font-extrabold text-red-400 tracking-widest uppercase flex items-center gap-2">
                     <span className="w-1.5 h-1.5 rounded-full bg-red-400 animate-ping" />
                     TARGET POSE
                  </span>
                  <div className="flex justify-between items-end">
                    <span className="text-white font-bold text-xl tracking-wide">
                      {FACE_ID_STEPS[currentStepIndex].label}
                    </span>
                    <span className="text-gray-400 text-xs font-mono bg-black/50 px-2 py-1 rounded border border-gray-700 font-bold">
                      {photosCountInCurrentStep}/2
                    </span>
                  </div>
                </div>
              )}

              <div className={`mt-auto p-4 rounded-xl border transition-all duration-300 flex flex-col gap-2 ${
                feedbackType === 'success' 
                  ? 'bg-green-900/30 border-green-500/40 shadow-[0_0_15px_rgba(16,185,129,0.05)]' 
                  : feedbackType === 'error'
                  ? 'bg-red-900/30 border-red-500/40 shadow-[0_0_15px_rgba(255,0,0,0.05)]'
                  : 'bg-amber-900/30 border-amber-500/40 shadow-[0_0_15px_rgba(245,158,11,0.05)]'
              }`}>
                <div className="flex items-center gap-2">
                  <span className={`w-2.5 h-2.5 rounded-full ${
                    feedbackType === 'success' 
                      ? 'bg-green-400 shadow-[0_0_8px_#10b981]' 
                      : feedbackType === 'error'
                      ? 'bg-red-400 shadow-[0_0_8px_#ff0000]'
                      : 'bg-amber-400 shadow-[0_0_8px_#f59e0b]'
                  }`} />
                  <span className={`text-[10px] font-bold uppercase tracking-wider ${
                    feedbackType === 'success' 
                      ? 'text-green-400' 
                      : feedbackType === 'error'
                      ? 'text-red-400'
                      : 'text-amber-400'
                  }`}>
                    {feedbackType === 'error' ? 'System Alert' : 'System Feedback'}
                  </span>
                </div>
                <p className={`text-sm font-semibold leading-snug ${
                  feedbackType === 'success' 
                    ? 'text-green-300' 
                    : feedbackType === 'error'
                    ? 'text-red-300'
                    : 'text-amber-300'
                }`}>
                  {faceStatusText}
                </p>
              </div>
            </div>

            <div className="flex-1 relative bg-black flex items-center justify-center p-6">
              <div className="relative overflow-hidden rounded-2xl border-4 border-gray-700 shadow-[0_0_50px_rgba(255,77,79,0.05)] flex items-center justify-center bg-black w-[640px] h-[480px]">
                
                <video
                  ref={videoRef}
                  autoPlay
                  playsInline
                  muted
                  className="block w-full h-full scale-x-[-1] object-cover"
                  style={{ minWidth: '480px', minHeight: '360px' }}
                />

                {flashEffect && (
                  <div 
                    className="absolute inset-0 bg-white pointer-events-none z-20 transition-opacity duration-75"
                    style={{ opacity: 0.8 }}
                  />
                )}

                {showRulesOverlay && (
                  <div className="absolute inset-0 bg-black/85 backdrop-blur-sm flex flex-col items-center justify-center p-6 z-40 animate-fade-in">
                    <div className="bg-gray-500/80 backdrop-blur-sm border border-gray-700 rounded-2xl p-6 w-[85%] max-w-md shadow-2xl flex flex-col gap-4">
                      <h3 className="text-white font-bold text-lg border-b border-gray-700 pb-3 text-center">
                        Pre-Registration Rules
                      </h3>
                      <div className="flex flex-col gap-3 mt-1">
                        <div className="flex gap-3 bg-black/30 border border-gray-700 p-3 rounded-xl">
                          <span className="w-5 h-5 rounded bg-red-900/30 border border-red-500/30 flex items-center justify-center text-[11px] font-bold text-red-400 shrink-0">1</span>
                          <p className="text-gray-300 text-xs leading-relaxed">Avoid heavy backlighting or deep shadows. Ensure your entire face is clearly illuminated.</p>
                        </div>
                        <div className="flex gap-3 bg-black/30 border border-gray-700 p-3 rounded-xl">
                          <span className="w-5 h-5 rounded bg-red-900/30 border border-red-500/30 flex items-center justify-center text-[11px] font-bold text-red-400 shrink-0">2</span>
                          <p className="text-gray-300 text-xs leading-relaxed">Temporarily remove eyeglasses, face masks, caps, or heavy accessories.</p>
                        </div>
                        <div className="flex gap-3 bg-black/30 border border-gray-700 p-3 rounded-xl">
                          <span className="w-5 h-5 rounded bg-red-900/30 border border-red-500/30 flex items-center justify-center text-[11px] font-bold text-red-400 shrink-0">3</span>
                          <p className="text-gray-300 text-xs leading-relaxed">Once you align into the requested pose, the ring turns <span className="text-green-400">GREEN</span> and auto-captures.</p>
                        </div>
                      </div>
                      <button
                        onClick={() => {
                          updateOverlayState(setShowRulesOverlay, showRulesOverlayRef, false)
                          isCapturingLockRef.current = false
                          setFaceStatusText('👀 Siap scan: posisikan wajah di tengah lingkaran dan ikuti instruksi pose.')
                          setFeedbackType('info')
                        }}
                        className="w-full bg-red-700 hover:bg-red-600 text-white font-bold py-3.5 rounded-xl transition-all text-sm mt-3 shadow-lg shadow-red-900/30"
                      >
                        I Understand, Ready to Scan!
                      </button>
                    </div>
                  </div>
                )}

                {showKacamataQuestion && (
                  <div className="absolute inset-0 bg-black/85 backdrop-blur-sm flex flex-col items-center justify-center p-6 z-40 animate-fade-in">
                    <div className="bg-gray-500/80 backdrop-blur-sm border border-gray-700 rounded-2xl p-6 w-[80%] max-w-sm text-center shadow-2xl flex flex-col gap-5">
                      <div className="w-16 h-16 rounded-full bg-blue-900/50 border border-blue-500/30 flex items-center justify-center text-2xl mx-auto text-blue-400">
                        ⎚-⎚
                      </div>
                      <div>
                        <h3 className="text-white font-bold text-lg">Do you wear glasses?</h3>
                        <p className="text-gray-300 text-xs mt-1.5 leading-relaxed">
                          To maximize identification accuracy under various conditions, please verify if you wear glasses.
                        </p>
                      </div>
                      <div className="grid grid-cols-2 gap-3 mt-2">
                        <button
                          type="button"
                          onClick={() => handleKacamataAnswer('no')}
                          className="bg-black/30 hover:bg-gray-700 text-white font-semibold py-3 rounded-xl transition-all border border-gray-700 text-sm"
                        >
                          No, I Don't.
                        </button>
                        <button
                          type="button"
                          onClick={() => handleKacamataAnswer('yes')}
                          className="bg-blue-600 hover:bg-blue-500 text-white font-semibold py-3 rounded-xl transition-all border border-blue-500 text-sm"
                        >
                          Yes, I Do!
                        </button>
                      </div>
                    </div>
                  </div>
                )}

                {showPutOnGlassesInstruction && (
                  <div className="absolute inset-0 bg-black/85 backdrop-blur-sm flex flex-col items-center justify-center p-6 z-40 animate-fade-in">
                    <div className="bg-gray-500/80 backdrop-blur-sm border border-gray-700 rounded-2xl p-6 w-[85%] max-w-sm text-center shadow-2xl flex flex-col gap-5">
                      <div className="w-16 h-16 rounded-full bg-amber-900/50 border border-amber-500/30 flex items-center justify-center text-2xl mx-auto text-amber-400 animate-bounce">
                        ⎚-⎚
                      </div>
                      <div>
                        <h3 className="text-white font-bold text-lg">Put Your Glasses On</h3>
                        <p className="text-gray-300 text-xs mt-1.5 leading-relaxed">
                          Please wear your glasses now before the second scanning phase begins. Make sure there is no strong glare on the lenses.
                        </p>
                      </div>
                      <button
                        type="button"
                        onClick={handleStartGlassesScanningPhase}
                        className="w-full bg-blue-600 hover:bg-blue-500 text-white font-semibold py-3.5 rounded-xl transition-all border border-blue-500 text-sm mt-2 shadow-lg shadow-blue-500/20 font-bold"
                      >
                        I'm Ready, Start Scan!
                      </button>
                    </div>
                  </div>
                )}

                <div className="absolute inset-0 flex items-center justify-center pointer-events-none z-10">
                  <div className={`w-64 h-64 border-[4px] rounded-full transition-all duration-300 ${
                    (isFaceValid || isCountingDown)
                      ? 'border-green-400 bg-green-500/5 shadow-[0_0_50px_rgba(16,185,129,0.2)]' 
                      : 'border-red-500 bg-red-500/5 shadow-[0_0_30px_rgba(255,77,79,0.15)]'
                  }`}>
                    <div className="absolute inset-0 rounded-full border border-dashed border-white/20 scale-95" />
                  </div>
                </div>

                {countdown !== null && countdown > 0 && (
                  <div className="absolute inset-0 flex items-center justify-center pointer-events-none z-15">
                    <div className="text-white font-bold" style={{ fontSize: '120px', textShadow: '0 0 60px rgba(255,255,255,0.5), 0 0 120px rgba(255,255,255,0.2)' }}>
                      {countdown}
                    </div>
                  </div>
                )}

                {countdown === 0 && (
                  <div className="absolute inset-0 flex items-center justify-center pointer-events-none z-15">
                    <div className="text-green-400 font-bold" style={{ fontSize: '80px', textShadow: '0 0 60px rgba(16,185,129,0.5)' }}>
                      📸
                    </div>
                  </div>
                )}
                
              </div>
            </div>

          </div>

          <canvas ref={canvasRef} className="hidden" />
        </div>
      )}
      
      <style>{`
        @keyframes fade-in {
          from { opacity: 0; }
          to { opacity: 1; }
        }
        .animate-fade-in {
          animation: fade-in 0.3s ease-out;
        }
      `}</style>
    </div>
  )
}