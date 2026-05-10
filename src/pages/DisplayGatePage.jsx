import { useState, useEffect, useRef } from 'react'
import logo from '../assets/logo-telkom.png'

function DisplayGatePage() {
  // ========== STATE MANAGEMENT ==========
  const [frame, setFrame] = useState(null)
  const [isConnected, setIsConnected] = useState(false)
  const [currentDetection, setCurrentDetection] = useState(null)
  const [showOverlay, setShowOverlay] = useState(false)
  const [stats, setStats] = useState({ granted: 0, denied: 0, total: 0 })
  const [recentLogs, setRecentLogs] = useState([])
  const [systemStatus, setSystemStatus] = useState('menunggu')
  const [fps, setFps] = useState(0)

  // Liveness state
  const [livenessPassed, setLivenessPassed] = useState(false)
  const [livenessChallenge, setLivenessChallenge] = useState('')
  const [livenessEar, setLivenessEar] = useState(0)
  const [livenessYaw, setLivenessYaw] = useState(0)
  const [livenessSmile, setLivenessSmile] = useState(0)

  // Refs
  const frameCountRef = useRef(0)
  const lastFpsUpdateRef = useRef(Date.now())
  const overlayTimeoutRef = useRef(null)
  const statusPrevRef = useRef(null)

  // Cooldown per NIM - sinkron dengan Dashboard & backend: 10 detik
  const nimCooldownRef = useRef({})
  const COOLDOWN_MS = 10000

  // ========== WEBSOCKET CONNECTION ==========
  useEffect(() => {
    const wsUrl = 'ws://100.89.141.47:8000/ws/detect'
    const ws = new WebSocket(wsUrl)

    ws.onopen = () => {
      console.log('✅ WebSocket Connected')
      setIsConnected(true)
      setSystemStatus('terhubung')
    }

    ws.onclose = () => {
      console.log('❌ WebSocket Disconnected')
      setIsConnected(false)
      setSystemStatus('putus')
    }

    ws.onerror = (error) => {
      console.error('WebSocket Error:', error)
      setSystemStatus('error')
    }

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data)
        if (data.image) setFrame(data.image)

        // FPS counter
        frameCountRef.current++
        const now = Date.now()
        if (now - lastFpsUpdateRef.current >= 1000) {
          setFps(frameCountRef.current)
          frameCountRef.current = 0
          lastFpsUpdateRef.current = now
        }

        // ========== AMBIL DATA LIVENESS ==========
        if (data.liveness) {
          setLivenessPassed(data.liveness.passed || false)
          setLivenessChallenge(data.liveness.challenge || '')
          setLivenessEar(data.liveness.ear || 0)
          setLivenessYaw(data.liveness.yaw || 0)
          setLivenessSmile(data.liveness.smile || 0)
        }

        const currentStatus = data.status // 'idle', 'scanning', 'granted', 'denied'
        const prevStatus = statusPrevRef.current

        // Hanya proses jika status berubah
        if (currentStatus !== prevStatus) {
          statusPrevRef.current = currentStatus

          if (currentStatus === 'scanning' || currentStatus === 'idle') {
            if (overlayTimeoutRef.current) clearTimeout(overlayTimeoutRef.current)
            setShowOverlay(false)
          }

          if (currentStatus === 'granted') {
            const detection = {
              nama: data.user?.nama || 'Tidak Dikenali',
              nim: data.user?.nim || '-',
              status: 'granted',
              confidence: data.user?.confidence || 0,
              waktu: new Date().toLocaleTimeString('id-ID')
            }
            setCurrentDetection(detection)
            setSystemStatus('granted')
            if (overlayTimeoutRef.current) clearTimeout(overlayTimeoutRef.current)
            setShowOverlay(true)
            overlayTimeoutRef.current = setTimeout(() => setShowOverlay(false), 2000)
          }

          if (currentStatus === 'denied') {
            const detection = {
              nama: 'Tidak Dikenali',
              nim: '-',
              status: 'denied',
              confidence: 0,
              waktu: new Date().toLocaleTimeString('id-ID')
            }
            setCurrentDetection(detection)
            setSystemStatus('denied')
            if (overlayTimeoutRef.current) clearTimeout(overlayTimeoutRef.current)
            setShowOverlay(true)
            overlayTimeoutRef.current = setTimeout(() => setShowOverlay(false), 2000)
          }
        }

        // Update logs & stats — cooldown 10 detik per NIM
        if (data.faces && data.faces.length > 0) {
          if (currentStatus === 'granted' || currentStatus === 'denied') {
            const isGranted = currentStatus === 'granted'
            const cooldownKey = isGranted ? (data.user?.nim || 'UNKNOWN') : 'DENIED'
            const now2 = Date.now()
            const lastTime = nimCooldownRef.current[cooldownKey] || 0

            if (now2 - lastTime >= COOLDOWN_MS) {
              nimCooldownRef.current[cooldownKey] = now2
              const nama = isGranted ? (data.user?.nama || 'Tidak Dikenali') : 'Tidak Dikenali'
              const nim = isGranted ? (data.user?.nim || '-') : '-'
              const confidence = isGranted ? (data.user?.confidence || 0) : 0
              const newLog = {
                id: now2,
                nama: nama,
                nim: nim,
                status: isGranted ? 'GRANTED' : 'DENIED',
                waktu: new Date().toLocaleTimeString('id-ID'),
                confidence: (confidence * 100).toFixed(1)
              }
              setRecentLogs(prev => [newLog, ...prev].slice(0, 10))
              setStats(prev => ({
                granted: prev.granted + (isGranted ? 1 : 0),
                denied: prev.denied + (!isGranted ? 1 : 0),
                total: prev.total + 1
              }))
            }
          }
        }
      } catch (err) {
        console.error('Error parsing WebSocket message:', err)
      }
    }

    return () => {
      if (overlayTimeoutRef.current) clearTimeout(overlayTimeoutRef.current)
      if (ws.readyState === WebSocket.OPEN) ws.close()
    }
  }, [])

  // ========== HELPER FUNCTIONS ==========
  const formatTimestamp = () => new Date().toLocaleTimeString('id-ID', {
    hour: '2-digit', minute: '2-digit', second: '2-digit'
  })

  const getStatusColor = () => {
    switch (systemStatus) {
      case 'granted': return 'text-green-400'
      case 'denied': return 'text-red-400'
      case 'terhubung': return 'text-emerald-400'
      case 'putus': return 'text-red-400'
      case 'error': return 'text-amber-400'
      default: return 'text-slate-400'
    }
  }

  const getStatusBadge = () => {
    switch (systemStatus) {
      case 'granted': return { bg: 'bg-green-500/20', border: 'border-green-500/50', text: 'AKSES DIBERIKAN', icon: '✓' }
      case 'denied': return { bg: 'bg-red-500/20', border: 'border-red-500/50', text: 'AKSES DITOLAK', icon: '✗' }
      case 'terhubung': return { bg: 'bg-emerald-500/20', border: 'border-emerald-500/50', text: 'SISTEM SIAP', icon: '●' }
      case 'putus': return { bg: 'bg-red-500/20', border: 'border-red-500/50', text: 'KONEKSI PUTUS', icon: '⚠' }
      case 'error': return { bg: 'bg-amber-500/20', border: 'border-amber-500/50', text: 'ERROR SISTEM', icon: '!' }
      default: return { bg: 'bg-slate-500/20', border: 'border-slate-500/50', text: 'MENUNGGU DETEKSI', icon: '○' }
    }
  }

  const statusBadge = getStatusBadge()

  // Instruksi challenge yang ramah user
  const getChallengeInstruction = () => {
    switch (livenessChallenge) {
      case 'BLINK': return '👁️ Kedipkan mata'
      case 'HEAD': return '↔️ Gelengkan kepala'
      case 'SMILE': return '😊 Senyum'
      default: return ''
    }
  }

  // ========== RENDER ==========
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900">
      {/* NAVBAR */}
      <nav className="bg-slate-900/80 backdrop-blur-md border-b border-slate-700 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <div className="w-10 h-10 bg-gradient-to-br from-red-600 to-red-800 rounded-xl flex items-center justify-center shadow-lg">
                <img src={logo} alt="Logo" className="w-6 h-6 object-contain" />
              </div>
              <div>
                <h1 className="text-white font-bold text-lg tracking-tight">
                  Face Recognition Access Gate
                </h1>
                <p className="text-slate-400 text-xs">
                  Telkom University — Sistem Autentikasi Biometrik
                </p>
              </div>
            </div>
            <div className="flex items-center gap-4">
              <div className="hidden md:flex items-center gap-2 bg-slate-800/50 px-4 py-2 rounded-full border border-slate-700">
                <div className={`w-2 h-2 rounded-full animate-pulse ${isConnected ? 'bg-emerald-500' : 'bg-red-500'}`} />
                <span className="text-slate-300 text-xs font-mono">
                  {isConnected ? `ONLINE · ${fps} FPS` : 'OFFLINE'}
                </span>
              </div>
              <div className="text-right">
                <p className="text-slate-400 text-[10px] uppercase tracking-wider">Waktu Sistem</p>
                <p className="text-white text-sm font-mono">{formatTimestamp()}</p>
              </div>
            </div>
          </div>
        </div>
      </nav>

      {/* MAIN CONTENT */}
      <div className="max-w-7xl mx-auto p-6">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">

          {/* LEFT COLUMN: VIDEO FEED */}
          <div className="lg:col-span-2 space-y-4">
            <div className="bg-slate-800/50 backdrop-blur-sm rounded-2xl border border-slate-700 overflow-hidden shadow-xl">
              <div className="px-5 py-3 bg-slate-800/80 border-b border-slate-700 flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <div className={`w-2 h-2 rounded-full animate-pulse ${isConnected ? 'bg-emerald-500' : 'bg-red-500'}`} />
                  <p className="text-slate-300 text-sm font-medium">Live Feed — Gate 4</p>
                </div>
                <div className="flex items-center gap-3 text-xs">
                  <span className="text-slate-500">Status:</span>
                  <span className={`font-mono ${getStatusColor()}`}>
                    {systemStatus.toUpperCase()}
                  </span>
                </div>
              </div>

              <div className="relative aspect-video bg-slate-900">
                {frame ? (
                  <img src={frame} alt="Live Feed" className="w-full h-full object-cover" />
                ) : (
                  <div className="absolute inset-0 flex flex-col items-center justify-center">
                    <div className="w-16 h-16 bg-slate-800 rounded-2xl flex items-center justify-center mb-3">
                      <svg className="w-8 h-8 text-slate-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
                      </svg>
                    </div>
                    <p className="text-slate-500 text-sm">Menunggu koneksi kamera...</p>
                    <p className="text-slate-600 text-xs mt-1">WebSocket: {isConnected ? 'Terhubung' : 'Menghubungkan...'}</p>
                  </div>
                )}

                {/* Badge LIVE */}
                <div className="absolute top-4 left-4 flex items-center gap-2 bg-black/70 backdrop-blur-sm rounded-full px-3 py-1.5">
                  <div className="w-2 h-2 bg-red-500 rounded-full animate-pulse" />
                  <span className="text-white text-xs font-bold tracking-widest">LIVE</span>
                </div>

                {/* Badge Liveness Challenge (hanya muncul saat ada challenge dan belum lulus) */}
                {!livenessPassed && livenessChallenge && (
                  <div className="absolute bottom-4 left-4 bg-yellow-600/80 backdrop-blur-sm rounded-full px-4 py-2 shadow-lg">
                    <span className="text-white text-sm font-bold">
                      {getChallengeInstruction()}
                    </span>
                  </div>
                )}

                {/* OVERLAY GRANTED / DENIED */}
                {showOverlay && currentDetection && (systemStatus === 'granted' || systemStatus === 'denied') && (
                  <div className="absolute inset-0 flex items-center justify-center bg-black/60 backdrop-blur-sm animate-in fade-in duration-300">
                    <div className={`text-center p-8 rounded-2xl border-2 shadow-2xl transform transition-all duration-300 scale-100 ${
                      systemStatus === 'granted'
                        ? 'bg-green-900/90 border-green-500'
                        : 'bg-red-900/90 border-red-500'
                    }`}>
                      <div className={`text-6xl mb-3 ${systemStatus === 'granted' ? 'text-green-400' : 'text-red-400'}`}>
                        {systemStatus === 'granted' ? '✓' : '✗'}
                      </div>
                      <h2 className={`font-bold text-2xl tracking-wider mb-2 ${
                        systemStatus === 'granted' ? 'text-green-400' : 'text-red-400'
                      }`}>
                        {systemStatus === 'granted' ? 'ACCESS GRANTED' : 'ACCESS DENIED'}
                      </h2>
                      {systemStatus === 'granted' && (
                        <>
                          <p className="text-white text-xl font-semibold">{currentDetection.nama}</p>
                          <p className="text-slate-300 text-sm">{currentDetection.nim}</p>
                          <p className="text-slate-400 text-xs mt-2">
                            Similarity: {(Number(currentDetection.confidence) * 100).toFixed(2)}%
                          </p>
                        </>
                      )}
                      {systemStatus === 'denied' && (
                        <p className="text-slate-300 text-sm">Wajah tidak dikenali dalam database</p>
                      )}
                    </div>
                  </div>
                )}
              </div>
            </div>

            <div className="lg:hidden bg-slate-800/50 rounded-2xl border border-slate-700 p-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <div className={`w-2 h-2 rounded-full animate-pulse ${isConnected ? 'bg-emerald-500' : 'bg-red-500'}`} />
                  <span className="text-slate-300 text-sm">Status Koneksi</span>
                </div>
                <span className={`text-xs font-mono ${isConnected ? 'text-emerald-400' : 'text-red-400'}`}>
                  {isConnected ? `${fps} FPS` : 'Terputus'}
                </span>
              </div>
            </div>
          </div>

          {/* RIGHT COLUMN: INFO PANELS */}
          <div className="space-y-5">

            {/* Panel 1: Status Sistem (ditambah Liveness) */}
            <div className="bg-slate-800/50 backdrop-blur-sm rounded-2xl border border-slate-700 p-5 shadow-xl">
              <div className="flex items-center justify-between mb-4">
                <p className="text-slate-400 text-[10px] font-bold uppercase tracking-wider">
                  Status Sistem
                </p>
                <div className={`px-2 py-1 rounded-lg text-[10px] font-mono font-bold ${statusBadge.bg} ${statusBadge.border} ${getStatusColor()}`}>
                  {statusBadge.text}
                </div>
              </div>
              <div className="flex items-center gap-4">
                <div className={`w-14 h-14 rounded-2xl flex items-center justify-center text-2xl ${statusBadge.bg} ${statusBadge.border}`}>
                  {statusBadge.icon}
                </div>
                <div>
                  <p className="text-slate-400 text-xs">Deteksi Terakhir</p>
                  {currentDetection ? (
                    <>
                      <p className="text-white font-semibold text-lg">{currentDetection.nama}</p>
                      <p className="text-white font-semibold text-lg">{currentDetection.nim}</p>
                      <p className="text-slate-500 text-xs mt-1">{currentDetection.waktu}</p>
                    </>
                  ) : (
                    <p className="text-slate-500 text-sm">Belum ada deteksi</p>
                  )}
                </div>
              </div>

              {/* Liveness Info */}
              <div className="mt-4 pt-3 border-t border-slate-700/50">
                <div className="flex items-center justify-between">
                  <p className="text-slate-400 text-[10px] uppercase tracking-wider">Liveness Check</p>
                  {!livenessPassed && livenessChallenge && (
                    <span className="text-yellow-400 text-[9px] font-mono animate-pulse">WAITING</span>
                  )}
                </div>
                {!livenessPassed ? (
                  <div className="mt-1">
                    <p className="text-yellow-400 text-xs font-mono flex items-center gap-1">
                      <span>⚠️</span> Challenge: {livenessChallenge || '—'}
                    </p>
                    <p className="text-slate-500 text-[9px] mt-1">Lakukan gerakan untuk membuka akses</p>
                  </div>
                ) : (
                  <p className="text-green-400 text-xs font-mono flex items-center gap-1 mt-1">
                    ✓ Liveness verified
                  </p>
                )}
                {/* Opsional: tampilkan nilai sensor (untuk debug, bisa dihapus) */}
                <div className="grid grid-cols-3 gap-2 mt-2 text-[9px] text-slate-500">
                  <div className="flex flex-col items-center bg-slate-900/30 rounded p-1">
                    <span>👁️ Blink</span>
                    <span className="font-mono">{livenessEar.toFixed(2)}</span>
                  </div>
                  <div className="flex flex-col items-center bg-slate-900/30 rounded p-1">
                    <span>🔄 Head</span>
                    <span className="font-mono">{livenessYaw.toFixed(2)}</span>
                  </div>
                  <div className="flex flex-col items-center bg-slate-900/30 rounded p-1">
                    <span>😊 Smile</span>
                    <span className="font-mono">{livenessSmile.toFixed(2)}</span>
                  </div>
                </div>
              </div>
            </div>

            {/* Panel 2: Statistik (tidak berubah) */}
            <div className="bg-slate-800/50 backdrop-blur-sm rounded-2xl border border-slate-700 p-5 shadow-xl">
              <p className="text-slate-400 text-[10px] font-bold uppercase tracking-wider mb-4">
                Statistik Hari Ini
              </p>
              <div className="grid grid-cols-3 gap-3">
                <div className="text-center p-3 bg-slate-900/50 rounded-xl">
                  <p className="text-2xl font-bold text-green-400">{stats.granted}</p>
                  <p className="text-slate-500 text-[10px] font-medium mt-1">GRANTED</p>
                </div>
                <div className="text-center p-3 bg-slate-900/50 rounded-xl">
                  <p className="text-2xl font-bold text-red-400">{stats.denied}</p>
                  <p className="text-slate-500 text-[10px] font-medium mt-1">DENIED</p>
                </div>
                <div className="text-center p-3 bg-slate-900/50 rounded-xl">
                  <p className="text-2xl font-bold text-blue-400">{stats.total}</p>
                  <p className="text-slate-500 text-[10px] font-medium mt-1">TOTAL</p>
                </div>
              </div>
            </div>

            {/* Panel 3: Activity Log (tidak berubah) */}
            <div className="bg-slate-800/50 backdrop-blur-sm rounded-2xl border border-slate-700 shadow-xl overflow-hidden">
              <div className="px-5 py-3 border-b border-slate-700">
                <p className="text-slate-400 text-[10px] font-bold uppercase tracking-wider">
                  Aktivitas Terbaru
                </p>
              </div>
              <div className="max-h-64 overflow-y-auto">
                {recentLogs.length === 0 ? (
                  <div className="p-8 text-center">
                    <p className="text-slate-600 text-sm">Belum ada aktivitas</p>
                    <p className="text-slate-700 text-xs mt-1">Sistem menunggu deteksi wajah</p>
                  </div>
                ) : (
                  <div className="divide-y divide-slate-700/50">
                    {recentLogs.map((log) => (
                      <div key={log.id} className="px-5 py-3 hover:bg-slate-700/30 transition-colors">
                        <div className="flex items-center justify-between">
                          <div>
                            <p className="text-white text-sm font-medium">{log.nama}</p>
                            <p className="text-white text-sm font-medium">{log.nim}</p>
                            <p className="text-slate-500 text-[10px] font-mono">{log.waktu}</p>
                          </div>
                          <div className="text-right">
                            <span className={`text-[10px] font-bold px-2 py-1 rounded-lg ${
                              log.status === 'GRANTED'
                                ? 'bg-green-500/20 text-green-400 border border-green-500/30'
                                : 'bg-red-500/20 text-red-400 border border-red-500/30'
                            }`}>
                              {log.status}
                            </span>
                            {log.confidence && log.status === 'GRANTED' && (
                              <p className="text-slate-600 text-[9px] mt-1">{log.confidence}%</p>
                            )}
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>

            {/* Panel 4: Informasi Sistem (tidak berubah) */}
            <div className="bg-slate-800/50 backdrop-blur-sm rounded-2xl border border-slate-700 p-4 shadow-xl">
              <div className="flex items-center justify-between text-xs">
                <div className="flex items-center gap-2">
                  <div className={`w-1.5 h-1.5 rounded-full ${isConnected ? 'bg-emerald-500' : 'bg-red-500'}`} />
                  <span className="text-slate-500">WebSocket</span>
                </div>
                <span className={`font-mono ${isConnected ? 'text-emerald-400' : 'text-red-400'}`}>
                  {isConnected ? 'TERHUBUNG' : 'TERPUTUS'}
                </span>
              </div>
              <div className="mt-3 pt-3 border-t border-slate-700/50">
                <p className="text-slate-600 text-[9px] text-center">
                  Face Recognition Gate System v2.0 — Anti Spoof
                </p>
              </div>
            </div>

          </div>
        </div>
      </div>
    </div>
  )
}

export default DisplayGatePage